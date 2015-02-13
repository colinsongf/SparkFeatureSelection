package org.lidiagroup.hmourit.tfg.discretization

import scala.collection.mutable
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._
import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.mllib.util.SearchUtils

/**
 * This class contains methods to calculate thresholds to discretize continuous values with the
 * method proposed by Fayyad and Irani in Multi-Interval Discretization of Continuous-Valued
 * Attributes (1993).
 *
 * @param data an RDD of LabeledPoint (in dense or sparse format)
 */
class EntropyMinimizationDiscretizer private (
    val data: RDD[LabeledPoint]) extends Serializable {

  //private val partitions = { x: Long => math.ceil(x.toFloat / elementsPerPartition).toInt }
  private val log2 = { x: Double => math.log(x) / math.log(2) }  
  private val isBoundary = (f1: Array[Long], f2: Array[Long]) => {
      (f1, f2).zipped.map(_ + _).filter(_ != 0).size > 1
  }
  private val maxLimitBins = Byte.MaxValue - Byte.MinValue + 1
  private val maxCandidates = 1e5
  
  val labels2Int = data.map(_.label).distinct.collect.zipWithIndex.toMap
  val nLabels = labels2Int.size
  
  /**
   * Get information about the attributes in order to perform a correct discretization process.
   * @param contFeat Indexes of features to be discretized (optional, in case not specified, they are calculated).
   * @param nFeatures Total number of input features
   * @param dense If the dataset is in dense format.
   * @return Indexes of the continuous features
   */  
  private def processContinuousAttributes(contFeat: Option[Seq[Int]], nFeatures: Int, dense: Boolean) = {
        // Function to calculate pairs according to the data format.
      def calcRawData = {
        dense match {
          case true =>
            data.flatMap({case LabeledPoint(label, values) =>
                for(k <- 0 until values.toArray.length) yield (k, values.toArray(k))
            })
          case false =>
            data.flatMap({case LabeledPoint(label, values) =>
                val v = values.asInstanceOf[SparseVector]
                for(i <- 0 until v.indices.size) yield (v.indices(i), v.values(i))
            })
        }    
      }
      
      // (Pre-processing) Count the number of features and what attributes are continuous
      contFeat match {
          case Some(s) => 
            // Attributes are in range 0..nfeat
            val intersect = (0 until nFeatures).seq.intersect(s)
            require(intersect.size == s.size)
            s.toArray
          case None =>        
            val freqCount = calcRawData
                .distinct
                .mapValues(d => 1L)
                .reduceByKey(_ + _)
                .filter{case (_, c) => c > maxLimitBins}
            val cvars = freqCount.sortByKey().keys.collect()
            cvars       
      }
  }
  
  /**
   * Calculates the initial candidate thresholds for a feature
   * @param data RDD (value, frequencies) of DISTINCT values for one particular feature.
   * @param firstElements first elements each partition (to be broadcasted to all partitions ) 
   * @return RDD of (boundary point, class frequencies between last and current candidate) pairs.
   */
  private def initialThresholds(
      points: RDD[((Int, Float), Array[Long])], 
      firstElements: Array[Option[(Int, Float)]]) = {
    
    val numPartitions = points.partitions.length
    val bcFirsts = points.context.broadcast(firstElements)      

    points.mapPartitionsWithIndex({ (index, it) =>
      
      if(it.hasNext) {
  
        var ((lastK, lastX), lastFreqs) = it.next()
        var result = Seq.empty[((Int, Float), Array[Long])]
        var accumFreqs = lastFreqs
      
        for (((k, x), freqs) <- it) {           
                if(k != lastK) {
                  // new attribute: add last point from the previous one
                  result = ((lastK, lastX), accumFreqs.clone) +: result
                  accumFreqs = Array.fill(nLabels)(0L)
                } else if(isBoundary(freqs, lastFreqs)) {
                  // new boundary point (mid point between this and the previous one)
                  result = ((lastK, (x + lastX) / 2), accumFreqs.clone) +: result
                  accumFreqs = Array.fill(nLabels)(0L)
                }
                
                lastK = k
                lastX = x
            lastFreqs = freqs
            accumFreqs = (accumFreqs, freqs).zipped.map(_ + _)
        }
       
        // Last point to end the set
        val lastPoint = if(index < (numPartitions - 1)) {
            bcFirsts.value(index + 1) match {
            case Some((k, x)) => 
              if(k != lastK) lastX else (x + lastX) / 2 
            case None => lastX // last point
            }
          }else{
            lastX // last partition
          } 
                    
        (((lastK, lastPoint), accumFreqs.clone) +: result).reverse.toIterator
        } else {
          Iterator.empty
        }             
    })
  }
    /**
   * Calculates class frequencies for each distinct point in the dataset
   * @param data RDD of (value, label) pairs.
   * @param nLabels Number of distinct labels in the dataset.
   * @return RDD of (point, class frequencies) pairs.
   *
   */
  private def countFrequencies(
    data: RDD[(Float, Int)],
  nLabels: Int) = {

    data.mapPartitions({ it =>
  
      def countFreq(
          it: Iterator[(Float, Int)],
          lastX: Float,
          accumFreqs: Array[Long]): Seq[(Float, Array[Long])] = {
  
          if (it.hasNext) {
            val (x, y) = it.next()
            if (x == lastX) {
            // same point than previous
              accumFreqs(y) += 1L
              countFreq(it, x, accumFreqs)
            } else {
              // new point
              val newAccum = Array.fill[Long](nLabels)(0L)
              newAccum(y) += 1L
              (lastX, accumFreqs.clone) +: countFreq(it, x, newAccum)
            }
          } else {
            Seq((lastX, accumFreqs))
          }
      }
  
      if (it.hasNext) {
        val (firstX, firstY) = it.next() // first element
        val accumFreqs = Array.fill[Long](nLabels)(0L)
        accumFreqs(firstY) += 1L
        countFreq(it, firstX, accumFreqs).toIterator
      } else {
        Iterator.empty
      }
    }, preservesPartitioning = true)

  } 
  
  /**
   * Returns a sequence of floats that define the intervals to discretize.
   *
   * @param candidates RDD of (value, label) pairs
   */
  private def getThresholds(
      candidates: RDD[(Float, Array[Long])], 
      maxBins: Int, 
      elementsPerPartition: Int): Seq[Float] = {

    val partitions = { x: Long => math.ceil(x.toFloat / elementsPerPartition).toInt }
    
    // Create queue
    val stack = new mutable.Queue[((Float, Float), Option[Float])]

    // Insert the extreme values in the stack
    stack.enqueue(((Float.NegativeInfinity, Float.PositiveInfinity), None))
    var result = Seq(Float.NegativeInfinity)

    // As long as there are more elements to evaluate, we continue
    while(stack.length > 0 && result.size < maxBins){

      val (bounds, lastThresh) = stack.dequeue

      var cands = candidates.filter({ case (th, _) => th > bounds._1 && th <= bounds._2 })
      val nCands = cands.count
      if (nCands > 0) {
        cands = cands.coalesce(partitions(nCands))

        evalThresholds(cands, lastThresh, nLabels) match {
          case Some(th) =>
            println("New point: " + th)
            result = th +: result
            stack.enqueue(((bounds._1, th), Some(th)))
            stack.enqueue(((th, bounds._2), Some(th)))
          case None => println("not fullfilled") /* criteria not fulfilled, finish */
        }
      }
    }
    (Float.PositiveInfinity +: result).sorted
  }
  
  /**
   * Returns a sequence of floats that define the intervals to discretize.
   *
   * @param candidates RDD of (value, label) pairs
   */
  private def getThresholds(
      candidates: Array[(Float, Array[Long])], 
      maxBins: Int): Seq[Float] = {

    // Create queue
    val stack = new mutable.Queue[((Float, Float), Option[Float])]

    // Insert first in the stack
    stack.enqueue(((Float.NegativeInfinity, Float.PositiveInfinity), None))
    var result = Seq(Float.NegativeInfinity)

    // While more elements to evaluate, continue
    while(stack.length > 0 && result.size < maxBins){

      val (bounds, lastThresh) = stack.dequeue
      val newCandidates = candidates.filter({ case (th, _) => th > bounds._1 && th <= bounds._2 })
      
      if (newCandidates.size > 0) {
        evalThresholds(newCandidates, lastThresh, nLabels) match {
          case Some(th) =>
            result = th +: result
            stack.enqueue(((bounds._1, th), Some(th)))
            stack.enqueue(((th, bounds._2), Some(th)))
          case None => /* criteria not fulfilled, finish */
        }
      }
    }
    (Float.PositiveInfinity +: result).sorted
  }

  /**
   * Selects one final thresholds among the candidates and returns two partitions to recurse
   * (calculation parallelized using several nodes)
   * @param candidates RDD of (candidate, class frequencies between last and current candidate)
   * @param lastSelected last selected threshold to avoid selecting it again
   */
  private def evalThresholds(
      candidates: RDD[(Float, Array[Long])],
      lastSelected : Option[Float],
      nLabels: Int) = {

    val numPartitions = candidates.partitions.size

    val sc = candidates.sparkContext

    // store total frequencies for each partition
    val totalsByPart = sc.runJob(candidates, { case it =>
      val accum = Array.fill(nLabels)(0L)
      for ((_, freqs) <- it) {
        for (i <- 0 until nLabels) accum(i) += freqs(i)
      }
      accum
    }: (Iterator[(Float, Array[Long])]) => Array[Long])
    
    var totals = Array.fill(nLabels)(0L)
    for (t <- totalsByPart) totals = (totals, t).zipped.map(_ + _)

    val bcTotalsByPart = sc.broadcast(totalsByPart)
    val bcTotals = sc.broadcast(totals)

    val result =
      candidates.mapPartitionsWithIndex({ (slice, it) =>

        // accumulate frequencies from left to right
        var leftTotal = Array.fill(nLabels)(0L)
        for (i <- 0 until slice) 
          leftTotal = (leftTotal, bcTotalsByPart.value(i)).zipped.map(_ + _)
        
        var entropyFreqs = Seq.empty[(Float, Array[Long], Array[Long], Array[Long])]

        for ((cand, freqs) <- it) {
          leftTotal = (leftTotal, freqs).zipped.map(_ + _)
          val rightTotal = (bcTotals.value, leftTotal).zipped.map(_ - _)
          entropyFreqs = (cand, freqs, leftTotal.clone, rightTotal) +: entropyFreqs
        }
        
        entropyFreqs.iterator
      })

    // calculate h(S)
    // s: number of elements
    // k: number of distinct classes
    // hs: entropy  
      
    val s  = totals.sum
    val hs = InfoTheory.entropy(totals.toSeq, s)
    val k  = totals.filter(_ != 0).size

    // select the best threshold according to the criteria
    val finalCandidates =
      result.flatMap({
        case (cand, _, leftFreqs, rightFreqs) =>

          val k1  = leftFreqs.filter(_ != 0).size
          val s1  = leftFreqs.sum
          val hs1 = InfoTheory.entropy(leftFreqs, s1)

          val k2  = rightFreqs.filter(_ != 0).size
          val s2  = rightFreqs.sum
          val hs2 = InfoTheory.entropy(rightFreqs, s2)

          val weightedHs = (s1 * hs1 + s2 * hs2) / s
          val gain       = hs - weightedHs
          val delta      = log2(math.pow(3, k) - 2) - (k * hs - k1 * hs1 - k2 * hs2)
          var criterion  = (gain - (log2(s - 1) + delta) / s) > -1e-5

          lastSelected match {
              case None =>
              case Some(last) => criterion = criterion && (cand != last)
          }

          if (criterion) Seq((weightedHs, cand)) else Seq.empty[(Double, Float)]
      })

      if (finalCandidates.count > 0) Some(finalCandidates.min._2) else None
  }
  
  /**
   * Selects one final thresholds among the candidates and returns two partitions to recurse
   * (calculation parallelized using only one node (the driver))
   * @param candidates RDD of (candidate, class frequencies between last and current candidate)
   * @param lastSelected last selected threshold to avoid selecting it again
   */
  private def evalThresholds(
      candidates: Array[(Float, Array[Long])],
      lastSelected : Option[Float],
      nLabels: Int): Option[Float] = {
    
    // Calculate total frequencies by label
    val totals = candidates
        .map(_._2)
        .reduce((freq1, freq2) => (freq1, freq2).zipped.map(_ + _))
    
    // Calculate partial frequencies (left and right to the candidate) by label
    var leftAccum = Array.fill(nLabels)(0L)
    var entropyFreqs = Seq.empty[(Float, Array[Long], Array[Long], Array[Long])]
    for(i <- 0 until candidates.size) {
      val (cand, freq) = candidates(i)
      leftAccum = (leftAccum, freq).zipped.map(_ + _)
      val rightTotal = (totals, leftAccum).zipped.map(_ - _)
      entropyFreqs = (cand, freq, leftAccum.clone, rightTotal) +: entropyFreqs
    }

    // calculate h(S)
    // s: number of elements
    // k: number of distinct classes
    // hs: entropy
    val s  = totals.sum
    val hs = InfoTheory.entropy(totals.toSeq, s)
    val k  = totals.filter(_ != 0).size

    // select best threshold according to the criteria
    val finalCandidates =
      entropyFreqs.flatMap({
        case (cand, _, leftFreqs, rightFreqs) =>

          val k1  = leftFreqs.filter(_ != 0).size
          val s1  = if (k1 > 0) leftFreqs.sum else 0
          val hs1 = InfoTheory.entropy(leftFreqs, s1)

          val k2  = rightFreqs.filter(_ != 0).size
          val s2  = if (k2 > 0) rightFreqs.sum else 0
          val hs2 = InfoTheory.entropy(rightFreqs, s2)

          val weightedHs = (s1 * hs1 + s2 * hs2) / s
          val gain       = hs - weightedHs
          val delta      = log2(math.pow(3, k) - 2) - (k * hs - k1 * hs1 - k2 * hs2)
          var criterion  = (gain - (log2(s - 1) + delta) / s) > -1e-5

          lastSelected match {
              case None =>
              case Some(last) => criterion = criterion && (cand != last)
          }

          if (criterion) {
            Seq((weightedHs, cand))
          } else {
            Seq.empty[(Double, Float)]
          }
      })
    
    // choose best candidates and partition data to make recursive calls
    if (finalCandidates.size > 0) Some(finalCandidates.min._2) else None
  }
 
  /**
   * Run the algorithm with the configured parameters on an input.
   * @param contFeat Indexes of features to be discretized (optional, in case not specified, they are calculated).
   * @param elementsPerPartition Maximum number of thresholds to treat in each RDD partition.
   * @param maxBins Maximum number of bins for each discretized feature.
   * @return A EntropyMinimizationDiscretizerModel with the thresholds to discretize.
   */
  def runAll(contFeat: Option[Seq[Int]], 
      elementsPerPartition: Int,
      maxBins: Int) = {

      val sc = data.context 
      val nInstances = data.count
      val bLabels2Int = sc.broadcast(labels2Int)
      val (dense, nFeatures) = data.first.features match {
        case v: DenseVector => 
          (true, v.size)
        case v: SparseVector =>         
            (false, v.size)
      }
      
      
      val continuousVars = processContinuousAttributes(contFeat, nFeatures, dense)
      
      println("Number of continuous attributes:" + continuousVars.distinct.size)
      println("Total number of attributes:" + nFeatures)
      
      if(continuousVars.isEmpty) 
          throw new IllegalStateException("There is no continuous attributes in the dataset")
      
      // Generate pairs ((attribute, value), class count)
      // In case of sparse data, we take into account whether 
      // the set of continuous attributes is too big to do an alternative process
      val featureValues = dense match{
        case true => 
          val bContinuousVars = sc.broadcast(continuousVars)
          data.flatMap({case LabeledPoint(label, values) =>
            val arr = values.toArray.map{case d => 
                // float precision
                d.toFloat
                //BigDecimal(d).setScale(6, BigDecimal.RoundingMode.HALF_UP).toFloat
            }
            
            bContinuousVars.value.map{ k =>
                val c = Array.fill[Long](nLabels)(0L)
                c(bLabels2Int.value(label)) = 1L
                ((k, arr(k)), c)
            }                   
          })
        case false =>
          val bContVars = sc.broadcast(continuousVars)
          
          data.flatMap({case LabeledPoint(label, values: SparseVector) =>
            val c = Array.fill[Long](nLabels)(0L)
            val arr = values.values.map{ case d => 
                d.toFloat
                //BigDecimal(d).setScale(6, BigDecimal.RoundingMode.HALF_UP).toFloat
            }
            c(bLabels2Int.value(label)) = 1L
            for(i <- 0 until values.indices.size 
                if SearchUtils.binarySearch(bContVars.value, values.indices(i))) 
              yield ((values.indices(i), arr(i)), c)            
            })
      }
    
      // Group elements by attribute and value (distinct values)
      val nonzeros = featureValues.reduceByKey{ case (v1, v2) => 
          (v1, v2).zipped.map(_ + _)
      }
      
      // Add zero elements just in case of sparse data
      val zeros = nonzeros
            .map{case ((k, p), v) => (k, v)}
            .reduceByKey{ case (v1, v2) =>  (v1, v2).zipped.map(_ + _)}
            .map{case (k, v) => ((k, 0.0F), v.map(s => nInstances - s))}
            .filter{case (k, v) => v.sum > 0}      
      val distinctValues = nonzeros.union(zeros)
    
      // Sort these values to perform the boundary points evaluation
      val sortedValues = distinctValues.sortByKey()   
          
      // Get the first elements by partition for the boundary points evaluation
      val firstElements = sc.runJob(sortedValues, { case it =>
          if (it.hasNext) Some(it.next()._1) else None
        }: (Iterator[((Int, Float), Array[Long])]) => Option[(Int, Float)])
      
      // Get only boundary points from the whole set of distinct values
      val initialCandidates = initialThresholds(sortedValues, firstElements)
              .map{case ((k, point), c) => (k, (point, c))}
              .cache() // It will be iterated through "big indexes" loop
      initialCandidates.checkpoint()
      
      // Divide RDD according to the number of candidates
      val bigIndexes = initialCandidates
          .countByKey()
          .filter{case (_, c) => c > maxCandidates}
      val bBigIndexes = sc.broadcast(bigIndexes)
      
      // Group by key those keys the small candidates and perform an iterative
      // and separate process for each big case.
      val smallCandidatesByAtt = initialCandidates
                      .filter{case (k, _) => !bBigIndexes.value.contains(k) }
                      .groupByKey()
                      .mapValues(_.toArray)
                      
      val smallThresholds = smallCandidatesByAtt
              .mapValues(points => getThresholds(points.toArray.sortBy(_._1), maxBins))
              
      //val bigInds = bigCandidates.keys.distinct.collect
      println("Number of big features:\t" + bigIndexes.size)
      var bigThresholds = Map.empty[Int, Seq[Float]]
      for (k <- bigIndexes.keys){ 
         val cands = initialCandidates.filter{case (k2, _) => k == k2}.values.sortByKey()
         bigThresholds += ((k, getThresholds(cands, maxBins, elementsPerPartition)))
      }
  
      // Join the thresholds and return them
      val bigThRDD = sc.parallelize(bigThresholds.toSeq)
      val thresholds = smallThresholds.union(bigThRDD)
                          .sortByKey() // Important
                          .collect
                          
      new EntropyMinimizationDiscretizerModel(thresholds)
  }

}

object EntropyMinimizationDiscretizer {

  /**
   * Train a EntropyMinimizationDiscretizerModel given an RDD of LabeledPoint's.
   * @param input RDD of LabeledPoint's.
   * @param continuousFeaturesIndexes Indexes of features to be discretized.
   * @param maxBins Maximum number of bins for feature.
   * @param elementsPerPartition Maximum number of thresholds to work with in each RDD partition.
   * @return A EntropyMinimizationDiscretizerModel with the thresholds to discretize.
   */
  def train(
      input: RDD[LabeledPoint],
      continuousFeaturesIndexes: Option[Seq[Int]],
      maxBins: Int = Byte.MaxValue - Byte.MinValue + 1,
      elementsPerPartition: Int = 10000)
    : EntropyMinimizationDiscretizerModel = {

    new EntropyMinimizationDiscretizer(input).runAll(continuousFeaturesIndexes, elementsPerPartition, maxBins)

  }

}
