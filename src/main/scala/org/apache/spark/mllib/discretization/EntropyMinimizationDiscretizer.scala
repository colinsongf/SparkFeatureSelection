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
 * @param data RDD of LabeledPoint (in dense or sparse format)
 */
class EntropyMinimizationDiscretizer private (
    val data: RDD[LabeledPoint]) extends Serializable {

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
   * @param contFeat Subset of indexes to be considered (in case it is not specified, they are calculated).
   * @param nFeatures Total number of input features.
   * @param dense If the dataset is dense or not.
   * @return Indexes of continuous features.
   */  
  private def processContinuousAttributes(contFeat: Option[Seq[Int]], nFeatures: Int, dense: Boolean) = {
        // It generates pairs according to the data format.
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
      
      // (Pre-processing) Count the number of features and which one are continuous
      contFeat match {
          case Some(s) => 
            // Attributes are in range 0..nfeat
            val intersect = (0 until nFeatures).seq.intersect(s)
            require(intersect.size == s.size)
            s.toArray
          case None =>        
            val countFeat = calcRawData
                .distinct
                .mapValues(d => 1L)
                .reduceByKey(_ + _)
                .filter{case (_, c) => c > maxLimitBins}
            val cvars = countFeat.sortByKey().keys.collect()
            cvars       
      }
  }
  
  /**
   * Calculates the initial candidate thresholds for a feature
   * @param points RDD with distinct points for all features ((index, point), class frequency vector) to be evaluated.
   * @param firstElements first elements by partition (used to evaluate points in the partitions limit) 
   * @return RDD of boundary points (point, class frequency vector between the previous and the current candidate).
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
   * Evaluates boundary points and selects the most relevant for discretization.
   * This version is for huge sets of candidate points (distributed version).
   * @param candidates RDD of candidates points (value, label).
   * @param maxBins Maximum number of bins to select (# thresholds = maxBins - 1)
   * @param elementsByPart Maximum number of elements to evaluate in each partition.
   * @return Sequence of threshold values.
   */
  private def getThresholds(
      candidates: RDD[(Float, Array[Long])], 
      maxBins: Int, 
      elementsByPart: Int): Seq[Float] = {

    val partitions = { x: Long => math.ceil(x.toFloat / elementsByPart).toInt }
    
    // Create queue
    val stack = new mutable.Queue[((Float, Float), Option[Float])]

    // Insert the extreme values in the stack
    stack.enqueue(((Float.NegativeInfinity, Float.PositiveInfinity), None))
    var result = Seq.empty[Float]

    // As long as there are more elements to evaluate, we continue
    while(stack.length > 0 && result.size < maxBins){

      val (bounds, lastThresh) = stack.dequeue

      var cands = candidates.filter({ case (th, _) => th > bounds._1 && th <= bounds._2 })
      val nCands = cands.count
      if (nCands > 0) {
        cands = cands.coalesce(partitions(nCands))
        //Selects one threshold among the candidates and returns two partitions to recurse
        evalThresholds(cands, lastThresh, nLabels) match {
          case Some(th) =>
            result = th +: result
            stack.enqueue(((bounds._1, th), Some(th)))
            stack.enqueue(((th, bounds._2), Some(th)))
          case None => /* criteria not fulfilled, finish */
        }
      }
    }
    result.sorted
  }
  
  /**
   * Evaluates boundary points and selects the most relevant for discretization.
   * In this version the evaluation is performed locally as the number of points is sufficiently small.
   * @param candidates RDD of candidates points (value, label).
   * @param maxBins Maximum number of bins to select (# thresholds = maxBins - 1)
   * @return Sequence of threshold values.
   */
  private def getThresholds(
      candidates: Array[(Float, Array[Long])], 
      maxBins: Int): Seq[Float] = {

    // Create queue
    val stack = new mutable.Queue[((Float, Float), Option[Float])]

    // Insert first in the stack
    stack.enqueue(((Float.NegativeInfinity, Float.PositiveInfinity), None))
    var result = Seq.empty[Float]

    while(stack.length > 0 && result.size < maxBins){

      val (bounds, lastThresh) = stack.dequeue
      // Filter candidates within the last range added
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
    //(Float.PositiveInfinity +: result).sorted
    result.sorted
  }

  /**
   * Calculates entropy minimization for candidate points in a range and select the best one 
   * according to that measure when required (distributed version).
   * @param candidates RDD of candidate points.
   * @param lastSelected last selected threshold.
   * @param nLabels Number of classes.
   * @return The point with the minimum entropy
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
   * Calculates entropy minimization for candidate points in a range and select the best one 
   * according to that measure when required (sequential version).
   * @param candidates Array of candidate points.
   * @param lastSelected last selected threshold.
   * @param nLabels Number of classes.
   * @return The point with the minimum entropy
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
   * Run the entropy minimization discretizer on input data previously initialized.
   * @param contFeat Feature indexes to discretize (in case not specified, they are calculated).
   * @param elementsByPart Maximum number of elements to evaluate in each partition.
   * @param maxBins Maximum number of bins by feature.
   * @return A EntropyMinimizationDiscretizerModel with the discretization thresholds.
   */
  def runAll(contFeat: Option[Seq[Int]], 
      elementsByPart: Int,
      maxBins: Int) = {

      val sc = data.context 
      val nInstances = data.count
      val bLabels2Int = sc.broadcast(labels2Int)
      val classDistrib = data.map(d => bLabels2Int.value(d.label)).countByValue()
      val bclassDistrib = sc.broadcast(classDistrib)
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
      
      // Add zero elements for sparse data
      val zeros = nonzeros
            .map{case ((k, p), v) => (k, v)}
            .reduceByKey{ case (v1, v2) =>  (v1, v2).zipped.map(_ + _)}
            .map{case (k, v) => 
              val v2 = for(i <- 0 until v.length) yield bclassDistrib.value(i) - v(i)
              ((k, 0.0F), v2.toArray)
            }
            .filter{case (_, v) => v.sum > 0}
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
              
      println("Number of big features:\t" + bigIndexes.size)
      var bigThresholds = Map.empty[Int, Seq[Float]]
      for (k <- bigIndexes.keys){ 
         val cands = initialCandidates.filter{case (k2, _) => k == k2}.values.sortByKey()
         bigThresholds += ((k, getThresholds(cands, maxBins, elementsByPart)))
      }
  
      // Join the thresholds and return them
      val bigThRDD = sc.parallelize(bigThresholds.toSeq)
      val thresholds = smallThresholds.union(bigThRDD)
                          .sortByKey() // Important!
                          .collect
                          
      new EntropyMinimizationDiscretizerModel(thresholds)
  }

}

object EntropyMinimizationDiscretizer {

  /**
   * Train a EntropyMinimizationDiscretizerModel given an RDD of LabeledPoint's.
   * @param input RDD of LabeledPoint's.
   * @param continuousFeaturesIndexes Indexes of features to be discretized. 
   *  In case of empty value, the algorithm selects those features with more than 256 (byte range) distinct values.
   * @param maxBins Maximum number of bins for feature.
   * @param elementsPerPartition Maximum number of elements to manage in each RDD partition.
   * @return A EntropyMinimizationDiscretizerModel with the subsequent thresholds for discretization.
   */
  def train(
      input: RDD[LabeledPoint],
      continuousFeaturesIndexes: Option[Seq[Int]],
      maxBins: Int = Byte.MaxValue - Byte.MinValue + 1,
      elementsByPart: Int = 10000)
    : EntropyMinimizationDiscretizerModel = {

    new EntropyMinimizationDiscretizer(input).runAll(continuousFeaturesIndexes, elementsByPart, maxBins)

  }

}
