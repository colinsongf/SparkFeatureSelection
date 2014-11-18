package org.lidiagroup.hmourit.tfg.discretization

import scala.collection.mutable
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.regression.LabeledPoint
import org.lidiagroup.hmourit.tfg.featureselection.InfoThCriterion
import org.apache.spark.RangePartitioner
import org.apache.spark.RangePartitioner
import org.apache.spark.Partitioner

/**
 * This class contains methods to calculate thresholds to discretize continuous values with the
 * method proposed by Fayyad and Irani in Multi-Interval Discretization of Continuous-Valued
 * Attributes (1993).
 *
 * @param continuousFeaturesIndexes Indexes of features to be discretized.
 * @param elementsPerPartition Maximum number of thresholds to treat in each RDD partition.
 * @param maxBins Maximum number of bins for each discretized feature.
 */
class EntropyMinimizationDiscretizer private (
    val continuousFeaturesIndexes: Seq[Int],
    val elementsPerPartition: Int,
    val maxBins: Int)
  extends Serializable {

  private val partitions = { x: Long => math.ceil(x.toDouble / elementsPerPartition).toInt }
  private val log2 = { x: Double => math.log(x) / math.log(2) }
  
  private val isBoundary = (f1: Array[Long], f2: Array[Long]) => {
	  	(f1, f2).zipped.map(_ + _).filter(_ != 0).size > 1
  }

  /**
   * Run the algorithm with the configured parameters on an input.
   * @param data RDD of LabeledPoint's.
   * @return A EntropyMinimizationDiscretizerModel with the thresholds to discretize.
   */
  def run(data: RDD[LabeledPoint]) = {
    val labels2Int = data.context.broadcast(data.map(_.label).distinct.collect.zipWithIndex.toMap)
    val nLabels = labels2Int.value.size
    var thresholds = Map.empty[Int, Seq[Double]]
    
    for (i <- continuousFeaturesIndexes) {
    	val featureValues = data.map({ case LabeledPoint(label, values) => 
   	  		val frequencies = Array.fill[Long](nLabels)(0L)
   	  		frequencies(labels2Int.value(label)) = 1L
   	  		(values.toArray(i), frequencies)       	  
    	})
    	
    	// Group class values by feature
    	val distinctValues = featureValues.reduceByKey((_,_).zipped.map(_ + _))
		
    	// Sort these values to perform the boundary points evaluation
    	val sortedValues = distinctValues.sortByKey() 	

      // get the first range before reaching a boundary point in each partition
      // (including their class frequencies)
      val sc = data.context
      	
		val partitionsHeads = sc.runJob(sortedValues, { case it =>
		  def getPartitionHead(
			    it: Iterator[(Double, Array[Long])],
			    lastX: Double,
			    lastFreqs: Array[Long],
			    accumFreqs: Array[Long]): Seq[(Double, Array[Long])] = {
				
				if (it.hasNext) {
					val (x, freqs) = it.next()
		    		if(isBoundary(freqs, lastFreqs)) {
		    			// There are some points between the first one and the first boundary point
		    			if(accumFreqs.filter(_ == 0).size < nLabels) Seq((lastX, accumFreqs), (x, freqs))  
		    			else Seq((x, freqs))
		    		} else {
		    			getPartitionHead(it, x, freqs, (accumFreqs, freqs).zipped.map(_ + _))
		    		}
				} else {
					// No boundary point have been found
					if(accumFreqs.filter(_ == 0).size < nLabels) Seq((lastX, accumFreqs))
					else Seq.empty[(Double, Array[Long])]
				}			        		
			}
			
			val (firstX, firstFreqs) = it.next() // first element
			val head = getPartitionHead(it, firstX, firstFreqs, Array.fill(nLabels)(0L))
			
			(Seq((firstX, firstFreqs)) ++ head)
			
		}: (Iterator[(Double, Array[Long])]) => Seq[(Double, Array[Long])])
	    
  	  	// Get the boundary points with its frequencies
  	  	val initialCandidates = initialThresholds(sortedValues, partitionsHeads, nLabels)

  	  	val nCandsApprox = initialCandidates.countApprox(2000).getFinalValue.high
  	  	val thesholdsByFeature = if(nCandsApprox < 1e5) {
  	  		getThresholds(initialCandidates.collect(), nLabels) 
  	  	} else {
    	  	getThresholds(initialCandidates.persist(), nLabels)
  	  	}
        
    	thresholds += ((i, thesholdsByFeature))
    }

    new EntropyMinimizationDiscretizerModel(thresholds)

  }
  
  /**
   * Calculates the initial candidate thresholds for a feature
   * @param data RDD (value, frequencies) of DISTINCT values for one particular feature.
   * @param partitionsHeads 
   * @return RDD of (boundary point, class frequencies between last and current candidate) pairs.
   */
  private def initialThresholds(
      points: RDD[(Double, Array[Long])], 
      partitionsHeads:  Array[Seq[(Double, Array[Long])]],
      nLabels: Int) = {
	  
	val sc = points.context
    val numPartitions = points.partitions.length
    val bcPartitionsHeads = sc.broadcast(partitionsHeads)   		

    points.mapPartitionsWithIndex({ (index, it) =>
      
  	  	// Point to the first element in order to start the process and
      	// avoid overlapping between consecutive partitions
  	  	val headIT = bcPartitionsHeads.value(index)

	  	var (lastX, lastFreqs) = if(index > 0) headIT.last else headIT(0)
	  	var result = Seq.empty[(Double, Array[Long])]
		var accumFreqs = lastFreqs.clone
		      
		// We forward main iterator until the end of its associated head
  	  	def findFirstNonRepetedElement(
  	      iter: Iterator[(Double, Array[Long])]): 
  	    	  Iterator[(Double, Array[Long])] = {
  	    
      	    if(iter.hasNext) {
      	    	val (x, _) = iter.next()
      	    	if(x == lastX) iter
      	    	else findFirstNonRepetedElement(iter)
      	    } else {
      	    	Iterator.empty
      	    }      	    
	  	}		  
	  	
  		val forwardIT = if(index > 0) {
  		  findFirstNonRepetedElement(it)
  		} else {
  		  it.next()
  		  it
  		}
  		
  	  	if(forwardIT.hasNext) {
		      // Afterwards, we follow the same process over the rest of the partition
		      for ((x, freqs) <- forwardIT) {		        
		          if(isBoundary(freqs, lastFreqs)) {
		        	  // new boundary point
		        	  result = ((x + lastX) / 2, accumFreqs.clone) +: result
		        	  accumFreqs = Array.fill(nLabels)(0L)
		          }
		          
		          lastX = x
				  lastFreqs = freqs
				  accumFreqs = (accumFreqs, freqs).zipped.map(_ + _)
		      }    
		   
	    	  // We continue with the head of the following partitions until 
		      // reaching the last boundary point
		      var boundaryPartition = false
		      while ((index + 1) < numPartitions && !boundaryPartition) {
		    	  val extendedIT = bcPartitionsHeads.value(index + 1).toIterator
		    	  for ((x, freqs) <- extendedIT) {
			          if(isBoundary(freqs, lastFreqs)) {
			        	  // last boundary point (we only need one more)
			        	  result = ((x + lastX) / 2, accumFreqs.clone) +: result
			        	  accumFreqs = Array.fill(nLabels)(0L)
			        	  // false in case all points in the partitions has the same class frequency
			        	  boundaryPartition = true 
			          }
			          
			          lastX = x
					  lastFreqs = freqs
					  accumFreqs = (accumFreqs, freqs).zipped.map(_ + _)
		    	  }	
		      }
  	  	}
  	  	
  	  	result.reverse.toIterator
      	      
    })
  }
  
  /**
   * Returns a sequence of doubles that define the intervals to make the discretization.
   *
   * @param candidates RDD of (value, label) pairs
   */
  private def getThresholds(candidates: RDD[(Double, Array[Long])], nLabels: Int): Seq[Double] = {

    // Create queue
    val stack = new mutable.Queue[((Double, Double), Option[Double])]

    // Insert the extreme values in the stack
    stack.enqueue(((Double.NegativeInfinity, Double.PositiveInfinity), None))
    var result = Seq(Double.NegativeInfinity)

    // As long as there are more elements to evaluate, we continue
    while(stack.length > 0 && result.size < this.maxBins){

      val (bounds, lastThresh) = stack.dequeue

      var cands = candidates.filter({ case (th, _) => th > bounds._1 && th <= bounds._2 })
      val nCands = cands.count
      if (nCands > 0) {
        cands = cands.coalesce(partitions(nCands))

        evalThresholds(cands, lastThresh, nLabels) match {
          case Some(th) =>
            result = th +: result
            stack.enqueue(((bounds._1, th), Some(th)))
            stack.enqueue(((th, bounds._2), Some(th)))
          case None => /* criteria not fulfilled, finish */
        }
      }
    }
    (Double.PositiveInfinity +: result).sorted
  }
  
    /**
   * Returns a sequence of doubles that define the intervals to make the discretization.
   *
   * @param candidates RDD of (value, label) pairs
   */
  private def getThresholds(
      candidates: Array[(Double, Array[Long])], 
      nLabels: Int): Seq[Double] = {

    // Create queue
    val stack = new mutable.Queue[((Double, Double), Option[Double])]

    // Insert first in the stack
    stack.enqueue(((Double.NegativeInfinity, Double.PositiveInfinity), None))
    var result = Seq(Double.NegativeInfinity)

    // While more elements to eval, continue
    while(stack.length > 0 && result.size < this.maxBins){

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
    (Double.PositiveInfinity +: result).sorted
  }

  /**
   * Selects one final thresholds among the candidates and returns two partitions to recurse
   * (calculation parallelized using several nodes)
   * @param candidates RDD of (candidate, class frequencies between last and current candidate)
   * @param lastSelected last selected threshold to avoid selecting it again
   */
  private def evalThresholds(
      candidates: RDD[(Double, Array[Long])],
      lastSelected : Option[Double],
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
    }: (Iterator[(Double, Array[Long])]) => Array[Long])
    
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
        
        var entropyFreqs = Seq.empty[(Double, Array[Long], Array[Long], Array[Long])]

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

          if (criterion) Seq((weightedHs, cand)) else Seq.empty[(Double, Double)]
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
      candidates: Array[(Double, Array[Long])],
      lastSelected : Option[Double],
      nLabels: Int): Option[Double] = {
    
    // Calculate total frequencies by label
    val totals = candidates
    		.map(_._2)
    		.reduce((freq1, freq2) => (freq1, freq2).zipped.map(_ + _))
    
    // Calculate partial frequencies (left and right to the candidate) by label
    var leftAccum = Array.fill(nLabels)(0L)
    var entropyFreqs = Seq.empty[(Double, Array[Long], Array[Long], Array[Long])]
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
            Seq.empty[(Double, Double)]
          }
      })
    
    // choose best candidates and partition data to make recursive calls
    if (finalCandidates.size > 0) Some(finalCandidates.min._2) else None
  }

}

object EntropyMinimizationDiscretizer {

  /**
   * Train a EntropyMinimizationDiscretizerModel given an RDD of LabeledPoint's.
   * @param input RDD of LabeledPoint's.
   * @param continuousFeaturesIndexes Indexes of features to be discretized.
   * @param maxBins Maximum number of bins for each discretized feature.
   * @param elementsPerPartition Maximum number of thresholds to treat in each RDD partition.
   * @return A EntropyMinimizationDiscretizerModel which has the thresholds to discretize.
   */
  def train(
      input: RDD[LabeledPoint],
      continuousFeaturesIndexes: Seq[Int],
      maxBins: Int = Int.MaxValue,
      elementsPerPartition: Int = 20000)
    : EntropyMinimizationDiscretizerModel = {

    new EntropyMinimizationDiscretizer(continuousFeaturesIndexes, elementsPerPartition, maxBins)
      .run(input)

  }

}
