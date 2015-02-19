package org.apache.spark.mllib.discretization

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.util.SearchUtils

/**
 * This class provides the methods to discretize data, given a list of thresholds.
 * @param thresholds Thresholds by feature used to discretize (each one must be sorted)
 *  
 */
class EntropyMinimizationDiscretizerModel (val thresholds: Array[(Int, Seq[Float])])
  extends DiscretizerModel[LabeledPoint] with Serializable {

  /**
   * Discretizes values in a given dataset using a set of thresholds.
   *
   * @param data RDD with continuous data.
   * @return RDD with data discretized (with bins from 1 to n).
   */
  override def discretize(data: RDD[LabeledPoint]) = {
    // thresholds must be sorted by key index to perform the evaluation
    val bc_thresholds = data.context.broadcast(thresholds)    
    data.map{
      	case LabeledPoint(label, values: SparseVector) =>
      	  	var newValues = Array.empty[Double]
      	  	val threshInds = bc_thresholds.value.map(_._1)
      			for(i <- 0 until values.indices.size) {
        				val ind = SearchUtils.binarySearch2(threshInds, values.indices(i))
        				if (ind == -1) {
        					newValues = values.values(i) +: newValues
        				} else {
        					newValues = assignDiscreteValue(values.values(i), 
        					    bc_thresholds.value(ind)._2).toDouble +: newValues
        				}
      	  	}
      	  	// the `index` array inside sparse vector object will not be changed,
      	  	// so we can re-use it to save memory.
      	    LabeledPoint(label, Vectors.sparse(values.size, values.indices, newValues))
        
  		  case LabeledPoint(label, values: DenseVector) =>
      	  	val threshInds = bc_thresholds.value.toMap
      	  	val newValues = values.toArray.zipWithIndex.map({ case (value, i) =>
      	  	  	threshInds.get(i) match {
    		        	case Some(th) => assignDiscreteValue(value, th)
    		        	case None => value
    		        }
	  	  	  })
      	  	
            LabeledPoint(label, Vectors.dense(newValues))
      }
  }


  /**
   * Discretizes a value with a set of intervals.
   *
   * @param value Value to be discretized
   * @param thresholds Thresholds used to assign a discrete value
   */
  private def assignDiscreteValue(value: Double, thresholds: Seq[Float]) = {
    if(thresholds.isEmpty) 1
        else if (value > thresholds.last) thresholds.size + 1
        else thresholds.indexWhere{value <= _} + 1
  }
  
  override def getThresholds() = thresholds

}