package org.apache.spark.mllib.feature

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._

/**
 * This class provides the methods to discretize data, given a list of thresholds.
 * @param thresholds Thresholds by feature used to discretize (each one must be sorted)
 *  
 */
class DiscretizerModel (val thresholds: Array[(Int, Seq[Float])]) extends VectorTransformer {
  
  /**
   * Discretizes values in a given example using a set of thresholds.
   *
   * @param data Vector.
   * @return RDD with data discretized (with bins from 1 to n).
   */
  override def transform(data: Vector) = {
    // thresholds must be sorted by key index to perform the evaluation  
    data match {
      case SparseVector(size, indices, values) =>
        var newValues = Array.empty[Double]
        var i = 0
        var j = 0
        while(i < indices.size && j < thresholds.size){
          val ival = indices(i)
          val th = thresholds(j)
          if (ival < th._1) {
            newValues = values(i) +: newValues
            i += 1
          } else if (ival > th._1) {
            j += 1
          } else {                  
            newValues = assignDiscreteValue(values(i), th._2).toDouble +: newValues
            j += 1
            i += 1
          }
        }
        // the `index` array inside sparse vector object will not be changed,
        // so we can re-use it to save memory.
        Vectors.sparse(size, indices, newValues)
        
        case DenseVector(values) =>
          var newValues = Array.empty[Double]
          var i = 0
          var j = 0
          while(i < values.size && j < thresholds.size){
            val th = thresholds(j)
            if (i < th._1) {
              newValues = values(i) +: newValues
              i += 1
            } else if (i > th._1) {
              j += 1
            } else {                  
              newValues = assignDiscreteValue(values(i), th._2).toDouble +: newValues
              j += 1
              i += 1
            }
          }
          Vectors.dense(newValues)
    }    
  }

  /**
   * Discretizes values in a given dataset using a set of thresholds.
   *
   * @param data RDD with continuous-valued vectors.
   * @return RDD with data discretized (with bins from 1 to n).
   */
  override def transform(data: RDD[Vector]) = {
    // thresholds must be sorted by key index to perform the evaluation
    val bc_thresholds = data.context.broadcast(thresholds)    
    data.map {
      case SparseVector(size, indices, values) =>
  	  	var newValues = Array.empty[Double]
        var i = 0
        var j = 0
        while(i < indices.size && j < bc_thresholds.value.size){
          val ival = indices(i)
          val th = bc_thresholds.value(j)
          if (ival < th._1) {
            newValues = values(i) +: newValues
            i += 1
          } else if (ival > th._1) {
            j += 1
          } else {                  
            newValues = assignDiscreteValue(values(i), th._2).toDouble +: newValues
            j += 1
            i += 1
          }
  	  	}
  	  	// the `index` array inside sparse vector object will not be changed,
  	  	// so we can re-use it to save memory.
        Vectors.sparse(size, indices, newValues)
        
        case DenseVector(values) =>
          var newValues = Array.empty[Double]
          var i = 0
          var j = 0
          while(i < values.size && j < bc_thresholds.value.size){
            val th = bc_thresholds.value(j)
            if (i < th._1) {
              newValues = values(i) +: newValues
              i += 1
            } else if (i > th._1) {
              j += 1
            } else {                  
              newValues = assignDiscreteValue(values(i), th._2).toDouble +: newValues
              j += 1
              i += 1
              }
          }
          Vectors.dense(newValues)
    }    
  }

  /**
   * Discretizes a value with a set of intervals.
   *
   * @param value Value to be discretized
   * @param thresholds Thresholds used to assign a discrete value
   */
  private def assignDiscreteValue(value: Double, thresholds: Seq[Float]) = {
    if(thresholds.isEmpty) 1 else if (value > thresholds.last) thresholds.size + 1
      else thresholds.indexWhere{value <= _} + 1
  }

}
