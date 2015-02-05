package org.lidiagroup.hmourit.tfg.discretization

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.util.SearchUtils

/**
 * This class provides the methods to discretize data with the given thresholds.
 * @param thresholds Thresholds used to discretize (must be sorted)
 *  
 */
class EntropyMinimizationDiscretizerModel (val thresholds: Array[(Int, Seq[Double])])
  extends DiscretizerModel[LabeledPoint] with Serializable {
  
  
  /*override def discretize(data: LabeledPoint): LabeledPoint = {
    
    val discFeatures = data.features match { 
      	case values: SparseVector =>
    	    val newValues = discretizeFeatures(values.indices, values.values, thresholds)
    	    Vectors.sparse(values.size, values.indices, newValues)
      	case values: DenseVector =>
      	  	val newValues = discretizeFeatures((0 until values.size).toArray, values.toArray, thresholds)
      	  	Vectors.dense(newValues)
    }
    
    LabeledPoint(data.label, discFeatures)
  }*/

  /**
   * Discretizes values for the given data set using the model trained.
   *
   * @param data RDD representing data points to discretize.
   * @return RDD with values discretized
   */
  override def discretize(data: RDD[LabeledPoint]) = {
    // must be sorted to perform the evaluation
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
    		        	case Some(th) => assignDiscreteValue(value, th).toDouble
    		        	case None => value
    		        }
	  	  	  })
      	  	
            LabeledPoint(label, Vectors.dense(newValues))
    }
  }


  /**
   * Discretizes a value with a set of intervals.
   *
   * @param value The value to be discretized
   * @param thresholds Thresholds used to assign a discrete value
   */
  private def assignDiscreteValue(value: Double, thresholds: Seq[Double]) = {
    var aux = thresholds.zipWithIndex
    while (value > aux.head._1) aux = aux.tail
    aux.head._2
  }
  
  override def getThresholds() = thresholds

}
