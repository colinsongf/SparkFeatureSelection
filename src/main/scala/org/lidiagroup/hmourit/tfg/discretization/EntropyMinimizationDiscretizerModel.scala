package org.lidiagroup.hmourit.tfg.discretization

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.DenseVector

/**
 * This class provides the methods to discretize data with the given thresholds.
 * @param thresholds Thresholds used to discretize.
 */
class EntropyMinimizationDiscretizerModel (val thresholds: Map[Int, Seq[Double]])
  extends DiscretizerModel[LabeledPoint] with Serializable {

  /**
   * Discretizes values for the given data set using the model trained.
   *
   * @param data Data point to discretize.
   * @return Data point with values discretized
   */
  override def discretize(data: LabeledPoint): LabeledPoint = {
    val dense = data.features match {
    	case _: DenseVector => true
    	case _ => false
    }
    val newValues = data.features.toArray.zipWithIndex.map({ case (value, i) =>
      val threshold = thresholds.get(i)
        threshold match {
        	case Some(th) => (i, assignDiscreteValue(value, th).toDouble)
        	case None => (i, value)
        }
    })
    if(dense) LabeledPoint(data.label, Vectors.dense(newValues.map(_._2).toArray)) else 
        LabeledPoint(data.label, Vectors.sparse(newValues.size, newValues.map(_._1), newValues.map(_._2)))
  }

  /**
   * Discretizes values for the given data set using the model trained.
   *
   * @param data RDD representing data points to discretize.
   * @return RDD with values discretized
   */
  override def discretize(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val bc_thresholds = data.context.broadcast(this.thresholds)
    val dense = data.first.features match {
    	case _: DenseVector => true
    	case _ => false
    }

    // applies thresholds to discretize every continuous feature
    data.map({ case LabeledPoint(label, values) =>
      val newValues = values.toArray.zipWithIndex.map({ case (value, i) =>
        val threshold = bc_thresholds.value.get(i)
        threshold match {
        	case Some(th) => (i, assignDiscreteValue(value, th).toDouble)
        	case None => (i, value)
        }
      })
      if(dense) LabeledPoint(label, Vectors.dense(newValues.map(_._2).toArray)) else 
        LabeledPoint(label, Vectors.sparse(newValues.size, newValues.map(_._1), newValues.map(_._2)))
    })
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
