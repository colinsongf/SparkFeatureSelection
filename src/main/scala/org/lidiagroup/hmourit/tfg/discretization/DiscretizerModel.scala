package org.lidiagroup.hmourit.tfg.discretization

import org.apache.spark.rdd.RDD
import java.io.Serializable
import breeze.linalg.{SparseVector => BSV}

/**
 * DiscretizerModel provides a template with the basic methods for future discretizers.
 */
trait DiscretizerModel[T] extends Serializable {

  /**
   * Discretizes values for the given data set using the model trained.
   *
   * @param data RDD representing data points to discretize.
   * @return RDD with values discretized
   */
  def discretize(data: RDD[T]): RDD[T]
  
  def getThresholds: Array[(Int, Seq[Double])]

}
