package org.lidiagroup.hmourit.tfg.featureselection

import org.apache.spark.rdd._

/**
 * FeatureSelectionModel provides an interface with basic methods for future Feature Selection
 * implementations.
 */
trait FeatureSelectionModel[T] extends Serializable{

  /**
   * Applies trained model to select the most relevant features of each element of the RDD.
   * according to a criterion.
   * @param data RDD elements to reduce.
   * @return RDD elements projected in the new dimensional space.
   */
  def select(data: RDD[T]): RDD[T]

  /**
   * Applies trained model to select the most relevant features of data
   * according to a criterion.
   * @param data Data point.
   * @return Data point projected in the new dimensional space.
   */
  def select(data: T): T
  
  /**
   * Get the current feature selection set and its scores.
   * @return Index and score for each selected feature.
   */
  def getSelection: Array[(Int, Double)]
}
