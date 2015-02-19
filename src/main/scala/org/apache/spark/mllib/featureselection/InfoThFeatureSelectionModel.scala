package org.apache.spark.mllib.featureselection

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._

/**
 * Info-theory selection model with the subset of selected features.
 * @param features Features that can be selected (initialized with zero score).
 */
class InfoThFeatureSelectionModel (val features: Array[(Int, Double)])
    extends FeatureSelectionModel[LabeledPoint] with Serializable {

  /**
   * Applies trained model to select the most relevant features of each element of the RDD.
   * according to a criterion.
   * @param data RDD elements to reduce.
   * @return RDD elements projected in the new dimensional space.
   */
  override def select(data: LabeledPoint): LabeledPoint = {
    data match {
      case LabeledPoint(label, values) =>
        val array = values.toArray
        LabeledPoint(label, 
            Vectors.dense(features.map(f => array(f._1))))
    }
  }

  /**
   * Applies trained model to select the most relevant features of data
   * according to a criterion.
   * @param data Data point.
   * @return Data point projected in the new dimensional space.
   * 
   */
  def select(data: RDD[LabeledPoint]): RDD[LabeledPoint] = { data.map(select(_)) }
  
  /**
   * Get the selected subset of features.
   * @return Array with the indexes of the selected features along with its scores.
   */
  override def getSelection: Array[(Int, Double)] = features
}
