package org.ugr.sci2s.mllib.test

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.SVMModel

trait ClassifierAdapter {
  
	def classify (
	    train: RDD[LabeledPoint], 
	    parameters: Map[String, String]): ClassificationModel
	    
	def algorithmInfo (parameters: Map[String, String]): String

}