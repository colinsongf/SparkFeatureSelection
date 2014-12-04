package org.ugr.sci2s.mllib.test

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.NaiveBayes
import org.ugr.sci2s.mllib.test.{MultiClassificationUtils => MCU}
import org.apache.spark.mllib.classification.ClassificationModel

object NBadapter extends ClassifierAdapter {
  
	def algorithmInfo (parameters: Map[String, String]): String = {
		  val lambda = parameters.getOrElse("cls-lambda", "1.0")
		  s"Algorithm: Naive Bayes (NB)\nlambda: $lambda\n\n"		
	}
  
	def classify (train: RDD[LabeledPoint], parameters: Map[String, String]) : ClassificationModel = {
  		val lambda = MCU.toDouble(parameters.getOrElse("cls-lambda", "1.0"), 1.0)
		  val model = NaiveBayes.train(train, lambda)
		  model
	}

}