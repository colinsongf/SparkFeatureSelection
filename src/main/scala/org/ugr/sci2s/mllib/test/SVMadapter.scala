package org.ugr.sci2s.mllib.test

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.ugr.sci2s.mllib.test.{MultiClassificationUtils => MCU}
import org.apache.spark.mllib.classification.ClassificationModel

object SVMadapter extends ClassifierAdapter {
  
	def algorithmInfo (parameters: Map[String, String]): String = {
  		val numIter = parameters.getOrElse("cls-numIter", "1") // default: 100 
  		val stepSize = parameters.getOrElse("cls-stepSize", "1.0")
  		val regParam = parameters.getOrElse("cls-regParam", "1.0")
  		val miniBatchFraction = parameters.getOrElse("cls-miniBatchFraction", "1.0")
		
  		s"Algorithm: Support Vector Machine (SVM)\n" + 
			s"numIter: $numIter\n" +
			s"stepSize: $stepSize\n" + 
			s"regParam: $regParam\n" +
			s"miniBatchFraction: $miniBatchFraction\n\n"		
	}
  
	private def calcThreshold(model: SVMModel, 
	    data: RDD[LabeledPoint]): Unit = {
	  
  		// Clear the default threshold.
		model.clearThreshold()
  		// Compute raw scores on the test set. 
		val scoreAndLabels = data.map { point =>
		  val score = model.predict(point.features)
		  (score, point.label)
		}
		
		// Get evaluation metrics.
		val metrics = new BinaryClassificationMetrics(scoreAndLabels)
		val measuresByThreshold = metrics.fMeasureByThreshold.toArray
		val maxThreshold = measuresByThreshold.maxBy{_._2}
		
		//println("Max (Threshold, Precision):" + maxThreshold)
		model.setThreshold(maxThreshold._1)			  
	}
  
	def classify (train: RDD[LabeledPoint], parameters: Map[String, String]): ClassificationModel = {
  		val numIter = MCU.toInt(parameters.getOrElse("cls-numIter", "1"), 1) // default: 100 
		val stepSize = MCU.toDouble(parameters.getOrElse("cls-stepSize", "1.0"), 1.0)
		val regParam = MCU.toDouble(parameters.getOrElse("cls-regParam", "1.0"), 1.0)
		val miniBatchFraction = MCU.toDouble(parameters.getOrElse("cls-miniBatchFraction", "1.0"), 1.0)
		val model = SVMWithSGD.train(train, numIter, stepSize, regParam, miniBatchFraction)
		calcThreshold(model, train)
		model
	}

}