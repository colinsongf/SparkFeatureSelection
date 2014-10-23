package org.lidiagroup.hmourit.tfg.examples
import org.apache.spark.mllib.classification._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.ugr.sci2s.mllib.test.{MultiClassificationUtils => MCU}
import org.lidiagroup.hmourit.tfg.discretization._
import org.lidiagroup.hmourit.tfg.featureselection._

object SVMtest {
  
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


	def main(args: Array[String]) {
		
		val initStartTime = System.nanoTime()

		if (args.length < 3) {
		  System.err.println("Usage: SVMtest <header-file> <data-dir> <output-dir> [<numIter> <stepSize> <regParam> <miniBatchFraction>]")
		  System.exit(1)
		}
		
		val headerFile = args(0)
		val dataDir = args(1)
		val outputDir = args(2) + "/"
		val nonDefaut = args.length > 3
		val numIter = if(nonDefaut) args(3).toInt else 100 
		val stepSize = if(nonDefaut) args(4).toDouble else 1.0 
		val regParam = if(nonDefaut) args(5).toDouble else 1.0 
		val miniBatchFraction = if(nonDefaut) args(6).toDouble else 1.0
		val conf = new SparkConf().setAppName("MLlib Benchmarking")
		
		//conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
		//conf.set("spark.kryo.registrator", "mypackage.MyRegistrator")
		val sc = new SparkContext(conf)
		
		def classify = (train: RDD[LabeledPoint]) => {
		  val model = SVMWithSGD.train(train, numIter, stepSize, regParam, miniBatchFraction)
		  calcThreshold(model, train)
		  model
		}
		
		def discretization = (train: RDD[LabeledPoint]) => {
			val discretizer = EntropyMinimizationDiscretizer.train(train,
		      0 until train.first.features.size, // continuous features 
		      10) // max number of values per feature
		    discretizer.discretize(train)
		}
		
		def featureSelect = (data: RDD[LabeledPoint]) => {
			// Feature Selection
			val criterion = new InfoThCriterionFactory("jmi")
			val model = InfoThFeatureSelection.train(criterion, 
		      data, //data 
		      100) // number of features to select
		    model.select(data)
		}
		
		// Output of the algorithm
		val algoInfo: String = s"Algorithm: Support Vector Machine (SVM)\n" + 
				s"numIter: $numIter\n" +
				s"stepSize: $stepSize\n" + 
				s"regParam: $regParam\n" +
				s"miniBatchFraction: $miniBatchFraction\n\n"
		
		MCU.executeExperiment(classify, Some(discretization), Some(featureSelect), 
		    sc, headerFile, dataDir, outputDir, algoInfo)
		sc.stop()
	}
    
}