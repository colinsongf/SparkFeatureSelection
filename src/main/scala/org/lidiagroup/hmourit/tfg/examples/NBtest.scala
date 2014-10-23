package org.lidiagroup.hmourit.tfg.examples
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.ugr.sci2s.mllib.test.{MultiClassificationUtils => MCU}
import org.lidiagroup.hmourit.tfg.discretization._
import org.lidiagroup.hmourit.tfg.featureselection._

object NBtest {

	def main(args: Array[String]) {
	  
		val initStartTime = System.nanoTime()

		if (args.length < 3) {
		  System.err.println("Usage: NBtest <header-file> <train-file> <test-file> <output-dir> [<lambda>]")
		  System.exit(1)
		}
		
		val headerFile = args(0)
		val trainFile = args(1)
		val testFile = args(2)
		val outputDir = args(3) + "/"
		val nonDefaut = args.length > 4
		val lambda = if (nonDefaut) args(4).toDouble else 1.0

		val conf = new SparkConf().setAppName("MLlib Benchmarking")
		val sc = new SparkContext(conf)
		
		def classify = (train: RDD[LabeledPoint]) => {
		  val model = NaiveBayes.train(train, lambda)
		  //calcThreshold(model, train)
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
		val algoInfo: String = s"Algorithm: Naive Bayes (NB)\n" + 
				s"lambda: $lambda\n\n"
				
		MCU.executeExperiment(classify, Some(discretization), Some(featureSelect), 
		    sc, headerFile, (trainFile, testFile), outputDir, algoInfo)
		sc.stop()
	}
    
}