package org.lidiagroup.hmourit.tfg.examples
import org.apache.spark.mllib.classification._
import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.ugr.sci2s.mllib.test.{MultiClassificationUtils => MCU}
import org.lidiagroup.hmourit.tfg.discretization._
import org.lidiagroup.hmourit.tfg.featureselection._

class MyRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[Double])
    kryo.register(classOf[Array[Double]])
    kryo.register(classOf[LabeledPoint])
    
  }
}

object NBtest {

	def main(args: Array[String]) {
	  
		val initStartTime = System.nanoTime()

		if (args.length < 3) {
		  System.err.println("Usage: NBtest <header-file> <train-file> <test-file> <output-dir> <fs-criterion> [<lambda>]")
		  System.exit(1)
		}	
		
		val headerFile = args(0)
		val trainFile = args(1)
		val testFile = args(2)
		val outputDir = args(3) + "/"
		val infoCriterion = args(4)
		val nonDefaut = args.length > 5
		val lambda = if (nonDefaut) args(5).toDouble else 1.0

		val conf = new SparkConf().setAppName("NBtest")
		val sc = new SparkContext(conf)
		
		def classify = (train: RDD[LabeledPoint]) => {
			val model = NaiveBayes.train(train, lambda)
			model
		}
		
		val ECBDLRangeContFeatures = (0 to 2) ++ (21 to 38) ++ (93 to 130) ++ (151 to 630)
		val irisRangeContFeatures = 0 to 3
		def discretization = (train: RDD[LabeledPoint]) => {
			val discretizer = EntropyMinimizationDiscretizer.train(train,
					ECBDLRangeContFeatures, // continuous features 
					10) // max number of values per feature
		    discretizer
		}
		
		def featureSelect = (data: RDD[LabeledPoint]) => {
			// Feature Selection
			val criterion = new InfoThCriterionFactory(infoCriterion)
			val model = InfoThFeatureSelection.train(criterion, 
		      data,
		      100) // number of features to select
		      //0) // without pool
		    model
		}

		
		// Output of the algorithm
		val algoInfo: String = s"Algorithm: Naive Bayes (NB)\n" + 
				s"lambda: $lambda\n\n"
				
		MCU.executeExperiment(sc, Some(discretization), Some(featureSelect),  Some(classify),
		    headerFile, (trainFile, testFile), outputDir, algoInfo)
		sc.stop()
	}
    
}