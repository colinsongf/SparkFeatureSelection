package org.ugr.sci2s.mllib.test

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

class MLlibRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[Double])
    kryo.register(classOf[Array[Double]])
    kryo.register(classOf[Byte])
    kryo.register(classOf[Array[Byte]])
    kryo.register(classOf[LabeledPoint])    
  }
}

object MainMLlibTest {

	def main(args: Array[String]) {
	  
		val initStartTime = System.nanoTime()
		
		val conf = new SparkConf().setAppName("MLlibTest")
		conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
		conf.set("spark.kryo.registrator", "org.ugr.sci2s.mllib.test.MLlibRegistrator")
		val sc = new SparkContext(conf)

		println("Usage: MLlibTest --header-file=\"hdfs://\" (--train-file=\"hdfs://\" --test-file=\"hdfs://\" " 
		    + "| --data-dir=\"hdfs://\") --output-dir=\"hdfs:\\ --disc=yes [--disc-nbis=10] --fs=yes [--fs-crit=mrmr "
		    + "--fs-nfeat=100 --fs-npool=30] --classifier=no|SVM|NB [--cls-lambda=1.0 --cls-numIter=1 --cls-stepSize = 1.0"
		    + "--cls-regParam=1.0 --cls-miniBatchFraction=1.0]")
		    
		// Create a table of parameters (parsing)
		val params = args.map({arg =>
		  	val param = arg.split("--|=").filter(_.size > 0)
		  	param.size match {
		  		case 2 =>  (param(0) -> param(1))
		  		case _ =>  ("" -> "")
		  	}
		}).toMap		
		
		val headerFile = params.get("header-file")
		val outputDir = params.get("output-dir")
		
		// Header file and output dir must be present
		(headerFile, outputDir) match {
			case (None, None) => 
			  System.err.println("Bad usage. Either header file or output dir is missing.")
			  System.exit(-1)
			case _ =>
		}
		
		// Discretization
		val disc = (train: RDD[LabeledPoint]) => {
			val ECBDLRangeContFeatures = (0 to 2) ++ (21 to 38) ++ (93 to 130) ++ (151 to 630)
			val irisRangeContFeatures = 0 to 3
			val nBins = MCU.toInt(params.getOrElse("disc-nbins", "10"), 10)
			val discretizer = EntropyMinimizationDiscretizer.train(train,
					ECBDLRangeContFeatures, // continuous features 
					nBins) // max number of values per feature
		    discretizer
		}
		
		val discretization = params.get("disc") match {
			case Some(s) if s matches "(?i)yes" => Some(disc)
			case _ => None
		}		
		
		// Feature Selection
		val fs = (data: RDD[LabeledPoint]) => {
			val criterion = new InfoThCriterionFactory(params.getOrElse("fs-criterion", "mrmr"))
			val nToSelect = MCU.toInt(params.getOrElse("fs-nfeat", "100"), 100)
			val nPool = MCU.toInt(params.getOrElse("fs-npool", "30"), 30) // 0 -> w/o pool
			val model = InfoThFeatureSelection.train(criterion, 
		      data,
		      nToSelect, // number of features to select
		      nPool) // number of features in pool
		    model
		}
		
		val featureSelection = params.get("fs") match {
			case Some(s) if s matches "(?i)yes" => Some(fs)
			case _ => None
		}		
		
		// Classification
		val (algoInfo, classification) = params.get("classifier") match {
		  	case Some(s) if s matches "(?i)no" => ("", None)
			case Some(s) if s matches "(?i)NB" => (NBadapter.algorithmInfo(params), 
			    		Some(NBadapter.classify(_: RDD[LabeledPoint], params)))
			case _ => (SVMadapter.algorithmInfo(params), // Default: SVM
			    		Some(SVMadapter.classify(_: RDD[LabeledPoint], params)))			    		
		}
				
		// Extract data files
		val dataFiles = params.get("data-dir") match {
			case Some(dataDir) => dataDir
			case _ =>
			  val trainFile = params.get("train-file")
			  val testFile = params.get("test-file")		
			  (trainFile, testFile) match {
					case (Some(tr), Some(tst)) => (tr, tst)
					case _ => 
					  System.err.println("Bad usage. Either train or test file is missing.")
					  System.exit(-1)
			  }
		}		
		
		// Perform the experiment
		MCU.executeExperiment(sc, discretization, featureSelection, classification,
					  headerFile.get, dataFiles, outputDir.get, algoInfo)
		
		sc.stop()
	}
    
}