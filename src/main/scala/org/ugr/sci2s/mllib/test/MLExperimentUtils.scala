package org.ugr.sci2s.mllib.test

import scala.util.Random
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.ConfusionMatrix
import org.apache.spark.annotation.Experimental
import org.apache.spark.SparkContext
import scala.collection.immutable.List
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.hadoop.mapreduce.lib.input.InvalidInputException
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.util.MLUtils

object MLExperimentUtils {
  
	    def toInt(s: String, default: Int): Int = {
  			try {
  				s.toInt
  			} catch {
  				case e:Exception => default
  			}
	    }	
	
	    def toDouble(s: String, default: Double): Double = {
  			try {
  				s.toDouble
  			} catch {
  				case e: Exception => default
  			}
		  }
  
  		private def parseThresholds (str: String): (Int, Seq[Float])= {
  			val tokens = str split "\t"
  			val points = tokens.slice(1, tokens.length).map(_.toFloat)
  			val attIndex = tokens(0).toInt
  			(attIndex, points.toSeq)
  		}
  		
  		private def parseSelectedAtts (str: String) = {
        val tokens = str split "\t"
        (tokens(0).toInt, tokens(1).toDouble)
  		}
  		
  		private def parsePredictions(str: String) = {
  			val tokens = str split "\t"
  			(tokens(0).toDouble, tokens(1).toDouble)
  		}
  
  		def computePredictions(model: ClassificationModelAdapter, data: RDD[LabeledPoint], threshold: Double = .5) =
			  data.map(point => (point.label, if(model.predict(point.features) >= threshold) 1.0 else 0.0))

 		  def computePredictions (model: ClassificationModelAdapter, data: RDD[LabeledPoint]) =
			  data.map(point => (point.label, model.predict(point.features)))
		
		  def computeAccuracy (valuesAndPreds: RDD[(Double, Double)]) = 
		    valuesAndPreds.filter(r => r._1 == r._2).count.toDouble / valuesAndPreds.count
		  
  		def computeAccuracyLabels (valuesAndPreds: RDD[(String, String)]) = 
		    valuesAndPreds.filter(r => r._1 == r._2).count.toDouble / valuesAndPreds.count
		
		private val positive = 1
		private val negative = 0	
		
		private def calcAggStatistics = (scores: Seq[Double]) => {
	  		val sum = scores.reduce(_ + _)
	  		val mean = sum / scores.length
	  		val devs = scores.map(score => (score - mean) * (score - mean))
	  		val stddev = Math.sqrt(devs.reduce(_ + _) / devs.length)
	  		(mean, stddev)
		}
		
		@Experimental
		private def doTraining (training: (RDD[LabeledPoint]) => ClassificationModelAdapter, isBinary: Boolean,
		    data: RDD[LabeledPoint]) {
			if(isBinary) training(data)
			else OVATraining(training, data)
		}
		
		@Experimental
		private def OVATraining (training: (RDD[LabeledPoint]) => ClassificationModelAdapter, 
		    data: RDD[LabeledPoint]): Array[(Double, Option[ClassificationModelAdapter])] = {
			// Histogram of labels
			val classHist = data.map(point => (point.label, 1L)).reduceByKey(_ + _).collect.sortBy(_._2)
				
			def toBinary = (point: LabeledPoint, label: Double) => {
				val cls = if (point.label == label) positive else negative
				new LabeledPoint(cls, point.features)
			}
			
			// We train models for each class except for the majority one
			val ovaModels: Array[(Double, Option[ClassificationModelAdapter])] = 
			  classHist.dropRight(1).map{ case (label, count) => {
					val binaryTr = data.map (toBinary(_, label))
					val oneModel = training(binaryTr)
					(label, Some(oneModel))
				}
			}
			
			// return the class labels and the binary classifier derived from each one
			val lastElem = Array((classHist.last._1, None: Option[ClassificationModelAdapter]))
			ovaModels ++ lastElem		
		}
		
		@Experimental
		private def computeOVAPredictions (ovaModels: Array[(Double, Option[ClassificationModelAdapter])], 
		    test: RDD[LabeledPoint], threshold: Double = 1.0): RDD[(Double, Double)] = {
		  
			def recursiveOVA (point: LabeledPoint, index: Int): Double = {
			  val (label, model) = ovaModels(index)
			  model match {
			  	case Some(m: ClassificationModelAdapter) =>
			  	  	val prediction = if(m.predict(point.features) >= threshold) positive else negative
			  	  	if (prediction == negative) recursiveOVA(point, index + 1) else label
			  	case None => ovaModels.last._1
			  }
			}
			
			test.map{point => {
			  (point.label, recursiveOVA(point, 0))} 
			}
			
		}
		
		private def discretization(
				discretize: (RDD[LabeledPoint]) => DiscretizerModel, 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int,
				save: Boolean = false) = {
		  
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val thresholds = sc.textFile(outputDir + "/discThresholds_" + iteration).filter(!_.isEmpty())
									.map(parseThresholds).collect
				
				val discAlgorithm = new DiscretizerModel(thresholds)
				val discTime = sc.textFile(outputDir + "/disc_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)
						.first
            // More efficient than by-instance version
		        val discData = discAlgorithm.transform(train.map(_.features))
              .zip(train.map(_.label))
              .map{case (v, l) => LabeledPoint(l, v)}
            val discTestData = discAlgorithm.transform(test.map(_.features))
              .zip(test.map(_.label))
              .map{case (v, l) => LabeledPoint(l, v)}
		        
            // Save discretized data 
		        if(save) {
             discData.map({lp => lp.features.toArray.mkString(",") + "," + lp.label})
               .saveAsTextFile(outputDir + "/disc_train_" + iteration + ".csv")
             discTestData.map({lp => lp.features.toArray.mkString(",") + "," + lp.label})
               .saveAsTextFile(outputDir + "/disc_test_" + iteration + ".csv")       
		        } 
        
				(discData, discTestData, discTime)			
				
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException =>
					val initStartTime = System.nanoTime()
					val discAlgorithm = discretize(train)
					val discTime = (System.nanoTime() - initStartTime) / 1e9
          // More efficient than by-instance version
          val discData = discAlgorithm.transform(train.map(_.features))
            .zip(train.map(_.label))
            .map{case (v, l) => LabeledPoint(l, v)}
          val discTestData = discAlgorithm.transform(test.map(_.features))
            .zip(test.map(_.label))
            .map{case (v, l) => LabeledPoint(l, v)}
		          
          // Save discretized data 
          if(save) {
             discData.map({lp => lp.features.toArray.mkString(",") + "," + lp.label})
               .saveAsTextFile(outputDir + "/disc_train_" + iteration + ".csv")
             discTestData.map({lp => lp.features.toArray.mkString(",") + "," + lp.label})
               .saveAsTextFile(outputDir + "/disc_test_" + iteration + ".csv")       
          } 
					
					// Save the obtained thresholds in a HDFS file (as a sequence)
					val thresholds = discAlgorithm.thresholds.toArray.sortBy(_._1)
					val output = thresholds.foldLeft("")((str, elem) => str + 
								elem._1 + "\t" + elem._2.mkString("\t") + "\n")
					val parThresholds = sc.parallelize(Array(output), 1)
					parThresholds.saveAsTextFile(outputDir + "/discThresholds_" + iteration)
					val strTime = sc.parallelize(Array(discTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "/disc_time_" + iteration)
					
					(discData, discTestData, discTime)
			}		
		}
		
		private def featureSelection(
				fs: (RDD[LabeledPoint]) => SelectorModel, 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int,
        save: Boolean) = {
		  
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val selectedAtts = sc.textFile(outputDir + "/FSscheme_" + iteration).filter(!_.isEmpty())
										.map(parseSelectedAtts).collect				
				val featureSelector = new SelectorModel(selectedAtts.map(_._1))
				
				val FSTime = sc.textFile(outputDir + "/fs_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)
						.first
         
        val redTrain = train.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
        val redTest = test.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
        
          // Save reduced data 
          if(save) {
             redTrain.map({lp => lp.features.toArray.mkString(",") + "," + lp.label})
               .saveAsTextFile(outputDir + "/fs_train_" + iteration + ".csv")
             redTest.map({lp => lp.features.toArray.mkString(",") + "," + lp.label})
               .saveAsTextFile(outputDir + "/fs_test_" + iteration + ".csv")     
          }         
        
				(redTrain, redTest, FSTime)
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException =>
					val initStartTime = System.nanoTime()
					val featureSelector = fs(train)
					val FSTime = (System.nanoTime() - initStartTime) / 1e9
          val redTrain = train.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
          val redTest = test.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
          
          // Save reduced data 
          if(save) {
             redTrain.map({lp => lp.features.toArray.mkString(",") + "," + lp.label})
               .saveAsTextFile(outputDir + "/fs_train_" + iteration + ".csv")
             redTest.map({lp => lp.features.toArray.mkString(",") + "," + lp.label})
               .saveAsTextFile(outputDir + "/fs_test_" + iteration + ".csv")       
          }    
					
					// Save the obtained FS scheme in a HDFS file (as a sequence)					
					val selectedAtts = featureSelector.selectedFeatures
          val output = selectedAtts.mkString("\n")
					//val output = selectedAtts.foldLeft("")((str, elem) => str + elem._1 + "\t" + elem._2 + "\n")
					val parFSscheme = sc.parallelize(Array(output), 1)
					parFSscheme.saveAsTextFile(outputDir + "/FSscheme_" + iteration)
					val strTime = sc.parallelize(Array(FSTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "/fs_time_" + iteration)
					
					(redTrain, redTest, FSTime)
			}
		}
		
		private def classification(
				classify: (RDD[LabeledPoint]) => ClassificationModelAdapter, 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int) = {
		  	
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val traValuesAndPreds = sc.textFile(outputDir + "/result_" + iteration + ".tra")
						.filter(!_.isEmpty())
						.map(parsePredictions)
						
				val tstValuesAndPreds = sc.textFile(outputDir + "/result_" + iteration + ".tst")
						.filter(!_.isEmpty())
						.map(parsePredictions)
						
				val classifficationTime = sc.textFile(outputDir + "/classification_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)	
						.first
				
				(traValuesAndPreds, tstValuesAndPreds, classifficationTime)
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException => 
          val nInstances = train.count() // to persist train and not to affect time measurements
					val initStartTime = System.nanoTime()	
					val classificationModel = classify(train)
					val classificationTime = (System.nanoTime() - initStartTime) / 1e9
					
					val traValuesAndPreds = computePredictions(classificationModel, train)
					val tstValuesAndPreds = computePredictions(classificationModel, test)
					
					// Print training fold results
					//val reverseConv = typeConv.last.map(_.swap) // for last class
					val outputTrain = traValuesAndPreds.map(t => t._1.toInt + "\t" + t._2.toInt)   
					outputTrain.saveAsTextFile(outputDir + "/result_" + iteration + ".tra")
					val outputTest = tstValuesAndPreds.map(t => t._1.toInt + "\t" + t._2.toInt)  
          //val outputTest = tstValuesAndPreds.map(t => reverseConv.getOrElse(t._1, "") + "\t" +
					//	    reverseConv.getOrElse(t._2, ""))    
					outputTest.saveAsTextFile(outputDir + "/result_" + iteration + ".tst")		
					val strTime = sc.parallelize(Array(classificationTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "/classification_time_" + iteration)
					
					(traValuesAndPreds, tstValuesAndPreds, classificationTime)
			}
		}

		/**
		 * Execute a MLlib experiment with three optional phases (discretization, feature selection and classification)
		 * @param sc Spark context
		 * @param discretize Optional function to discretize a dataset
		 * @param featureSelect Optional function to reduce the set of features
		 * @param classify Optional function to classify a dataset
		 * @param inputData File or directory path where the data set files are placed
		 * @param outputDir HDFS output directory for the experiment
		 * @param algoInfo Some basis information about the algorithm to be executed
		 */		
		def executeExperiment(
		    sc: SparkContext,
		    discretize: (Option[(RDD[LabeledPoint]) => DiscretizerModel], Boolean), 
		    featureSelect: (Option[(RDD[LabeledPoint]) => SelectorModel], Boolean), 
		    classify: Option[(RDD[LabeledPoint]) => ClassificationModelAdapter],
		    inputData: (Any, String, Boolean), 
		    outputDir: String, 
		    algoInfo: String) {

			def getDataFiles(dirPath: String, k: Int): Array[(String, String)] = {
				val subDir = dirPath.split("/").last
				def genName = (i: Int) => dirPath + "/" + subDir.replaceFirst("fold", i.toString)
				val cvdata = for (i <- 1 to k) yield (genName(i) + "tra.data", genName(i) + "tst.data")
				cvdata.toArray
			}
      
		      val (headerFile, dataFiles, format, dense) = inputData match {
		        case ((header: Option[String], dataPath: String, kfold: Int), format: String, dense: Boolean) => 
		          (header, getDataFiles(dataPath, kfold), format, dense)
		        case ((header: Option[String], train: String, test: String), format: String, dense: Boolean) => 
		          (header, Array((train, test)), format, dense)
		      } 
			
		      val samplingRate = 1.0      
    		  // Create the function to read the labeled points
    		  val readFile = format match {
		          case s if s matches "(?i)LibSVM" => 
		            (filePath: String) => {
		              val svmData = MLUtils.loadLibSVMFile(sc, filePath)
		              val data = if(samplingRate < 1.0) svmData.sample(false, samplingRate) else svmData
		              if(dense) {
		                data.map{case LabeledPoint(label, features) => 
		                  new LabeledPoint(label, Vectors.dense(features.toArray))}
		              } else {
		                data
		              }
		            }
		          case _ => 
		           (filePath: String) => {
		              val typeConversion = KeelParser.parseHeaderFile(sc, headerFile.get) 
		              val bcTypeConv = sc.broadcast(typeConversion)
		              val lines = sc.textFile(filePath: String)
		              val data = if(samplingRate < 1.0) lines.sample(false, samplingRate) else lines
		              data.map(line => KeelParser.parseLabeledPoint(bcTypeConv.value, line))              
		           }
		      }      
      
			val info = Map[String, String]("algoInfo" -> algoInfo)
			val times = scala.collection.mutable.Map[String, Seq[Double]] ("FullTime" -> Seq(),
			    "DiscTime" -> Seq(),
			    "FSTime" -> Seq(),
			    "ClsTime" -> Seq())
			
			val nFolds = dataFiles.length
			var predictions = Array.empty[(RDD[(Double, Double)], RDD[(Double, Double)])]
			    
			val accTraResults = Seq.empty[(Double, Double)]
			for (i <- 0 until nFolds) {
								var initAllTime = System.nanoTime()
                
				val (trainFile, testFile) = dataFiles(i)
				val trainData = readFile(trainFile)
				val testData = readFile(testFile)
				
				// Discretization
				var trData = trainData; var tstData = testData
        trData.cache() // Data are called repeatedly in MDLP discretizer
				var taskTime = 0.0
				discretize match { 
				  case (Some(disc), b) => 
				    val (discTrData, discTstData, discTime) = discretization(
								disc, trData, tstData, outputDir, i, save = b) 
					trData = discTrData
					tstData = discTstData
					taskTime = discTime
				  case _ => /* criteria not fulfilled, do not discretize */
				}				
				times("DiscTime") = times("DiscTime") :+ taskTime
        trData.unpersist()

				// Feature Selection
				featureSelect match { 
				  case (Some(fs), b) => 
				    val (fsTrainData, fsTestData, fsTime) = 
				      featureSelection(fs, trData, tstData, outputDir, i, save = b)
					trData = fsTrainData
					tstData = fsTestData
					taskTime = fsTime
				  case _ => taskTime = 0.0 /* criteria not fulfilled, do not do select */
				}
				times("FSTime") = times("FSTime") :+ taskTime
				
				//Classification
        val nFeatures = trData.first().features.size
        println("Number of features: " + nFeatures)
        
				classify match { 
				  case Some(cls) => 
				    val (traValuesAndPreds, tstValuesAndPreds, classificationTime) = 
				  		classification(cls, trData.cache, tstData.cache, outputDir, i)
					taskTime = classificationTime
					/* Confusion matrix for the test set */
					//confusionMatrices :+ ConfusionMatrix.apply(tstValuesAndPreds, typeConversion.last)
					predictions = predictions :+ (traValuesAndPreds, tstValuesAndPreds)
				  case None => taskTime = 0.0 /* criteria not fulfilled, do not classify */
				}
				times("ClsTime") = times("ClsTime") :+ taskTime
				
				//trainData.unpersist(); testData.unpersist()
				
				var fullTime = (System.nanoTime() - initAllTime) / 1e9
				times("FullTime") = times("FullTime") :+ fullTime
			}
			
			// Print the aggregated results
			printResults(sc, outputDir, predictions, info, getTimeResults(times.toMap))
		}
    
	    private def getTimeResults(timeResults: Map[String, Seq[Double]]) = {
	        "Mean Discretization Time:\t" + 
	            timeResults("DiscTime").sum / timeResults("DiscTime").size + " seconds.\n" +
	        "Mean Feature Selection Time:\t" + 
	            timeResults("FSTime").sum / timeResults("FSTime").size + " seconds.\n" +
	       "Mean Classification Time:\t" + 
	            timeResults("ClsTime").sum / timeResults("ClsTime").size + " seconds.\n" +
	        "Mean Execution Time:\t" + 
	            timeResults("FullTime").sum / timeResults("FullTime").size + " seconds.\n" 
	    }

		private def printResults(
				sc: SparkContext,
				outputDir: String, 
				predictions: Array[(RDD[(Double, Double)], RDD[(Double, Double)])], 
				info: Map[String, String],
				timeResults: String) {
         
	        var output = timeResults
	        if(!predictions.isEmpty){
	            // Statistics by fold
	            output += info.get("algoInfo").get + "Accuracy Results\tTrain\tTest\n"
	      			val traFoldAcc = predictions.map(_._1).map(computeAccuracy)
	      			val tstFoldAcc = predictions.map(_._2).map(computeAccuracy)		
	      			// Print fold results into the global result file
	      			for (i <- 0 until predictions.size){
	      				output += s"Fold $i:\t" +
	      					traFoldAcc(i) + "\t" + tstFoldAcc(i) + "\n"
	      			} 
	      			
	      			// Aggregated statistics
	      			val (aggAvgAccTr, aggStdAccTr) = calcAggStatistics(traFoldAcc)
	      			val (aggAvgAccTst, aggStdAccTst) = calcAggStatistics(tstFoldAcc)
	      			output += s"Avg Acc:\t$aggAvgAccTr\t$aggAvgAccTst\n"
	      			output += s"Svd acc:\t$aggStdAccTr\t$aggStdAccTst\n"
	      					
	      			// Confusion Matrix
	      			val aggTstConfMatrix = ConfusionMatrix.apply(predictions.map(_._2).reduceLeft(_ ++ _))			    
	    		    output += "Test Confusion Matrix\n" + aggTstConfMatrix.toString
	    		    output += aggTstConfMatrix.fValue.foldLeft("\t")((str, t) => str + "\t" + t._1) + "\n"
	    		    
	    		    output += aggTstConfMatrix.fValue.foldLeft("F-Measure:")((str, t) => str + "\t" + t._2) + "\n"
	    		    output += aggTstConfMatrix.precision.foldLeft("Precision:")((str, t) => str + "\t" + t._2) + "\n"
	    		    output += aggTstConfMatrix.recall.foldLeft("Recall:")((str, t) => str + "\t" + t._2) + "\n"
	    		    
	    		    val aggTraConfMatrix = ConfusionMatrix.apply(predictions.map(_._1).reduceLeft(_ ++ _))
	    		    output += "Train Confusion Matrix\n" + aggTraConfMatrix.toString
	        }
			println(output)
			
			val hdfsOutput = sc.parallelize(Array(output), 1)
			hdfsOutput.saveAsTextFile(outputDir + "/globalResult.txt")
		}
}
