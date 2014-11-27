package org.ugr.sci2s.mllib.test

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.ConfusionMatrix
import org.apache.spark.annotation.Experimental
import org.apache.spark.SparkContext
import scala.util.Random
import org.lidiagroup.hmourit.tfg.discretization.EntropyMinimizationDiscretizer
import org.lidiagroup.hmourit.tfg.featureselection.InfoThCriterionFactory
import org.lidiagroup.hmourit.tfg.featureselection.InfoThFeatureSelection
import scala.collection.immutable.List
import scala.collection.mutable.LinkedList
import org.lidiagroup.hmourit.tfg.discretization.EntropyMinimizationDiscretizerModel
import org.lidiagroup.hmourit.tfg.featureselection.InfoThFeatureSelectionModel
import org.lidiagroup.hmourit.tfg.discretization.DiscretizerModel
import org.lidiagroup.hmourit.tfg.featureselection.FeatureSelectionModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.hadoop.mapreduce.lib.input.InvalidInputException
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.EmptyRDD
import org.apache.spark.mllib.util.ConfusionMatrixWithDict
import org.apache.spark.mllib.util.ConfusionMatrixWithDict
import org.apache.spark.mllib.regression.LabeledPoint

object MultiClassificationUtils {
  
  		private def parseThresholds (str: String): (Int, Seq[Double])= {
			val tokens = str split "\t"
			val points = tokens.slice(1, tokens.length).map(_.toDouble)
			val attIndex = tokens(0).toInt
			(attIndex, points.toSeq)
  		}
  		
  		private def parseSelectedAtts (str: String) = {
			val tokens = str split "\t"
			val attIndex = tokens(0).toInt
			(attIndex)
  		}
  		
  		private def parsePredictions(str: String) = {
			val tokens = str split "\t"
			(tokens(0).toDouble, tokens(1).toDouble)
  		}  		
  
  		def computePredictions(model: ClassificationModel, data: RDD[LabeledPoint], threshold: Double = .5) =
			data.map(point => (point.label, if(model.predict(point.features) >= threshold) 1. else 0.))

 		def computePredictions(model: ClassificationModel, data: RDD[LabeledPoint]) =
			data.map(point => (point.label, model.predict(point.features)))
		
		def computeAccuracy (valuesAndPreds: RDD[(Double, Double)]) = 
		  valuesAndPreds.filter(r => r._1 == r._2).count.toDouble / valuesAndPreds.count
		  
  		def computeAccuracyLabels (valuesAndPreds: RDD[(String, String)]) = 
		  valuesAndPreds.filter(r => r._1 == r._2).count.toDouble / valuesAndPreds.count
		
		private val possitive = 1
		private val negative = 0	
		
		private def calcAggStatistics = (scores: Seq[Double]) => {
	  		val sum = scores.reduce(_ + _)
	  		val mean = sum / scores.length
	  		val devs = scores.map(score => (score - mean) * (score - mean))
	  		val stddev = Math.sqrt(devs.reduce(_ + _) / devs.length)
	  		(mean, stddev)
		}
		
		@Experimental
		private def doTraining (training: (RDD[LabeledPoint]) => ClassificationModel, isBinary: Boolean,
		    data: RDD[LabeledPoint]) {
			if(isBinary) training(data)
			else OVATraining(training, data)
		}
		
		@Experimental
		private def OVATraining (training: (RDD[LabeledPoint]) => ClassificationModel, 
		    data: RDD[LabeledPoint]): Array[(Double, Option[ClassificationModel])] = {
			// Histogram of labels
			val classHist = data.map(point => (point.label, 1L)).reduceByKey(_ + _).collect.sortBy(_._2)
				
			def toBinary = (point: LabeledPoint, label: Double) => {
				val cls = if (point.label == label) possitive else negative
				new LabeledPoint(cls, point.features)
			}
			
			// We train models for each class except for the majority one
			val ovaModels: Array[(Double, Option[ClassificationModel])] = 
			  classHist.dropRight(1).map{ case (label, count) => {
					val binaryTr = data.map (toBinary(_, label))
					val oneModel = training(binaryTr)
					(label, Some(oneModel))
				}
			}
			
			// return the class labels and the binary classifier derived from each one
			val lastElem = Array((classHist.last._1, None: Option[ClassificationModel]))
			ovaModels ++ lastElem		
			//result.foreach(f => print(f.toString))
		}
		
		@Experimental
		private def computeOVAPredictions (ovaModels: Array[(Double, Option[ClassificationModel])], 
		    test: RDD[LabeledPoint], threshold: Double = 1.): RDD[(Double, Double)] = {
		  
			def recursiveOVA (point: LabeledPoint, index: Int): Double = {
			  val (label, model) = ovaModels(index)
			  model match {
			  	case Some(m: ClassificationModel) =>
			  	  	val prediction = if(m.predict(point.features) >= threshold) possitive else negative
			  	  	if (prediction == negative) recursiveOVA(point, index + 1) else label
			  	case None => ovaModels.last._1
			  }
			}
			
			test.map{point => {
			  (point.label, recursiveOVA(point, 0))} 
			}
			
		}
		
		private def discretization(
				discretize: (RDD[LabeledPoint]) => (DiscretizerModel[LabeledPoint], RDD[LabeledPoint]), 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int,
				saveData: Boolean = false) = {
		  
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val thresholds = sc.textFile(outputDir + "discThresholds_" + iteration).filter(!_.isEmpty())
									.map(parseThresholds).collect.toMap
				
				val discAlgorithm = new EntropyMinimizationDiscretizerModel(thresholds)
				val discTime = sc.textFile(outputDir + "disc_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)
						.first
				(discAlgorithm.discretize(train), discAlgorithm.discretize(test), discTime)			
				
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException =>
					val initStartTime = System.nanoTime()
					val (discAlgorithm, discData) = discretize(train)
					val discTime = (System.nanoTime() - initStartTime) / 1e9
					val thresholds = discAlgorithm.getThresholds.toArray.sortBy(_._1)
					val discTestData = discAlgorithm.discretize(test)
					
					// Save discretized data 
					if(saveData) {
 						val strTrainDisc = discData.map({case LabeledPoint(label, features) => 
							features.toArray.map(_.toInt).mkString(",") + "," + label.toInt
						})
						
						strTrainDisc.saveAsTextFile(outputDir + "disc_train_" + iteration + ".csv")
						
						val strTstDisc = discTestData.map({case LabeledPoint(label, features) => 
							features.toArray.map(_.toInt).mkString(",") + "," + label.toInt
						})
						
						strTstDisc.saveAsTextFile(outputDir + "disc_tst_" + iteration + ".csv")						
					}
					
					// Save the obtained thresholds in a HDFS file (as a sequence)
					val output = thresholds.foldLeft("")((str, elem) => str + 
								elem._1 + "\t" + elem._2.mkString("\t") + "\n")
					val parThresholds = sc.parallelize(Array(output), 1)
					parThresholds.saveAsTextFile(outputDir + "discThresholds_" + iteration)
					val strTime = sc.parallelize(Array(discTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "disc_time_" + iteration)
					
					(discData, discTestData, discTime)
			}		
		}
		
		private def featureSelection(
				fs: (RDD[LabeledPoint]) => (FeatureSelectionModel[LabeledPoint], RDD[LabeledPoint]), 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int) = {
		  
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val selectedAtts = sc.textFile(outputDir + "FSscheme_" + iteration).filter(!_.isEmpty())
										.map(parseSelectedAtts).collect				
				val featureSelector = new InfoThFeatureSelectionModel(selectedAtts)
				
				val FSTime = sc.textFile(outputDir + "fs_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)
						.first
				(featureSelector.select(train), featureSelector.select(test), FSTime)
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException =>
  					val initStartTime = System.nanoTime()
					val (featureSelector, reductedData) = fs(train)
					val FSTime = (System.nanoTime() - initStartTime) / 1e9
					val selectedAtts = featureSelector.getSelection
					// Save the obtained FS scheme in a HDFS file (as a sequence)
					val output = selectedAtts.mkString("\n")
					val parFSscheme = sc.parallelize(Array(output), 1)
					parFSscheme.saveAsTextFile(outputDir + "FSscheme_" + iteration)
					val strTime = sc.parallelize(Array(FSTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "fs_time_" + iteration)
					
					(reductedData, featureSelector.select(test), FSTime)
			}
		}
		
		private def classification(
				classify: (RDD[LabeledPoint]) => ClassificationModel, 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				typeConv: Array[Map[String, Double]],
				iteration: Int) = {
		  
		  	
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val traValuesAndPreds = sc.textFile(outputDir + "result_" + iteration + ".tra")
						.filter(!_.isEmpty())
						.map(parsePredictions)
						
				val tstValuesAndPreds = sc.textFile(outputDir + "result_" + iteration + ".tst")
						.filter(!_.isEmpty())
						.map(parsePredictions)
						
				val classifficationTime = sc.textFile(outputDir + "classification_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)	
						.first
				
				(traValuesAndPreds, tstValuesAndPreds, classifficationTime)
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException => 
					val initStartTime = System.nanoTime()	
					val classificationModel = classify(train)
					val classificationTime = (System.nanoTime() - initStartTime) / 1e9
					
					val traValuesAndPreds = computePredictions(classificationModel, train)
					val tstValuesAndPreds = computePredictions(classificationModel, test)
					
					// Print training fold results
					val reverseConv = typeConv.last.map(_.swap) // for last class
					val outputTrain = traValuesAndPreds.map(t => reverseConv.getOrElse(t._1, "") + "\t" +
						    reverseConv.getOrElse(t._2, ""))   
					outputTrain.saveAsTextFile(outputDir + "result_" + iteration + ".tra")
					val outputTest = tstValuesAndPreds.map(t => reverseConv.getOrElse(t._1, "") + "\t" +
						    reverseConv.getOrElse(t._2, ""))    
					outputTest.saveAsTextFile(outputDir + "result_" + iteration + ".tst")		
					val strTime = sc.parallelize(Array(classificationTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "classification_time_" + iteration)
					
					(traValuesAndPreds, tstValuesAndPreds, classificationTime)
			}
		}
		
		/**
		 * Execute a MLlib experiment with three optional phases (discretization, feature selection and classification)
		 * @param sc Spark context
		 * @param kfold Number of folds in which is divided the dataset
		 * @param discretize Optional function to discretize a dataset
		 * @param featureSelect Optional function to reduce the set of features
		 * @param classify Optional function to classify a dataset
		 * @param headerFile Header file with the basis information about the dataset (arff format)
		 * @param inputData File name or data directory where the data files are placed
		 * @param outputDir HDFS directory for the experiment output
		 * @param algoInfo Some basis information to be written
		 */
		
		def executeExperiment(
		    sc: SparkContext,
		    discretize: Option[(RDD[LabeledPoint]) => (DiscretizerModel[LabeledPoint], RDD[LabeledPoint])], 
		    featureSelect: Option[(RDD[LabeledPoint]) => (FeatureSelectionModel[LabeledPoint], RDD[LabeledPoint])], 
		    classify: Option[(RDD[LabeledPoint]) => ClassificationModel],
		    headerFile: String, 
		    inputData: Object, 
		    outputDir: String, 
		    algoInfo: String) {

			def getDataFiles(dirPath: String, k: Int): Array[(String, String)] = {
				val subDir = dirPath.split("/").last
				def genName = (i: Int) => dirPath + "/" + subDir.replaceFirst("fold", i.toString)
				val cvdata = for (i <- 1 to k) yield (genName(i) + "tra.data", genName(i) + "tst.data")
				cvdata.toArray
			}
			
			// Load training data
			val dataInfo = sc.textFile(headerFile).collect.reduceLeft(_ + "\n" + _)
					
			val typeConversion = KeelParser.parseHeaderFile(sc, headerFile)	
			val reverseConv = typeConversion.last.map(_.swap) // for last class
			val binary = typeConversion.last.size < 3
			
			//if(!binary) throw new IllegalArgumentException("Data class not binary...")
			val bcTypeConv = sc.broadcast(typeConversion).value
			val dataFiles = inputData match {
			  case (dataPath: String, kfold: Int) => getDataFiles(dataPath, kfold)
			  case (train: String, test: String) => Array((train, test))
			}			
			
			val info = Map[String, String]("algoInfo" -> algoInfo, "dataInfo" -> dataInfo)
			val times = scala.collection.mutable.Map[String, Seq[Double]] ("FullTime" -> Seq(),
			    "DiscTime" -> Seq(),
			    "FSTime" -> Seq(),
			    "ClsTime" -> Seq())
			
			val nFolds = dataFiles.length
			//val confusionMatrices = Seq.empty[ConfusionMatrixWithDict]
			var predictions = Array.empty[(RDD[(Double, Double)], RDD[(Double, Double)])]
			    
			val accTraResults = Seq.empty[(Double, Double)]
			for (i <- 0 until nFolds) {
								var initAllTime = System.nanoTime()
				val (trainFile, testFile) = dataFiles(i)
				val trainData = sc.textFile(trainFile).
						//sample(false, samplingRate, seed.nextLong).
						map(line => (KeelParser.parseLabeledPoint(bcTypeConv, line)))
								
				val testData = sc.textFile(testFile).
						//sample(false, samplingRate, seed.nextLong).
						map(line => (KeelParser.parseLabeledPoint(bcTypeConv, line)))
				trainData.persist(StorageLevel.MEMORY_AND_DISK_SER) 
				testData.persist(StorageLevel.MEMORY_AND_DISK_SER)
				
				// Discretization
				var trData = trainData; var tstData = testData
				var taskTime = 0.0
				discretize match { 
				  case Some(disc) => 
				    val (discTrData, discTstData, discTime) = discretization(
								disc, trData, tstData, outputDir, i, saveData=true)
					trData = discTrData
					tstData = discTstData
					taskTime = discTime
				  case None => /* criteria not fulfilled, do not discretize */
				}				
				times("DiscTime") = times("DiscTime") :+ taskTime

				// Feature Selection
				featureSelect match { 
				  case Some(fs) => 
				    val (fsTrainData, fsTestData, fsTime) = 
				      featureSelection(fs, trData, tstData, outputDir, i)
					trData = fsTrainData
					tstData = fsTestData
					taskTime = fsTime
				  case None => taskTime = 0.0 /* criteria not fulfilled, do not do FS */
				}
				times("FSTime") = times("FSTime") :+ taskTime
				
				//Classification
				classify match { 
				  case Some(cls) => 
				    val (traValuesAndPreds, tstValuesAndPreds, classifficationTime) = 
				  		classification(cls, trData, tstData, outputDir, typeConversion, i)
					taskTime = classifficationTime
					/* Confusion matrix for the test set */
					//confusionMatrices :+ ConfusionMatrix.apply(tstValuesAndPreds, typeConversion.last)
					predictions = predictions :+ (traValuesAndPreds, tstValuesAndPreds)
				  case None => taskTime = 0.0 /* criteria not fulfilled, do not classify */
				}
				times("ClsTime") = times("ClsTime") :+ taskTime
				
				trainData.unpersist(); testData.unpersist()
				
				var fullTime = (System.nanoTime() - initAllTime) / 1e9
				times("FullTime") = times("FullTime") :+ fullTime
			}
			
			// Print the aggregated results
			printResults(outputDir, predictions, info, times.toMap, typeConversion)
		}

		private def printResults(
				outputDir: String, 
				predictions: Array[(RDD[(Double, Double)], RDD[(Double, Double)])], 
				info: Map[String, String],
				timeResults: Map[String, Seq[Double]],
				typeConv: Array[Map[String,Double]]) {
		  
  			val revConv = typeConv.last.map(_.swap) // for last class
			var output = info.get("algoInfo").get + "Accuracy Results\tTrain\tTest\n"
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
			output += "Mean Discretization Time:\t" + 
					timeResults("DiscTime").sum / timeResults("DiscTime").size + " seconds.\n"
			output += "Mean Feature Selection Time:\t" + 
					timeResults("FSTime").sum / timeResults("FSTime").size + " seconds.\n"
			output += "Mean Classification Time:\t" + 
					timeResults("ClsTime").sum / timeResults("ClsTime").size + " seconds.\n"
			output += "Mean Execution Time:\t" + 
					timeResults("FullTime").sum / timeResults("FullTime").size + " seconds.\n"
					
			// Confusion Matrix		
			val str = typeConv.last.mkString("\n")
			println("Conversor: " + str)
			val aggTstConfMatrix = ConfusionMatrix.apply(predictions.map(_._2).reduceLeft(_ ++ _), 
			    typeConv.last)			    
		    output += "Test Confusion Matrix\n" + aggTstConfMatrix.toString
		    output += aggTstConfMatrix.fValue.foldLeft("\t")((str, t) => str + "\t" + t._1) + "\n"
		    
		    output += aggTstConfMatrix.fValue.foldLeft("F-Measure:")((str, t) => str + "\t" + t._2) + "\n"
		    output += aggTstConfMatrix.precision.foldLeft("Precision:")((str, t) => str + "\t" + t._2) + "\n"
		    output += aggTstConfMatrix.recall.foldLeft("Recall:")((str, t) => str + "\t" + t._2) + "\n"
		    output += aggTstConfMatrix.specificity.foldLeft("Specificity:")((str, t) => str + "\t" + t._2) + "\n\n"
		    
		    val aggTraConfMatrix = ConfusionMatrix.apply(predictions.map(_._1).reduceLeft(_ ++ _), 
			    typeConv.last)
		    output += "Train Confusion Matrix\n" + aggTraConfMatrix.toString
			println(output)
			
			val sc = predictions(0)._1.context
			val hdfsOutput = sc.parallelize(Array(output), 1)
			hdfsOutput.saveAsTextFile(outputDir + "globalResult.txt")
		}
}
