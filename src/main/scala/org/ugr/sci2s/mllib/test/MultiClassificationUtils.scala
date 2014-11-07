package org.ugr.sci2s.mllib.test

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.ConfusionMatrix
import java.io.PrintWriter
import java.io.File
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

object MultiClassificationUtils {
  
  		def parseThresholds (str: String): (Int, Seq[Double])= {
			val tokens = str split "\t"
			val points = tokens.slice(1, tokens.length).map(_.toDouble)
			val attIndex = tokens(0).toInt
			(attIndex, points.toSeq)
  		}
  		
  		def parseSelectedAtts (str: String) = {
			val tokens = str split "\t"
			val attIndex = tokens(0).toInt
			(attIndex)
  		}
  		
  		def parsePredictions(str: String) = {
			val tokens = str split "\t"
			(tokens(0), tokens(1))
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
				
		def calcAggStatistics = (scores: Seq[Double]) => {
	  		val sum = scores.reduce(_ + _)
	  		val mean = sum / scores.length
	  		val devs = scores.map(score => (score - mean) * (score - mean))
	  		val stddev = Math.sqrt(devs.reduce(_ + _) / devs.length)
	  		(mean, stddev)
		}
		
		@Experimental
		def doTraining (training: (RDD[LabeledPoint]) => ClassificationModel, isBinary: Boolean,
		    data: RDD[LabeledPoint]) {
			if(isBinary) training(data)
			else OVATraining(training, data)
		}
		
		@Experimental
		def OVATraining (training: (RDD[LabeledPoint]) => ClassificationModel, 
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
		def computeOVAPredictions (ovaModels: Array[(Double, Option[ClassificationModel])], 
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
		
		def discretization(discretize: Option[(RDD[LabeledPoint]) => (DiscretizerModel[LabeledPoint], RDD[LabeledPoint])], 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int) = {
		  
			val sc = train.context
		  
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
					val (discAlgorithm, discData) = discretize.get(train)
					val discTime = (System.nanoTime() - initStartTime) / 1e9
					val thresholds = discAlgorithm.getThresholds
					// Save the obtained thresholds in a HDFS file (as a sequence)
					val output = thresholds.foldLeft("")((str, elem) => str + 
								elem._1 + "\t" + elem._2.mkString("\t") + "\n")
					val parThresholds = sc.parallelize(Array(output), 1)
					parThresholds.saveAsTextFile(outputDir + "discThresholds_" + iteration)
					val strTime = sc.parallelize(discTime.toString, 1)
					strTime.saveAsTextFile(outputDir + "disc_time_" + iteration)
					
					(discData, discAlgorithm.discretize(test), discTime)
			}		
		}
		
		def featureSelection(fs: Option[(RDD[LabeledPoint]) => (FeatureSelectionModel[LabeledPoint], RDD[LabeledPoint])], 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int) = {
		  
			val sc = train.context
		  				
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
					val (featureSelector, reductedData) = fs.get(train)
					val FSTime = (System.nanoTime() - initStartTime) / 1e9
					val selectedAtts = featureSelector.getSelection
					// Save the obtained FS scheme in a HDFS file (as a sequence)
					val output = selectedAtts.mkString("\n")
					val parFSscheme = sc.parallelize(Array(output), 1)
					parFSscheme.saveAsTextFile(outputDir + "FSscheme_" + iteration)
					val strTime = sc.parallelize(FSTime.toString, 1)
					strTime.saveAsTextFile(outputDir + "fs_time_" + iteration)
					
					(reductedData, featureSelector.select(test), FSTime)
			}
		}
		
		def classification(classify: (RDD[LabeledPoint]) => ClassificationModel, 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				typeConv: Array[Map[String, Double]],
				iteration: Int) = {
		  				
			try {
				val sc = train.context
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
					val classifficationTime = (System.nanoTime() - initStartTime) / 1e9
					
					val traValuesAndPreds = computePredictions(classificationModel, train)
					val tstValuesAndPreds = computePredictions(classificationModel, test)
					
					// Print training fold results
					val reverseConv = typeConv.last.map(_.swap) // for last class
					val outputTrain = traValuesAndPreds.map(t => reverseConv.getOrElse(t._1, "") + " " +
						    reverseConv.getOrElse(t._2, ""))   
					outputTrain.saveAsTextFile(outputDir + "result_" + iteration + ".tra")
					val outputTest = tstValuesAndPreds.map(t => reverseConv.getOrElse(t._1, "") + " " +
						    reverseConv.getOrElse(t._2, ""))    
					outputTest.saveAsTextFile(outputDir + "result_" + iteration + ".tst")				
					(traValuesAndPreds, tstValuesAndPreds, classifficationTime)
			}
		}
		
		def executeExperiment(classify: (RDD[LabeledPoint]) => ClassificationModel, 
		    discretize: Option[(RDD[LabeledPoint]) => (DiscretizerModel[LabeledPoint], RDD[LabeledPoint])], 
		    featureSelect: Option[(RDD[LabeledPoint]) => (FeatureSelectionModel[LabeledPoint], RDD[LabeledPoint])], 
		    sc: SparkContext, 
		    headerFile: String, 
		    inputData: Object, 
		    outputDir: String, 
		    algoInfo: String) {
  		
			val samplingRate = 0.01
			val seed = new Random
			
			def getDataFiles(dirPath: String): Array[(String, String)] = {
				val k = 5
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
			  case s: String => getDataFiles(s)
			  case (train: String, test: String) => Array((train, test))
			}			
			
			val info = Map[String, String]("algoInfo" -> algoInfo, "dataInfo" -> dataInfo)
			val times = scala.collection.mutable.Map[String, Seq[Double]] ("FullTime" -> Seq(),
			    "DiscTime" -> Seq(),
			    "FSTime" -> Seq(),
			    "ClsTime" -> Seq())
			    
			val accTraResults = Seq.empty[(Double, Double)]
			for (i <- 1 to dataFiles.length) {
								var initAllTime = System.nanoTime()
				val (trainFile, testFile) = dataFiles(i)
				val trainData = sc.textFile(trainFile).
						//sample(false, samplingRate, seed.nextLong).
						map(line => (KeelParser.parseLabeledPoint(bcTypeConv, line)))
				val testData = sc.textFile(testFile).
						//sample(false, samplingRate, seed.nextLong).
						map(line => (KeelParser.parseLabeledPoint(bcTypeConv, line)))
				trainData.persist(StorageLevel.MEMORY_ONLY_SER) 
				testData.persist(StorageLevel.MEMORY_ONLY_SER)
				
				// Discretization
				var trData = trainData; var tstData = testData
				var taskTime = 0.0
				discretize match { 
				  case Some(x) => 
				    val (discTrData, discTstData, discTime) = discretization(
								discretize, trData, tstData, outputDir, i)
					trData = discTrData
					tstData = discTstData
					taskTime = discTime
				  case None => /* criteria not fulfilled, do not discretize */
				}				
				times("DiscTime") = times("DiscTime") :+ taskTime

				// Feature Selection
				val haveToFS = featureSelect match { 
				  case Some(x) => 
				    val (fsTrainData, fsTestData, fsTime) = featureSelection(
					    featureSelect, trData, tstData, outputDir, i)
					trData = fsTrainData
					tstData = fsTestData
					taskTime = fsTime
				  case None => taskTime = 0.0 /* criteria not fulfilled, do not do FS */
				}
				times("FSTime") = times("FSTime") :+ taskTime
				
				// Run training algorithm to build the model
				val results = classification(classify, trainData, testData, 
				    outputDir, typeConversion, i)
				
				trainData.unpersist(); testData.unpersist()
				
				var fullTime = (System.nanoTime() - initAllTime) / 1e9
				times("FullTime") = times("FullTime") :+ fullTime
			}
			
			// Do the output (fold results files and global result file)
			//val dir = new File("results/" + outputDir); dir.mkdirs()
			printResults(outputDir, accResults, info, times.toMap, typeConversion)
		}

		def printResults(
				outputDir: String, 
				accResults: Array[(RDD[(Double, Double)], RDD[(Double, Double)])], 
				info: Map[String, String],
				timeResults: Map[String, Seq[Double]],
				typeConv: Array[Map[String,Double]]) {
		  
  			val revConv = typeConv.last.map(_.swap) // for last class
  			
			var output = info.get("algoInfo").get + "Accuracy Results\tTrain\tTest\n"
			val traFoldAcc= accResults.map(_._1).map(computeAccuracy)
			val tstFoldAcc = accResults.map(_._2).map(computeAccuracy)
			
			// Aggregated statistics
			val (aggAvgAccTr, aggStdAccTr) = calcAggStatistics(traFoldAcc)
			val (aggAvgAccTst, aggStdAccTst) = calcAggStatistics(tstFoldAcc)
			
			
			val aggConfMatrix = ConfusionMatrix.apply(accResults.map(_._2).reduceLeft(_ union _), 
			    typeConv.last)			    
			
			// Output of the algorithm
			(0 until accResults.size).map { i =>
				// Print training fold results
				/*val (foldTrPred, foldTstPred) = accResults(i)
				val strResults = foldTrPred.map(t => revConv.getOrElse(t._1, default) + " " +
					    revConv.getOrElse(t._2, default))   
				strResults.coalesce(1, true).
					saveAsTextFile(outputDir + "result" + i + ".tra")
				val strResults2 = foldTstPred.map(t => revConv.getOrElse(t._1, default) + " " +
					    revConv.getOrElse(t._2, default))    
				strResults2.coalesce(1, true)
					.saveAsTextFile(outputDir + "result" + i + ".tst") /
					
				// Print fold results into the global result file*/
				output += s"Fold $i:\t" +
					traFoldAcc(i) + "\t" + tstFoldAcc(i) + "\n"
			} 
			
  			
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
		    output += aggConfMatrix.fValue.foldLeft("\t")((str, t) => str + "\t" + t._1) + "\n"
		    output += aggConfMatrix.fValue.foldLeft("F-Measure:")((str, t) => str + "\t" + t._2) + "\n"
		    output += aggConfMatrix.precision.foldLeft("Precision:")((str, t) => str + "\t" + t._2) + "\n"
		    output += aggConfMatrix.recall.foldLeft("Recall:")((str, t) => str + "\t" + t._2) + "\n"
		    output += aggConfMatrix.specificity.foldLeft("Specificity:")((str, t) => str + "\t" + t._2) + "\n"
		    output += aggConfMatrix.toString
			println(output)
			
			val hdfsOutput = accResults(0)._1.context.parallelize(Array(output), 1)
			hdfsOutput.saveAsTextFile(outputDir + "globalResult.txt")
  			//val writer = new PrintWriter(new File(outputDir + "globalResult.txt"))
			//writer.write(output); writer.close()
		}
}
