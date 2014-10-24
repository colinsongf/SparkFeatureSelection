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

object MultiClassificationUtils {
  
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
			//val typeconv = typeConversion.asInstanceOf[Array[scala.collection.immutable.Map[String,scala.Double]]]
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
			
			val accResults = dataFiles.map { case (trainFile, testFile) => {
			  
				var initAllTime = System.nanoTime()
				val trainData = sc.textFile(trainFile).
						//sample(false, samplingRate, seed.nextLong).
						map(line => (KeelParser.parseLabeledPoint(bcTypeConv, line)))
				val testData = sc.textFile(testFile).
						//sample(false, samplingRate, seed.nextLong).
						map(line => (KeelParser.parseLabeledPoint(bcTypeConv, line)))
				trainData.persist(); testData.persist()
				
				// Discretization
				var initStartTime = System.nanoTime()
				var trData = trainData
				var tstData = testData
				
				val haveToDisc = discretize match { 
				  case Some(i) => true
				  case None => false
				}
				
				var discTime = 0.0
				if(haveToDisc) {
					initStartTime = System.nanoTime()
					val (discAlgorithm, discData) = discretize.get(trainData)
					trData = discData
					discTime = (System.nanoTime() - initStartTime) / 1e9
					tstData = discAlgorithm.discretize(tstData)
				}				
				times("DiscTime") = times("DiscTime") :+ discTime

				// Feature Selection
				val haveToFS = featureSelect match { 
				  case Some(i) => true
				  case None => false
				}
				
				var FSTime = 0.0
				if(haveToFS) {
					initStartTime = System.nanoTime()
					val (featureSelector, reductedData) = featureSelect.get(trData)
					trData = reductedData
					FSTime = (System.nanoTime() - initStartTime) / 1e9
					tstData = featureSelector.select(tstData)
				}
				times("FSTime") = times("FSTime") :+ FSTime
				
				// Run training algorithm to build the model
				initStartTime = System.nanoTime()
				val classificationModel = classify(trData)
				val classifficationTime = (System.nanoTime() - initStartTime) / 1e9		
				times("ClsTime") = times("ClsTime") :+ classifficationTime
				
				val traValuesAndPreds = computePredictions(classificationModel, trData)
				val tstValuesAndPreds = computePredictions(classificationModel, tstData)
				
				trainData.unpersist(); testData.unpersist()
				
				var fullTime = (System.nanoTime() - initAllTime) / 1e9
				times("FullTime") = times("FullTime") :+ fullTime
				
				(traValuesAndPreds, tstValuesAndPreds)}
			}
			
			// Do the output (fold results files and global result file)
			//val dir = new File("results/" + outputDir); dir.mkdirs()
			doOutput(outputDir, accResults, info, times.toMap, typeConversion)
		}

		
		def doOutput (outputDir: String, 
				accResults: Array[(RDD[(Double, Double)], RDD[(Double, Double)])], 
				info: Map[String, String],
				timeResults: Map[String, Seq[Double]],
				typeConv: Array[Map[String,Double]]) {
		  
  			val revConv = typeConv.last.map(_.swap) // for last class
  			val default = "error"
		  
			var output = info.get("algoInfo").get + "Accuracy Results\tTrain\tTest\n"
			val traFoldAcc= accResults.map(_._1).map(computeAccuracy)
			val tstFoldAcc = accResults.map(_._2).map(computeAccuracy)
			
			// Aggregated statistics
			val (aggAvgAccTr, aggStdAccTr) = calcAggStatistics(traFoldAcc)
			val (aggAvgAccTst, aggStdAccTst) = calcAggStatistics(tstFoldAcc)
			
			
			//val aggConfMatrix = ConfusionMatrix.apply(accResults.map(_._2).reduceLeft(_ union _))
			val aggConfMatrix = ConfusionMatrix.apply(accResults.map(_._2).reduceLeft(_ union _), 
			    typeConv.last)			    
			
			// Output of the algorithm
			(0 until accResults.size).map { i =>
				// Print training fold results
				val (foldTrPred, foldTstPred) = accResults(i)
				val strResults = foldTrPred.map(t => revConv.getOrElse(t._1, default) + " " +
					    revConv.getOrElse(t._2, default))   
				strResults.coalesce(1, true).
					saveAsTextFile(outputDir + "result" + i + ".tra")
				val strResults2 = foldTstPred.map(t => revConv.getOrElse(t._1, default) + " " +
					    revConv.getOrElse(t._2, default))    
				strResults2.coalesce(1, true)
					.saveAsTextFile(outputDir + "result" + i + ".tst")
				// Print fold results into the global result file*/
				output += s"Fold $i:\t" +
					traFoldAcc(i) + "\t" + tstFoldAcc(i) + "\n"
			} 
			
  			val value = aggConfMatrix.fValue.foldLeft("")((str, nstr) => str + "\t" + nstr)
  			
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