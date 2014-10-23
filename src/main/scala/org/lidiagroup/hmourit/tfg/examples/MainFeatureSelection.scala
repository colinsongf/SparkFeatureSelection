package org.lidiagroup.hmourit.tfg.examples

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.lidiagroup.hmourit.tfg._
import org.lidiagroup.hmourit.tfg.generators._
import org.lidiagroup.hmourit.tfg.discretization._
import org.lidiagroup.hmourit.tfg.featureselection._
import org.lidiagroup.hmourit.tfg.featureselection.InfoThCriterionFactory
import org.ugr.sci2s.mllib.test.KeelParser
import scala.util.Random

object MainFeatureSelection extends App {
	override def main(args: Array[String]) {
		
		var initAllTime = System.nanoTime()
		if (args.length < 3) {
			System.err.println("Usage: FS <header-file> <train-file> <test-file>")
			System.exit(1)
		}
		
		val headerFile = args(0)
		val trainFile = args(1)
		val testFile = args(2)
		val nonDefaut = args.length > 3
		val conf = new SparkConf()
					.setMaster("local[*]")
					.setAppName("test")
		val sc = new SparkContext(conf)
		
		val samplingRate = 0.01
		val seed = new Random
		
		// Load training data
		val dataInfo = sc.textFile(headerFile).collect.reduceLeft(_ + "\n" + _)
				
		val typeConversion = KeelParser.parseHeaderFile(sc, headerFile)	
		//val typeconv = typeConversion.asInstanceOf[Array[scala.collection.immutable.Map[String,scala.Double]]]
		val binary = typeConversion.last.size < 3
		//if(!binary) throw new IllegalArgumentException("Data class not binary...")
		val bcTypeConv = sc.broadcast(typeConversion).value
		
		val trainData = sc.textFile(trainFile).
					sample(false, samplingRate, seed.nextLong).
					map(line => (KeelParser.parseLabeledPoint(bcTypeConv, line)))
		val testData = sc.textFile(testFile).
					sample(false, samplingRate, seed.nextLong).
					map(line => (KeelParser.parseLabeledPoint(bcTypeConv, line)))
					
		var initStartTime = System.nanoTime()
		val discretizer = EntropyMinimizationDiscretizer.train(trainData,
		      0 until trainData.first.features.size, // continuous features 
		      10) // max number of values per feature
		val disData = discretizer.discretize(trainData)
		val discTime = (System.nanoTime() - initStartTime) / 1e9
		// Feature Selection
		val criterion = new InfoThCriterionFactory("jmi")
		val model = InfoThFeatureSelection.train(criterion, 
		      disData, //data 
		      100) // number of features to select
		
		val reducedData = model.select(disData)
		val FSTime = (System.nanoTime() - initStartTime) / 1e9
		val fullTime = (System.nanoTime() - initAllTime) / 1e9
		
		print(discretizer.thresholds.toString)
		println("Full time:\t" + fullTime)
		println("Disc time:\t" + discTime)
		println("FS time:\t" + FSTime)		
		//print(disData)
	}	
}