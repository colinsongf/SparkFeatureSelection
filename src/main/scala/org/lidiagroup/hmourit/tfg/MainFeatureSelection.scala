package org.lidiagroup.hmourit.tfg

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.lidiagroup.hmourit.tfg._
import org.lidiagroup.hmourit.tfg.generators._
import org.lidiagroup.hmourit.tfg.discretization._
import org.lidiagroup.hmourit.tfg.featureselection._
import org.lidiagroup.hmourit.tfg.featureselection.InfoThCriterionFactory

object MainFeatureSelection extends App {
	override def main(args: Array[String]) {
		val conf = new SparkConf()
						.setMaster("local[*]").setAppName("test")
		
		//conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
		//conf.set("spark.kryo.registrator", "mypackage.MyRegistrator")
		val sc = new SparkContext(conf)

		// Discrete data
		/*val disGenerator = new DiscreteDataGenerator
		val disData = disGenerator.generateDataWithDT(sc,
		      10, // number of relevant features
		      0.01, // noise of the relevant features 
		      (1 to 9).map( _ * 0.1), // noise of the redundant features
		      20, // number of random features
		      10, // max number of values per feature
		      5, // max depth of the generating tree
		      2, // number of labels
		      10000) // number of data points */
		
		// Continuous data, it should be discretized
		val conGenerator = new GaussianGenerator
		val conData = conGenerator.generate(sc,
		      1000, // number of data points 
		      10, // number of relevant features 
		      9, // number of redundant features 
		      20, // number of random features 
		      2.0, // max norm of mean vectors 
		      0.2, // min variance in each dimension 
		      1.0, // max variance in each dimension
		      0.5) // red noise
		
		val discretizer = EntropyMinimizationDiscretizer.train(conData,
		      0 until conData.first.features.size, // continuous features 
		      10) // max number of values per feature
		
		val disData = discretizer.discretize(conData)
		
		// Feature Selection
		val criterion = new InfoThCriterionFactory("jmi")
		val model = InfoThFeatureSelection.train(criterion, 
		      disData, //data 
		      5) // number of features to select
		
		print(model.features.toString()) // selected features
		
		val reducedData = model.select(disData)
	}	
}