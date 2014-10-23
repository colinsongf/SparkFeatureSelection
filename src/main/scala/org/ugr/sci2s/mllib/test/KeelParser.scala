package org.ugr.sci2s.mllib.test

import keel.Dataset._
import java.util.ArrayList
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object KeelParser {
  
	def parseHeaderFile (sc: SparkContext, file: String): Array[Map[String, Double]] = {
  	  val header = sc.textFile(file)
  	  // Java code classes
  	  var arr: ArrayList[String] = new ArrayList[String]
  	  // Important to collect and work with arrays instead of RDD's
  	  for(x <- header.toArray) arr.add(x)
  	 
  	  new InstanceSet().parseHeaderFromString(arr, true)
  	  
  	  var conv: Array[Map[String, Double]] = new Array[Map[String, Double]](Attributes.getNumAttributes)
  	  for(i <- 0 until Attributes.getNumAttributes) {
  		  conv(i) = Map()
  		  if(Attributes.getAttribute(i).getType == Attribute.NOMINAL){
  			  val values: java.util.Vector[String] = Attributes.getAttribute(i).getNominalValuesList()
  					  .asInstanceOf[java.util.Vector[String]]
			  val gen = for (j <- 0 until values.size) 
			    yield (values.get(j) -> j.toDouble / (values.size - 1))
			  conv(i) = gen.toMap
  		  }    	
  	  }
  	  
  	  conv
  	}
  
	def parseLabeledPoint (conv: Array[Map[String, Double]], str: String): LabeledPoint = {
	  
		val tokens = str split ","
		var x = new Array[Double](tokens.length)
		var y = 0.0
		
		for(i <- 0 until tokens.length) {
		  val value = if(conv(i).isEmpty) tokens(i).toDouble else conv(i)(tokens(i))
		  if(i < (tokens.length - 1)) x(i) = value else y = value
		}
		
		new LabeledPoint(y, Vectors.dense(x))
	}
	
	def parsePoint (conv: Array[Map[String, Double]], str: String, omitLast: Boolean): org.apache.spark.mllib.linalg.Vector = {
	  
		val tokens = str split ","
		var x = new Array[Double](tokens.length)
		var y = 0.0
		val vsize = if (omitLast) tokens.length - 1 else tokens.length
		
		for(i <- 0 until vsize) {
		  x(i) = if(conv(i).isEmpty) tokens(i).toDouble else conv(i)(tokens(i))
		}
		
		Vectors.dense(x)
	}
}