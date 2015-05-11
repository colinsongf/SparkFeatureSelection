/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import breeze.linalg._
import breeze.numerics._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM}

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.Partitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import scala.collection.mutable.HashSet
import org.apache.spark.SparkException

/**
 * Information Theory function and distributed primitives.
 */
object InfoTheory {
  
  private var classCol: Array[Byte] = null
  private var marginalProb: RDD[(Int, BDV[Float])] = null
  private var jointProb: RDD[(Int, BDM[Float])] = null
  
  /**
   * Calculate entropy for the given frequencies.
   *
   * @param freqs Frequencies of each different class
   * @param n Number of elements
   * 
   */
  private[feature] def entropy(freqs: Seq[Long], n: Long) = {
    freqs.aggregate(0.0)({ case (h, q) =>
      h + (if (q == 0) 0  else (q.toDouble / n) * (math.log(q.toDouble / n) / math.log(2)))
    }, { case (h1, h2) => h1 + h2 }) * -1
  }

  /**
   * Calculate entropy for the given frequencies.
   *
   * @param freqs Frequencies of each different class
   */
  private[feature] def entropy(freqs: Seq[Long]): Double = {
    entropy(freqs, freqs.reduce(_ + _))
  }  
  
  /**
   * Method that calculates mutual information (MI) and conditional mutual information (CMI) 
   * simultaneously for several variables. Indexes must be disjoint.
   *
   * @param data RDD of data (first element is the class attribute)
   * @param varX Indexes of primary variables (must be disjoint with Y and Z)
   * @param varY Indexes of secondary variable (must be disjoint with X and Z)
   * @param nInstances    Number of instances
   * @param nFeatures Number of features (including output ones)
   * @return  RDD of (primary var, (MI, CMI))
   * 
   */
  def computeMI(
      rawData: RDD[(Long, Byte)],
      varX: Seq[Int],
      varY: Int,
      nInstances: Long,      
      nFeatures: Int) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = rawData.context
    val label = nFeatures - 1
    // A boolean vector that shows that X variables are involved on this computation
    val fselected = Array.ofDim[Boolean](nFeatures)
    fselected(label) = true // output feature
    varX.map(fselected(_) = true)
    val bFeatSelected = sc.broadcast(fselected)
    val data = rawData.filter({ case (k, _) => bFeatSelected.value((k % nFeatures).toInt)})
     
    // Broadcast vector for Y variable
    val yCol: Array[Byte] = if(varY == label){
	    // classCol corresponds with output attribute which is re-used in the iteration 
      classCol = data.filter({ case (k, _) => k % nFeatures == varY}).values.collect()
      classCol
    }  else {
      data.filter({ case (k, _) => k % nFeatures == varY}).values.collect()
    }    
    
    val histograms = computeHistograms(data, yCol, nFeatures)
    val jointTable = histograms.mapValues(_.map(_.toFloat / nInstances))
    val marginalTable = jointTable.mapValues(h => sum(h(::, *)).toDenseVector)
      
    // If y corresponds with output feature, we save for CMI computation
    if(varY == label) {
      marginalProb = marginalTable.cache()
      jointProb = jointTable.cache()
    }
    
    val yProb = marginalTable.lookup(varY)(0)
    computeMutualInfo(histograms.filter{case (k, _) => k != label}, yProb, nInstances)
  }
  
  private def computeHistograms(
      data: RDD[(Long, Byte)],
      yCol: Array[Byte],
      nFeatures: Long) = {
    
    val byCol = data.context.broadcast(yCol)    
    data.mapPartitions({ it =>
      val max = 256
      var result = Map.empty[Int, BDM[Long]]
      for((k, x) <- it) {
        val feat = (k % nFeatures).toInt; val inst = (k / nFeatures).toInt
        val m = result.getOrElse(feat, BDM.zeros[Long](max, max))        
        m(x, byCol.value(inst)) += 1
        result += feat -> m
      }
      result.toIterator
    }).reduceByKey(_ + _)
  }
  
  private def computeMutualInfo(
      data: RDD[(Int, BDM[Long])],
      yProb: BDV[Float],
      n: Long) = {
    
    
    val byProb = data.context.broadcast(yProb)
    
      val local = data.collect()
      for((_, m) <- local) {
        var mi = 0.0d
        val xProb = sum(m(*, ::)).map(_.toFloat / n)
        for(i <- 0 until m.rows){
          for(j <- 0 until m.cols){
            val pxy = m(i, j).toFloat / n
            val px = xProb(i)
            val py = byProb.value(j)
            mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
          }
        } 
        mi
      }
       
    
    
    data.mapValues({ m =>
      var mi = 0.0d
      val xProb = sum(m(*, ::)).map(_.toFloat / n)
      for(i <- 0 until m.rows){
        for(j <- 0 until m.cols){
          val pxy = m(i, j).toFloat / n
          val py = byProb.value(j); val px = xProb(i)
          if(pxy != 0 && px != 0 && py != 0)
            mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
        }
      } 
      mi        
    })  
  }
  
  def computeMIandCMI(
      data: RDD[(Long, Byte)],
      varX: Seq[Int],
      varY: Int,
      varZ: Int,
      nInstances: Long,      
      nFeatures: Int) = {
    
    
      // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = data.context
    // A boolean vector that shows that X variables are involved on this computation
    val fselected = Array.ofDim[Boolean](nFeatures + 1)
    fselected(nFeatures) = true 
    varX.map(fselected(_) = true)
    val bFeatSelected = sc.broadcast(fselected)    
    val fdata = data.filter({ case (k, _) => bFeatSelected.value((k / nInstances).toInt)})
     
    // Broadcast vector for Y and Z variable
	  val yCol = fdata.filter({ case (k, _) => k / nInstances == varY}).values.collect()
	  val zCol = if(classCol == null){
  		fdata.filter({ case (k, _) => k / nInstances == varZ}).values.collect()
  	} else {
  		classCol
  	} 	    
    
	  val histograms3d = computeConditionalHistograms(fdata, yCol, zCol, nFeatures)
    
    // Get sum by columns for varY
    //val (_ , histY) = histograms.filter(_ == varY).first
    //val yAcc = sum(histY(::, *)).toArray // More compatible structure to serialize
      
    // If y corresponds with output feature, it is saved for CMI computations
    computeConditionalMutualInfo(histograms3d, varY, varZ, nInstances)
 }

  private def computeConditionalHistograms(
    data: RDD[(Long, Byte)],
    yCol: Array[Byte],
    zCol: Array[Byte],
    nFeatures: Long) = {
    
      val byCol = data.context.broadcast(yCol)
      val bzCol = data.context.broadcast(zCol)
      val zdim = zCol.length
      
      data.mapPartitions({ it =>
        val max = 256
        var result = Map.empty[Int, BDV[BDM[Long]]]
        for((k, x) <- it) {
          val feat = (k % nFeatures).toInt; val inst = (k / nFeatures).toInt
          val m = result.getOrElse(feat, BDV.fill[BDM[Long]](zdim){BDM.zeros[Long](max, max)})
          val y = byCol.value(inst); val z = bzCol.value(inst)
          m(z)(x, y) += 1
          result += feat -> m
        }
        result.toIterator
      }).reduceByKey(_ + _)
  }
  
  private def computeConditionalMutualInfo(
      data: RDD[(Int, BDV[BDM[Long]])],
      varY: Int,
      varZ: Int,
      n: Long) = {

	  if(jointProb == null || marginalProb == null) 
	    throw new SparkException("Histograms lost. Unexpected exception")
    val sc = data.context
	  val yProb = sc.broadcast(marginalProb.lookup(varY)(0))
  	val zProb = sc.broadcast(marginalProb.lookup(varZ)(0))
	  val yzProb = sc.broadcast(jointProb.lookup(varY)(0))

    data.mapValues({ m =>
      var cmi = 0.0d; var mi = 0.0d
      // Aggregate matrices by row (X)
      val aggX = m.map(h1 => sum(h1(::, *)).toDenseVector)
      // Use the previous variable to sum up and so obtaining X accumulators 
      val xProb = aggX.reduce(_ + _).apply(0).map(_.toFloat / n)
      // Aggregate all matrices in Z to obtain the unique XY matrix
      val xyProb = m.reduce(_ + _).apply(0).map(_.toFloat / n)  
      val xzProb = aggX.map(_.map(_.toFloat / n))
      
      for(z <- 0 until m.length){
        for(x <- 0 until m(z).rows){
          for(y <- 0 until m(z).cols) {
            val pz = zProb.value(z)
  		      val pxyz = (m(z)(x, y).toFloat / n) / pz
  		      val pxz = xzProb(z)(x) / pz
  		      val pyz = yzProb.value(y, z) / pz
  		      cmi += pz * pxyz * (math.log(pxyz / (pxz * pyz)) / math.log(2))
      		  if (z == 0) { // Do MI computations only once
              val px = xProb(x)
              val pxy = xyProb(x, y)
              val py = yProb.value(y)
              mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
  		      }
          }            
        }
      } 
      (mi, cmi)        
    })  
  }
  
}
