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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.SparkException

import scala.collection.immutable.HashMap
import scala.collection.immutable.LongMap
import scala.collection.mutable

/**
 * Information Theory function and distributed primitives.
 */
object InfoTheory {
  
  private var classCol: Array[Array[Byte]] = null
  private var classColSparse: HashMap[Long, Byte] = null
  private var marginalProb: RDD[(Int, BDV[Float])] = null
  private var jointProb: RDD[(Int, BDM[Float])] = null
  private var classHist: Map[Byte, Long] = null
  
  private def computeFrequency(data: HashMap[Long, Byte], nInstances: Long) = {
    val tmp = data.values.groupBy(l => l).map(t => (t._1, t._2.size.toLong))
    tmp.get(0) match {
      case Some(_) => tmp
      case None => 
        val lastElem = (0: Byte, nInstances - tmp.values.sum) 
        tmp + lastElem
    }
  }
  
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
  def computeMISparse(
      rawData: RDD[(Int, HashMap[Long, Byte])],
      varX: Seq[Int],
      varY: Int,
      nInstances: Long,      
      nFeatures: Int,
      counter: Map[Int, Int]) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = rawData.context
    val label = nFeatures - 1
    // A boolean vector that indicates the variables involved on this computation
    val fselected = Array.ofDim[Boolean](nFeatures)
    fselected(varY) = true // output feature
    varX.map(fselected(_) = true)
    val bFeatSelected = sc.broadcast(fselected)
    // Filter data by these variables
    val data = rawData.filter({ case (k, _) => bFeatSelected.value(k)})
     
    // Broadcast Y vector
    val ycol = data.lookup(varY)(0)    
    val yhist = if(varY == label) {
      classColSparse = ycol
      classHist = computeFrequency(classColSparse, nInstances)
      classHist
    } else {
      computeFrequency(ycol, nInstances)
    }
    
    val histograms = computeHistogramsSparse(data, (varY, ycol), yhist, counter, nInstances)
    val jointTable = histograms.mapValues(_.map(_.toFloat / nInstances))
    val marginalTable = jointTable.mapValues(h => sum(h(*, ::)).toDenseVector)
      
    // If y corresponds with output feature, we save for CMI computation
    if(varY == label) {
      marginalProb = marginalTable.cache()
      jointProb = jointTable.cache()      
    }
        
    // Remove output feature from the computations
    val fdata = histograms.filter{case (k, _) => k != label}
    val marginalY = marginalTable.lookup(varY)(0)
    computeMutualInfo(fdata, marginalY, nInstances)
  }
  
  private def computeHistogramsSparse(
      data:  RDD[(Int, HashMap[Long, Byte])],
      ycol: (Int, HashMap[Long, Byte]),
      yhist: Map[Byte, Long],
      counter: Map[Int, Int],
      nInstances: Long) = {
    
    val maxSize = 256
    val bycol = data.context.broadcast(ycol._2)    
    val bCounter = data.context.broadcast(counter) 
    val ys = counter.getOrElse(ycol._1, maxSize).toInt
    // To avoid serializing the whole object
      
    data.map({ case (feat, xcol) =>        
      val result = BDM.zeros[Long](
          bCounter.value.getOrElse(feat, maxSize).toInt, ys)
      
      val histCls = mutable.HashMap.empty ++= yhist // clone
      for ( (inst, x) <- xcol){     
        val y = bycol.value.getOrElse(inst, 0: Byte)     
        histCls += y -> (histCls(y) - 1)
        result(x, y) += 1
      }
      // Zeros count
      histCls.foreach({ case (c, q) => result(0, c) += q })
      feat -> result
    })
  }
  
  private def computeHistogramsSparse2(
      data:  RDD[(Int, HashMap[Long, Byte])],
      ycol: (Int, HashMap[Long, Byte]),
      nInstances: Long,
      counter: Map[Int, Int]) = {
    
    val maxSize = 256 
    val bycol = data.context.broadcast(ycol._2)    
    val bCounter = data.context.broadcast(counter) 
    val ys = counter.getOrElse(ycol._1, maxSize).toInt
      
    data.mapPartitions({ it =>
      var result = Map.empty[Int, BDM[Long]]
      val data = it.toArray
      (0 until nInstances.toInt).map({ inst =>
        val yval = bycol.value.getOrElse(inst, 0: Byte)
        data.map({ case (feat, xcol) =>
          val mat = result.getOrElse(feat, 
            BDM.zeros[Long](bCounter.value.getOrElse(feat, maxSize).toInt, ys)) 
          val xval = xcol.getOrElse(inst, 0: Byte)          
          mat(xval, yval) += 1
          result += feat -> mat
        })
      })
      result.toIterator
    }).reduceByKey(_ + _)
  }
  
  def computeMIandCMISparse(
      rawData: RDD[(Int, HashMap[Long, Byte])],
      varX: Seq[Int],
      varY: Int,
      varZ: Int,
      nInstances: Long,      
      nFeatures: Int,
      counter: Map[Int, Int]) = {    
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = rawData.context
    val label = nFeatures - 1
    // A boolean vector that indicates the variables involved on this computation
    val fselected = Array.ofDim[Boolean](nFeatures)
    fselected(varY) = true; fselected(varZ) = true
    varX.map(fselected(_) = true)
    val bFeatSelected = sc.broadcast(fselected)
    // Filter data by these variables
    val data = rawData.filter({ case (k, _) => bFeatSelected.value(k)})
     
    // Prepare Y and Z vector
    val ycol = data.lookup(varY)(0)    
    val zcol = if(varZ == label) classColSparse else data.lookup(varZ)(0)   
    val zhist = if(varZ == label) classHist else computeFrequency(zcol, nInstances)
    
    // Compute conditional histograms for all variables
    // Then, we remove those not present in X set
    val histograms3d = computeConditionalHistogramsSparse(
        data, (varY, ycol), (varZ, zcol), zhist, counter, nInstances)
        .filter{case (k, _) => k != varZ && k != varY}
    
    // Compute CMI and MI for all X variables
    computeConditionalMutualInfo(histograms3d, varY, varZ, nInstances)
 }
  
  private def computeConditionalHistogramsSparse(
    data: RDD[(Int, HashMap[Long, Byte])],
    ycol: (Int, HashMap[Long, Byte]),
    zcol: (Int, HashMap[Long, Byte]),
    zhist: Map[Byte, Long],
    counter: Map[Int, Int],
    nInstances: Long) = {
    
      val bycol = data.context.broadcast(ycol._2)
      val bzcol = data.context.broadcast(zcol._2)
      val bCounter = data.context.broadcast(counter)
      //val bClassHist = data.context.broadcast(classHist)
      val ys = counter.getOrElse(ycol._1, 256)
      val zs = counter.getOrElse(zcol._1, 256)
      
      
       /* val (feat, xcol) = data.first()
        val result = BDV.fill[BDM[Long]](zs){
          BDM.zeros[Long](bCounter.value.getOrElse(feat, 256), ys)
        }
        
        val histCls = mutable.HashMap.empty ++= zhist // clone
        for ( (inst, x) <- xcol){     
          val y = bycol.value.getOrElse(inst, 0: Byte)
          val z = bzcol.value.getOrElse(inst, 0: Byte)          
          histCls += z -> (histCls(z) - 1)
          result(z)(x, y) += 1
        }
        // Zeros count
        histCls.foreach({ case (c, q) => result(c)(0, 0) += q })
        println("xcol: " + xcol.size)
        println("zcol: " + zcol._2.size)
        println("histCls: " + histCls)*/
      
      data.map({ case (feat, xcol) =>        
        val result = BDV.fill[BDM[Long]](zs){
          BDM.zeros[Long](bCounter.value.getOrElse(feat, 256), ys)
        }
        
        val histCls = mutable.HashMap.empty ++= zhist // clone
        for ( (inst, x) <- xcol){     
          val y = bycol.value.getOrElse(inst, 0: Byte)
          val z = bzcol.value.getOrElse(inst, 0: Byte)          
          histCls += z -> (histCls(z) - 1)
          result(z)(x, y) += 1
        }
        // Zeros count
        histCls.foreach({ case (c, q) => result(c)(0, 0) += q })
        feat -> result
    })
  }  
    
  private def computeConditionalHistogramsSparse2(
    data: RDD[(Int, HashMap[Long, Byte])],
    ycol: (Int, HashMap[Long, Byte]),
    zcol: (Int, HashMap[Long, Byte]),
    nInstances: Long,
    counter: Map[Int, Int]) = {
    
      val bycol = data.context.broadcast(ycol._2)
      val bzcol = data.context.broadcast(zcol._2)
      val bCounter = data.context.broadcast(counter)
      val ys = counter.getOrElse(ycol._1, 256)
      val zs = counter.getOrElse(zcol._1, 256)
      
      data.mapPartitions({ it =>
        var result = Map.empty[Int, BDV[BDM[Long]]]
        val data = it.toArray
        (0 until nInstances.toInt).map({ inst =>      
          val y = bycol.value.getOrElse(inst, 0: Byte)
          val z = bzcol.value.getOrElse(inst, 0: Byte)
          data.map({ case (feat, xcol) =>
            val m = result.getOrElse(feat, 
                BDV.fill[BDM[Long]](zs){BDM.zeros[Long](bCounter.value.getOrElse(feat, 256), ys)})
            val x = xcol.getOrElse(inst, 0: Byte)
            m(z)(x, y) += 1            
            result += feat -> m
          })
        })
        result.toIterator
      }).reduceByKey(_ + _)
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
      rawData: RDD[(Int, (Int, Array[Byte]))],
      varX: Seq[Int],
      varY: Int,
      nInstances: Long,      
      nFeatures: Int,
      counter: Map[Int, Int]) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = rawData.context
    val label = nFeatures - 1
    // A boolean vector that indicates the variables involved on this computation
    val fselected = Array.ofDim[Boolean](nFeatures)
    fselected(varY) = true // output feature
    varX.map(fselected(_) = true)
    val bFeatSelected = sc.broadcast(fselected)
    // Filter data by these variables
    val data = rawData.filter({ case (k, _) => bFeatSelected.value(k)})
     
    // Broadcast Y vector
    val yvals = data.lookup(varY)
    var ycol = Array.ofDim[Array[Byte]](yvals.length)
    yvals.foreach({ case (b, v) => ycol(b) = v })
    
    // classCol corresponds with output attribute, which is re-used in the iteration
    if(varY == label) classCol = ycol
    
    val histograms = computeHistograms(data, (varY, ycol), nFeatures, counter)
    val jointTable = histograms.mapValues(_.map(_.toFloat / nInstances))
    val marginalTable = jointTable.mapValues(h => sum(h(*, ::)).toDenseVector)
      
    // If y corresponds with output feature, we save for CMI computation
    if(varY == label) {
      marginalProb = marginalTable.cache()
      jointProb = jointTable.cache()
    }
    
    val yProb = marginalTable.lookup(varY)(0)
    // Remove output feature from the computations
    val fdata = histograms.filter{case (k, _) => k != label}
    computeMutualInfo(fdata, yProb, nInstances)
  }
  
  private def computeHistograms(
      data:  RDD[(Int, (Int, Array[Byte]))],
      ycol: (Int, Array[Array[Byte]]),
      nFeatures: Long,
      counter: Map[Int, Int]) = {
    
    val maxSize = 256 
    val bycol = data.context.broadcast(ycol._2)    
    val bCounter = data.context.broadcast(counter) 
    val ys = counter.getOrElse(ycol._1, maxSize).toInt
      
    data.mapPartitions({ it =>
      var result = Map.empty[Int, BDM[Long]]
      for((feat, (block, arr)) <- it) {
        val m = result.getOrElse(feat, 
            BDM.zeros[Long](bCounter.value.getOrElse(feat, maxSize).toInt, ys)) 
        for(i <- 0 until arr.length) 
          m(arr(i), bycol.value(block)(i)) += 1
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
    data.mapValues({ m =>
      var mi = 0.0d
      // Aggregate by row (x)
      val xProb = sum(m(*, ::)).map(_.toFloat / n)
      for(i <- 0 until m.rows){
        for(j <- 0 until m.cols){
          val pxy = m(i, j).toFloat / n
          val py = byProb.value(j); val px = xProb(i)
          if(pxy != 0 && px != 0 && py != 0) // To avoid NaNs
            mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
        }
      } 
      mi        
    })  
  }
  
  def computeMIandCMI(
      rawData: RDD[(Int, (Int, Array[Byte]))],
      varX: Seq[Int],
      varY: Int,
      varZ: Int,
      nInstances: Long,      
      nFeatures: Int,
      counter: Map[Int, Int]) = {    
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = rawData.context
    val label = nFeatures - 1
    // A boolean vector that indicates the variables involved on this computation
    val fselected = Array.ofDim[Boolean](nFeatures)
    fselected(varY) = true; fselected(varZ) = true
    varX.map(fselected(_) = true)
    val bFeatSelected = sc.broadcast(fselected)
    // Filter data by these variables
    val data = rawData.filter({ case (k, _) => bFeatSelected.value(k)})
     
    // Prepare Y and Z vector
    val yvals = data.lookup(varY)
	  val ycol = Array.ofDim[Array[Byte]](yvals.length)
    yvals.foreach({ case (b, v) => ycol(b) = v })
    
	  val zcol = if(classCol != null) classCol else throw new SparkException(
        "Output column must be computed and cached.")    
    
    // Compute conditional histograms for all variables
    // Then, we remove those not present in X set
	  val histograms3d = computeConditionalHistograms(
        data, (varY, ycol), (varZ, zcol), nFeatures, counter)
        .filter{case (k, _) => k != varZ && k != varY}
    
    // Compute CMI and MI for all X variables
    computeConditionalMutualInfo(histograms3d, varY, varZ, nInstances)
 }

  private def computeConditionalHistograms(
    data: RDD[(Int, (Int, Array[Byte]))],
    ycol: (Int, Array[Array[Byte]]),
    zcol: (Int, Array[Array[Byte]]),
    nFeatures: Long,
    counter: Map[Int, Int]) = {
    
      val bycol = data.context.broadcast(ycol._2)
      val bzcol = data.context.broadcast(zcol._2)
      val bCounter = data.context.broadcast(counter)
      val ys = counter.getOrElse(ycol._1, 256)
      val zs = counter.getOrElse(zcol._1, 256)
      
      data.mapPartitions({ it =>
        var result = Map.empty[Int, BDV[BDM[Long]]]
        for((feat, (block, arr)) <- it) {
          // We create a vector (z) of matrices (x,y) to represent a 3-dim matrix
          val m = result.getOrElse(feat, 
              BDV.fill[BDM[Long]](zs){BDM.zeros[Long](bCounter.value.getOrElse(feat, 256), ys)})
          for(i <- 0 until arr.length){
            val y = bycol.value(block)(i)
            val z = bzcol.value(block)(i)
            m(z)(arr(i), y) += 1
          }
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
	    throw new SparkException("First, you must compute simple mutual info. (computeMI).")
    val sc = data.context
	  val yProb = sc.broadcast(marginalProb.lookup(varY)(0))
  	val zProb = sc.broadcast(marginalProb.lookup(varZ)(0))
	  val yzProb = sc.broadcast(jointProb.lookup(varY)(0))    

    data.mapValues({ m =>
      var cmi = 0.0d; var mi = 0.0d
      // Aggregate matrices by row (X)
      val aggX = m.map(h1 => sum(h1(*, ::)).toDenseVector)
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
            if(pxz != 0 && pyz != 0 && pxyz != 0)
  		        cmi += pz * pxyz * (math.log(pxyz / (pxz * pyz)) / math.log(2))
      		  if (z == 0) { // Do MI computations only once
              val px = xProb(x)
              val pxy = xyProb(x, y)
              val py = yProb.value(y)
              if(pxy != 0 && px != 0 && py != 0)
                mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
  		      }
          }            
        }
      } 
      (mi, cmi)        
    })  
  }
  
}
