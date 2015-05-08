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
  private var marginalHist: RDD[(Int, BDV[Float])] = null
  private var jointHist: RDD[(Int, BDM[Float])] = null
  
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
  
  /* Pair generator for dense data */
  private def DenseGenerator(
      bv: BV[Byte], 
      varX: Broadcast[Seq[Int]],
      varY: Int,
      varZ: Option[Int]) = {
    
     val zval = varZ match {case Some(z) => Some(bv(z)) case None => None}     
     var pairs = Seq.empty[((Int, Byte, Byte, Option[Byte]), Long)]
     
     for(xind <- varX.value){
       pairs = ((xind, bv(xind), bv(varY), zval), 1L) +: pairs
     }     
     pairs
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
    val nd = nInstances.toFloat
    // A boolean vector that shows that X variables are involved on this computation
    val fselected = Array.ofDim[Boolean](nFeatures + 1)
    fselected(nFeatures) = true 
    varX.map(fselected(_) = true)
    val bFeatSelected = sc.broadcast(fselected)    
    val data = rawData.filter({ case (k, _) => bFeatSelected.value((k / nInstances).toInt)})
     
    // Broadcast vector for Y variable
    val auxCol: Seq[Byte] = if(varY == nFeatures){
	// Auxcol corresponds with output attribute, so we save it 
      classCol = data.filter({ case (k, _) => k / nInstances == varY}).values.collect()
      classCol
    }  else {
      data.filter({ case (k, _) => k / nFeatures == varY}).values.collect()
    }    
    
    val histograms = computeHistograms(data, auxCol, nInstances)
    val prob = histograms.mapValues(_.map(_ / nd))
    
    // Get sum by columns for varY
    //val (_ , histY) = histograms.filter(_ == varY).first
    //val yAcc = sum(histY(::, *)).toArray // More compatible structure to serialize
      
    // If y corresponds with output feature, it is saved for CMI computations
    if(varY == nFeatures) {
      marginalHist = prob.mapValues(h => sum(h(::, *)).toDenseVector).cache()
      jointHist = prob.cache()
    }
    
    val yHist = marginalHist.lookup(varY)(0)
    computeMutualInfo(histograms, yHist, nInstances)
  }
  
  private def computeHistograms(
    data: RDD[(Long, Byte)],
    auxCol: Seq[Byte],
    nInstances: Long) = {
    
      val bAuxCol = data.context.broadcast(auxCol)      
      data.mapPartitions({ it =>
        val max = 256
        var result = Map.empty[Int, BDM[Long]]
        for((k, x) <- it) {
          val feat = (k / nInstances).toInt
          val m = result.getOrElse(feat, BDM.zeros[Long](max, max))
          val inst = k % nInstances
          m(x, bAuxCol.value(inst.toInt)) += 1L
          result += feat -> m
        }
        result.toIterator
      }).reduceByKey(_ + _)
  }
  
  private def computeMutualInfo(
    data: RDD[(Int, BDM[Long])],
    yAcc: BDV[Float],
    n: Long) = {
      
	val byAcc = data.context.broadcast(yAcc)
      data.mapValues({ m =>
        var mi = 0.0d
        for(i <- 0 until m.rows){
          val px = sum(m(i, ::))
          for(j <- 0 until (m.cols - 1)){
            val pxy = m(i, j).toDouble / n
            val py = byAcc.value(j)
            mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
          }
        } 
        mi        
      })  
  }
  
 /* private def getMI(
    combinations: RDD[((Int, Byte, Byte, Option[Byte]), Long)],
    n: Long,
    saveProb: Boolean = false) = {
    
      val freqy = combinations.context.broadcast(marginalClass)      
      combinations.mapPartitions({ it => 
        val elems = it.toArray
        var freqx = Map.empty[Int, Map[Byte, Long]]
        var freqxy = Map.empty[Int, Map[(Byte, Byte), Long]]
        
        // Compute freq counters for marginal and joint probabilities (all inputs)
        elems.map{ case ((kx, x, y, _), q) =>
          var smap = freqx.getOrElse(kx, Map.empty)
          smap += x -> (smap.getOrElse(x, 0L) + q)
          freqx += kx -> smap
          var smap2 = freqxy.getOrElse(kx, Map.empty)
          smap2 += (x, y) -> (smap2.getOrElse((x,y), 0L) + q)
          freqxy += kx -> smap2
        }
        
        // Get mutual informations values using previous frequency counter
        var result = Map.empty[Int, Double]
        val minst = elems.map{ case ((kx, x, y, _), q) =>           
          val px = freqx.getOrElse(kx, Map.empty).getOrElse(x, 0L).toDouble / n
          val py = freqy.value.getOrElse(y, 0L).toDouble / n
          val pxy = q.toDouble / n
          val mi = (pxy * (math.log(pxy / (px * py)) / math.log(2))) + result.getOrElse(kx, 0.0d)
          result += kx -> mi
        }
        
        result.map({case (k, v) => k -> (v, freqx(k), freqxy(k))}).toIterator       
        // Group instances by key and compute the final tuple result        
        /*var result = minst.groupBy(_._1).map{ case (k, a) =>
          val mi = a.map(_._2).sum
          (k, (mi, freqx(k), freqxy(k)))
        }*/       
      })      
  } */
  
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
    
	val histograms3d = computeConditionalHistograms(fdata, yCol, zCol, nInstances)
    
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
    nInstances: Long) = {
    
      val byCol = data.context.broadcast(yCol)
      val bzCol = data.context.broadcast(zCol)
      val zdim = zCol.length
      
      data.mapPartitions({ it =>
        val max = 256
        var result = Map.empty[Int, BDV[BDM[Long]]]
        for((k, x) <- it) {
          val feat = (k / nInstances).toInt
		  val inst = k % nInstances
          val m = result.getOrElse(feat, BDV.fill[BDM[Long]](zdim){BDM.zeros[Long](max, max)})
          val y = byCol.value(inst.toInt); val z = bzCol.value(inst.toInt)
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
    nInstances: Long) = {

		if(jointHist == null || marginalHist == null) 
		  throw new SparkException("Histograms lost. Unexpected exception")
		val sc = data.context		
        val nd = nInstances.toDouble
		val yProb = sc.broadcast(marginalHist.lookup(varY)(0))
		val zProb = sc.broadcast(marginalHist.lookup(varZ)(0))
		val yzProb = sc.broadcast(jointHist.lookup(varY)(0))
	
      data.mapValues({ m =>
        var cmi = 0.0d; var mi = 0.0d
        // Aggregate matrices by row (X)
        val aggX = m.map(h1 => sum(h1(::, *)).toDenseVector)
        // Use the previous variable to do a summatory to obtain X accumulators 
        val xProb = aggX.reduce(_ + _).apply(0).map(_ / nd)
        // Aggregate all matrices in Z to obtain the unique XY matrix
        val xyProb = m.reduce(_ + _).apply(0).map(_ / nd)  
        val xzProb = aggX.map(_.map(_ / nd))
        
        for(z <- 0 until m.length){
          for(x <- 0 until m(z).rows){
            for(y <- 0 until (m(z).cols - 1)) {
              val pz = zProb.value(z)
    		  val pxyz = (m(z)(x, y) / nd) / pz
    		  val pxz = xzProb(z)(x) / pz
    		  val pyz = yzProb.value(y, z) / pz
    		  cmi += pz * pxyz * (math.log(pxyz / (pxz * pyz)) / math.log(2))
    		  if (z == 0) {
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
      

  
  /*private def getCMIandMI(
    combinations: RDD[((Int, Byte, Byte, Option[Byte]), Long)],
    n: Long,
    varY: Int,
    varZ: Int) = {
      if(joints == null || marginals == null || marginalClass == null) throw new Exception()
      
      val sc = combinations.context
      val freqyz = sc.broadcast(joints.lookup(varY)(0))
      val freqy = sc.broadcast(marginals.lookup(varY)(0))
      val freqz = sc.broadcast(marginalClass)
      
      combinations.mapPartitions({ it => 
        
        val elems = it.toArray
        var freqx = Map.empty[Int, Map[Byte, Long]]
        var freqxz = Map.empty[Int, Map[(Byte, Byte), Long]]
        var freqxy = Map.empty[Int, Map[(Byte, Byte), Long]]
        
        // Compute frequency counters for marginal and joint probabilities (xz, xy, x)
        elems.map{ case ((kx, x, y, z), q) =>
          var smap = freqx.getOrElse(kx, Map.empty)
          smap += x -> (smap.getOrElse(x, 0L) + q)
          freqx += kx -> smap
          var smap2 = freqxy.getOrElse(kx, Map.empty)
          smap2 += (x, y) -> (smap2.getOrElse((x,y), 0L) + q)
          freqxy += kx -> smap2
          smap2 = freqxz.getOrElse(kx, Map.empty)
          smap2 += (x, z.get) -> (smap2.getOrElse((x,z.get), 0L) + q)
          freqxz += kx -> smap2          
        }
        
        // Get mutual information values using previous frequency counters
        var result = Map.empty[Int, (Double, Double)]
        // Use a set to avoid re-computing MI values for same (x,y) combinations
        var miVisited = HashSet.empty[(Int, Float, Float)]
        val minst = elems.map{ case ((kx, x, y, z), qxyz) =>             
            val mi = if(!miVisited.contains((kx, x, y))) {
              val px = freqx.getOrElse(kx, Map.empty).getOrElse(x, 0L).toDouble / n
              val py = freqy.value.getOrElse(y, 0L).toDouble / n            
              val pxy = freqxy.getOrElse(kx, Map.empty).getOrElse((x,y), 0L).toDouble / n
              miVisited.add((kx, x, y))
              pxy * (math.log(pxy / (px * py)) / math.log(2))              
            } else {
              0.0d
            }
            
            val pz = freqz.value.getOrElse(z.get, 0L).toDouble / n
            val pxz = (freqxz.getOrElse(kx, Map.empty).getOrElse((x,z.get), 0L).toDouble / n) / pz
            val pyz = (freqyz.value.getOrElse((y,z.get), 0L).toDouble / n) / pz
            val pxyz = (qxyz.toDouble / n) / pz
            val cmi = pz * pxyz * (math.log(pxyz / (pxz * pyz)) / math.log(2))            
            
            val (omi, ocmi) = result.getOrElse(kx, (0.0d, 0.0d))       
            result += kx -> (omi + mi, ocmi + cmi)
        }
        
        result.toIterator
      })      
  } */
  
  class Key1Partitioner(numParts: Int) extends Partitioner {
    override def numPartitions: Int = numParts
    override def getPartition(key: Any): Int = {
      val (indx, _, _, _) = key.asInstanceOf[(Int, Float, Float, Option[Float])]
      val code = indx % numPartitions
      if (code < 0) {
        code + numPartitions  // Make it non-negative
      } else {
        code
      }
    }
    // Java equals method to let Spark compare our Partitioner objects
    override def equals(other: Any): Boolean = other match {
      case fep: Key1Partitioner =>
        fep.numPartitions == numPartitions
      case _ =>
        false
    }
  }
  
}
