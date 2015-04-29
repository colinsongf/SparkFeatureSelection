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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.Partitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import scala.collection.mutable.HashSet

/**
 * Information Theory function and distributed primitives.
 */
object InfoTheory {
  
  private var marginalClass: scala.collection.Map[Byte, Long] = null
  private var marginals: RDD[(Int, Map[Byte, Long])] = null
  private var joints: RDD[(Int, (Map[(Byte, Byte), Long]))] = null  
  
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
   * @param varZ Indexes of conditional variable (must be disjoint  with X and Y)
   * @param n    Number of instances
   * @return     RDD of (primary var, (MI, CMI))
   * 
   */
  def computeMI(
      data: RDD[BV[Byte]],
      varX: Seq[Int],
      varY: Int,
      n: Long,      
      nFeatures: Int) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = data.context
    val bvarX = sc.broadcast(varX)
    
    // Common function to generate pairs, it choose between sparse and dense processing 
    data.first match {
      case v: BDV[Byte] =>
        val generator = DenseGenerator(_: BV[Byte], bvarX, varY, None)
        marginalClass = data.map(_(varY)).countByValue()
        val comb = data.flatMap(generator).reduceByKey(new Key1Partitioner(4), _ + _)
        val miStruct = getMI(comb, n)
        val miValues = miStruct.mapValues(_._1)
        marginals = miStruct.mapValues(_._2).cache()
        joints = miStruct.mapValues(_._3).cache()
        miValues
      case v: BSV[Byte] =>     
        // Not implemented yet!
        throw new NotImplementedError()
    }
  }
  
  private def getMI(
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
  }
  
  def computeCMIandMI(
      data: RDD[BV[Byte]],
      varX: Seq[Int],
      varY: Int,
      varZ: Int,
      n: Long,      
      nFeatures: Int) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = data.context
    val bvarX = sc.broadcast(varX)
    
    // Common function to generate pairs, it choose between sparse and dense processing 
    data.first match {
      case v: BDV[Byte] =>
        val generator = DenseGenerator(_: BV[Byte], bvarX, varY, Some(varZ))
        val comb = data.flatMap(generator).reduceByKey(new Key1Partitioner(4), _ + _)
        getCMIandMI(comb, n, varY, varZ)
      case v: BSV[Byte] =>
        // Not implemented yet!
        throw new NotImplementedError()
    }
  }
  
  private def getCMIandMI(
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
  }
  
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
