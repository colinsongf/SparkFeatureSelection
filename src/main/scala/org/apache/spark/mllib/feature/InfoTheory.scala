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

/**
 * Information Theory function and distributed primitives.
 */
object InfoTheory {
  
  private var marginalOutput: scala.collection.Map[Float, Long] = null
  private var marginals: RDD[(Int, Map[Float, Long])] = null
  private var joints: RDD[(Int, (Map[(Float, Float), Long]))] = null  
  
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
      bv: BV[Float], 
      varX: Broadcast[Seq[Int]],
      varY: Int,
      varZ: Option[Int]) = {
    
     val zval = varZ match {case Some(z) => Some(bv(z)) case None => None}     
     var pairs = Seq.empty[((Int, Float, Float, Option[Float]), Long)]
     
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
  def getRelevances(
      data: RDD[BV[Float]],
      varX: Seq[Int],
      varY: Int,
      n: Long,      
      nFeatures: Int) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = data.context
    val bvarX = sc.broadcast(varX)
    println ("Total reg: " + n)
    
    // Common function to generate pairs, it choose between sparse and dense processing 
    data.first match {
      case v: BDV[Float] =>
        val generator = DenseGenerator(_: BV[Float], bvarX, varY, None)
        marginalOutput = data.map(_(varY)).countByValue()
        val comb = data.flatMap(generator).reduceByKey(new Key1Partitioner(600), _ + _)
/*        val comb2 = comb.mapPartitionsWithIndex({ (index, it) =>
          val arr = it.toArray.map(k => (index, k._1._1))
          arr.toIterator
        }, preservesPartitioning = true)
        println(comb2.collect.mkString("\n"))*/
        val relevCalcs = computeRelevances(comb, n)
        val relevances = relevCalcs.mapValues(_._1)
        // We omit repeated label marginal prob from the complete set
        marginals = relevCalcs.mapValues(_._2).cache()
        joints = relevCalcs.mapValues(_._3).cache()
        relevances
      case v: BSV[Float] =>     
        // Not implemented yet!
        throw new NotImplementedError()
    }
  }
  
  private def computeRelevances(
    combinations: RDD[((Int, Float, Float, Option[Float]), Long)],
    n: Long) = {
    
      val freqy = combinations.context.broadcast(marginalOutput)
      
      val test = combinations.collect
        var freqx = Map.empty[Int, Map[Float, Long]]
        var freqxz = Map.empty[Int, Map[(Float, Float), Long]]
        
        // Compute freq counters for marginal and joint probabilities (all inputs)
        test.map{ case ((kx, x, z, _), q) =>
          var smap = freqx.getOrElse(kx, Map.empty)
          smap += x -> (smap.getOrElse(x, 0L) + q)
          freqx += kx -> smap
          var smap2 = freqxz.getOrElse(kx, Map.empty)
          smap2 += (x, z) -> (smap2.getOrElse((x,z), 0L) + q)
          freqxz += kx -> smap2
        }
        
        // Get mutual informations values using previous frequency counter
        val minst = test.map{ case ((kx, x, z, _), q) =>           
          val px = freqx.getOrElse(kx, Map.empty).getOrElse(x, 0L).toDouble / n
          val pz = freqy.value.getOrElse(z, 0L).toDouble / n
          val pxz = q.toDouble / n
          println("Tuples: " + kx + "," + x + "," + z + px + "," + pz + "," + pxz)
          (kx, pxz * (math.log(pxz / (px * pz)) / math.log(2)))
        }
        
        println("MInfo: " + minst.mkString("\n"))
        
        // Group instances by key and compute the final tuple result        
        var result = minst.groupBy(_._1).map{ case (k, a) =>
          val mi = a.map(_._2).sum / n
          (k, (mi, freqx(k), freqxz(k)))
        }
        
        println("Final result: " + result.map(t => (t._1, t._2._1)).mkString("\n"))
      
      combinations.mapPartitions({ it => 
        val elems = it.toArray
        var freqx = Map.empty[Int, Map[Float, Long]]
        var freqxy = Map.empty[Int, Map[(Float, Float), Long]]
        
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
        val minst = elems.map{ case ((kx, x, y, _), q) =>           
          val px = freqx.getOrElse(kx, Map.empty).getOrElse(x, 0L).toDouble / n
          val py = freqy.value.getOrElse(y, 0L).toDouble / n
          val pxy = q.toDouble / n
          (kx, pxy * (math.log(pxy / (px * py)) / math.log(2)))
        }
        
        // Group instances by key and compute the final tuple result        
        var result = minst.groupBy(_._1).map{ case (k, a) =>
          val mi = a.map(_._2).sum / n
          (k, (mi, freqx(k), freqxy(k)))
        }
        
        result.toIterator
      })      
  }
  
  def getRedundancies(
      data: RDD[BV[Float]],
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
      case v: BDV[Float] =>
        val generator = DenseGenerator(_: BV[Float], bvarX, varY, Some(varZ))
        val comb = data.flatMap(generator).reduceByKey(new Key1Partitioner(600), _ + _)
        val relevances = computeRedundancies(comb, n, varY, varZ)
        relevances.cache()
      case v: BSV[Float] =>
        // Not implemented yet!
        throw new NotImplementedError()
    }
  }
  
  private def computeRedundancies(
    combinations: RDD[((Int, Float, Float, Option[Float]), Long)],
    n: Long,
    varY: Int,
    varZ: Int) = {
      if(joints == null || marginals == null) throw new Exception()
      
      val sc = combinations.context
      val freqyz = sc.broadcast(joints.lookup(varY)(0))
      val freqy = sc.broadcast(marginals.lookup(varY)(0))
      val freqz = sc.broadcast(marginals.lookup(varZ)(0))
      val numPartitions = combinations.partitions.length
      
      combinations.mapPartitions({ it => 
        
        val elems = it.toArray
        var freqx = Map.empty[Int, Map[Float, Long]]
        var freqxz = Map.empty[Int, Map[(Float, Float), Long]]
        var freqxy = Map.empty[Int, Map[(Float, Float), Long]]
        
        // Compute freqograms for marginal and joint probabilities (xz, xy, x)
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
        
        // Get mutual informations values using previous freqograms
        var result = Map.empty[Int, (Double, Double)]
        val minst = elems.map{ case ((kx, x, y, z), qxyz) => 
            val px = freqx.getOrElse(kx, Map.empty).getOrElse(x, 0L).toDouble / n
            val py = freqy.value.getOrElse(y, 0L).toDouble / n            
            val pxy = freqxy.getOrElse(kx, Map.empty).getOrElse((x,y), 0L).toDouble / n
            val mi = pxy * (math.log(pxy / (px * py)) / math.log(2))
            
            val pz = freqz.value.getOrElse(z.get, 0L).toDouble / n
            val pxz = freqxz.getOrElse(kx, Map.empty).getOrElse((x,z.get), 0L).toDouble / n / pz
            val pyz = freqyz.value.getOrElse((y,z.get), 0L).toDouble / n / pz
            val pxyz = qxyz.toDouble / n / pz
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
