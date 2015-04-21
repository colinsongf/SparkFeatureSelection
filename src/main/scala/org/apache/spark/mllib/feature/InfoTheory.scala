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

  private val log2 = { x: Double => math.log(x) / math.log(2) } 
  
  private var marginals: RDD[(Int, Map[Double, Double])] = null
  private var joints: RDD[(Int, Map[(Double, Double), Double])] = null  
  
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
      lp: LabeledPoint, 
      varX: Broadcast[Seq[Int]],
      varY: Int,
      varZ: Option[Int]) = {
    
     val feat = lp.features
     val zval = varZ match {case Some(z) => Some(feat(z)) case None => None}     
     var pairs = Seq.empty[((Int, Double, Double, Option[Double]), Long)]
     
     for(xind <- varX.value){
       pairs = ((xind, feat(xind), feat(varY), zval), 1L) +: pairs
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
      data: RDD[LabeledPoint],
      varX: Seq[Int],
      varY: Int,
      varZ: Option[Int],
      n: Long,      
      nFeatures: Int, 
      inverseX: Boolean = false) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = data.context
    val bvarX = sc.broadcast(varX)
    
    // Common function to generate pairs, it choose between sparse and dense processing 
    data.first.features match {
      case v: DenseVector =>
        val generator = DenseGenerator(_: LabeledPoint, bvarX, varY, varZ)
        val comb = data.flatMap(generator).reduceByKey(new Key1Partitioner(1000), _ + _)
        val relevCalcs = computeRelevances(comb, n, nFeatures)
        relevCalcs.cache()
        val relevances = relevCalcs.mapValues(_._1)
        marginals = relevCalcs.mapValues(_._2).cache()
        joints = relevCalcs.mapValues(_._3).cache()
        relevCalcs.unpersist()  
        relevances
      case v: SparseVector =>     
        // Not implemented yet!
        throw new NotImplementedError()
    }
  }
  
  private def computeRelevances(
    combinations: RDD[((Int, Double, Double, Option[Double]), Long)],
    n: Long,
    indz: Int) = {
      val numPartitions = combinations.partitions.length
      combinations.mapPartitionsWithIndex({ (index, it) => 
        var newit = Seq.empty[(Int, (Float, Map[Double, Double], 
              Map[(Double, Double), Double]))]
        if (it.hasNext) {     
          var mapx = Map.empty[Double, Double]
          var mapz = Map.empty[Double, Double]
          var mapxz = Map.empty[(Double, Double), Double]
          val computeMI = (x: Double, z: Double, q: Double) => {
            val px = mapx.getOrElse(x, 0d) / n
            val pz = mapz.getOrElse(z, 0d) / n
            val pxz = q / n
            pxz * (math.log(pxz / px * pz) / math.log(2)) 
          }
          
          var currx = it.next._1._1
          for(((kx, x, z, _), q) <- it) {
            if(currx != kx) {              
              val minfo = mapxz.map{ case ((x, z), q) => 
                computeMI(x, z, q) }.sum.toFloat
              newit = (kx, (minfo, mapx, mapxz)) +: newit
              // Last calculation of P(Z), so we add it
              if(index == numPartitions - 1) {
                newit = (indz, (0.0f, mapx, 
                    Map.empty[(Double, Double), Double])) +: newit
              }
              mapx = mapx.empty; mapz = mapz.empty; mapxz = mapxz.empty
              currx = kx 
            }
            mapx += x -> (mapx.getOrElse(x, 0d) + q)
            mapz += z -> (mapz.getOrElse(z, 0d) + q)
            mapxz += (x, z) -> (mapxz.getOrElse((x, z), 0d) + q)
          }
          val minfo = mapxz.map{ case ((x, z), q) => 
                computeMI(x, z, q) }.sum.toFloat
          newit = (currx, (minfo, mapx, mapxz)) +: newit          
        }
        
        newit.toIterator
      })      
  }
  
  def getRedundancies(
      data: RDD[LabeledPoint],
      varX: Seq[Int],
      varY: Int,
      varZ: Int,
      n: Long,      
      nFeatures: Int, 
      inverseX: Boolean = false) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = data.context
    val bvarX = sc.broadcast(varX)
    
    // Common function to generate pairs, it choose between sparse and dense processing 
    data.first.features match {
      case v: DenseVector =>
        val generator = DenseGenerator(_: LabeledPoint, bvarX, varY, Some(varZ))
        val comb = data.flatMap(generator).reduceByKey(new Key1Partitioner(1000), _ + _)
        val relevances = computeRedundancies(comb, n, varY, varZ, nFeatures)
        relevances.cache()
      case v: SparseVector =>     
        // Not implemented yet!
        throw new NotImplementedError()
    }
  }
  
  private def computeRedundancies(
    combinations: RDD[((Int, Double, Double, Option[Double]), Long)],
    n: Long,
    varY: Int,
    varZ: Int,
    indz: Int) = {
      if(joints == null || marginals == null) throw new Exception()
      val sc = combinations.context
      val mpyz = sc.broadcast(joints.lookup(varY)(0))
      val mpy = sc.broadcast(marginals.lookup(varY)(0))
      val mpz = sc.broadcast(marginals.lookup(varZ)(0))
      val numPartitions = combinations.partitions.length
      
      combinations.mapPartitions({ it => 
        val elems = it.toArray
        var newit = Seq.empty[(Int, (Float, Float))]
        
        if (!elems.isEmpty) {     
          var mapxy = Map.empty[(Double, Double), Double]
          var mapxz = Map.empty[(Double, Double), Double]
          var mapx = Map.empty[Double, Double]
          val computeMIandCMI = (x: Double, y: Double, z: Double, qxyz: Double) => {
            val px = mapx.getOrElse(x, 0d) / n
            val py = mpy.value.getOrElse(y, 0d) / n
            val pz = mpz.value.getOrElse(z, 0d) / n
            val pxy = mapxy.getOrElse((x,y), 0d) / n
            val pxz = mapxz.getOrElse((x,z), 0d) / n
            val pyz = mpyz.value.getOrElse((y, z), 0d) / n
            val pxyz = qxyz / n
            (pxz * (math.log(pxz / px * pz) / math.log(2)),
                pxyz * (math.log(pz * pxyz / pxz * pyz) / math.log(2)))
          }
                    
          var (currx, idx) = (elems(0)._1._1, 0)
          for(i <- 1 until elems.size) {
            val ((kx, x, y, z), q) = elems(i)
            if(currx != kx) {
              var mi = 0.0; var cmi = 0.0
              for(j <- idx until i) {
                val ((_, x, y, z), q) = elems(j)
                val psum = computeMIandCMI(x, y, z.get, q)
                mi += psum._1; cmi += psum._2
              }              
              newit = (kx, (mi.toFloat, cmi.toFloat)) +: newit
              mapxy = mapxy.empty; mapxz = mapxz.empty; mapx = mapx.empty 
              currx = kx; idx = i                      
            }  
            mapx += x -> (mapx.getOrElse(x, 0d) + q)
            mapxz += (x,z.get) -> (mapxz.getOrElse((x,z.get), 0d) + q)
            mapxy += (x,y) -> (mapxy.getOrElse((x,y), 0d) + q)
          }
          
          var mi = 0.0; var cmi = 0.0
          for(j <- idx until elems.size) {
            val ((_, x, y, z), q) = elems(j)
            val psum = computeMIandCMI(x, y, z.get, q)
            mi += psum._1; cmi += psum._2
          }
              
          newit = (currx, (mi.toFloat, cmi.toFloat)) +: newit          
        }
        newit.toIterator
      })      
  }
  
  class Key1Partitioner(numParts: Int) extends Partitioner {
    override def numPartitions: Int = numParts
    override def getPartition(key: Any): Int = {
      val (indx, _, _, _) = key.asInstanceOf[(Int, Double, Double, Option[Double])]
      val code = (indx.hashCode % numPartitions)
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
