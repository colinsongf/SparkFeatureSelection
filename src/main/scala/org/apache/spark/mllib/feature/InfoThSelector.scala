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


import scala.collection.immutable.TreeMap
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkException
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{Vector, DenseVector, SparseVector}
import org.apache.spark.annotation.Experimental
import org.apache.spark.Logging
import org.apache.spark.mllib.feature.{InfoThCriterionFactory => FT}
import org.apache.spark.mllib.feature.{InfoTheory => IT}
import scala.collection.immutable.LongMap
import scala.collection.immutable.HashMap
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.HashPartitioner

/**
 * Train a info-theory feature selection model according to a criterion.
 * 
 * @param criterionFactory Factory to create info-theory measurements for each feature.
 * @param data RDD of LabeledPoint (discrete data).
 * 
 */
class InfoThSelector private[feature] (val criterionFactory: FT) extends Serializable with Logging {

  // Pool of criterions
  private type Pool = RDD[(Int, InfoThCriterion)]
  // Case class for criterions by feature
  protected case class F(feat: Int, crit: Double) 
  private case class ColumnarData(dense: RDD[(Int, (Int, Array[Byte]))], 
      sparse: RDD[(Int, Map[Long, Byte])],
      isDense: Boolean)

  /**
   * Perform a info-theory selection process without pool optimization.
   * 
   * @param data Columnar data (last element is the class attribute).
   * @param nToSelect Number of features to select.
   * @return A list with the most relevant features and its scores.
   * 
   */
  private[feature] def selectFeatures(
      data: ColumnarData, 
      nToSelect: Int,
      nInstances: Long,
      nFeatures: Int) = {
    
    val label = nFeatures - 1
    /*val (distinctByFeat, relevances) = if(data.isDense) {
      val counterByKey = data.dense.mapValues({ case (_, v) => v.max + 1})
          .reduceByKey((m1, m2) => if(m1 > m2) m1 else m2)
          .collectAsMap()
          .toMap
        // calculate relevance
      val MiAndCmi = IT.computeMI(
        data.dense, 0 until label, label, nInstances, nFeatures, counterByKey)
        (counterByKey, MiAndCmi)
    } else {
      //val nInstances = data.sparse.count() / nFeatures
      val counterByKey: Map[Int, Int] = data.sparse
        .mapValues(v => if(v.valuesIterator.isEmpty) 1 else v.valuesIterator.max + 1)
        .collectAsMap()
        .toMap
      // calculate relevance
      // println("Spare data: " + data.sparse.first()._2.toString())
      val MiAndCmi = IT.computeMISparse(
        data.sparse, 0 until label, label, nInstances, nFeatures, counterByKey)
      (counterByKey, MiAndCmi)
    } */
    val (it, relevances) = if(data.isDense) {
      val it = InfoTheory2.initializeDense(data.dense, label, nInstances, nFeatures)
      (it, it.relevances)
    } else {
      val it = InfoTheory2.initializeSparse(data.sparse, label, nInstances, nFeatures)
      (it, it.relevances)
    }

    // Init all (less output attribute) criterions to a bad score
    val pool = Array.fill[InfoThCriterion](nFeatures - 1) {
      val crit = criterionFactory.getCriterion.init(Float.NegativeInfinity)
      crit.setValid(false)
    }
    
    relevances.collect().foreach{ case (x, mi) => pool(x) = criterionFactory.getCriterion.init(mi.toFloat) }
    
    // Print most relevant features
    val strRels = relevances.sortBy(_._2, false).take(nToSelect)
      .map({case (f, mi) => (f + 1) + "\t" + "%.4f" format mi})
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels) 
    
    // get maximum and select it
    val (max, mid) = pool.zipWithIndex.maxBy(_._1.relevance)
    var selected = Seq(F(mid, max.score))
    pool(mid).setValid(false)

    var moreFeat = true
    while (selected.size < nToSelect && moreFeat) {
      // update pool
      val ids = for (i <- 0 until pool.length if pool(i).valid) yield i
      val redundancies = it match {
        case dit: InfoTheoryDense => dit.getRedundancies(ids, selected.head.feat)
        case sit: InfoTheorySparse => sit.getRedundancies(ids, selected.head.feat)
      }    
      /*
      if(data.isDense)
        IT.computeMIandCMI(data.dense, ids, selected.head.feat, 
          label, nInstances, nFeatures, distinctByFeat) // Maybe we can remove counter
      } else {
        IT.computeMIandCMISparse(data.sparse, ids, selected.head.feat, 
          label, nInstances, nFeatures)
      }*/
      
      val red = redundancies.collect()
      
      //val red = redundancies.collect()
      /*.foreach({ case (k, (mi, cmi)) =>
        pool(k).update(mi.toFloat, cmi.toFloat)
      })*/

      /*pool.foreach({ case (k, crit) =>
        redundancies.get(k) match {
          case Some((mi, cmi)) => crit.update(mi, cmi)
          case None => /* Never happens */
        }
      })*/      
      
      // get maximum and save it
      var (maxi, max) = (-1, Float.NegativeInfinity)
      /*for(i <- 0 until pool.length if pool(i).valid; sc = pool(i).score) {
        if(sc > max) maxi = i; max = sc
      }*/
      
      for((k, (mi, cmi)) <- red if pool(k).valid) {
        pool(k).update(mi.toFloat, cmi.toFloat)
        val sc = pool(k).score
        if(sc > max){
          maxi = k; max = sc
        } 
      }
      
      // select the best feature and remove from the whole set of features
      if(maxi != -1){
        selected = F(maxi, max) +: selected
        pool(maxi).setValid(false)
      } else {
        moreFeat = false
      }
    }
    selected.reverse
  }

  private[feature] def run(
      data: RDD[LabeledPoint], 
      nToSelect: Int, 
      numPartitions: Int) = {  
    
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
      
    val requireByteValues = (v: Vector) => {        
      val values = v match {
        case sv: SparseVector =>
          sv.values
        case dv: DenseVector =>
          dv.values
      }
      val condition = (value: Double) => value <= Byte.MaxValue && 
        value >= Byte.MinValue && value % 1 == 0.0
      if (!values.forall(condition(_))) {
        throw new SparkException(s"Info-Theoretic Framework requires positive values in range [0, 255]")
      }           
    }
        
    val first = data.first
    val dense = first.features.isInstanceOf[DenseVector]    
    var nInstances = 0L
    val nFeatures = first.features.size + 1
    
    val colData = if(dense) {
      
      val nPart = if(numPartitions == 0) nFeatures else numPartitions
      if(nPart > nFeatures) {
        logWarning("Number of partitions should be less than the number of features in the dataset."
          + " At least, less than 2x nPartitions.")
      }
      
      val columnarData: RDD[(Int, (Int, Array[Byte]))] = data.mapPartitionsWithIndex({ (index, it) =>
        val data = it.toArray
        val mat = Array.ofDim[Byte](nFeatures, data.length)
        var j = 0
        for(reg <- data) {
          requireByteValues(reg.features)
          for(i <- 0 until reg.features.size) mat(i)(j) = reg.features(i).toByte
          mat(reg.features.size)(j) = reg.label.toByte
          j += 1
        }
        val chunks = for(i <- 0 until nFeatures) yield (i -> (index, mat(i)))
        chunks.toIterator
      })      
      // Sort to group all chunks for the same feature closely. It will avoid to shuffle too much histograms
      val denseData = columnarData.sortByKey(numPartitions = nPart).persist(StorageLevel.MEMORY_ONLY)
      val c = denseData.count() // Important to cache the data!
      //println("Number of chunks: " + c)
      
      nInstances = denseData.lookup(0).map(_._2.length).reduce(_ + _)
      ColumnarData(denseData, null, true)      
    } else {      
      
      val nPart = if(numPartitions == 0) data.conf.getInt("spark.default.parallelism", 5) else numPartitions
      //val distinct = data.flatMap(_.features.asInstanceOf[SparseVector].indices).distinct().count()
      //val maxindex = data.flatMap(_.features.asInstanceOf[SparseVector].indices).max()
      val classMap = data.map(_.label).distinct.collect().zipWithIndex.toMap
      nInstances = data.count()
      
      val sparseData = data.zipWithIndex().flatMap ({ case (lp, r) => 
          requireByteValues(lp.features)
          val sv = lp.features.asInstanceOf[SparseVector]
          val output = (nFeatures - 1) -> (r, classMap(lp.label).toByte)
          val inputs = for(i <- 0 until sv.indices.length) 
            yield (sv.indices(i), (r, sv.values(i).toByte))
          output +: inputs           
      })
      
      /*val sparseData2 = data.zipWithUniqueId().flatMap ({ case (lp, inst) =>        
        requireByteValues(lp.features)
        val v = lp.features.asInstanceOf[SparseVector]
        val inputs = (0 until v.indices.length).map({ i =>            
          (v.indices(i), inst) -> v.values(i).toByte   
        })
        val output = (nFeatures - 1, inst) -> classMap(lp.label).toByte
        output +: inputs      
      }).sortByKey().persist(StorageLevel.MEMORY_ONLY)   
      val c2 = sparseData2.count()*/
      
      /*val sparseData2 = data.zipWithUniqueId().mapPartitions ({ it =>        
        val featCols = Array.fill(nFeatures){ HashMap[Long, Byte]() }
        for ((lp, inst) <- it) {
          requireByteValues(lp.label, lp.features)
          featCols(nFeatures - 1) += inst -> classMap(lp.label).toByte
          val v = lp.features.asInstanceOf[SparseVector]
          (0 until v.indices.length).map({ i =>            
            featCols(v.indices(i)) += inst -> v.values(i).toByte   
          })          
        }
        featCols.zipWithIndex.map({case (col, idx) => (idx, col)}).toIterator            
      }).reduceByKey(_ ++ _).persist(StorageLevel.MEMORY_ONLY)   
      val c2 = sparseData2.count()*/
      
      val columnarData = sparseData.groupByKey(new HashPartitioner(nPart))
        .mapValues({ a =>
          val map: Map[Long, Byte] = if(a.size != nInstances) {
            HashMap(a.toArray:_*)
          } else {
            TreeMap(a.toArray:_*)
          }
          map
        })
        .persist(StorageLevel.MEMORY_ONLY)
      val c3 = columnarData.count()
      
      //val classData = indexed.map({case (l, r) => r -> l}).sortByKey().collect()
      
      /*val columnarData2 = sparseData.groupByKey().mapValues({ a => 
          val sorted = a.toArray.sortBy(_._1)
          val vector: BV[Byte] = if(a.size > nInstances / 2){
            new BDV(sorted.map(_._2.toByte))
          } else {   
            new BSV(sorted.map(_._1.toInt), sorted.map(_._2.toByte), nInstances.toInt)
          }   
          vector
        }).persist(StorageLevel.MEMORY_ONLY)
      val c3 = columnarData2.count()*/
      
       // Partitioner in order to increase the performance of lookups
      /*val columnarData = sparseData
        .aggregateByKey(HashMap.empty[Long, Byte], new HashPartitioner(numPartitions))(
            {case (f, e) => f + e}, _ ++ _)
        .persist(StorageLevel.MEMORY_ONLY)    
      val c4 = columnarData.count()*/
      
      ColumnarData(null, columnarData, false)
    }
    
    require(nToSelect < nFeatures)  
    
    val selected = selectFeatures(colData, nToSelect, nInstances, nFeatures)
          
    if(dense) {
      colData.dense.unpersist()
    } else {
      colData.sparse.unpersist()
    }
  
    // Print best features according to the mRMR measure
    val out = selected.map{case F(feat, rel) => (feat + 1) + "\t" + "%.4f".format(rel)}.mkString("\n")
    println("\n*** mRMR features ***\nFeature\tScore\n" + out)
    // Features must be sorted
    new SelectorModel(selected.map{case F(feat, rel) => feat}.sorted.toArray)
  }
}

object InfoThSelector {

  /**
   * Train a feature selection model according to a given criterion
   * and return a subset of data.
   *
   * @param   criterionFactory Initialized criterion to use in this selector
   * @param   data RDD of LabeledPoint (discrete data as integers in range [0, 255]).
   * @param   nToSelect maximum number of features to select
   * @param   numPartitions number of partitions to structure the data.
   * @return  A feature selection model which contains a subset of selected features
   * 
   * Note: LabeledPoint data must be integer values in double representation 
   * with a maximum of 256 distinct values. In this manner, data can be transformed
   * to byte class directly, making the selection process much more efficient.
   * 
   * Note: numPartitions must be less or equal to the number of features to achieve a better performance.
   * In this way, the number of histograms to be shuffled is reduced. 
   * 
   */
  def train(
      criterionFactory: FT, 
      data: RDD[LabeledPoint],
      nToSelect: Int = 25,
      numPartitions: Int = 0) = {
    new InfoThSelector(criterionFactory).run(data, nToSelect, numPartitions)
  }
}
