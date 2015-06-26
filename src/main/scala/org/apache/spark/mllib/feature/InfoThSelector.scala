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
import scala.collection.immutable.BitSet

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
      sparse: RDD[(Int, BV[Byte])],
      isDense: Boolean)
      
    /**
   * Perform a info-theory selection process without pool optimization.
   * 
   * @param data Columnar data (last element is the class attribute).
   * @param nToSelect Number of features to select.
   * @return A list with the most relevant features and its scores.
   * 
   */
  private[feature] def selectFeatures2(
      data: ColumnarData, 
      nToSelect: Int,
      nInstances: Long,
      nFeatures: Int) = {
    
    val label = nFeatures - 1
    val (it, relevances) = if(data.isDense) {
      val it = InfoTheory.initializeDense(data.dense, label, nInstances, nFeatures)
      (it, it.relevances)
    } else {
      val it = InfoTheory.initializeSparse(data.sparse, label, nInstances, nFeatures)
      (it, it.relevances)
    }
    
    object FeatOrdering extends Ordering[(Int, InfoThCriterion)] {
      def compare(a:(Int, InfoThCriterion), b:(Int, InfoThCriterion)) = a._2.score compare b._2.score
    }
    
    var pool = relevances.mapValues{ mi => criterionFactory.getCriterion.init(mi) }
      .partitionBy(new HashPartitioner(400)).cache()
    
    // Print most relevant features
    val strRels = pool.top(nToSelect)(FeatOrdering)
      .map({case (f, crit) => (f + 1) + "\t" + "%.4f" format crit.score})
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels) 
    
    // get maximum and select it
    
    val (mid, mcrit) = pool.max()(FeatOrdering)
    var selected = Seq(F(mid, mcrit.score))
    var moreFeat = true
    
    while (selected.size < nToSelect && moreFeat) {
      // update pool
      val redundancies = it match {
        //case dit: InfoTheoryDense => dit.getRedundancies(ids, selected.head.feat)
        case sit: InfoTheorySparse => sit.getRedundancies(selected.head.feat)
      }
      
      val newpool = pool.leftOuterJoin(redundancies).mapValues({case (crit, upd) => 
        upd match {
          case Some((mi, cmi)) => crit.update(mi, cmi)
          case None => crit
        }  
      }).cache()
      
      // get maximum and save it      
      val bsel = BitSet(selected.map(_.feat):_*)
      val (mid, mcrit) = newpool.filter({case (k, _) => !bsel.contains(k)}).max()(FeatOrdering)
      selected = F(mid, mcrit.score) +: selected
      if(bsel.size + 1 == nFeatures - 1) moreFeat = false
      
      pool.unpersist(false)
      pool = newpool
    }
    pool.unpersist()
    selected.reverse
  }


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
    val (it, relevances) = if(data.isDense) {
      val it = InfoTheory.initializeDense(data.dense, label, nInstances, nFeatures)
      (it, it.relevances)
    } else {
      val it = InfoTheory.initializeSparse(data.sparse, label, nInstances, nFeatures)
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
      val redundancies = it match {
        case dit: InfoTheoryDense => 
          //val ids = for (i <- 0 until pool.length if pool(i).valid) yield i
          dit.getRedundancies(selected.head.feat)
        case sit: InfoTheorySparse => sit.getRedundancies(selected.head.feat)
      }
      
      redundancies.collect().par.foreach({case (k, (mi, cmi)) =>
         pool(k).update(mi.toFloat, cmi.toFloat) 
      })   
      
      // get maximum and save it
      /*var (maxi, max) = (-1, Float.NegativeInfinity)      
      for((k, (mi, cmi)) <- red if pool(k).valid) {
        pool(k).update(mi.toFloat, cmi.toFloat)
        val sc = pool(k).score
        if(sc > max){
          maxi = k; max = sc
        } 
      }*/
      
      val (max, maxi) = pool.par.zipWithIndex.filter(_._1.valid).maxBy(_._1)
      
      // select the best feature and remove from the whole set of features
      if(maxi != -1){
        selected = F(maxi, max.score) +: selected
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
    val nInstances = data.count()
    val nFeatures = first.features.size + 1
    
    val colData = if(dense) {
      
      val nPart = if(numPartitions == 0) nFeatures else numPartitions
      if(nPart > nFeatures) {
        logWarning("Number of partitions should be less than the number of features in the dataset."
          + " At least, less than 2x nPartitions.")
      }
      
      val classMap = data.map(_.label).distinct.collect().zipWithIndex.map(t => t._1 -> t._2.toByte).toMap
      val columnarData: RDD[(Int, (Int, Array[Byte]))] = data.mapPartitionsWithIndex({ (index, it) =>
        val data = it.toArray
        val mat = Array.ofDim[Byte](nFeatures, data.length)
        var j = 0
        for(reg <- data) {
          requireByteValues(reg.features)
          for(i <- 0 until reg.features.size) mat(i)(j) = reg.features(i).toByte
          mat(reg.features.size)(j) = classMap(reg.label)
          j += 1
        }
        val chunks = for(i <- 0 until nFeatures) yield (i -> (index, mat(i)))
        chunks.toIterator
      })      
      // Sort to group all chunks for the same feature closely. It will avoid to shuffle too much histograms
      val denseData = columnarData.sortByKey(numPartitions = nPart).persist(StorageLevel.MEMORY_ONLY)
      
      //nInstances = denseData.lookup(0).map(_._2.length).reduce(_ + _)
      ColumnarData(denseData, null, true)      
    } else {      
      
      val nPart = if(numPartitions == 0) data.conf.getInt("spark.default.parallelism", 750) else numPartitions
      val classMap = data.map(_.label).distinct.collect().zipWithIndex.map(t => t._1 -> t._2.toByte).toMap
      val sparseData = data.zipWithIndex().flatMap ({ case (lp, r) => 
          requireByteValues(lp.features)
          val sv = lp.features.asInstanceOf[SparseVector]
          val output = (nFeatures - 1) -> (r, classMap(lp.label))
          val inputs = for(i <- 0 until sv.indices.length) 
            yield (sv.indices(i), (r, sv.values(i).toByte))
          output +: inputs           
      })
      
      val columnarData = sparseData.groupByKey(new HashPartitioner(nPart))
        .mapValues({a => 
          if(a.size >= nInstances) {
            val init = Array.fill[Byte](nInstances.toInt)(0)
            val result: BV[Byte] = new BDV(init)
            a.foreach({case (k, v) => result(k.toInt) = v})
            result
          } else {
            val init = a.toArray.sortBy(_._1)
            new BSV(init.map(_._1.toInt), init.map(_._2), nInstances.toInt)
          }
        })
        .persist(StorageLevel.MEMORY_ONLY)
      
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
