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

import scala.collection.immutable.TreeMap
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkException
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{Vector, DenseVector, SparseVector}
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.feature.{InfoThCriterionFactory => FT}
import org.apache.spark.mllib.feature.{InfoTheory => IT}

/**
 * Train a info-theory feature selection model according to a criterion.
 * 
 * @param criterionFactory Factory to create info-theory measurements for each feature.
 * @param data RDD of LabeledPoint (discrete data).
 * 
 */
class InfoThSelector private[feature] (val criterionFactory: FT) extends Serializable {

  // Pool of criterions
  private type Pool = RDD[(Int, InfoThCriterion)]
  // Case class for criterions by feature
  protected case class F(feat: Int, crit: Double) 
  private case class ColumnarData(dense: RDD[(Int, (Int, Array[Byte]))], 
      sparse: RDD[(Int, TreeMap[Long, Byte])],
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
    val (distinctByFeat, relevances) = {
      val counterByKey = data.dense.mapValues({ case (_, v) => v.max + 1})
        .reduceByKey((m1, m2) => if(m1 > m2) m1 else m2).collectAsMap().toMap
      // calculate relevance
      val MiAndCmi = IT.computeMI(
        data.dense, 0 until label, label, nInstances, nFeatures, counterByKey)
      (counterByKey, MiAndCmi)
    }
    //} else {
      //val nInstances = data.sparse.count() / nFeatures
      //val counterByKey: Map[Int, Int] = null
      // calculate relevance
      //val MiAndCmi = IT.computeMI(
      //  data.dense, 0 until label, label, nInstances, nFeatures, counterByKey)
      //(0, counterByKey, MiAndCmi)
    //}    

    var pool = relevances.map{case (x, mi) => (x, criterionFactory.getCriterion.init(mi))}
      .collectAsMap()  
    // Print most relevant features
    val strRels = relevances.collect().sortBy(-_._2)
      .take(nToSelect)
      .map({case (f, mi) => (f + 1) + "\t" + "%.4f" format mi})
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels) 
    
    // get maximum and select it
    val firstMax = pool.maxBy(_._2.score)
    var selected = Seq(F(firstMax._1, firstMax._2.score))
    pool = pool - firstMax._1

    while (selected.size < nToSelect) {
      // update pool
      val newMiAndCmi = IT.computeMIandCMI(data.dense, pool.keys.toSeq, selected.head.feat, 
          label, nInstances, nFeatures, distinctByFeat) 
          .map({ case (x, crit) => (x, crit) })
          .collectAsMap()
      pool.foreach({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some((mi, cmi)) => crit.update(mi, cmi)
          case None => /* Never happens */
        }
      })      
      
      // get maximum and save it
      val max = pool.maxBy(_._2.score)
      // select the best feature and remove from the whole set of features
      selected = F(max._1, max._2.score) +: selected
      pool = pool - max._1 
    }
    selected.reverse
  }

  private[feature] def run(
      data: RDD[LabeledPoint], 
      nToSelect: Int, 
      numPartitions: Int) = {   
      
    val requireByteValues = (l: Double, v: Vector) => {        
      val values = v match {
        case sv: SparseVector =>
          sv.values
        case dv: DenseVector =>
          dv.values
      }
      val condition = (value: Double) => value <= Byte.MaxValue && 
        value >= Byte.MinValue && value % 1 == 0.0
      if (!values.forall(condition(_)) || !condition(l)) {
        throw new SparkException(s"Info-Theoretic Framework requires positive values in range [0, 255]")
      }           
    }
        
    val features = data.first.features
    val nAllFeatures = features.size + 1
    val dense = features.isInstanceOf[DenseVector]
    val nPart = if(numPartitions == 0) nAllFeatures else numPartitions
    var nInstances = 0; var nFeatures = 0
    
    val colData = if(dense) {
      val columnarData: RDD[(Int, (Int, Array[Byte]))] = data.mapPartitionsWithIndex({ (index, it) =>
        val data = it.toArray
        val nfeat = data(0).features.size + 1
        val mat = Array.ofDim[Byte](nfeat, data.length)
        var j = 0
        for(reg <- data) {
          requireByteValues(reg.label, reg.features)
          for(i <- 0 until reg.features.size) mat(i)(j) = reg.features(i).toByte
          mat(reg.features.size)(j) = reg.label.toByte
          j += 1
        }
        val chunks = for(i <- 0 until nfeat) yield (i -> (index, mat(i)))
        chunks.toIterator
      })      
      
      val denseData = columnarData.sortByKey(numPartitions = nPart).persist(StorageLevel.MEMORY_ONLY)  
      val str = columnarData.first()._2
      println("First raw: " + str._1 + " " + str._2.mkString(","))
      //val c = columnarData.count()
      //val denseData = columnarData.sortByKey(numPartitions = nPart) // put numPartitions parameter      
      nInstances = columnarData.lookup(0).map(_._2.length).reduce(_ + _)
      
      ColumnarData(columnarData, null, true)      
    } else {
      val sparseData = data.zipWithUniqueId().flatMap ({
        case (LabeledPoint(label, values: SparseVector), r) => 
          requireByteValues(label, values)
          val inputs = for(i <- values.indices) yield (i, (r, values(i).toByte))
          val output = Array((values.size, (r, label.toByte)))
          output ++ inputs     
      }).groupByKey(numPartitions = nPart)
      .mapValues(a => TreeMap(a.toArray:_*))
      .persist(StorageLevel.MEMORY_ONLY)
      ColumnarData(null, sparseData, false)
    }
    
    require(nToSelect < nAllFeatures)  
    
    val selected = selectFeatures(colData, nToSelect, nInstances, nAllFeatures)
          
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
   */
  def train(
      criterionFactory: FT, 
      data: RDD[LabeledPoint],
      nToSelect: Int = 25,
      numPartitions: Int = 0) = {
    new InfoThSelector(criterionFactory).run(data, nToSelect, numPartitions)
  }
}
