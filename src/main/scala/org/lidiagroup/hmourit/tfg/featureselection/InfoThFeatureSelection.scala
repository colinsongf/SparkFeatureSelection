package org.lidiagroup.hmourit.tfg.featureselection

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext._
import org.lidiagroup.hmourit.tfg.featureselection.{InfoTheory => IT}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.annotation.Experimental

class InfoThFeatureSelection private (
    val criterionFactory: InfoThCriterionFactory,
    var poolSize:  Int = 30)
  extends Serializable {

  private type Pool = RDD[(Int, InfoThCriterion)]
  private case class F(feat: Int, crit: Double)
  
  def calcMutualInformation(partitions: Array[RDD[Array[Double]]],
      varX: Seq[Int],
      varY: Int,
      varZ: Option[Int]) = {
    
	  partitions
    	.map(part => IT.miAndCmi(part, varX, varY, varZ, part.count).toSeq)
    	.reduce(_ ++ _)
    	.groupBy(_._1)
    	.mapValues(_.map(_._2))
    	.mapValues(_.foldLeft(0.0,0.0){ 
    	  case ((smi, scmi), (mi, cmi)) => (smi + mi, scmi + cmi)
    	  })
  }

  def setPoolSize(poolSize: Int) = {
    this.poolSize = poolSize
    this
  }
  	@Experimental
    private def selectFeaturesWithoutPool(
      data: RDD[Array[Double]],
      nToSelect: Int,
      nFeatures: Int,
      label: Int,
      nPartitions: Int)
    : Seq[F] = {
    
    val weights = Array.fill[Double](nPartitions)(nPartitions / 100f)
    val partitions = data.randomSplit(weights)
    
    // calculate relevance
    var pool = calcMutualInformation(partitions, 1 to nFeatures, label, None)
		.map({ case (k, (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) })
		.toArray
		
    // get maximum and select it
    var max = pool.maxBy(_._2.score)
    var selected = Seq(F(max._1, max._2.score))
    var toSelect = pool.map(_._1) diff Seq(max._1)

    while (selected.size < nToSelect) {
      // update pool
      val newMiAndCmi = calcMutualInformation(partitions, toSelect, selected.head.feat, Some(label))
      pool = pool.flatMap({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some((mi, cmi)) => Seq((k, crit.update(mi, cmi)))
          case None => Seq.empty[(Int, InfoThCriterion)]
        }
      })

      // look for maximum
      max = pool.maxBy(_._2.score)

      // select feature
      selected = F(max._1, max._2.score) +: selected
      toSelect = toSelect diff Seq(max._1)

    }

    selected.reverse
  }

  private def selectFeaturesWithoutPool(
      data: RDD[Array[Double]],
      nToSelect: Int,
      nFeatures: Int,
      label: Int,
      nElements: Long)
    : Seq[F] = {
    
    // Experimental
    val nPartitions = 10
    val weights = Array.fill[Double](nPartitions)(nPartitions / 100f)
    val partitions = data.randomSplit(weights)
    
    // calculate relevance
    var pool = IT.miAndCmi(data, 1 to nFeatures, label, None, nElements)
      .map({ case (k, (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) })
      .toArray

    // get maximum and select it
    var max = pool.maxBy(_._2.score)
    var selected = Seq(F(max._1, max._2.score))
    var toSelect = pool.map(_._1) diff Seq(max._1)

    while (selected.size < nToSelect) {
      // update pool
      val newMiAndCmi = IT.miAndCmi(data, toSelect, selected.head.feat, Some(label), nElements)
      pool = pool.flatMap({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some((mi, cmi)) => Seq((k, crit.update(mi, cmi)))
          case None => Seq.empty[(Int, InfoThCriterion)]
        }
      })

      // look for maximum
      max = pool.maxBy(_._2.score)

      // select feature
      selected = F(max._1, max._2.score) +: selected
      toSelect = toSelect diff Seq(max._1)

    }

    selected.reverse
  }

  private def selectFeaturesWithPool(
      data: RDD[Array[Double]],
      nToSelect: Int,
      nFeatures: Int,
      label: Int,
      nElements: Long)
    : Seq[F] = {

    // calculate relevance
    var rels = IT.miAndCmi(data, 1 to nFeatures, label, None, nElements)
        .toArray
        .map({ case (k, (mi, _)) => (k, mi) })
        .sortBy(-_._2)

    // extract pool
    val initialPoolSize = math.min(math.max(poolSize, nToSelect), rels.length)
    var pool = rels.take(initialPoolSize).map({ case (k, mi) =>
      (k, criterionFactory.getCriterion.init(mi))
    })
    var min = pool.last._2.asInstanceOf[InfoThCriterion with Bound]
    var toSelect = pool.map(_._1)
    rels = rels.drop(initialPoolSize)

    // select feature with top relevancy
    var max = pool.head
    var selected = Seq(F(max._1, max._2.score))
    toSelect = toSelect diff Seq(max._1)

    while (selected.size < nToSelect) {

      // update pool
      val newMiAndCmi = IT.miAndCmi(data, toSelect, selected.head.feat, Some(label), nElements)
      pool = pool.flatMap({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some((mi, cmi)) => Seq((k, crit.update(mi, cmi)))
          case None => Seq.empty[(Int, InfoThCriterion)]
        }
      })

      // look for maximum
      max = pool.maxBy(_._2.score)

      // increase pool if necessary
      while (max._2.score < min.bound && toSelect.size + selected.size < nFeatures) {

        // increase pool
        val realPoolSize = math.min(poolSize, rels.length)
        pool ++= rels.take(realPoolSize).map({ case (k, mi) => (k, criterionFactory.getCriterion.init(mi)) })
        rels = rels.drop(realPoolSize)
        min = pool.last._2.asInstanceOf[InfoThCriterion with Bound]

        // do missed calculations
        for (i <- (pool.length - realPoolSize) until pool.length) {
          val missed_calc = IT.miAndCmi(data, selected.map(_.feat), i, Some(label), nElements)
          missed_calc.foreach({ case (_, (mi, cmi)) => pool(i)._2.update(mi, cmi)})
          toSelect = pool(i)._1 +: toSelect
        }

        // look for maximum
        max = pool.maxBy(_._2.score)

      }
      // select feature
      selected = F(max._1, max._2.score) +: selected
      toSelect = toSelect diff Seq(max._1)

    }

    selected.reverse

  }

  def run(data: RDD[LabeledPoint], nToSelect: Int): InfoThFeatureSelectionModel = {
    val nFeatures = data.first.features.size

    if (nToSelect > nFeatures) {
      throw new IllegalArgumentException("data doesn't have so many features")
    }

    val array = data.map({ case LabeledPoint(label, values) => (label +: values.toArray) })
    val nElements = array.count()
    
    /*var array = data.map({ case LabeledPoint(label, values) => (label +: values.toArray) }).persist(StorageLevel.MEMORY_ONLY_SER)
    val nElements = array.count()
    array = array.coalesce(214, true)
    */
    var selected = Seq.empty[F]
    criterionFactory.getCriterion match {
      case _: InfoThCriterion with Bound if poolSize != 0 =>
        selected = selectFeaturesWithPool(array, nToSelect, nFeatures, 0, 10)
      case _: InfoThCriterion =>
        selected = selectFeaturesWithoutPool(array, nToSelect, nFeatures, 0, array.count())
      case _ =>
    }

    new InfoThFeatureSelectionModel(selected.map({ case F(feat, rel) => feat - 1 }).toArray)
  }
}

object InfoThFeatureSelection {

  def train(criterionFactory: InfoThCriterionFactory,
      data: RDD[LabeledPoint],
      nToSelect: Int,
      poolSize: Int = 30) = {
    new InfoThFeatureSelection(criterionFactory, poolSize).run(data, nToSelect)
  }
}
