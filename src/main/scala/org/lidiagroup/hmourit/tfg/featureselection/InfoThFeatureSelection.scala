package org.lidiagroup.hmourit.tfg.featureselection

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext._
import org.lidiagroup.hmourit.tfg.featureselection.{InfoTheory => IT}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.annotation.Experimental
import scala.collection.mutable.ArrayBuffer

class InfoThFeatureSelection private (
    val criterionFactory: InfoThCriterionFactory,
    var poolSize:  Int = 30)
  extends Serializable {

  private type Pool = RDD[(Int, InfoThCriterion)]
  private case class F(feat: Int, crit: Double)
  
  def calulateMutualInformation(
       applyMI: Seq[Int] => scala.collection.Map[Int,(Double, Double)],
       varX: Seq[Int],
       miniBatchFraction: Float = .5f) : Array[(Int, (Double, Double))] = {
    
	  val miniBatchSize = if (varX.length > 100) math.round(
	    varX.length * miniBatchFraction) else varX.length
	  val featuresWindow = varX.grouped(miniBatchSize).toArray
	  val resultsByWindow = for (w <- featuresWindow) yield applyMI(w).toArray
	  resultsByWindow
	  	.reduce(_ ++ _)	  
  }

  def setPoolSize(poolSize: Int) = {
    this.poolSize = poolSize
    this
  }
  
   private def selectFeaturesWithoutPool(
      data: RDD[Array[Double]],
      nToSelect: Int,
      nFeatures: Int,
      label: Int,
      nElements: Long,
      miniBatchFraction: Float = 1.f)
    : Seq[F] = {    
        

    // calculate relevance
    val calcMI = IT.miAndCmi(data, _: Seq[Int], label, None, nElements)
    val MiAndCmi = calulateMutualInformation(calcMI,  1 to nFeatures, miniBatchFraction)
	    
	var pool = MiAndCmi.map({ case (k, (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) })
	
    // get maximum and select it
    var max = pool.maxBy(_._2.score)
    var selected = Seq(F(max._1, max._2.score))
    var toSelect = pool.map(_._1) diff Seq(max._1)

    while (selected.size < nToSelect) {
      // update pool
      val calcMI = IT.miAndCmi(data, _: Seq[Int], selected.head.feat, Some(label), nElements)
	  val newMiAndCmi = calulateMutualInformation(calcMI, toSelect, miniBatchFraction)
			  .toMap
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
      nElements: Long, 
      miniBatchFraction: Float = 1.f)
    : Seq[F] = {

    // calculate relevance
    val calcMI = IT.miAndCmi(data, _: Seq[Int], label, None, nElements)
    var rels = calulateMutualInformation(calcMI, 1 to nFeatures, miniBatchFraction)
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

    // select feature with the maximum relevance
    var max = pool.head
    var selected = Seq(F(max._1, max._2.score))
    toSelect = toSelect diff Seq(max._1)

    while (selected.size < nToSelect) {

      // update pool
      val calcMI = IT.miAndCmi(data, _: Seq[Int], selected.head.feat, Some(label), nElements)
	  val newMiAndCmi = calulateMutualInformation(calcMI, toSelect, miniBatchFraction).toMap
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
        	val calcMI = IT.miAndCmi(data, _: Seq[Int], i, Some(label), nElements).toMap
			val missedMiAndCmi = calulateMutualInformation(calcMI, selected.map(_.feat), miniBatchFraction)
			missedMiAndCmi.foreach({ case (_, (mi, cmi)) => pool(i)._2.update(mi, cmi)})
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

    val array = data.map({ case LabeledPoint(label, values) => (label +: values.toArray) })//.cache()
    val nElements = array.count()
    
    var selected = Seq.empty[F]
    criterionFactory.getCriterion match {
      case _: InfoThCriterion with Bound if poolSize != 0 =>
        selected = selectFeaturesWithPool(array, nToSelect, nFeatures, 0, nElements, .5f)
      case _: InfoThCriterion =>
        selected = selectFeaturesWithoutPool(array, nToSelect, nFeatures, 0, nElements, .5f)
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
