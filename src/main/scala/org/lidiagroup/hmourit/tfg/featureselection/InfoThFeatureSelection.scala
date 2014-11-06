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
  
  def calculateMutualInformationWithSplitting(partitions: Array[RDD[Array[Double]]],
      varX: Seq[Int],
      varY: Int,
      varZ: Option[Int]) = {
    
	  val miPerPartitions = for(part <- partitions) yield IT.miAndCmi(
	      part, varX, varY, varZ, part.count)
	      
	    var output = "\n\n\nPool\n"
		output += miPerPartitions.map(_.map{case (k, (mi, cmi)) => k + "\t" + mi + "," + cmi}).mkString("\n\n\n")
		val miOutput = partitions(0).context.parallelize(Array(output), 1)
	    miOutput.saveAsTextFile("hdfs://localhost:8020/user/sramirez/results/subSetRos_test16/boostrapMI")
	      
	  miPerPartitions
	  	.map(_.toSeq)
	  	.reduce(_ ++ _)
	  	.groupBy(_._1)
	  	.mapValues(_.map(_._2))
    	.mapValues(_.foldLeft(0.0,0.0){ 
    	  case ((smi, scmi), (mi, cmi)) => (smi + mi, scmi + cmi)
    	  })
  }
  
  def calulateMutualInformation(
       applyMI: Seq[Int] => scala.collection.Map[Int,(Double, Double)],
       varX: Seq[Int],
       miniBatchFraction: Float = .5f) : Array[(Int, (Double, Double))] = {
    
	  val miniBatchSize = if (varX.length > 100) math.round(
	    varX.length * miniBatchFraction) else varX.length
	  val featuresWindow = varX.sliding(miniBatchSize)
	  featuresWindow
    	.map(w => {
	    	applyMI(w)
	    		.toArray
	    })
	    .reduce(_ ++ _)
  }

  def setPoolSize(poolSize: Int) = {
    this.poolSize = poolSize
    this
  }
  	@Experimental
    private def selectFeaturesWithoutSplits(
      data: RDD[Array[Double]],
      nToSelect: Int,
      nFeatures: Int,
      label: Int,
      nPartitions: Int)
    : Seq[F] = {
    
    val weights = Array.fill[Double](nPartitions)(nPartitions / 100f)
    val partitions = data.randomSplit(weights)
    
    var output = partitions
    	.map(_.map(v => (v(0), 1L)))
    	.map(_.groupByKey())
    	.map(_.mapValues(_.sum))
    	.map(_.collectAsMap.toString())
    	.mkString("\n")   	
    
    val parOutput = data.context.parallelize(Array(output), 1)
    parOutput.saveAsTextFile("hdfs://localhost:8020/user/sramirez/results/subSetRos_test16/partitions")
    
    // calculate relevance
    var pool = calculateMutualInformationWithSplitting(partitions, 1 to nFeatures, label, None)
		.map({ case (k, (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) })
		.toArray
		
    var pool2 = IT.miAndCmi(data, 1 to nFeatures, label, None, data.count)
      .map({ case (k, (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) })
      .toArray
      
	output = "\n\n\nPool\n"
	output += pool.map(t => t._1 + "\t" + t._2.score).mkString("\n")
	output += "\n\n\nPool2\n"
	output += pool2.map(t => t._1 + "\t" + t._2.score).mkString("\n")
	var miOutput = data.context.parallelize(Array(output), 1)
    miOutput.saveAsTextFile("hdfs://localhost:8020/user/sramirez/results/subSetRos_test16/miCalcs")
    

    // get maximum and select it
    var max = pool.maxBy(_._2.score)
    var selected = Seq(F(max._1, max._2.score))
    var toSelect = pool.map(_._1) diff Seq(max._1)

    while (selected.size < nToSelect) {
      // update pool
      val newMiAndCmi = calculateMutualInformationWithSplitting(partitions, toSelect, selected.head.feat, Some(label))
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
  	
   @Experimental
   private def selectFeaturesWithoutPool(
      data: RDD[Array[Double]],
      nToSelect: Int,
      nFeatures: Int,
      label: Int,
      nElements: Long,
      miniBatchFraction: Float)
    : Seq[F] = {    
        

    // calculate relevance
    val varX = 1 to nFeatures
    val calcMI = IT.miAndCmi(data, _: Seq[Int], label, None, nElements)
    val MiAndCmi = calulateMutualInformation(calcMI, varX, miniBatchFraction)
	    
	val pool = MiAndCmi.map({ case (k, (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) })
	
    // get maximum and select it
    var max = pool.maxBy(_._2.score)
    var selected = Seq(F(max._1, max._2.score))
    var toSelect = pool.map(_._1) diff Seq(max._1)

    while (selected.size < nToSelect) {
      // update pool
	  val miniBatchSize = if (toSelect.length > 100) math.round(
	  toSelect.length * miniBatchFraction) else toSelect.length
	  val featuresWindow = toSelect.sliding(miniBatchSize)
      val newMiAndCmi = featuresWindow
    	.map(w => {
	    	IT.miAndCmi(data, w, selected.head.feat, Some(label), nElements)
	    		.toArray
	    })
	    .reduce(_ ++ _)
	    .toMap
      pool.map({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some((mi, cmi)) => (k, crit.update(mi, cmi))
          case _ => /* Don't update any criterion */
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
    
    // calculate relevance
    val pool = IT.miAndCmi(data, 1 to nFeatures, label, None, nElements)
      .map({ case (k, (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) })
      .toArray

    // get maximum and select it
    var max = pool.maxBy(_._2.score)
    var selected = Seq(F(max._1, max._2.score))
    var toSelect = pool.map(_._1) diff Seq(max._1)

    while (selected.size < nToSelect) {
      // update pool
      val newMiAndCmi = IT.miAndCmi(data, toSelect, selected.head.feat, Some(label), nElements)
      pool.map({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some((mi, cmi)) => (k, crit.update(mi, cmi))
          case _ => /* Don't update any criterion */
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

    // select feature with the maximum relevance
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
        selected = selectFeaturesWithPool(array, nToSelect, nFeatures, 0, array.count())
      case _: InfoThCriterion =>
        selected = selectFeaturesWithoutPool(array, nToSelect, nFeatures, 0, 10, .2f)
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
