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
  
  def calcMutualInformation(
       applyMI: Seq[Int] => scala.collection.Map[Int,(Double, Double)], 
       varX: Seq[Int], 
       miniBatchFraction: Float = .5f) : Array[(Int, (Double, Double))] = {
    
	  val miniBatchSize = if (varX.length > 100) math.round(varX.length * miniBatchFraction) else varX.length
	  val featuresWindow = varX.grouped(miniBatchSize).toArray
	  val resultsByWindow = for (w <- featuresWindow) yield applyMI(w).toArray
	  resultsByWindow.reduce(_ ++ _)	  
  }

  def setPoolSize(poolSize: Int) = {
    this.poolSize = poolSize
    this
  }
  
   private def selectFeaturesWithoutPool(
      data: RDD[Array[Byte]],
      nToSelect: Int,
      nFeatures: Int,
      label: Int,
      nElements: Long,
      miniBatchFraction: Float)
    : Seq[F] = {            

    // calculate relevance
    val calcMI = IT.miAndCmi(data, _: Seq[Int], label, None, nElements)
    val MiAndCmi = calcMutualInformation(calcMI, 1 to nFeatures, miniBatchFraction)
	var pool = MiAndCmi.map({ case (k, (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) }).toMap
	
	// Print most relevant features
    val strMRMR = MiAndCmi.sortBy(-_._2._1)
    	.take(nToSelect)
    	.map({case (f, (mi, _)) => f + "\t" + "%.4f" format mi})
    	.mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strMRMR)
	
    // get maximum and select it
    val firstMax = pool.maxBy(_._2.score)
    var selected = Seq(F(firstMax._1, firstMax._2.score))
    pool = pool - firstMax._1

    while (selected.size < nToSelect) {
      // update pool
      val calcMI = IT.miAndCmi(data, _: Seq[Int], selected.head.feat, Some(label), nElements)
  	  val newMiAndCmi = calcMutualInformation(calcMI, pool.keys.toSeq, miniBatchFraction).toMap
      pool.foreach({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some((mi, cmi)) => crit.update(mi, cmi)
          case None => 
        }
      })

      // get maximum and select it
      val max = pool.maxBy(_._2.score)

      // select feature
      selected = F(max._1, max._2.score) +: selected
      pool = pool - max._1
      val strSelected = selected.reverse
    	.map({case F(f, c) => f + "\t" + "%.4f" format c})
    	.mkString("\n")
      println("\n*** Selected features ***\nFeature\tScore\n" + strSelected)
      
    }
    
    selected.reverse
  }

  private def selectFeaturesWithPool(
      data: RDD[Array[Byte]],
      nToSelect: Int,
      nFeatures: Int,
      label: Int,
      nElements: Long, 
      miniBatchFraction: Float)
    : Seq[F] = {

    // calculate relevance
    val calcMI = IT.miAndCmi(data, _: Seq[Int], label, None, nElements)
    var orderedRels = calcMutualInformation(calcMI, 1 to nFeatures, miniBatchFraction)
    	.map({ case (k, (mi, _)) => (k, mi) })
        .sortBy(-_._2)
        
    // Print most relevant features
    val strRels = orderedRels
    	.take(nToSelect)
    	.map({case (f, c) => f + "\t" + "%.4f" format c}).mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)
	
    // extract pool
    val initialPoolSize = math.min(math.max(poolSize, nToSelect), orderedRels.length)
    var pool = orderedRels.take(initialPoolSize).map({ case (k, mi) =>
    	(k, criterionFactory.getCriterion.init(mi))
    }).toMap
    orderedRels = orderedRels.drop(initialPoolSize)

    // select feature with the maximum relevance
    var max = pool.maxBy(_._2.score)
    var selected = Seq(F(max._1, max._2.score))
    pool = pool - max._1

    while (selected.size < nToSelect) {

      // update pool
      val calcMI = IT.miAndCmi(data, _: Seq[Int], selected.head.feat, Some(label), nElements)
	  val newMiAndCmi = calcMutualInformation(calcMI, pool.keys.toSeq, miniBatchFraction).toMap
      pool.foreach({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some((mi, cmi)) => crit.update(mi, cmi)
          case None => 
        }
      })
      
      // look for maximum and bound
      max = pool.maxBy(_._2.score)
      var min = pool.minBy(_._2.score)._2.asInstanceOf[InfoThCriterion with Bound]
      
      // increase pool if necessary
      while (max._2.score < min.bound && orderedRels.size > 0) { 
        
        val realPoolSize = math.min(poolSize, orderedRels.length)        
        val newFeatures = orderedRels.take(realPoolSize).map{ case (k, mi) => (k, criterionFactory.getCriterion.init(mi)) }
        
        // do missed calculations (for each previously selected attribute)
        selected.foreach({case F(feat, c) =>
        	val calcMI = IT.miAndCmi(data, _: Seq[Int], feat, Some(label), nElements)
			val missedMiAndCmi = calcMutualInformation(calcMI, newFeatures.map(_._1), miniBatchFraction).toMap
			newFeatures.foreach{ case (feat, crit) => 
			  missedMiAndCmi.get(feat) match {
  				case Some((mi, cmi)) => crit.update(mi, cmi)
  				case None => 
			  }		  
        	}
        })
        
        // Add new features to the pool and remove them from the other set
        pool ++= newFeatures
        orderedRels = orderedRels.drop(realPoolSize)
        
        // look for maximum
        max = pool.maxBy(_._2.score)        
        min = pool.minBy(_._2.score)._2.asInstanceOf[InfoThCriterion with Bound]   
      }
      
      // select feature
      selected = F(max._1, max._2.score) +: selected
      pool = pool - max._1
      val strSelected = selected.reverse
    	.map({case F(f, c) => f + "\t" + "%.4f" format c})
    	.mkString("\n")
      println("\n*** Selected features ***\nFeature\tScore\n" + strSelected)
    }

    selected.reverse

  }
  
  @Experimental
  private def discreteDataToByte2(discData: RDD[LabeledPoint]): RDD[Array[Byte]] = {
       val nFeatures = discData.first.features.size + 1
	   val arrData = discData.map({ case LabeledPoint(label, values) => (label +: values.toArray) })
	   val distinct = (0 until nFeatures).map(i => arrData.map(d => d(i)).distinct)
	   
	   // Normalize to [0, 1]
	   val dict = (0 until nFeatures).map({i => 
	   		require(distinct(i).count < 257)
	   		distinct(i).collect.zipWithIndex.toMap
	   })
	   
	   // Normalize to [-126, 127]
	   arrData.map({ case d =>
	   		(0 until nFeatures).map({i => 
	   		  val x = dict(i).getOrElse(d(i), 0)
	   		  ((x * 255) - 126).toByte 	   		  
	   		}).toArray
	   })	   
  }
  
  private def discreteDataToByte(discData: RDD[LabeledPoint]): RDD[Array[Byte]] = {
	  discData.map({ case LabeledPoint(label, values) => 
	     (label.toByte +: values.toArray.map(_.toByte)) 
	  })	   
  }  

  def run(data: RDD[LabeledPoint], nToSelect: Int, miniBatchFraction: Float): InfoThFeatureSelectionModel = {
    
    val nFeatures = data.first.features.size

    if (nToSelect > nFeatures) {
      throw new IllegalArgumentException("data doesn't have so many features")
    }
    
    val byteData = discreteDataToByte(data).persist(StorageLevel.MEMORY_ONLY_SER)
    //val array = data.map({ case LabeledPoint(label, values) => (label +: values.toArray) })//.cache()
    val nElements = byteData.count()
    
    val selected = criterionFactory.getCriterion match {
      case _: InfoThCriterion with Bound if poolSize != 0 =>
        selectFeaturesWithPool(byteData, nToSelect, nFeatures, 0, nElements, miniBatchFraction)
      case _: InfoThCriterion =>
        selectFeaturesWithoutPool(byteData, nToSelect, nFeatures, 0, nElements, miniBatchFraction)
      case _ => Seq.empty[F]
    }
    
    byteData.unpersist()
    
    // Print best features according to the mRMR measure
    val strMRMR = selected.map({case F(f, c) => f + "\t" + "%.4f" format c}).mkString("\n")
    println("\n*** mRMR features ***\nFeature\tScore\n" + strMRMR)

    new InfoThFeatureSelectionModel(selected.map({ case F(feat, rel) => feat - 1 }).toArray)
  }
}

object InfoThFeatureSelection {

  def train(criterionFactory: InfoThCriterionFactory,
      data: RDD[LabeledPoint],
      nToSelect: Int,
      poolSize: Int = 30,
      miniBatchFraction: Float = .5f) = {
    new InfoThFeatureSelection(criterionFactory, poolSize).run(data, nToSelect, miniBatchFraction)
  }
}
