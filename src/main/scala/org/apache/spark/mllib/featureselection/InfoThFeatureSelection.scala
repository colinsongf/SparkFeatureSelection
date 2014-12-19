package org.apache.spark.mllib.featureselection

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.featureselection.{InfoTheory => IT}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.linalg.convert

/**
 * Train a info-theory feature selection model according to a criterion.
 * @param criterionFactory Factory to create info-theory measurements for each feature.
 * @param poolSize In case of using pool optimization, it indicates pool increments.
 */
class InfoThFeatureSelection private (
    val criterionFactory: InfoThCriterionFactory,
    var poolSize:  Int = 30)
  extends Serializable {

  private type Pool = RDD[(Int, InfoThCriterion)]
  protected case class F(feat: Int, crit: Double)
  
  /**
   * Wrapper method to calculate mutual information (MI) and conditional mutual information (CMI) 
   * on several X variables with the capability of splitting the calculations into several chunks.
   * @param applyMI Partial function that calculates MI and CMI on a subset of variables
   * @param X variables to subset
   * @param miniBatchFraction Percentage of simultaneous features to calculate MI and CMI (just in case).
   * @return MI and CMI results for each X variable
   */
  protected def calcMutualInformation(
       applyMI: Seq[Int] => scala.collection.Map[(Int, Int),(Double, Double)], 
       varX: Seq[Int], 
       miniBatchFraction: Float) = {
    
	  val miniBatchSize = math.round(varX.length * miniBatchFraction)
	  val featuresWindow = varX.grouped(miniBatchSize).toArray
	  val resultsByWindow = for (w <- featuresWindow) yield applyMI(w).toArray
	  resultsByWindow.reduce(_ ++ _)	  
  }

  
  def setPoolSize(poolSize: Int) = {
    this.poolSize = poolSize
    this
  }
  
  /**
   * Method that trains a info-theory selection model without using pool optimization.
   * @param data Data points represented by a byte array (first element is the class attribute).
   * @param nToSelect Number of features to select.
   * @param nElements Number of instances.
   * @param miniBatchFraction Fraction of data to be used in each iteration (just in case).
   * @return A list with the most relevant features and its scores
   */
   protected def selectFeaturesWithoutPool(
      data: RDD[BV[Byte]],
      nToSelect: Int,
      nElements: Long,
      nFeatures: Int,
      miniBatchFraction: Float)
    : Seq[F] = {            
     
    val label = 0
    
    // calculate relevance
    val calcMI = IT.miAndCmi(data, _: Seq[Int], Seq(label), None, nElements, nFeatures)
    val MiAndCmi = calcMutualInformation(calcMI, 1 to nFeatures, miniBatchFraction)
	  var pool = MiAndCmi.map({ case ((x, y), (mi, _)) => (x, criterionFactory.getCriterion.init(mi)) }).toMap
	
	  // Print most relevant features
    val strMRMR = MiAndCmi.sortBy(-_._2._1)
    	.take(nToSelect)
    	.map({case ((f, _), (mi, _)) => f + "\t" + "%.4f" format mi})
    	.mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strMRMR)
	
    // get maximum and select it
    val firstMax = pool.maxBy(_._2.score)
    var selected = Seq(F(firstMax._1, firstMax._2.score))
    pool = pool - firstMax._1

    while (selected.size < nToSelect) {
      // update pool
      val calcMI = IT.miAndCmi(data, _: Seq[Int], Seq(selected.head.feat), Some(label), nElements, nFeatures)
  	  val newMiAndCmi = calcMutualInformation(calcMI, pool.keys.toSeq, miniBatchFraction)
          .map({ case ((x, _), crit) => (x, crit) })
          .toMap
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

 /**
   * Method that trains a info-theory selection model using pool optimization.
   * @param data Data points represented by a byte array (first element is the class attribute).
   * @param nToSelect Number of features to select.
   * @param nElements Number of instances.
   * @param miniBatchFraction Fraction of data to be used in each iteration (just in case).
   * @return A list with the most relevant features and its scores
   */
  protected def selectFeaturesWithPool(
      data: RDD[BV[Byte]],
      nToSelect: Int,
      nElements: Long,
      nFeatures: Int,
      miniBatchFraction: Float)
    : Seq[F] = {
    
    val label = 0
    
    // calculate relevance
    val calcMI = IT.miAndCmi(data, _: Seq[Int], Seq(label), None, nElements, nFeatures)
    var orderedRels = calcMutualInformation(calcMI, 1 to nFeatures, miniBatchFraction)
    	.map({ case ((k, _), (mi, _)) => (k, mi) })
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
      val calcMI = IT.miAndCmi(data, _: Seq[Int], Seq(selected.head.feat), Some(label), nElements, nFeatures)
	    val newMiAndCmi = calcMutualInformation(calcMI, pool.keys.toSeq, miniBatchFraction)
            .map({ case ((x, _), crit) => (x, crit) })
            .toMap
      pool.foreach({ case (k, crit) =>
          newMiAndCmi.get(k) match {
            case Some((mi, cmi)) => crit.update(mi, cmi)
            case None => 
          }
      })
      
      // look for maximum and bound
      max = pool.maxBy(_._2.score)
      var min = pool.minBy(_._2.relevance)._2.asInstanceOf[InfoThCriterion with Bound]
      
      // increase pool if necessary
      while (max._2.score < min.bound && orderedRels.size > 0) {
                
        val realPoolSize = math.min(poolSize, orderedRels.length)        
        val newFeatures = orderedRels.take(realPoolSize)
                .map{ case (k, mi) => (k, criterionFactory.getCriterion.init(mi)) }
                .toMap
        
        // do missed calculations (for each previously selected attribute)
        val calcMI = IT.miAndCmi(data, _: Seq[Int], selected.map(_.feat), Some(label), nElements, nFeatures)
        val missedMiAndCmi = calcMutualInformation(calcMI, newFeatures.keys.toSeq, miniBatchFraction)
        
        missedMiAndCmi.foreach{ case ((feat, _), (mi, cmi)) => 
            newFeatures.get(feat) match {
              case Some(crit) => crit.update(mi, cmi)
              case None => 
            }     
        }
        
        // Add new features to the pool and remove them from the other set
        pool ++= newFeatures.toSeq
        orderedRels = orderedRels.drop(realPoolSize)
        
        // look for maximum
        max = pool.maxBy(_._2.score)        
        min = pool.minBy(_._2.relevance)._2.asInstanceOf[InfoThCriterion with Bound]
      }
      
      // select feature
      selected = F(max._1, max._2.score) +: selected
      pool = pool - max._1
      val strSelected = selected.reverse
    	    .map({case F(f, c) => f + "\t" + "%.4f" format c})
    	    .mkString("\n")
      println("\n*** Selected features ***\nFeature\tScore\n" + strSelected)
      println("Pool size: " + pool.size)
    }

    selected.reverse
  }
  
  /**
   * Transform discrete labeled input data in byte array data.
   * 
   * @param discData RDD of LabeledPoint. 
   * @return RDD of byte array
   * 
   * Note: LabeledPoint must be integer values in double representation 
   * with a maximum range value of 256. In this manner, data can be transformed
   * to byte directly. 
   */
  private def discreteDataToByte(discData: RDD[LabeledPoint], nFeatures: Int): RDD[BV[Byte]] = {
    val byteData: RDD[BV[Byte]] = discData.first().features match {
      case _: SparseVector => 
         discData.map({ case LabeledPoint(label, values) =>
            val sv = values.asInstanceOf[SparseVector]
            new BSV( 0 +: sv.indices.map(_ + 1), 
                label.toByte +: sv.toArray.map(_.toByte), 
                sv.size + 1)
          })
      case _: DenseVector =>
        discData.map({ case LabeledPoint(label, values) => 
            val sv = values.asInstanceOf[DenseVector]
            new BDV(label.toByte +: sv.toArray.map(_.toByte))
        })
    }
    
    byteData
  }  
  
  protected def run(
      data: RDD[LabeledPoint], 
      nToSelect: Int, 
      miniBatchFraction: Float): 
        InfoThFeatureSelectionModel = {
    
    val first = data.first
    val nFeatures = first.features match {
      case _: SparseVector => 
         val lastFeat = data.map(_.features.asInstanceOf[SparseVector].indices.max).max
         lastFeat + 1
      case f: DenseVector => f.size + 1
    }
    
    if (nToSelect > nFeatures) {
      throw new IllegalArgumentException("data doesn't have so many features")
    }
    
    val byteData = discreteDataToByte(data, nFeatures).persist(StorageLevel.MEMORY_ONLY_SER)
    val nElements = byteData.count()
    
    val selected = criterionFactory.getCriterion match {
      case _: InfoThCriterion with Bound if poolSize != 0 =>
        selectFeaturesWithPool(byteData, nToSelect, nElements, nFeatures, miniBatchFraction)
      case _: InfoThCriterion =>
        selectFeaturesWithoutPool(byteData, nToSelect, nElements, nFeatures, miniBatchFraction)
      case _ => Seq.empty[F]
    }
    
    byteData.unpersist()
    
    // Print best features according to the mRMR measure
    val strMRMR = selected.map({case F(f, c) => f + "\t" + "%.4f" format c}).mkString("\n")
    println("\n*** mRMR features ***\nFeature\tScore\n" + strMRMR)

    new InfoThFeatureSelectionModel(selected.map({ case F(feat, rel) => (feat - 1, rel) }).toArray)
  }
}

object InfoThFeatureSelection {

  /**
   * Train a feature selection model according to a given criterion and return a feature selection subset of the data.
   *
   * @param   data RDD of LabeledPoint (discrete data with a maximum range of 256).
   * @param   nToSelect maximum number of features to select
   * @param   poolSize number of features to be used in pool optimization in order to alleviate calculations.
   * @param   miniBatchFraction Percentage of simultaneous features to calculate MI and CMI (just in case).
   * @return  InfoThFeatureSelectionModel a feature selection model which contains a subset of selected features
   * 
   * Note: LabeledPoint data must be integer values in double representation 
   * with a maximum range value of 256. In this manner, data can be transformed
   * to byte directly. 
   */
  def train(criterionFactory: InfoThCriterionFactory,
      data: RDD[LabeledPoint],
      nToSelect: Int,
      poolSize: Int = 100,
      miniBatchFraction: Float = 1.0f) = {
    new InfoThFeatureSelection(criterionFactory, poolSize).run(data, nToSelect, miniBatchFraction)
  }
}
