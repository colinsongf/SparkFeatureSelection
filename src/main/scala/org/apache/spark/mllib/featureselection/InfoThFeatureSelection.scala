package org.apache.spark.mllib.featureselection

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.featureselection.{InfoTheory => IT}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.annotation.Experimental

/**
 * Train a info-theory feature selection model according to a criterion.
 * @param criterionFactory Factory to create info-theory measurements for each feature.
 * @param poolSize In case of using pool optimization, it indicates pool increments.
 */
class InfoThFeatureSelection private (
    val criterionFactory: InfoThCriterionFactory,
    val data: RDD[LabeledPoint])
  extends Serializable {

  private type Pool = RDD[(Int, InfoThCriterion)]
  protected case class F(feat: Int, crit: Double)
    
	val (nFeatures, isDense) = data.first.features match {
		case v: SparseVector => (v.size, false)
		case v: DenseVector => (v.size, true)			  
	}
  
  	val byteData: RDD[BV[Byte]] = data.map {
      case LabeledPoint(label, values: SparseVector) => 
            new BSV[Byte](0 +: values.indices.map(_ + 1), label.toByte +: values.values.toArray.map(_.toByte), values.indices.size + 1)
      case LabeledPoint(label, values: DenseVector) => 
            new BDV[Byte](label.toByte +: values.toArray.map(_.toByte))
    }
  
    
  /**
   * Wrapper method to calculate mutual information (MI) and conditional mutual information (CMI) 
   * on several X variables with the capability of splitting the calculations into several chunks.
   * @param applyMI Partial function that calculates MI and CMI on a subset of variables
   * @param X variables to subset
   * @param miniBatchFraction Percentage of simultaneous features to calculate MI and CMI (just in case).
   * @return MI and CMI results for each X variable
   */
  @deprecated
    protected def calcMutualInformation(
       applyMI: Seq[Int] => RDD[((Int, Int),(Double, Double))], 
       varX: Seq[Int], 
       miniBatchFraction: Float) = {
    	  val miniBatchSize = math.round(varX.length * miniBatchFraction)
		  val it = varX.grouped(miniBatchSize)
		  var results = Seq.empty[RDD[((Int, Int), (Double, Double))]]
		  while (it.hasNext) applyMI(it.next) +: results
		  for(i <- 1 until results.size) results(0).union(results(i))
		  results(0)
  }
    
	implicit val orderedByScore = new Ordering[(Int, InfoThCriterion)] {
	    override def compare(a: (Int, InfoThCriterion), b: (Int, InfoThCriterion)) = {
	    	a._2.score.compare(b._2.score) 
	    }
	}
	
	implicit val orderedByRelevance = new Ordering[(Int, InfoThCriterion)] {
	    override def compare(a: (Int, InfoThCriterion), b: (Int, InfoThCriterion)) = {
	    	a._2.relevance.compare(b._2.relevance) 
	    }
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
      nToSelect: Int)
    : Seq[F] = {            
    
    val nElements = data.count()
    val nFeatures = data.first.size - 1
    val label = 0
    
    // calculate relevance
    val MiAndCmi = IT.miAndCmi(data, 1 to nFeatures, Seq(label), None, nElements, nFeatures)
    var pool = MiAndCmi.map({ case ((x, y), (mi, _)) => (x, criterionFactory.getCriterion.init(mi)) }).collectAsMap()
  
    // Print most relevant features
    val strRels = MiAndCmi.collect().sortBy(-_._2._1)
      .take(nToSelect)
      .map({case ((f, _), (mi, _)) => f + "\t" + "%.4f" format mi})
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)
  
    // get maximum and select it
    val firstMax = pool.maxBy(_._2.score)
    var selected = Seq(F(firstMax._1, firstMax._2.score))
    pool = pool - firstMax._1

    while (selected.size < nToSelect) {
      // update pool
      val newMiAndCmi = IT.miAndCmi(data, pool.keys.toSeq, Seq(selected.head.feat), 
          Some(label), nElements, nFeatures) 
                          .map({ case ((x, _), crit) => (x, crit) })
                          .collectAsMap()
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
      poolSize: Int)
    : Seq[F] = {
    
    val label = 0
    val nElements = data.count()
    
    // calculate relevance
    var orderedRels = IT.miAndCmi(data,1 to nFeatures, 
        Seq(label), None, nElements, nFeatures).collect
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
      val newMiAndCmi = IT.miAndCmi(data, pool.keys.toSeq, Seq(selected.head.feat), 
          Some(label), nElements, nFeatures)
            .collectAsMap()
            .map({ case ((x, _), crit) => (x, crit) })
            
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
        val missedMiAndCmi = IT.miAndCmi(data, newFeatures.keys.toSeq, 
            selected.map(_.feat), Some(label), nElements, nFeatures)        
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
  

  protected def run(nToSelect: Int,
      poolSize: Int = 30): 
        InfoThFeatureSelectionModel = {
    
    if (nToSelect > nFeatures) {
      throw new IllegalArgumentException("data doesn't have so many features")
    }    
    
    byteData.persist(StorageLevel.MEMORY_ONLY_SER)
    
    val selected = criterionFactory.getCriterion match {
      case _: InfoThCriterion with Bound if poolSize != 0 =>
        selectFeaturesWithPool(byteData, nToSelect, poolSize)
      case _: InfoThCriterion =>
        selectFeaturesWithoutPool(byteData, nToSelect)
    }
    
    byteData.unpersist()
    
    // Print best features according to the mRMR measure
    val strMRMR = selected.map({case F(feat, rel) => feat + "\t" + "%.4f" format rel}).mkString("\n")
    println("\n*** mRMR features ***\nFeature\tScore\n" + strMRMR)

    new InfoThFeatureSelectionModel(selected.map({case F(feat, rel) => (feat - 1, rel) }).toArray)
  }
}

object InfoThFeatureSelection {

  /**
   * Train a feature selection model according to a given criterion and return a feature selection subset of the data.
   *
   * @param   data RDD of LabeledPoint (discrete data with a maximum range of 256).
   * @param   nToSelect maximum number of features to select
   * @param   poolSize number of features to be used in pool optimization in order to alleviate calculations.
   * @return  InfoThFeatureSelectionModel a feature selection model which contains a subset of selected features
   * 
   * Note: LabeledPoint data must be integer values in double representation 
   * with a maximum range value of 256. In this manner, data can be transformed
   * to byte directly. 
   */
  def train(criterionFactory: InfoThCriterionFactory,
      data: RDD[LabeledPoint],
      nToSelect: Int = 100,
      poolSize: Int = 100) = {
    new InfoThFeatureSelection(criterionFactory, data).run(nToSelect, poolSize)
  }
}
