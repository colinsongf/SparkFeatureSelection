package org.lidiagroup.hmourit.tfg.featureselection

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext._
import org.lidiagroup.hmourit.tfg.featureselection.{InfoTheory => IT}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}

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
  
    val nElements = byteData.count()
    
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
   * Trains a info-theory selection model without using pool optimization.
   * @param data Data points represented by a byte array (first element is the class attribute).
   * @param nToSelect Number of features to select.
   * @param nElements Number of instances.
   * @return A list with the most relevant features and its scores
   */
   protected def selectFeaturesWithoutPool(
      data: RDD[BV[Byte]],
      nToSelect: Int,
      poolSize: Int)
    	: RDD[(Int, Double)] = {            
     
    require(nToSelect < nFeatures)
    val label = 0
    val sc = data.context
    
    // calculate relevance
    val MiAndCmi = if(isDense) 
      IT.miAndCmi(data, 1 to nFeatures, Seq(label), None, nElements, nFeatures) 
    else
      IT.miAndCmi(data, Seq(label), Seq(label), None, nElements, nFeatures, true)
    
    // Init criterions and sort by key
    var pool = MiAndCmi
    	.map({ case ((x, _), (mi, _)) => (x, criterionFactory.getCriterion.init(mi)) })
    
	// Print most relevant features
    val strRels = pool
    	.top(nToSelect)(orderedByRelevance)
    	.map({case (f, c) => f + "\t" + "%.4f" format c.relevance}).mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)
	
    // get maximum and select it
    val max = pool.max()(orderedByScore)
    var lastSelected = (max._1, max._2.score)
    var selected = sc.parallelize(Seq(lastSelected))
    pool = pool.subtractByKey(selected)
    var nSelected = 1
    
    while (nSelected < nToSelect) {
      // update pool
      val newMiAndCmi = if(isDense) {
  		  IT.miAndCmi(data, pool.map(_._1).collect, Seq(lastSelected._1), 
  				  Some(label), nElements, nFeatures) 
  	  } else {
  		  // As label = 0, it is not necessary to sort
  		  val inverseX = label +: selected.sortByKey().map(_._1).collect
  		  IT.miAndCmi(data, inverseX, Seq(lastSelected._1), Some(label), nElements, nFeatures, true)
  	  }
      
  	  // Update criterions in the pool
  	  pool = pool.leftOuterJoin(newMiAndCmi.map({ case ((x, _), crit) => (x, crit) }))
  	  			 .mapValues{
	  	          case (crit, Some((mi, cmi))) => crit.update(mi, cmi)
			  	      case (crit, None) => crit
		      	  }

      // get the maximum and add it to the final set
      val max = pool.max()(orderedByScore)
      lastSelected = (max._1, max._2.score)
      val newSelected = sc.parallelize(Seq(lastSelected))
      selected = selected.union(newSelected)
      pool = pool.subtractByKey(newSelected)
      nSelected = nSelected + 1
      /*val strSelected = selected
    	    .map({case (f, c) => f + "\t" + "%.4f" format c})
    	    .collect
    	    .mkString("\n")
      println("\n*** Selected features ***\nFeature\tScore\n" + strSelected)*/
    }
    
    selected
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
    : RDD[(Int, Double)] = {
    
    require(nToSelect < nFeatures)
    val label = 0
    val sc = data.context
    
    // calculate relevance
    val MiAndCmi = if(isDense) 
      IT.miAndCmi(data, 1 to nFeatures, Seq(label), None, nElements, nFeatures) 
    else
      IT.miAndCmi(data, Seq(label), Seq(label), None, nElements, nFeatures, true)
    var wholeSet = MiAndCmi.map({ case ((k, _), (mi, _)) => (k, criterionFactory.getCriterion.init(mi)) })
	        
    // Print most relevant features
    val strRels = wholeSet
    	.top(nToSelect)(orderedByRelevance)
    	.map({case (f, c) => f + "\t" + "%.4f" format c.relevance}).mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)
	
    // extract pool
    val initialPoolSize = math.max(poolSize, nToSelect)
    // var indixes must be sorted before applying MI function
    var pool = sc.parallelize(wholeSet.top(initialPoolSize)(orderedByRelevance)).sortByKey()
	  wholeSet = wholeSet.subtractByKey(pool)
	  var leftRels = nFeatures - initialPoolSize

    // select feature with the maximum relevance
    var max = pool.max()(orderedByScore)
    var lastSelected = (max._1, max._2.score)
    var selected = sc.parallelize(Seq(lastSelected))
    pool = pool.filter{case (k, _) => k != max._1}
    //pool = pool.subtractByKey(selected)
    var nSelected = 1
    
    while (nSelected < nToSelect) {

      // update pool (varX must be sorted)
      val newMiAndCmi = IT.miAndCmi(data, 
          pool.map(_._1).collect, 
          Seq(lastSelected._1), 
          Some(label), 
          nElements, 
          nFeatures).map({ case ((x, _), crit) => (x, crit) })
            
	    pool = pool.leftOuterJoin(newMiAndCmi).mapValues{
			  	      case (crit, Some((mi, cmi))) => crit.update(mi, cmi)
			  	      case (crit, None) => crit
		      	  }
	  
      // look for maximum and bound
      max = pool.max()(orderedByScore)
      var bound = pool.min()(orderedByRelevance)._2
    		    .asInstanceOf[InfoThCriterion with Bound].bound
            
      // increase pool if necessary
      while (max._2.score < bound && leftRels > 0) {
        
        // Select a new subset to be added to the pool        
        val realPoolSize = math.min(poolSize, leftRels)
        var newFeatures = sc.parallelize(wholeSet.top(realPoolSize)(orderedByRelevance))

        // do missed calculations (for each previously selected attribute)
        val missedMiAndCmi = IT.miAndCmi(data, 
        							newFeatures.sortByKey().map(_._1).collect, 
        							selected.sortByKey().map(_._1).collect, 
        							Some(label), 
        							nElements, 
                      nFeatures)
									.map({ case ((x, _), crit) => (x, crit) })
									.groupByKey()
		
		    newFeatures = newFeatures.leftOuterJoin(missedMiAndCmi).mapValues{
					  	    case (crit, Some(it)) => 
					  	      	for((mi, cmi) <- it) crit.update(mi, cmi)
					  	      	crit
					  	    case (crit, None) => crit
  		  }
        
        // Add new features to the pool and remove them from the relevances set
        pool = pool.union(newFeatures).sortByKey()
        wholeSet = wholeSet.subtractByKey(newFeatures)
        leftRels = leftRels - realPoolSize
        
        // calculate again maximum and bound
        max = pool.max()(orderedByScore)
        bound = pool.min()(orderedByRelevance)._2
        			.asInstanceOf[InfoThCriterion with Bound].bound
      }
      
      // select feature
      lastSelected = (max._1, max._2.score)
      val newSelected = sc.parallelize(Seq(lastSelected))
	    selected = selected.union(newSelected)
      //pool = pool.subtractByKey(newSelected)
      pool = pool.filter{case (k, _) => k != max._1}
      nSelected = nSelected + 1
      
      /*val strSelected = selected
    	    .map({case (f, c) => f + "\t" + "%.4f" format c})
    	    .collect
    	    .mkString("\n")
      println("\n*** Selected features ***\nFeature\tScore\n" + strSelected)*/
    }

    selected
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
        selectFeaturesWithoutPool(byteData, nToSelect, poolSize)
    }
    
    byteData.unpersist()
    
    // Print best features according to the mRMR measure
    val strMRMR = selected.collect.map({case (f, c) => f + "\t" + "%.4f" format c}).mkString("\n")
    println("\n*** mRMR features ***\nFeature\tScore\n" + strMRMR)

    new InfoThFeatureSelectionModel(selected.map({ case (feat, rel) => (feat - 1, rel) }).toArray)
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
