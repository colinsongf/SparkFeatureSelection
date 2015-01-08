package org.lidiagroup.hmourit.tfg.featureselection

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.util.{SearchUtils => SU}
import breeze.linalg.{DenseVector => BDV}
import breeze.linalg.{SparseVector => BSV}
import breeze.linalg.{Vector => BV}
import org.apache.spark.mllib.util.{SearchUtils => SU}

/**
 * Object with some Information Theory methods.
 */
object InfoTheory {

	private val log2 = { x: Double => math.log(x) / math.log(2) }  
	
    private val createCombiner: ((Byte, Long)) => (Long, Long, Long, Long, Long, Long, Long) = {
      case (1, q) => (q, 0, 0, 0, 0, 0, 0)
      case (2, q) => (0, q, 0, 0, 0, 0, 0)
      case (3, q) => (0, 0, q, 0, 0, 0, 0)
      case (4, q) => (0, 0, 0, q, 0, 0, 0)
      case (5, q) => (0, 0, 0, 0, q, 0, 0)
      case (6, q) => (0, 0, 0, 0, 0, q, 0)
      case (7, q) => (0, 0, 0, 0, 0, 0, q)
    }

    private val mergeValues: ((Long, Long, Long, Long, Long, Long, Long), (Byte, Long)) => 
        (Long, Long, Long, Long, Long, Long, Long) = {
      case ((qxz, qyz, qxyz, qz, qxy, qx, qy), (ref, q)) =>
        ref match {
          case 1 => (qxz + q, qyz, qxyz, qz, qxy, qx, qy)
          case 2 => (qxz, qyz + q, qxyz, qz, qxy, qx, qy)
          case 3 => (qxz, qyz, qxyz + q, qz, qxy, qx, qy)
          case 4 => (qxz, qyz, qxyz, qz + q, qxy, qx, qy)
          case 5 => (qxz, qyz, qxyz, qz, qxy + q, qx, qy)
          case 6 => (qxz, qyz, qxyz, qz, qxy, qx + q, qy)
          case 7 => (qxz, qyz, qxyz, qz, qxy, qx, qy + q)
        }
    }

    private val mergeCombiners: ((Long, Long, Long, Long, Long, Long, Long), (Long, Long, Long, Long, Long, Long, Long)) => 
      (Long, Long, Long, Long, Long, Long, Long) = {
      case ((qxz1, qyz1, qxyz1, qz1, qxy1, qx1, qy1), (qxz2, qyz2, qxyz2, qz2, qxy2, qx2, qy2)) =>
        (qxz1 + qxz2, qyz1 + qyz2, qxyz1 + qxyz2, qz1 + qz2, qxy1 + qxy2, qx1 + qx2, qy1 + qy2)
    }
  
    /* Pair generator for sparse data */
  	private def SparseGenerator(v: BV[Byte], 
  	    fx: Broadcast[Int => Boolean],
        varY: Broadcast[Seq[Int]],
        varZ: Broadcast[Option[Int]]) =  {
      
       val sv = v.asInstanceOf[BSV[Byte]]
       val (withZ, zind) = varZ.value match {
         case Some(v) => (true, v)
         case None => (false, -1)
       }
       
       var zVal: Option[Byte] = None
       var xValues = Seq.empty[(Int, Byte)]
       var yValues = Seq.empty[(Int, Byte)]
       
       // Generate pairs using X and Y values generated before
       var combinations = Seq.empty[((Any, Byte, Any), (scala.collection.immutable.Set[_], Long))]       
       
       for(offset <- 0 until sv.activeSize) {
         
            val index: Int = sv.indexAt(offset)
            val value: Byte = sv.valueAt(offset)
            
            if (withZ && index == zind) {
              zVal = Some(value)
            } else if (SU.binarySearch(varY.value, index)) {
              yValues = (index, value) +: yValues
            // This X index is involved in calculation?
            } else if (fx.value(index)) {
              xValues = (index, value) +: xValues
            }
       }
       
       /* Generate combinations */
     zVal match {
   		case Some(z) => 
            var xycomb = Seq.empty[(Int, Byte, Int, Byte)]
   		  		for((xind, x) <- xValues){
			         for((yind, y) <- yValues) {
    		          val comb = Seq((((xind, yind), 3:Byte /*"xyz"*/, (x, y, z)), (Set.empty, 1L)),
					            (((xind, yind), 5:Byte /*"xy"*/ , (x, y)), (Set.empty,  1L)))
			            combinations = comb ++: combinations
                  xycomb = (xind, x, yind, y) +: xycomb
			         }
		        }
            
            for((xind, x) <- xValues){
              val yset = yValues.toSet
              combinations = ((xind, 6:Byte /*"x"*/, x), (yset, 1L)) +: combinations
              combinations = ((xind, 1:Byte /*"xz"*/ , (x, z)), (yset, 1L)) +: combinations
            }
            
            for((yind, y) <- yValues){
              val xset = xValues.toSet
              combinations = ((yind, 7:Byte /*"y"*/, y), (xset, 1L)) +: combinations
              combinations = ((yind, 2:Byte /*"yz"*/ , (y, z)), (xset, 1L)) +: combinations
            }
            
            combinations = ((None, 4:Byte /*"z"*/  , z), (xycomb.toSet, 1L)) +: combinations

   		case None =>
            for((xind, x) <- xValues){
               for((yind, y) <- yValues) {
                  combinations = (((xind, yind), 5:Byte /*"xy"*/ , (x, y)), (Set.empty,  1L)) +: combinations
               }
            }
            
            for((xind, x) <- xValues){
              val yset = yValues.toSet
              combinations = ((xind, 6:Byte /*"x"*/, x), (yset, 1L)) +: combinations
            }
            
            for((yind, y) <- yValues){
              val xset = xValues.toSet
              combinations = ((yind, 7:Byte /*"y"*/, y), (xset, 1L)) +: combinations
            }
       }
       
       combinations      
    }
  
    /* Pair generator for dense data */
    private def DenseGenerator(v: BV[Byte], 
        varX: Broadcast[Seq[Int]],
        varY: Broadcast[Seq[Int]],
        varZ: Broadcast[Option[Int]]) = {
      
       val dv = v.asInstanceOf[BDV[Byte]]
       val zval = varZ.value match {case Some(z) => Some(v(z)) case None => None}
       
       var pairs = Seq.empty[((Any, Byte, Byte, Option[Byte]), Long)]
       val multY = varY.value.length > 1
       for(xind <- varX.value){
         for(yind <- varY.value) {
        	 val indexes = if(multY) 
        	   (xind, yind) 
        	 else 
        	   xind
             pairs = ((indexes, dv(xind), dv(yind), zval), 1L) +: pairs
         }
       }  
       
       pairs
    } 
    
  
  /**
   * Calculates mutual information (MI) and conditional mutual information (CMI) simultaneously
   * for several variables (X's) with others (Y's), conditioned by an optional variable Z. 
   * Indexes must be disjoint.
   *
   * @param data RDD of data containing the variables (first element is the class attribute)
   * @param varX Indexes of variables (must be sorted and disjoint with Y and Z)
   * @param varY Indexes of the second variable (must be sorted and disjoint with X and Z)
   * @param varZ Indexes of the conditioning values (disjoint with X and Y)
   * @param n    Number of instances
   * @return     RDD of (variable, (MI, CMI))
   */
  def miAndCmi(
      data: RDD[BV[Byte]],
      varX: Seq[Int],
      varY: Seq[Int],
      varZ: Option[Int],
      n: Long,      
      nFeatures: Int, 
      inverseX: Boolean = false) = {
    
    // Pre-requisites
    require(varX.size > 0 && varY.size > 0)

    // Broadcast variables
    val sc = data.context
    val bvarX = sc.broadcast(varX)
    val bvarY = sc.broadcast(varY)
    val bvarZ = sc.broadcast(varZ)
    
    /* Common function to generate pairs, it choose between sparse and dense fetch 
     * (full or indexed, respectively)
     */
    data.first() match {
      case v: BDV[Byte] =>
	    val generator = DenseGenerator(_: BV[Byte], bvarX, bvarY, bvarZ)
	    calculateMIDenseData(data, generator, varY(0), n)
      case v: BSV[Byte] =>        
        val bfX = if(inverseX) sc.broadcast(!SU.binarySearch(bvarX.value, _: Int)) else sc.broadcast(SU.binarySearch(bvarX.value, _: Int))
          
        val combGenerator = SparseGenerator(_: BV[Byte], bfX, bvarY, bvarZ)
        calculateMISparseData(data, combGenerator, varY(0), n)
    }
  }
  
  private def calculateMISparseData(data: RDD[BV[Byte]],
      combGenerator: BV[Byte] => Seq[((Any, Byte, Any), (scala.collection.immutable.Set[_], Long))],
      firstY: Int,
      n: Long) = {
	
		val combinations = data.flatMap(combGenerator)
    
    //println("Comb size: " + combinations.count())
    //println("Top comb: " + combinations.take(100).mkString("\n"))
					            
	    // Count frequencies for each combination
	    val grouped_frequencies =
	      combinations.reduceByKey({
	        case ((keys1, q1), (keys2, q2)) => (keys1 ++ keys2, q1 + q2)
	      })
	      // Separate by origin of combinations
	      .flatMap({
	        case ((k, ref, value), (keys, q))  => 
	          val none: Option[Byte] = None
	          ref match {
	            case 1 /*"xz"*/ => 
	              val (x, z) = value.asInstanceOf[(Byte, Byte)]
	              for ((yind, y) <- keys) yield (((k, yind.asInstanceOf[Int]), x, 
	            		  y.asInstanceOf[Byte], Some(z)), (1:Byte, q))
	            case 2 /*"yz"*/ =>
	              val (y, z) = value.asInstanceOf[(Byte, Byte)]
	              for ((xind, x) <- keys) yield (((xind.asInstanceOf[Int], k), 
	            		  x.asInstanceOf[Byte], y, Some(z)), (2:Byte, q))
	            case 3 /*"xyz"*/ =>
	              val (x, y, z) = value.asInstanceOf[(Byte, Byte, Byte)]
	              Seq(((k, x, y, Some(z)), (3:Byte, q)))
	            case 4 /*"z"*/ =>
	              val z = value.asInstanceOf[Byte]
	              for ((xind, x, yind, y) <- keys) yield 
	                (((xind.asInstanceOf[Int], yind.asInstanceOf[Int]), 
	                		x.asInstanceOf[Byte], y.asInstanceOf[Byte], Some(z)), (4:Byte, q))
	            case 5 /*"xy"*/ =>
	                val (x, y) = value.asInstanceOf[(Byte, Byte)]
	                Seq(((k, x, y, none), (5:Byte, q)))
	            case 6 /*"x"*/ =>
	              val x = value.asInstanceOf[Byte]
	              for ((yind, y) <- keys) yield (((k, yind.asInstanceOf[Int]), 
	                  x, y.asInstanceOf[Byte], none), (6:Byte, q))
	            case 7 /*"y"*/ =>
	              val y = value.asInstanceOf[Byte]
	              for ((xind, x) <- keys) yield (((xind.asInstanceOf[Int], k), 
	                  x.asInstanceOf[Byte], y, none), (7:Byte, q))
	          }        
	      })
	      // Group by origin
	      .combineByKey(createCombiner, mergeValues, mergeCombiners)
	      
        //println("Top gfreq: " + grouped_frequencies.take(100).mkString("\n"))
	    grouped_frequencies.map({
	      case ((k, _, _, z), (qxz, qyz, qxyz, qz, qxy, qx, qy)) =>
	        // Select id
	        val finalKey = k match {
	          case (kx: Int, ky: Int) => (kx, ky)
	          case kx: Int => (kx, firstY)
	        }           
	        // Choose between MI or CMI
	        z match {
	          case Some(_) =>
	              val pz = qz.toDouble / n
	              val pxyz = (qxyz.toDouble / n) / pz
	              val pxz = (qxz.toDouble / n) / pz
	              val pyz = (qyz.toDouble / n) / pz
	              (finalKey, (0.0, pz * pxyz * log2(pxyz / (pxz * pyz))))
	          case None => 
	              val pxy = qxy.toDouble / n
	              val px = qx.toDouble / n
	              val py = qy.toDouble / n
	              (finalKey, (pxy * log2(pxy / (px * py)), 0.0))
	        }
	    })
	    // Compute results for each X
	    .reduceByKey({ case ((mi1, cmi1), (mi2, cmi2)) => (mi1 + mi2, cmi1 + cmi2) })
  }
  
  private def calculateMIDenseData(data: RDD[BV[Byte]],
      pairsGenerator: BV[Byte] => Seq[((Any, Byte, Any, Option[Byte]), Long)],
      firstY: Int,
      n: Long) = {
	
		val combinations = data.flatMap(pairsGenerator)
			 .reduceByKey(_ + _)
			 // Split each combination keeping instance keys
			 .flatMap {
			      case ((k, x, y, Some(z)), q) =>          
			        Seq(((k, 1:Byte /*"xz"*/ , (x, z)),    (Set(y), q)),
			            ((k, 2:Byte /*"yz"*/ , (y, z)),    (Set(x), q)),
			            ((k, 3:Byte /*"xyz"*/, (x, y, z)), (Set.empty, q)),
			            ((k, 4:Byte /*"z"*/  , z),         (Set((x, y)), q)),
			            ((k, 5:Byte /*"xy"*/ , (x, y)),    (Set.empty,  q)),
			            ((k, 6:Byte /*"x"*/  , x),         (Set(y), q)),
			            ((k, 7:Byte /*"y"*/  , y),         (Set(x), q)))
			      case ((k, x, y, None), q) =>
			        Seq(((k, 5:Byte /*"xy"*/ , (x, y)),    (Set.empty,  q)),
			            ((k, 6:Byte /*"x"*/  , x),         (Set(y), q)),
			            ((k, 7:Byte /*"y"*/  , y),         (Set(x), q)))
		    }
	
	    // Count frequencies for each combination
	    val grouped_frequencies =
	      combinations.reduceByKey({
	        case ((keys1, q1), (keys2, q2)) => (keys1 ++ keys2, q1 + q2)
	      })
	      // Separate by origin of combinations
	      .flatMap({
	        case ((k, ref, value), (keys, q))  => 
	          val none: Option[Byte] = None
	          ref match {
	            case 1 => 
	              val (x, z) = value.asInstanceOf[(Byte, Byte)]
	              for (y <- keys) yield ((k, x, y.asInstanceOf[Byte], Some(z)), (1:Byte, q))
	            case 2 =>
	              val (y, z) = value.asInstanceOf[(Byte, Byte)]
	              for (x <- keys) yield ((k, x.asInstanceOf[Byte], y, Some(z)), (2:Byte, q))
	            case 3 =>
	              val (x, y, z) = value.asInstanceOf[(Byte, Byte, Byte)]
	              Seq(((k, x, y, Some(z)), (3:Byte, q)))
	            case 4 =>
	              val z = value.asInstanceOf[Byte]
	              for ((x, y) <- keys) yield 
	                ((k, x.asInstanceOf[Byte], y.asInstanceOf[Byte], Some(z)), (4:Byte, q))
	            case 5 =>
	                val (x, y) = value.asInstanceOf[(Byte, Byte)]
	                Seq(((k, x, y, none), (5:Byte, q)))
	            case 6 =>
	              val x = value.asInstanceOf[Byte]
	              for (y <- keys) yield ((k, x, y.asInstanceOf[Byte], none), (6:Byte, q))
	            case 7 =>
	              val y = value.asInstanceOf[Byte]
	              for (x <- keys) yield ((k, x.asInstanceOf[Byte], y, none), (7:Byte, q))
	          }        
	      })
	      // Group by origin
	      .combineByKey(createCombiner, mergeValues, mergeCombiners)
	      
	    grouped_frequencies.map({
	      case ((k, _, _, z), (qxz, qyz, qxyz, qz, qxy, qx, qy)) =>
	        // Select id
	        val finalKey = k match {
	          case (kx: Int, ky: Int) => (kx, ky)
	          case kx: Int => (kx, firstY)
	        }           
	        // Choose between MI or CMI
	        z match {
	          case Some(_) =>
	              val pz = qz.toDouble / n
	              val pxyz = (qxyz.toDouble / n) / pz
	              val pxz = (qxz.toDouble / n) / pz
	              val pyz = (qyz.toDouble / n) / pz
	              (finalKey, (0.0, pz * pxyz * log2(pxyz / (pxz * pyz))))
	          case None => 
	              val pxy = qxy.toDouble / n
	              val px = qx.toDouble / n
	              val py = qy.toDouble / n
	              (finalKey, (pxy * log2(pxy / (px * py)), 0.0))
	        }
	    })
	    // Compute results for each X
	    .reduceByKey({ case ((mi1, cmi1), (mi2, cmi2)) => (mi1 + mi2, cmi1 + cmi2) })
  }
  
  

}
