package org.apache.spark.mllib.featureselection

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, StorageVector => BSTV}
import org.apache.spark.broadcast.Broadcast

/**
 * Object with some Information Theory methods.
 */
object InfoTheory {

  private val log2 = { x: Double => math.log(x) / math.log(2) }  
  
  /**
   * Calculates mutual information (MI) and conditional mutual information (CMI) simultaneously
   * for several variables (X's) with others (Y's), conditioned by an optional variable Z. 
   * Indexes must be disjoint.
   *
   * @param data RDD of data containing the variables (first element is the class attribute)
   * @param varX Indexes of variables
   * @param varY Indexes of the second variable
   * @param varZ Indexes of the conditioning values
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
      denseData: Boolean) = {
    
    // Pre-requisites
    require(varX.size > 0 && varY.size > 0)  
    require(varX.intersect(varY).size == 0)
    
     varZ match {
        case Some(z) => 
          val seqZ = Seq(z)
          require(varX.intersect(Seq(z)).size == 0 && 
              varY.intersect(Seq(z)).size == 0)
          case _ => (1 to nFeatures).diff(varX) ++ varY           
    }

    val sc = data.context
    val bMultY = sc.broadcast(varY.length > 1)
    val bFirstY = sc.broadcast(varY(0))
    val bvarY = sc.broadcast(varY)
    val bvarZ = sc.broadcast(varZ)    
        
    /* Pair generator for sparse data */
    def SparseGenerator(v: BV[Byte], 
        bvarX: Broadcast[Seq[Int]], 
        bisXPoint: Broadcast[Int => Boolean]): 
          Seq[((Any, Byte, Byte, Option[Byte]), Long)] =  {
      
       val sv = v.asInstanceOf[BSV[Byte]]
       val (withZ, varZ) = bvarZ.value match {
         case Some(v) => (true, v)
         case None => (false, -1)
       }
       
       var zval: Option[Byte] = None
       var xValues = Seq.empty[(Int, Byte)]
       var yValues = Seq.empty[(Int, Byte)] 
       
       for(offset <- 0 until sv.activeSize) {
         
            val index: Int = sv.indexAt(offset)
            val value: Byte = sv.valueAt(offset)
            
            // This X index is involved in calculation?
            if (bisXPoint.value(index)) {
              xValues = (index, value) +: xValues
            // Same for a Y index
            } else if (bvarY.value.contains(index)) {
              yValues = (index, value) +: yValues
            } else if (withZ && index == varZ) {
              zval = Some(value)
            }
        }  
       
       // Generate pairs using X and Y values generated before
       var pairs = Seq.empty[((Any, Byte, Byte, Option[Byte]), Long)]
       for((xind, xval) <- xValues){
         for((yind, yval) <- yValues) {
           if(bMultY.value) pairs = (((xind, yind), xval, yval, zval), 1L) +: pairs else
             pairs = ((xind, xval, yval, zval), 1L) +: pairs
         }
       }     
       
       pairs      
    }
    
    /* Pair generator for dense data */
    def DenseGenerator(v: BV[Byte], 
        bvarX: Broadcast[Seq[Int]]): 
          Seq[((Any, Byte, Byte, Option[Byte]), Long)] = {
      
       val dv = v.asInstanceOf[BDV[Byte]]
       val zval = bvarZ.value match {case Some(z) => Some(v(z)) case None => None}
       
       var pairs = Seq.empty[((Any, Byte, Byte, Option[Byte]), Long)]
       for(xind <- bvarX.value){
         for(yind <- bvarY.value) {
           if(bMultY.value) pairs =(((xind, yind), dv(xind), dv(yind), zval), 1L) +: pairs else
             pairs = ((xind, dv(xind), dv(yind), zval), 1L) +: pairs
         }
       }     
       
       pairs
    }

    /* Common function to generate pairs, it choose between sparse and dense fetch 
     * (full or indexed, respectively)
     */
    val pairsGenerator = denseData match {
      case true =>
        val bvarX = sc.broadcast(varX)
        DenseGenerator(_: BV[Byte], bvarX)
      case false =>        
        /* In order to alleviate the amount of X indexes used here, we choose between
         * the whole set of X indexes and its opposite.
         */
        val nonSelectVars = varZ match {
          case Some(z) => (1 to nFeatures).diff(varX) ++ varY ++ Seq(z)           
          case _ => (1 to nFeatures).diff(varX) ++ varY           
        }
        val reverseX = nonSelectVars.length < varX.length
        val auxVarX = if(reverseX) nonSelectVars.toSeq else varX
        val bauxVarX = sc.broadcast(auxVarX)
        val bisPoint = sc.broadcast(if(reverseX) !bauxVarX.value.contains(_: Int) else 
            bauxVarX.value.contains(_: Int))
        SparseGenerator(_: BV[Byte], bauxVarX, bisPoint)
    }
    
    val combinations = data
      .flatMap(pairsGenerator)
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

    val createCombiner: ((Byte, Long)) => (Long, Long, Long, Long, Long, Long, Long) = {
      case (1, q) => (q, 0, 0, 0, 0, 0, 0)
      case (2, q) => (0, q, 0, 0, 0, 0, 0)
      case (3, q) => (0, 0, q, 0, 0, 0, 0)
      case (4, q) => (0, 0, 0, q, 0, 0, 0)
      case (5, q) => (0, 0, 0, 0, q, 0, 0)
      case (6, q) => (0, 0, 0, 0, 0, q, 0)
      case (7, q) => (0, 0, 0, 0, 0, 0, q)
    }

    val mergeValues: ((Long, Long, Long, Long, Long, Long, Long), (Byte, Long)) => 
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

    val mergeCombiners: ((Long, Long, Long, Long, Long, Long, Long), (Long, Long, Long, Long, Long, Long, Long)) => 
      (Long, Long, Long, Long, Long, Long, Long) = {
      case ((qxz1, qyz1, qxyz1, qz1, qxy1, qx1, qy1), (qxz2, qyz2, qxyz2, qz2, qxy2, qx2, qy2)) =>
        (qxz1 + qxz2, qyz1 + qyz2, qxyz1 + qxyz2, qz1 + qz2, qxy1 + qxy2, qx1 + qx2, qy1 + qy2)
    }

    // Count frequencies for each combination
    val grouped_frequencies =
      combinations.reduceByKey({
        case ((keys1, q1), (keys2, q2)) => (keys1 ++ keys2, q1 + q2)
      })
      // Separate by origin of combinations
      .flatMap({
        case ((k, ref, id), (keys, q))  => 
          val none: Option[Byte] = None
          ref match {
            case 1 => 
              val (x, z) = id.asInstanceOf[(Byte, Byte)]
              for (y <- keys) yield ((k, x, y.asInstanceOf[Byte], Some(z)), (1:Byte, q))
            case 2 =>
              val (y, z) = id.asInstanceOf[(Byte, Byte)]
              for (x <- keys) yield ((k, x.asInstanceOf[Byte], y, Some(z)), (2:Byte, q))
            case 3 =>
              val (x, y, z) = id.asInstanceOf[(Byte, Byte, Byte)]
              Seq(((k, x, y, Some(z)), (3:Byte, q)))
            case 4 =>
              val z = id.asInstanceOf[Byte]
              for ((x, y) <- keys) yield 
                ((k, x.asInstanceOf[Byte], y.asInstanceOf[Byte], Some(z)), (4:Byte, q))
            case 5 =>
                val (x, y) = id.asInstanceOf[(Byte, Byte)]
                Seq(((k, x, y, none), (5:Byte, q)))
            case 6 =>
              val x = id.asInstanceOf[Byte]
              for (y <- keys) yield ((k, x, y.asInstanceOf[Byte], none), (6:Byte, q))
            case 7 =>
              val y = id.asInstanceOf[Byte]
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
          case kx: Int => (kx, bFirstY.value)
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
    .reduceByKeyLocally({ case ((mi1, cmi1), (mi2, cmi2)) => (mi1 + mi2, cmi1 + cmi2) })
  }

}
