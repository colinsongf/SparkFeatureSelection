package org.lidiagroup.hmourit.tfg.featureselection

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Object with some Information Theory methods.
 */
object InfoTheory {

  private val log2 = { x: Double => math.log(x) / math.log(2) }
  
    /**
   * Calculates mutual information (MI) and conditional mutual information (CMI) simultaneously
   * for several variables (X) with another one (Y) conditioned by variables Z.
   *
   * @param data RDD of data containing the variables
   * @param varX Indexes of variables
   * @param varY Index of the second variable
   * @param varZ Indexes of the conditioning values
   * @param n    Number of instances
   * @return     RDD of (variable, (MI, CMI))
   */
  def miAndCmi(
      data: RDD[Array[Byte]],
      varX: Seq[Int],
      varY: Seq[Int],
      varZ: Option[Int],
      n: Long) = {

    require(varX.size != 0)
    
    
    val none: Option[Byte] = None
    val xyComb = for(x <- varX; y <- varY) yield (x, y)

    val combinations = data.flatMap {
        case d =>
          xyComb.map{case (kx, ky) => ((kx,
              ky,
              d(kx),
              d(ky), 
              varZ match {case Some(z) => Some(d(z)) case None => None}), 
              1L)}
    }.reduceByKey(_ + _)
    // Split each combination keeping instance keys
    .flatMap {
        case ((kx, ky, x, y, z), q) =>
          
          Seq(((kx, ky, 1:Byte /*"xz"*/ , (x, z)),    (Set(y), q)),
              ((kx, ky, 2:Byte /*"yz"*/ , (y, z)),    (Set(x), q)),
              ((kx, ky, 3:Byte /*"xyz"*/, (x, y, z)), (Set.empty, q)),
              ((kx, ky, 4:Byte /*"z"*/  , z),         (Set((x, y)), q)),
              ((kx, ky, 5:Byte /*"xy"*/ , (x, y)),    (Set.empty,  q)),
              ((kx, ky, 6:Byte /*"x"*/  , x),         (Set(y), q)),
              ((kx, ky, 7:Byte /*"y"*/  , y),         (Set(x), q)))
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
        case ((kx, ky, ref, id), (keys, q))  => 
          ref match {
            case 1 => 
              val (x, z) = id.asInstanceOf[(Byte, Option[Byte])]
              for (y <- keys) yield ((kx, ky, x, y.asInstanceOf[Byte], z), (1:Byte, q))
            case 2 =>
              val (y, z) = id.asInstanceOf[(Byte, Option[Byte])]
              for (x <- keys) yield ((kx, ky, x.asInstanceOf[Byte], y, z), (2:Byte, q))
            case 3 =>
              val (x, y, z) = id.asInstanceOf[(Byte, Byte, Option[Byte])]
              Seq(((kx, ky, x, y, z), (3:Byte, q)))
            case 4 =>
              val z = id.asInstanceOf[Option[Byte]]
              for ((x, y) <- keys) yield 
                ((kx, ky, x.asInstanceOf[Byte], y.asInstanceOf[Byte], z), (4:Byte, q))
            case 5 =>
                val (x, y) = id.asInstanceOf[(Byte, Byte)]
                Seq(((kx, ky, x, y, none), (5:Byte, q)))
            case 6 =>
              val x = id.asInstanceOf[Byte]
              for (y <- keys) yield ((kx, ky, x, y.asInstanceOf[Byte], none), (6:Byte, q))
            case 7 =>
              val y = id.asInstanceOf[Byte]
              for (x <- keys) yield ((kx, ky, x.asInstanceOf[Byte], y, none), (7:Byte, q))
          }        
      })
      // Group by origin
      .combineByKey(createCombiner, mergeValues, mergeCombiners)
      
    grouped_frequencies.map({
      case ((kx, ky, _, _, Some(_)), (qxz, qyz, qxyz, qz, _, _, _)) =>
        val pz = qz.toDouble / n
        val pxyz = (qxyz.toDouble / n) / pz
        val pxz = (qxz.toDouble / n) / pz
        val pyz = (qyz.toDouble / n) / pz

        ((kx, ky), (0.0, pz * pxyz * log2(pxyz / (pxz * pyz))))

      case ((kx, ky, _, _, None), (_, _, _, _, qxy, qx, qy)) =>
        val pxy = qxy.toDouble / n
        val px = qx.toDouble / n
        val py = qy.toDouble / n

        ((kx, ky), (pxy * log2(pxy / (px * py)), 0.0))
    })
    // Compute results for each x
    .reduceByKeyLocally({ case ((mi1, cmi1), (mi2, cmi2)) => (mi1 + mi2, cmi1 + cmi2) })
  }

}
