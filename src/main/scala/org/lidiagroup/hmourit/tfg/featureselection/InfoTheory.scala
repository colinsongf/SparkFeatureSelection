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
      varY: Int,
      varZ: Option[Int],
      n: Long) = {

    require(varX.size != 0)

	val combinations = data.flatMap {
        case d =>
          varX.map(k => ((k, 
              d(k),
              d(varY), 
              varZ match {case Some(z) => Some(d(z)) case None => None}), 
              1L))
	}.reduceByKey(_ + _)
	// Split each combination keeping instance keys
    .flatMap {
        case ((k, x, y, z), q) =>
          val key_cmi = (k, x, y, Some(z))
          val key_mi = (k, x, y, None)

          Seq(((k, 1:Byte /*"xz"*/ , (x, z)),    (Seq(key_cmi), q)),
              ((k, 2:Byte /*"yz"*/ , (y, z)),    (Seq(key_cmi), q)),
              ((k, 3:Byte /*"xyz"*/, (x, y, z)), (Seq(key_cmi), q)),
              ((k, 4:Byte /*"z"*/  , z),         (Seq(key_cmi), q)),
              ((k, 5:Byte /*"xy"*/ , (x, y)),    (Seq(key_mi),  q)),
              ((k, 6:Byte /*"x"*/  , x),         (Seq(key_mi),  q)),
              ((k, 7:Byte /*"y"*/  , y),         (Seq(key_mi),  q)))
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
        case ((_, ref, _), (keys, q)) => for (key <- keys.distinct) yield (key, (ref, q))
      })
      // Group by origin
      .combineByKey[(Long, Long, Long, Long, Long, Long, Long)](createCombiner, mergeValues, mergeCombiners)

    grouped_frequencies.map({
      case ((k, _, _, Some(_)), (qxz, qyz, qxyz, qz, _, _, _)) =>
        val pz = qz.toDouble / n
        val pxyz = (qxyz.toDouble / n) / pz
        val pxz = (qxz.toDouble / n) / pz
        val pyz = (qyz.toDouble / n) / pz

        (k, (0.0, pz * pxyz * log2(pxyz / (pxz * pyz))))

      case ((k, _, _, None), (qxz, qyz, qxyz, qz, qxy, qx, qy)) =>
        val pxy = qxy.toDouble / n
        val px = qx.toDouble / n
        val py = qy.toDouble / n

        (k, (pxy * log2(pxy / (px * py)), 0.0))
    })
    // Compute results for each x
    .reduceByKeyLocally({ case ((mi1, cmi1), (mi2, cmi2)) => (mi1 + mi2, cmi1 + cmi2) })
  }

}
