package org.apache.spark.mllib.featureselection

/**
 * Factory class that generates a wide range of info-theory criterions [1] to 
 * perform a feature selection phase on data.
 * 
 * [1] Brown, G., Pocock, A., Zhao, M. J., & Luján, M. (2012). 
 * "Conditional likelihood maximization: a unifying framework for information theoretic feature selection." 
 * The Journal of Machine Learning Research, 13(1), 27-66.
 * 
 * @param criterion String that specifies the criterion to be used (options: JMI, mRMR, ICAP, CMIM and IF).
 * @return An initialized info-theory criterion.
 */

class InfoThCriterionFactory(val criterion: String) extends Serializable {
	
	val JMI  = "jmi"
  val MRMR = "mrmr"
  val ICAP = "icap"
  val CMIM = "cmim"
  val IF   = "if"

  /** Generates a specific info-theory criterion
   *  
   */
	def getCriterion: InfoThCriterion = {
		criterion match {
      case JMI  => new Jmi
      case MRMR => new Mrmr
      case ICAP => new Icap
      case CMIM => new Cmim
      case IF   => new If
      case _    => throw new IllegalArgumentException("criterion unknown")
    }
	}

}