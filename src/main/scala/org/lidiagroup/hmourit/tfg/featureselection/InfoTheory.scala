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
	private var basicCounts: RDD[((Int, Byte, Byte), (Long, Long, Long))] = null
	
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
        varZ: Broadcast[Option[Int]],
        byPairs: Boolean) =  {
      
       val sv = v.asInstanceOf[BSV[Byte]]
       val (withZ, zind) = varZ.value match {
         case Some(v) => (true, v)
         case None => (false, -1)
       }
       
       var zVal: Option[Byte] = None
       var xValues = Seq.empty[(Int, Byte)]
       var yValues = Seq.empty[(Int, Byte)]
       
       // Generate pairs using X and Y values generated before       
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
       
       var pairs = Seq.empty[(Any, (Set[_], Long))]    
       if(byPairs) {
	  		for((xind, x) <- xValues){
		         for((yind, y) <- yValues) {
		            pairs = ((xind, x, y), (Set.empty, 1L)) +: pairs
		         }
	  		}
	  		
	  		for((xind, x) <- xValues){
	  			pairs = ((xind, x), (yValues.toSet, 1L)) +: pairs		  			
	  		} 
	  		
	  		for((_, y) <- yValues){
	  			pairs = (y, (xValues.toSet, 1L)) +: pairs		  			
	  		}
         
       } else {
         zVal match {
         	case Some(z) => 
         	  for((xind, x) <- xValues){
		         for((yind, y) <- yValues) {
		        	 /* In CMI Y could be an input attribute */
		        	 pairs = ((xind, yind, x, y, z), (Set.empty, 1L)) +: pairs
		         }
         	  }
         	case None => /* Do nothing */    
         }	  		
       }
       
       pairs      
    }
  
    /* Pair generator for dense data */
    private def DenseGenerator(v: BV[Byte], 
        varX: Broadcast[Seq[Int]],
        varY: Broadcast[Seq[Int]],
        varZ: Broadcast[Option[Int]],
        byPairs: Boolean) = {
      
       val dv = v.asInstanceOf[BDV[Byte]]
       val zval = varZ.value match {case Some(z) => Some(v(z)) case None => None}
       
       var pairs = Seq.empty[(Any, (Set[_], Long))]
       val multY = varY.value.length > 1
       if (byPairs) {
    	   for(xind <- varX.value){
		         for(yind <- varY.value) {
		        	 val seq = Seq((dv(yind), (Set((xind, dv(xind))), 1L)),
			        				((xind, dv(xind)), (Set(dv(yind)), 1L)),
			        				((xind, dv(xind), dv(yind)), (Set.empty, 1L))) 
    				pairs =  seq ++: pairs
		         }
	    	   }  
       } else {
         zval match {
         	case Some(z) => 
	    	   for(xind <- varX.value){
		         for(yind <- varY.value) {
		        	 /*val indexes = if(multY) 
		        	   (xind, yind) 
		        	 else 
		        	   xind*/
		             pairs = ((xind, yind, dv(xind), dv(yind), zval), (Set.empty, 1L)) +: pairs
		         }
	    	   }  
         	case None => /* Do nothing */ 
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
    val generator = data.first match {
		case v: BSV[Byte] => 
	        val bfX = if(inverseX) 
	          sc.broadcast(!SU.binarySearch(bvarX.value, _: Int)) 
	        else 
	          sc.broadcast(SU.binarySearch(bvarX.value, _: Int))
	        SparseGenerator(_: BV[Byte], bfX, bvarY, bvarZ, _: Boolean)
		case v: BDV[Byte] => DenseGenerator(_: BV[Byte], bvarX, bvarY, bvarZ, _: Boolean)		  
	}
    
   val result = varZ match {
      	case None =>
      	  	val (result, bc) = calculateInitialMI(data, generator(_: BV[Byte], true), varY(0), n)
      	  	basicCounts = bc.cache
      	  	result
      	case Some(_) => 
      	  	calculateMIAndCMI(data, generator(_: BV[Byte], false), varY(0), n)
      	  	
    }
	    
    result
  }
  
  /*private def calculateMISparseData(data: RDD[BV[Byte]],
      pairsGenerator: BV[Byte] => Seq[(Any, Long)],
      firstY: Int,
      n: Long) = {
    
		val combinations = data.flatMap(pairsGenerator).reduceByKey(_ + _)
			 // Split each combination keeping instance keys
			 .flatMap {
			      case ((xind: Int, yind: Int, x: Byte, y: Byte, Some(z: Byte)), q) => 
			        println("no me toques los huevos")
			        Seq((((3:Byte /*"xyz"*/, (xind, x, yind, y, z)), (Seq.empty, q)),
			            ((5:Byte /*"xy"*/ , (xind, x, yind, y)),    (Seq.empty,  q))))
			            
			      case ((xind: Int, yind: Int, x: Byte, y: Byte, None), q) =>
			        Seq(((5:Byte /*"xy"*/ , (xind, x, yind, y)),    (Seq.empty,  q)))
			            
			      case ((ref: Byte, ind: Int, value: Byte, None), q) => 
			        ref match {
		        		case 1 /*"x"*/ => 
		        		  Seq(((6:Byte /*"x"*/, (ind, value)), (Seq.empty, q)))
		        		case 2 /*"y"*/ => 
		        		  Seq(((7:Byte /*"y"*/, (ind, value)), (Seq.empty, q)))
		        	}
			        
		        case ((ref: Byte, ind: Int, value: Byte, Some(z: Byte)), q) => 
			        ref match {
		        		case 1 /*"x"*/ => 
		        		 Seq(((1:Byte /*"xz"*/ , (ind, value, z)), (Seq.empty, q)),
					        ((4:Byte /*"z"*/  , z),         (Seq.empty, q)),
					        ((6:Byte /*"x"*/  , (ind, value)),  (Seq.empty, q))) 
		        		case 2 /*"y"*/ => 
		        		  Seq(((2:Byte /*"yz"*/ , (ind, value, z)),    (Seq.empty, q)),
					        ((4:Byte /*"z"*/  , z),         (Seq.empty, q)),
					        ((7:Byte /*"y"*/  , (ind, value)),         (Seq.empty, q))) 
		        	}
		        case (Some(z: Byte), q) => 
			        Seq(((4:Byte /*"z"*/, z), (Seq.empty, q)))
		    }
    	
	      	// Separate by origin of combinations
			val comb = combinations.reduceByKey({
	        	case ((keys1, q1), (keys2, q2)) => (keys1 ++ keys2, q1 + q2)
	      	})
	      	
	      	println("Top combinations: " + combinations.take(100).mkString("\n"))
	    	val grouped_frequencies = comb
	      	.flatMap({
		        case ((ref, k), (keys, q))  =>
		          println("Caseado")
		          println("ref igual a: " + ref)
		          val none: Option[Byte] = None
		          ref match {
		            case 1 /*"xz"*/ => 
		              val (xind, x, z) = k.asInstanceOf[(Int, Byte, Byte)]
		              for ((yind, y) <- keys.distinct) yield (((xind, yind.asInstanceOf[Int]), x, 
		            		  y.asInstanceOf[Byte], Some(z)), (1:Byte, q))
		            case 2 /*"yz"*/ =>
		              val (yind, y, z) = k.asInstanceOf[(Int, Byte, Byte)]
		              for ((xind, x) <- keys.distinct) yield (((xind.asInstanceOf[Int], yind), 
		            		  x.asInstanceOf[Byte], y, Some(z)), (2:Byte, q))
		            case 3 /*"xyz"*/ =>
		              println("Hola")
		              val (xind, x, yind, y, z) = k.asInstanceOf[(Int, Byte, Int, Byte, Byte)]
		              Seq((((xind, yind), x, y, Some(z)), (3:Byte, q)))
		            case 4 /*"z"*/ =>
		              val z = k.asInstanceOf[Byte]
		              for ((xind, x, yind, y) <- keys.distinct) yield (((xind.asInstanceOf[Int], yind.asInstanceOf[Int]), 
		                		x.asInstanceOf[Byte], y.asInstanceOf[Byte], Some(z)), (4:Byte, q))
		            case 5 /*"xy"*/ =>
		              	println("Hola xy")
		                val (xind, x, yind, y) = k.asInstanceOf[(Int, Byte, Int, Byte)]
		                Seq((((xind, yind), x, y, none), (5:Byte, q)))
		            case 6 /*"x"*/ =>
		              val (xind, x) = k.asInstanceOf[(Int, Byte)]
		              for ((yind, y) <- keys.distinct) yield (((xind, yind.asInstanceOf[Int]), 
		                  x, y.asInstanceOf[Byte], none), (6:Byte, q))
		            case 7 /*"y"*/ =>
		              val (yind, y) = k.asInstanceOf[(Int, Byte)]
		              for ((xind, x) <- keys.distinct) yield (((xind.asInstanceOf[Int], yind), 
		                  x.asInstanceOf[Byte], y, none), (7:Byte, q))
		        }        
	      })
	      // Group by origin
	      .combineByKey(createCombiner, mergeValues, mergeCombiners)
	      
	      	println("Grouped size: " + grouped_frequencies.count)
	      	println("Top grouped: " + grouped_frequencies.take(100).mkString("\n"))
	      
	    grouped_frequencies.map({
	      case ((k, _, _, z), (qxz, qyz, qxyz, qz, qxy, qx, qy)) =>
	        // Select id        
	        // Choose between MI or CMI
	        z match {
	          case Some(_) =>
	              val pz = qz.toDouble / n
	              val pxyz = (qxyz.toDouble / n) / pz
	              val pxz = (qxz.toDouble / n) / pz
	              val pyz = (qyz.toDouble / n) / pz
	              (k, (0.0, pz * pxyz * log2(pxyz / (pxz * pyz))))
	          case None => 
	              val pxy = qxy.toDouble / n
	              val px = qx.toDouble / n
	              val py = qy.toDouble / n
	              (k, (pxy * log2(pxy / (px * py)), 0.0))
	        }
	    })
	    // Compute results for each X
	    .reduceByKey({ case ((mi1, cmi1), (mi2, cmi2)) => (mi1 + mi2, cmi1 + cmi2) })
  }*/
  
  private def calculateInitialMI(data: RDD[BV[Byte]],
      pairsGenerator: BV[Byte] => Seq[(Any, (Set[_], Long))],
      firstY: Int,
      n: Long) = {
    
    		  //println("Top flat: " + data.flatMap(pairsGenerator).collect.mkString("\n"))
	    val combinations = data.flatMap(pairsGenerator).reduceByKey({
	        case ((keys1, q1), (keys2, q2)) => (keys1 ++ keys2, q1 + q2)
	      })
	      
	      
      	println("Top comb: " + combinations.collect.mkString("\n"))
      	
           val createCombiner: ((Byte, Long)) => (Long, Long, Long) = {
		      case (1, q) => (q, 0, 0)
		      case (2, q) => (0, q, 0)
		      case (3, q) => (0, 0, q)
		    }
		
		    val mergeValues: ((Long, Long, Long), (Byte, Long)) => 
		        (Long, Long, Long) = {
		      case ((qxy, qx, qy), (ref, q)) =>
		        ref match {
		          case 1 => (qxy + q, qx, qy)
		          case 2 => (qxy, qx + q, qy)
		          case 3 => (qxy, qx, qy + q)
		        }
		    }
		
		    val mergeCombiners: ((Long, Long, Long), (Long, Long, Long)) => 
		      (Long, Long, Long) = {
		      case ((qxy1, qx1, qy1), (qxy2, qx2, qy2)) =>
		        (qxy1 + qxy2, qx1 + qx2, qy1 + qy2)
		    }
  
    		  
		val grouped_frequencies = combinations
			 // Split each combination keeping instance keys
			 .flatMap {
			      case ((xind: Int, x: Byte, y: Byte), (_, q)) =>  
			        	Seq(((xind, x, y), (1:Byte, q)))
			      case ((xind: Int, x: Byte), (keys, q)) =>  
			        for ((_, y) <- keys) yield ((xind, x, y.asInstanceOf[Byte]), (2:Byte, q))
			      case (y: Byte, (keys, q)) =>
			        for ((xind, x) <- keys) yield ((xind.asInstanceOf[Int], 
			            x.asInstanceOf[Byte], y), (3:Byte, q))
		    }
	      	// Group by origin
	        .combineByKey(createCombiner, mergeValues, mergeCombiners)
	      
	    val results = grouped_frequencies.map({
	      	case ((k, _, _), (qxy, qx, qy)) =>       
	              val pxy = qxy.toDouble / n
	              val px = qx.toDouble / n
	              val py = qy.toDouble / n
	              ((k, firstY), (pxy * log2(pxy / (px * py)), 0.0))
	    }).reduceByKey({ case ((mi1, cmi1), (mi2, cmi2)) => (mi1 + mi2, cmi1 + cmi2) })
	    
	    println("Grouped size: " + grouped_frequencies.count)
      	println("Top grouped: " + grouped_frequencies.take(100).mkString("\n"))
	    
	    (results, grouped_frequencies)
  }
  
  private def calculateMIAndCMI(data: RDD[BV[Byte]],
      pairsGenerator: BV[Byte] => Seq[(Any, (Set[_], Long))],
      firstY: Int,
      n: Long) = {
    
		val xyzcomb = data.flatMap(pairsGenerator)
			 .reduceByKey({
			 	case ((_, q1), (_, q2)) => (Set.empty, q1 + q2)
			 })
			 .mapValues{case (_, q) => (3:Byte, q)}.asInstanceOf[RDD[((Int, Int, Byte, Byte, Byte), (Byte, Long))]]
			 
		val basicPairs = xyzcomb.flatMap{
		  case ((xind: Int, yind: Int, x: Byte, y: Byte, z: Byte), _) => 
			  Seq(((xind, x, z), (1:Byte, yind, y)),
			  		((yind, y, z), (2: Byte, xind, x)))		  																
		}
		
		val basic_frequencies = basicPairs
					.join(basicCounts)
					.map({ case ((ind, value, z), ((ref, ind2, value2), (q12, q1, q2))) =>
					  ref match {
					    /*(qxyz, qxy, qxz, qyz, qx, qy, qz)*/
					  	case 1 => ((ind, ind2, value, value2, z), (q12, 0L, 0L, q1, 0L, q2, 0L))
					  	case 2 => ((ind2, ind, value2, value, z), (0L, q12, 0L, 0L, 0L, 0L, q2))
					  }
					})
					
		val xycomb = xyzcomb.map {
			      case ((xind: Int, yind: Int, x: Byte, y: Byte, z: Byte), (_, q)) => 			        
			            ((xind, yind, x, y), (Set(z), q))
	    		}.reduceByKey({
	    			case ((keys1, q1), (keys2, q2)) => (keys1 ++ keys2, q1 + q2)
	    		}).flatMap({
	    			case ((xind, yind, x, y), (keys, q)) => 		
	    				for (z <- keys) yield ((xind, yind, x, y, z), (5:Byte, q))
	    		})

	    		
	    val allfreqs = xyzcomb.union(xycomb)
	    				.combineByKey(createCombiner, mergeValues, mergeCombiners)
	    				.union(basic_frequencies)
	    val grouped_frequencies = allfreqs.reduceByKey(mergeCombiners)
	    
	    grouped_frequencies.map({
	      case ((kx, ky, _, _, z), (qxz, qyz, qxyz, qz, qxy, qx, qy)) =>      
	        	  // CMI
	              val pz = qz.toDouble / n
	              val pxyz = (qxyz.toDouble / n) / pz
	              val pxz = (qxz.toDouble / n) / pz
	              val pyz = (qyz.toDouble / n) / pz
	              // MI
	              val pxy = qxy.toDouble / n
	              val px = qx.toDouble / n
	              val py = qy.toDouble / n
	              ((kx, ky), (pxy * log2(pxy / (px * py)), 
	                  pz * pxyz * log2(pxyz / (pxz * pyz))))
	    })
	    // Compute results for each X
	    .reduceByKey({ case ((mi1, cmi1), (mi2, cmi2)) => (mi1 + mi2, cmi1 + cmi2) })
  }
  
  private def calculateMIDenseData(data: RDD[BV[Byte]],
      pairsGenerator: BV[Byte] => Seq[((Any, Byte, Any, Option[Byte]), Long)],
      firstY: Int,
      n: Long) = {
    		  
		  val pairs = data.flatMap(pairsGenerator)

    	println("Count pairs: " + pairs.count())
    	println("Count pairs2: " + pairs.reduceByKey(_ + _).count())
		val combinations = data.flatMap(pairsGenerator)
			 .reduceByKey(_ + _)
			 // Split each combination keeping instance keys
			 .flatMap {
			      case ((k, x, y, Some(z)), q) =>          
			        Seq(((k, 1:Byte /*"xz"*/ , (x, z)),    (Seq(y), q)),
			            ((k, 2:Byte /*"yz"*/ , (y, z)),    (Seq(x), q)),
			            ((k, 3:Byte /*"xyz"*/, (x, y, z)), (Seq.empty, q)),
			            ((k, 4:Byte /*"z"*/  , z),         (Seq((x, y)), q)),
			            ((k, 5:Byte /*"xy"*/ , (x, y)),    (Seq.empty,  q)),
			            ((k, 6:Byte /*"x"*/  , x),         (Seq(y), q)),
			            ((k, 7:Byte /*"y"*/  , y),         (Seq(x), q)))
			      case ((k, x, y, None), q) =>
			        Seq(((k, 5:Byte /*"xy"*/ , (x, y)),    (Seq.empty,  q)),
			            ((k, 6:Byte /*"x"*/  , x),         (Seq(y), q)),
			            ((k, 7:Byte /*"y"*/  , y),         (Seq(x), q)))
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
	              for (y <- keys.distinct) yield ((k, x, y.asInstanceOf[Byte], Some(z)), (1:Byte, q))
	            case 2 =>
	              val (y, z) = value.asInstanceOf[(Byte, Byte)]
	              for (x <- keys.distinct) yield ((k, x.asInstanceOf[Byte], y, Some(z)), (2:Byte, q))
	            case 3 =>
	              val (x, y, z) = value.asInstanceOf[(Byte, Byte, Byte)]
	              Seq(((k, x, y, Some(z)), (3:Byte, q)))
	            case 4 =>
	              val z = value.asInstanceOf[Byte]
	              for ((x, y) <- keys.distinct) yield 
	                ((k, x.asInstanceOf[Byte], y.asInstanceOf[Byte], Some(z)), (4:Byte, q))
	            case 5 =>
	                val (x, y) = value.asInstanceOf[(Byte, Byte)]
	                Seq(((k, x, y, none), (5:Byte, q)))
	            case 6 =>
	              val x = value.asInstanceOf[Byte]
	              for (y <- keys.distinct) yield ((k, x, y.asInstanceOf[Byte], none), (6:Byte, q))
	            case 7 =>
	              val y = value.asInstanceOf[Byte]
	              for (x <- keys.distinct) yield ((k, x.asInstanceOf[Byte], y, none), (7:Byte, q))
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
