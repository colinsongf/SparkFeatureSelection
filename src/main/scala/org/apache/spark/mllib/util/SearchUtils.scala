package org.apache.spark.mllib.util

object SearchUtils {
  
	def binarySearch(array: Seq[Int], value: Int) = {
     
	    def binarySearchRecursive(start: Int, end: Int): Boolean = {
	      var i = (start + end) / 2
	      if(start > end) {
	        return false
	      }
	       
	      array(i) match {
	        case x if (x == value) => true
	        case x if (x > value) => binarySearchRecursive(start, i - 1)
	        case x if (x < value) => binarySearchRecursive(i + 1, end)
	      }
	    }
	    binarySearchRecursive(0, array.size - 1)
	}
	
	def binarySearch2(array: Seq[Int], value: Int) = {
     
	    def binarySearchRecursive(start: Int, end: Int): Int = {
	      var i = (start + end) / 2
	      if(start > end) {
	        return -1
	      }
	       
	      array(i) match {
	        case x if (x == value) => i
	        case x if (x > value) => binarySearchRecursive(start, i - 1)
	        case x if (x < value) => binarySearchRecursive(i + 1, end)
	      }
	    }
	    binarySearchRecursive(0, array.size - 1)
	}
  

}