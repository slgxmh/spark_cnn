package com.slgxmh.cnn.nn

import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("GitHub push counter")
      .master("local")
      .getOrCreate();
    val sc = spark.sparkContext
  }
}
