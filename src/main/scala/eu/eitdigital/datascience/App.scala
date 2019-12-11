package eu.eitdigital.datascience

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

object App {
  
  def main(args : Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)

    val spark = SparkSession
      .builder()
      .appName("Big Data SparkSQL Session")
      .getOrCreate()

    import spark.implicits._

    val _dataFrame = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(args(0))
      .drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
        "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")

     val dataFrame = _dataFrame.withColumn("DepTime", _dataFrame.col("DepTime").cast(DataTypes.IntegerType))
        .withColumn("CRSElapsedTime", _dataFrame.col("CRSElapsedTime").cast(DataTypes.IntegerType))
        .withColumn("ArrDelay", _dataFrame.col("ArrDelay").cast(DataTypes.IntegerType))
        .withColumn("DepDelay", _dataFrame.col("DepDelay").cast(DataTypes.IntegerType))
        .withColumn("TaxiOut", _dataFrame.col("TaxiOut").cast(DataTypes.IntegerType))

    println(s"Lines in the document: ${dataFrame.count()}")

  }

}
