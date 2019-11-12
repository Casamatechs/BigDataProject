package eu.eitdigital.datascience

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object App {
  
  def main(args : Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)

    val spark = SparkSession
      .builder()
      .appName("Big Data SparkSQL Session")
      .getOrCreate()

    import spark.implicits._

    val dataFrame = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("file:///home/carlos/Documents/Big Data/BigDataProject/src/main/resources/2008.csv")
      .drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")

    println(s"Lines in the document: ${dataFrame.count()}")

  }

}
