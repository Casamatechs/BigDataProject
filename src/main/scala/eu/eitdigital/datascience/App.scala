package eu.eitdigital.datascience

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object App {

  def main(args: Array[String]) {

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
      .load("/home/carlos/Documents/Big Data/BigDataProject/src/main/resources/2008.csv")
      .drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
        "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
      .withColumn("DepTime", $"DepTime".cast(DataTypes.IntegerType))
      .withColumn("CRSElapsedTime", $"CRSElapsedTime".cast(DataTypes.IntegerType))
      .withColumn("ArrDelay", $"ArrDelay".cast(DataTypes.IntegerType))
      .withColumn("DepDelay", $"DepDelay".cast(DataTypes.IntegerType))
      .withColumn("TaxiOut", $"TaxiOut".cast(DataTypes.IntegerType))
      .withColumn("FlightID", concat($"UniqueCarrier", lit(""), $"FlightNum"))
      .filter($"ArrDelay".isNotNull)

    val indexer = new StringIndexer()
      .setInputCol("FlightID")
      .setOutputCol("indexFlightID")
        .fit(dataFrame)
        .transform(dataFrame)

    val scaler = new MinMaxScaler()
        .setInputCol("indexFlightID")
        .setOutputCol("scaledFlightID")
        .fit(indexer)
        .transform(indexer)


    println(s"Lines in the document: ${dataFrame.count()}")

  }

}
