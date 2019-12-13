package eu.eitdigital.datascience

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
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

    // Data loading and pre-processing
    val dataFrame = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(args(0))
      // We drop the forbidden variables
      .drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
        "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
      // We cast the variables to their right types
      .withColumn("DepTime", $"DepTime".cast(DataTypes.IntegerType))
      .withColumn("CRSElapsedTime", $"CRSElapsedTime".cast(DataTypes.IntegerType))
      .withColumn("ArrDelay", $"ArrDelay".cast(DataTypes.IntegerType))
      .withColumn("DepDelay", $"DepDelay".cast(DataTypes.IntegerType))
      .withColumn("TaxiOut", $"TaxiOut".cast(DataTypes.IntegerType))
      .withColumn("FlightID", concat($"UniqueCarrier", lit(""), $"FlightNum"))
      // We remove the rows missing the response variable
      .filter($"ArrDelay".isNotNull)

    val indexer = new StringIndexer()
      .setInputCol("FlightID")
      .setOutputCol("indexFlightID")
        .fit(dataFrame)
        .transform(dataFrame)

    val assembler = new VectorAssembler()
        .setInputCols(Array("DepDelay", "DayOfWeek", "indexFlightID"))
        .setOutputCol("features")

    val prepared_df = assembler.transform(indexer)

    val linearRegression = new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("ArrDelay")
        .setMaxIter(10)
        .setElasticNetParam(0.8)

    val lrModel = linearRegression.fit(prepared_df)

    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary

    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")




    println(s"Lines in the document: ${dataFrame.count()}")

  }

}
