package eu.eitdigital.datascience

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
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

    // We do a selection of columns to transform

    val columnsToIndex = Array("Origin", "Dest", "FlightID")

    val inds = columnsToIndex.map(
      colName => new StringIndexer()
        .setInputCol(colName)
        .setOutputCol("index_".concat(colName))
        .fit(dataFrame)
    )

    // We transform the categorical variables into doubles
    val preparedDf = new Pipeline().setStages(inds).fit(dataFrame).transform(dataFrame)

    val assembler = new VectorAssembler()
      .setInputCols(Array("DepDelay", "DayOfWeek", "index_Origin", "index_Dest", "index_FlightID"))
      .setOutputCol("features")

    val linearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setRegParam(0.1)
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    // We prepare the data to train and test the ML model
    val split = preparedDf.randomSplit(Array(0.8, 0.2))
    val training_set = split(0)
    val test_set = split(1)

    val pipeline = new Pipeline().setStages(Array(assembler, linearRegression))

    val lrModel = pipeline.fit(training_set)
    val predictions = lrModel.transform(test_set)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
  }

}
