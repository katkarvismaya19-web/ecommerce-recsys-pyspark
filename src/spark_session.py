"""Shared SparkSession factory. Runs locally with all cores; point master at a cluster to scale out."""
from pyspark.sql import SparkSession


def get_spark(app_name: str = "ecommerce-recsys", master: str = "local[*]") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")
    return spark
