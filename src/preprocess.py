"""Step 1: clean the raw Online Retail II log and build implicit-feedback interactions.

Input : data/online_retail_II.csv
Output: data/clean.parquet         (cleaned transaction lines)
        data/interactions.parquet  (one row per customer x product x invoice)
        results/cleaning_report.json
"""
import json
import os

from pyspark.sql import functions as F

from spark_session import get_spark

RAW = "data/online_retail_II.csv"
os.makedirs("results", exist_ok=True)

spark = get_spark("preprocess")

raw = (
    spark.read.option("header", True).option("inferSchema", False).csv(RAW)
    .withColumnRenamed("Customer ID", "CustomerID")
    .withColumn("Quantity", F.col("Quantity").cast("int"))
    .withColumn("Price", F.col("Price").cast("double"))
    .withColumn("InvoiceDate", F.to_timestamp("InvoiceDate"))
    .withColumn("CustomerID", F.col("CustomerID").cast("double").cast("int"))
)

report = {"raw_rows": raw.count()}

# 1. Exact duplicate lines
df = raw.dropDuplicates()
report["after_dedup"] = df.count()

# 2. Guest checkouts have no customer, so they cannot be personalised
df = df.filter(F.col("CustomerID").isNotNull())
report["after_drop_null_customer"] = df.count()

# 3. Cancellations (invoice starts with C), returns and zero/negative prices
df = df.filter(~F.col("Invoice").startswith("C"))
df = df.filter((F.col("Quantity") > 0) & (F.col("Price") > 0))
report["after_drop_cancel_neg"] = df.count()

# 4. Non-product codes (postage, manual adjustments, bank charges, fees, vouchers).
#    Real product codes start with a digit.
df = df.filter(F.col("StockCode").rlike("^[0-9]"))
report["clean_rows"] = df.count()

df = df.withColumn("Revenue", F.col("Quantity") * F.col("Price"))
df.write.mode("overwrite").parquet("data/clean.parquet")

# Implicit feedback: one event per customer x product x invoice.
# Purchase *frequency* is the signal (not quantity) so one bulk wholesale order
# does not dominate the confidence score.
inter = (
    df.groupBy("CustomerID", "StockCode", "Invoice")
    .agg(F.min("InvoiceDate").alias("ts"), F.sum("Quantity").alias("qty"))
)
inter.write.mode("overwrite").parquet("data/interactions.parquet")

report["clean_customers"] = df.select("CustomerID").distinct().count()
report["clean_products"] = df.select("StockCode").distinct().count()
report["clean_invoices"] = df.select("Invoice").distinct().count()
report["interaction_events"] = inter.count()
pairs = inter.select("CustomerID", "StockCode").distinct().count()
report["unique_customer_product_pairs"] = pairs
report["matrix_density_pct"] = round(100 * pairs / (report["clean_customers"] * report["clean_products"]), 3)

print(json.dumps(report, indent=2))
with open("results/cleaning_report.json", "w") as f:
    json.dump(report, f, indent=2)
spark.stop()
