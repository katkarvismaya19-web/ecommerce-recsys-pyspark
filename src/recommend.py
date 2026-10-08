"""Step 4: serve top-N recommendations for a customer from the saved ALS model.

Usage:  python src/recommend.py 12347            # top 10, previously bought items excluded
        python src/recommend.py 12347 --k 5 --include-seen
"""
import argparse

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F

from spark_session import get_spark

ap = argparse.ArgumentParser()
ap.add_argument("customer_id", type=int)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--include-seen", action="store_true")
args = ap.parse_args()

spark = get_spark("recommend")
model = ALSModel.load("models/als_final")
items = spark.read.parquet("models/item_index.parquet")
clean = spark.read.parquet("data/clean.parquet")
names = clean.groupBy("StockCode").agg(F.first("Description").alias("Description"))

seen = clean.filter(F.col("CustomerID") == args.customer_id).select("StockCode").distinct()
users = spark.createDataFrame([(args.customer_id,)], ["user"])
recs = model.recommendForUserSubset(users, args.k + (0 if args.include_seen else 500))
if recs.count() == 0:
    print(f"Customer {args.customer_id} was not in the training data (cold start). Showing best-sellers instead.")
    out = (clean.groupBy("StockCode").agg(F.countDistinct("CustomerID").alias("score"))
           .join(names, "StockCode").orderBy(F.desc("score")).limit(args.k))
else:
    out = (recs.select(F.explode("recommendations").alias("r"))
           .select(F.col("r.item").alias("item"), F.col("r.rating").alias("score"))
           .join(items, "item").join(names, "StockCode"))
    if not args.include_seen:
        out = out.join(seen, "StockCode", "left_anti")
    out = out.orderBy(F.desc("score")).limit(args.k)

print(f"\nTop {args.k} recommendations for customer {args.customer_id}:")
for row in out.select("StockCode", "Description", "score").collect():
    print(f"  {row.StockCode:<8} {row.Description.strip():<40} score={row.score:.3f}")

print("\nWhat this customer bought most often (for context):")
hist = (clean.filter(F.col("CustomerID") == args.customer_id).groupBy("StockCode", "Description")
        .agg(F.countDistinct("Invoice").alias("orders")).orderBy(F.desc("orders")).limit(5))
for row in hist.collect():
    print(f"  {row.StockCode:<8} {row.Description.strip():<40} orders={row.orders}")
spark.stop()
