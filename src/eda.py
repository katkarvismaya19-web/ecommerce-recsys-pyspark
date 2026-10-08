"""Step 2: exploratory analysis on the cleaned data (Spark aggregations, matplotlib plots).

Output: results/eda.json, results/figures/*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as F

from spark_session import get_spark

spark = get_spark("eda")
df = spark.read.parquet("data/clean.parquet")
inter = spark.read.parquet("data/interactions.parquet")
out = {}

# Monthly revenue and active customers
monthly = (
    df.withColumn("month", F.date_format("InvoiceDate", "yyyy-MM"))
    .groupBy("month")
    .agg(F.round(F.sum("Revenue"), 0).alias("revenue"), F.countDistinct("CustomerID").alias("customers"))
    .orderBy("month").toPandas()
)
monthly = monthly[monthly.month != "2011-12"]  # partial month (data ends 9 Dec)
out["monthly"] = monthly.to_dict(orient="records")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly.month, monthly.revenue / 1e3, marker="o")
ax.set_ylabel("Revenue (GBP thousands)")
ax.set_title("Monthly revenue, Dec 2009 - Nov 2011")
ax.tick_params(axis="x", rotation=60)
fig.tight_layout(); fig.savefig("results/figures/monthly_revenue.png", dpi=150); plt.close(fig)

# Long tail: how concentrated are purchases across products?
item_pop = inter.groupBy("StockCode").agg(F.countDistinct("CustomerID").alias("buyers")).toPandas()
item_pop = item_pop.sort_values("buyers", ascending=False).reset_index(drop=True)
cum = item_pop.buyers.cumsum() / item_pop.buyers.sum()
out["products_for_50pct_of_purchases"] = int((cum < 0.5).sum() + 1)
out["products_for_80pct_of_purchases"] = int((cum < 0.8).sum() + 1)
out["total_products"] = int(len(item_pop))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(item_pop) + 1) / len(item_pop) * 100, cum * 100)
ax.set_xlabel("% of products (most popular first)"); ax.set_ylabel("% of customer-product purchases")
ax.set_title("Product popularity long tail"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/figures/long_tail.png", dpi=150); plt.close(fig)

# Customer activity: distinct products bought per customer
cust = inter.groupBy("CustomerID").agg(
    F.countDistinct("StockCode").alias("products"), F.countDistinct("Invoice").alias("orders")
).toPandas()
out["customer_products_median"] = float(cust.products.median())
out["customer_orders_median"] = float(cust.orders.median())
out["customers_single_order_pct"] = round(100 * float((cust.orders == 1).mean()), 1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log10(cust.products), bins=40)
ax.set_xlabel("log10(distinct products bought)"); ax.set_ylabel("Customers")
ax.set_title("Products per customer")
fig.tight_layout(); fig.savefig("results/figures/products_per_customer.png", dpi=150); plt.close(fig)

# Top products
top = (
    df.groupBy("StockCode").agg(F.first("Description").alias("Description"), F.countDistinct("CustomerID").alias("buyers"))
    .orderBy(F.desc("buyers")).limit(10).toPandas()
)
out["top_products"] = top.to_dict(orient="records")

uk = df.groupBy((F.col("Country") == "United Kingdom").alias("uk")).agg(F.sum("Revenue").alias("rev")).toPandas()
out["uk_revenue_pct"] = round(100 * float(uk[uk.uk].rev.iloc[0] / uk.rev.sum()), 1)

print(json.dumps({k: v for k, v in out.items() if k != "monthly"}, indent=2, default=str))
with open("results/eda.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
spark.stop()
