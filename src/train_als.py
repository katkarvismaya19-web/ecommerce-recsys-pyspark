"""Step 3: train implicit-feedback ALS with a time-based split and compare against baselines.

Split (no future leakage):
  tuning : train < 2011-06-01, validate on 2011-06-01 .. 2011-08-31
  final  : train < 2011-09-01, test     on 2011-09-01 .. 2011-12-09

Two evaluation tasks for every customer active in both windows:
  "all"       - predict every product the customer buys in the test window (repeats included)
  "discovery" - predict only products the customer has never bought before (previously-bought items are excluded)

Output: results/metrics.json, results/tuning.csv, results/figures/model_comparison.png, models/als_final
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

from metrics import coverage, precision_recall_ndcg
from spark_session import get_spark

K = 10
SEED = 42
spark = get_spark("train-als")
events = spark.read.parquet("data/interactions.parquet")

# Integer ids required by Spark ALS
items = events.select("StockCode").distinct().orderBy("StockCode").rdd.zipWithIndex() \
    .map(lambda x: (x[0][0], int(x[1]))).toDF(["StockCode", "item"])
events = events.join(items, "StockCode").withColumnRenamed("CustomerID", "user").cache()
N_ITEMS = items.count()


def split(cut, end):
    train = events.filter(F.col("ts") < cut)
    test = events.filter((F.col("ts") >= cut) & (F.col("ts") < end))
    # confidence = number of separate orders containing the product
    train_r = train.groupBy("user", "item").agg(F.countDistinct("Invoice").cast("float").alias("rating")).cache()
    return train_r, test


def to_sets(df):
    pdf = df.select("user", "item").distinct().toPandas()
    return pdf.groupby("user")["item"].apply(set).to_dict()


def build_tasks(train_r, test):
    hist = to_sets(train_r)
    test_sets = to_sets(test)
    users = [u for u in test_sets if u in hist]
    all_truth = {u: test_sets[u] for u in users}
    disc_truth = {u: test_sets[u] - hist[u] for u in users}
    disc_truth = {u: s for u, s in disc_truth.items() if s}
    return hist, all_truth, disc_truth


def popularity_recs(train_r, users, hist, exclude_seen):
    ranked = train_r.groupBy("item").count().orderBy(F.desc("count")).limit(2000).toPandas()["item"].tolist()
    out = {}
    for u in users:
        seen = hist.get(u, set()) if exclude_seen else set()
        out[u] = [i for i in ranked if i not in seen][:K]
    return out


def history_recs(train_r, users):
    """Repeat-purchase baseline: the customer's own most frequently re-ordered products."""
    pdf = train_r.filter(F.col("user").isin(list(users))).toPandas()
    pdf = pdf.sort_values(["user", "rating"], ascending=[True, False])
    return pdf.groupby("user")["item"].apply(lambda s: s.tolist()[:K]).to_dict()


def als_recs(model, users, hist, exclude_seen):
    uf = model.userFactors.toPandas().set_index("id")
    itf = model.itemFactors.toPandas().set_index("id")
    U = np.vstack(uf.loc[[u for u in users if u in uf.index], "features"].values)
    uid = [u for u in users if u in uf.index]
    V = np.vstack(itf["features"].values)
    iid = itf.index.to_numpy()
    scores = U @ V.T
    out = {}
    for row, u in enumerate(uid):
        s = scores[row].copy()
        if exclude_seen:
            seen_mask = np.isin(iid, list(hist.get(u, ())))
            s[seen_mask] = -np.inf
        top = np.argpartition(-s, K)[:K]
        out[u] = iid[top[np.argsort(-s[top])]].tolist()
    return out


def fit(train_r, rank, reg, alpha):
    als = ALS(userCol="user", itemCol="item", ratingCol="rating", implicitPrefs=True,
              rank=rank, regParam=reg, alpha=alpha, maxIter=15, coldStartStrategy="drop",
              nonnegative=False, seed=SEED, checkpointInterval=5)
    return als.fit(train_r)


# ---------- 1. Hyper-parameter tuning on the validation window ----------
train_r, val = split("2011-06-01", "2011-09-01")
hist_v, all_v, disc_v = build_tasks(train_r, val)
grid = list(itertools.product([10, 20, 50], [0.01, 0.1], [1.0, 5.0, 20.0]))
rows = []
for rank, reg, alpha in grid:
    t0 = time.time()
    m = fit(train_r, rank, reg, alpha)
    rec_d = als_recs(m, list(disc_v), hist_v, exclude_seen=True)
    res = precision_recall_ndcg(rec_d, disc_v, K)
    rows.append({"rank": rank, "regParam": reg, "alpha": alpha, **res, "fit_s": round(time.time() - t0, 1)})
    print(rows[-1], flush=True)
tuning = pd.DataFrame(rows).sort_values(f"ndcg@{K}", ascending=False)
tuning.to_csv("results/tuning.csv", index=False)
best = tuning.iloc[0]
best_params = {"rank": int(best["rank"]), "regParam": float(best.regParam), "alpha": float(best.alpha)}
print("BEST (by discovery NDCG@10 on validation):", best_params)
train_r.unpersist()

# ---------- 2. Final model on everything before 1 Sep 2011, test on Sep-Dec 2011 ----------
train_r, test = split("2011-09-01", "2011-12-31")
hist, all_t, disc_t = build_tasks(train_r, test)
t0 = time.time()
final = fit(train_r, best_params["rank"], best_params["regParam"], best_params["alpha"])
train_time = round(time.time() - t0, 1)
final.write().overwrite().save("models/als_final")
items.write.mode("overwrite").parquet("models/item_index.parquet")

users_all = list(all_t)
results = {
    "all": {
        "Popularity": popularity_recs(train_r, users_all, hist, exclude_seen=False),
        "Repeat history": history_recs(train_r, users_all),
        "ALS (implicit)": als_recs(final, users_all, hist, exclude_seen=False),
    },
    "discovery": {
        "Popularity": popularity_recs(train_r, list(disc_t), hist, exclude_seen=True),
        "ALS (implicit)": als_recs(final, list(disc_t), hist, exclude_seen=True),
    },
}
truths = {"all": all_t, "discovery": disc_t}
metrics = {"best_params": best_params, "final_train_seconds": train_time,
           "train_pairs": train_r.count(), "n_items": N_ITEMS, "tasks": {}}
for task, models in results.items():
    metrics["tasks"][task] = {}
    for name, recs in models.items():
        r = precision_recall_ndcg(recs, truths[task], K)
        r["catalogue_coverage_pct"] = coverage(recs, N_ITEMS, K)
        metrics["tasks"][task][name] = r
print(json.dumps(metrics, indent=2))
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Comparison chart (also reproducible from results/metrics.json via scripts/plot_results.py)
import subprocess, sys
subprocess.run([sys.executable, "scripts/plot_results.py"], check=True)
spark.stop()
