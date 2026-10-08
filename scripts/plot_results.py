"""Plot model comparison from results/metrics.json -> results/figures/model_comparison.png"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

m = json.load(open("results/metrics.json"))["tasks"]
K = 10
COLORS = {"Popularity": "#9aa5b1", "Repeat history": "#f08c00", "ALS (implicit)": "#3b5bdb"}
METRICS = [f"precision@{K}", f"recall@{K}", f"ndcg@{K}"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, task, title in zip(axes, ["all", "discovery"], ["All test purchases", "New products only (discovery)"]):
    names = list(m[task]); w = 0.8 / len(names); x = np.arange(len(METRICS))
    for j, n in enumerate(names):
        ax.bar(x + j * w, [m[task][n][k] for k in METRICS], w, label=n, color=COLORS[n])
    ax.set_xticks(x + w * (len(names) - 1) / 2, ["Precision@10", "Recall@10", "NDCG@10"])
    ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=.3)
fig.suptitle("Test period Sep-Dec 2011", fontsize=10, color="#555")
fig.tight_layout(); fig.savefig("results/figures/model_comparison.png", dpi=150)
print("saved results/figures/model_comparison.png")
