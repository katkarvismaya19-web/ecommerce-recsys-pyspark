"""Render the pipeline architecture as a PNG for the blog (results/figures/architecture.png)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

stages = [
    ("1. Ingest", "Raw CSV, 1,067,371 lines -> Spark DataFrame"),
    ("2. Clean", "Drop duplicates, guest orders, cancellations, non-products -> 776,577 lines (Parquet)"),
    ("3. Build implicit feedback", "Confidence = orders per customer-product pair; 480,971 pairs, 1.78% dense"),
    ("4. Time-based split", "Tune on Jun-Aug 2011; train < 1 Sep 2011, test Sep-Dec 2011"),
    ("5. Train ALS (Spark MLlib)", "Implicit-feedback ALS, 18-configuration grid search"),
    ("6. Evaluate and recommend", "Precision / Recall / NDCG@10 vs baselines; top-10 list per customer"),
]
fig, ax = plt.subplots(figsize=(9, 7.2))
ax.set_xlim(0, 10); ax.set_ylim(0, len(stages) * 1.2 + 0.4); ax.axis("off")
for i, (name, body) in enumerate(stages):
    y = (len(stages) - 1 - i) * 1.2 + 0.4
    hl = i == 4
    ax.add_patch(FancyBboxPatch((0.5, y), 9, 0.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#e8eefc" if hl else "white", ec="#3b5bdb" if hl else "#555", lw=2 if hl else 1.2))
    ax.text(0.75, y + 0.53, name, fontsize=11.5, weight="bold", va="center")
    ax.text(0.75, y + 0.22, body, fontsize=9.5, color="#444", va="center")
    if i < len(stages) - 1:
        ax.annotate("", xy=(5, y - 0.38), xytext=(5, y), arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
ax.set_title("End-to-end PySpark recommendation pipeline", fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig("results/figures/architecture.png", dpi=150)
