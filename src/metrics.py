"""Ranking metrics for top-K recommendation (binary relevance)."""
import numpy as np


def precision_recall_ndcg(recs: dict, truth: dict, k: int = 10) -> dict:
    """recs: user -> ranked list of items; truth: user -> set of relevant items."""
    p, r, n, h = [], [], [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for u, rel in truth.items():
        if not rel:
            continue
        top = recs.get(u, [])[:k]
        hits = np.array([1.0 if i in rel else 0.0 for i in top] + [0.0] * (k - len(top)))
        idcg = discounts[: min(len(rel), k)].sum()
        p.append(hits.sum() / k)
        r.append(hits.sum() / len(rel))
        n.append((hits * discounts).sum() / idcg)
        h.append(1.0 if hits.sum() > 0 else 0.0)
    return {
        f"precision@{k}": round(float(np.mean(p)), 4),
        f"recall@{k}": round(float(np.mean(r)), 4),
        f"ndcg@{k}": round(float(np.mean(n)), 4),
        f"hit_rate@{k}": round(float(np.mean(h)), 4),
        "users_evaluated": len(p),
    }


def coverage(recs: dict, n_items: int, k: int = 10) -> float:
    shown = {i for lst in recs.values() for i in lst[:k]}
    return round(100 * len(shown) / n_items, 2)
