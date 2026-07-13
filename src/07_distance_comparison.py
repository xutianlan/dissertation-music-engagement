"""
Step 7: Robustness Check - Alternative Distance Metrics
Dissertation: Predicting User Engagement on a Music Streaming Platform
Author: Tianlan Xu (25332222)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import kendalltau
from scipy.spatial.distance import jensenshannon

FEATURE_COLS = [
    "morning_ratio", "afternoon_ratio", "evening_ratio", "night_ratio",
    "n_scrobbles", "days_active", "daily_rate",
    "n_unique_artists", "n_unique_tracks", "artist_diversity", "track_diversity"
]

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading features...")
features = pd.read_csv("data/raw/features_per_chunk.csv")

# ── 2. Distance functions ──────────────────────────────────────────────────────
def get_cosine_dist(a, b):
    return cosine_distances(a.reshape(1,-1), b.reshape(1,-1))[0][0]

def get_euclidean_dist(a, b):
    return euclidean_distances(a.reshape(1,-1), b.reshape(1,-1))[0][0]

def get_js_dist(a, b):
    # Shift to positive then normalise to probability distribution
    a = a - a.min() + 1e-9
    b = b - b.min() + 1e-9
    a = a / a.sum()
    b = b / b.sum()
    return jensenshannon(a, b)

METRICS = {
    "Cosine"     : get_cosine_dist,
    "Euclidean"  : get_euclidean_dist,
    "Jensen-Shannon": get_js_dist,
}

# ── 3. Run pipeline for each metric ───────────────────────────────────────────
def run_with_metric(features, metric_name, dist_fn):
    records = []
    for user_id, group in features.groupby("user_id"):
        group = group.sort_values("chunk_id").reset_index(drop=True)
        if len(group) < 2:
            continue
        X = StandardScaler().fit_transform(group[FEATURE_COLS].values)
        for i in range(len(X)-1):
            dist = dist_fn(X[i], X[i+1])
            records.append({
                "user_id"   : user_id,
                "chunk_to"  : group.loc[i+1, "chunk_id"],
                "distance"  : dist,
            })

    distances = pd.DataFrame(records)
    cutoff = distances["distance"].quantile(0.75)
    distances["is_drift"] = (distances["distance"] >= cutoff).astype(int)

    # Merge labels onto features
    fl = features.merge(
        distances[["user_id","chunk_to","is_drift"]].rename(columns={"chunk_to":"chunk_id"}),
        on=["user_id","chunk_id"], how="left"
    )
    fl["is_drift"] = fl["is_drift"].fillna(0).astype(int)
    threshold = fl["daily_rate"].median()
    fl["high_engagement"] = (fl["daily_rate"] > threshold).astype(int)

    # SHAP per group
    def get_ranking(data):
        X = data[FEATURE_COLS].drop(columns=["daily_rate"])
        y = data["high_engagement"]
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        if hasattr(sv, 'ndim') and sv.ndim == 3:
            sv = sv[:, :, 1]
        return pd.Series(np.abs(sv).mean(axis=0), index=X.columns)

    drift_df  = fl[fl["is_drift"]==1]
    stable_df = fl[fl["is_drift"]==0]

    feat_drift  = get_ranking(drift_df)
    feat_stable = get_ranking(stable_df)

    cols = feat_drift.index.tolist()
    tau, pvalue = kendalltau(
        feat_drift[cols].rank(ascending=False),
        feat_stable[cols].rank(ascending=False)
    )

    print(f"[{metric_name:15s}] Drift={distances['is_drift'].sum():4d} | "
          f"Stable={(distances['is_drift']==0).sum():4d} | "
          f"Tau={tau:.4f} | p={pvalue:.4f}")
    return tau

# ── 4. Run all metrics ─────────────────────────────────────────────────────────
print("\n=== Distance Metric Comparison ===")
results = {}
for name, fn in METRICS.items():
    results[name] = run_with_metric(features, name, fn)

# ── 5. Summary table ───────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
summary = pd.DataFrame({
    "Distance Metric": list(results.keys()),
    "Kendall Tau"    : [round(v, 4) for v in results.values()],
})
print(summary.to_string(index=False))

# ── 6. Bar chart ───────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"serif", "font.size":11,
                     "axes.spines.top":False, "axes.spines.right":False})

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.bar(summary["Distance Metric"], summary["Kendall Tau"],
              color=colors, alpha=0.85, width=0.5)
ax.set_ylim(0, 1.05)
ax.axhline(0.7, color="gray", linestyle="--", linewidth=1,
           label="Threshold (τ=0.7)")
ax.set_ylabel("Kendall's Tau")
ax.set_title("Feature Importance Consistency Across Distance Metrics")
ax.legend()
for bar, val in zip(bars, summary["Kendall Tau"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.4f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("data/raw/fig4_distance_comparison.png", dpi=150)
plt.close()
print("\n✅ Distance comparison complete.")
print("   → data/raw/fig4_distance_comparison.png")