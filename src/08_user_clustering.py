"""
Step 8: User Clustering - K-Means on behavioural profiles
Dissertation: Predicting User Engagement on a Music Streaming Platform
Author: Tianlan Xu (25332222)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kendalltau

FEATURE_COLS = [
    "morning_ratio", "afternoon_ratio", "evening_ratio", "night_ratio",
    "n_scrobbles", "days_active", "daily_rate",
    "n_unique_artists", "n_unique_tracks", "artist_diversity", "track_diversity"
]

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
features = pd.read_csv("data/raw/features_labelled.csv")

# ── 2. Build user-level profile (mean of all chunks per user) ──────────────────
print("Building user profiles...")
user_profiles = features.groupby("user_id")[FEATURE_COLS].mean()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_profiles)

# ── 3. K-Means with k=3 ───────────────────────────────────────────────────────
print("Running K-Means (k=3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
user_profiles["cluster"] = kmeans.fit_predict(X_scaled)

# Label clusters by daily_rate (high/mid/low engagement)
cluster_means = user_profiles.groupby("cluster")["daily_rate"].mean().sort_values(ascending=False)
label_map = {cluster_means.index[0]: "Heavy listeners",
             cluster_means.index[1]: "Moderate listeners",
             cluster_means.index[2]: "Light listeners"}
user_profiles["cluster_label"] = user_profiles["cluster"].map(label_map)

print("\nCluster sizes:")
print(user_profiles["cluster_label"].value_counts())
print("\nCluster profiles (mean daily_rate):")
print(user_profiles.groupby("cluster_label")["daily_rate"].mean().round(1))

# ── 4. Merge cluster labels onto chunk-level data ──────────────────────────────
features = features.merge(
    user_profiles[["cluster_label"]].reset_index(),
    on="user_id", how="left"
)
threshold = features["daily_rate"].median()
features["high_engagement"] = (features["daily_rate"] > threshold).astype(int)

# ── 5. SHAP + Kendall's Tau per cluster ───────────────────────────────────────
def get_shap_ranking(data):
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

print("\n=== Kendall's Tau by Cluster ===")
cluster_taus = {}
for cluster_label in ["Heavy listeners", "Moderate listeners", "Light listeners"]:
    data = features[features["cluster_label"] == cluster_label]
    drift_df  = data[data["is_drift"] == 1]
    stable_df = data[data["is_drift"] == 0]

    if len(drift_df) < 10 or len(stable_df) < 10:
        print(f"[{cluster_label}] Not enough data, skipping.")
        continue

    feat_drift  = get_shap_ranking(drift_df)
    feat_stable = get_shap_ranking(stable_df)
    cols = feat_drift.index.tolist()
    tau, pvalue = kendalltau(
        feat_drift[cols].rank(ascending=False),
        feat_stable[cols].rank(ascending=False)
    )
    cluster_taus[cluster_label] = tau
    print(f"[{cluster_label:20s}] n={len(data):4d} | Tau={tau:.4f} | p={pvalue:.4f}")

# ── 6. Visualisation ───────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"serif", "font.size":11,
                     "axes.spines.top":False, "axes.spines.right":False})

# Fig 5a: Cluster profiles radar-style bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
colors = ["#4C72B0", "#DD8452", "#55A868"]
display_features = ["daily_rate", "n_unique_artists", "artist_diversity",
                    "morning_ratio", "evening_ratio", "night_ratio"]

for ax, (label, color) in zip(axes, zip(
        ["Heavy listeners", "Moderate listeners", "Light listeners"], colors)):
    data = user_profiles[user_profiles["cluster_label"] == label]
    means = data[display_features].mean()
    means_scaled = (means - means.min()) / (means.max() - means.min() + 1e-9)
    ax.barh(display_features, means_scaled, color=color, alpha=0.8)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Normalised mean value")
    ax.set_xlim(0, 1)

plt.suptitle("User Cluster Behavioural Profiles", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("data/raw/fig5a_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 5b: Tau by cluster
if cluster_taus:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(cluster_taus.keys())
    taus   = list(cluster_taus.values())
    bars = ax.bar(labels, taus, color=colors[:len(labels)], alpha=0.85, width=0.4)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1, label="Threshold (τ=0.7)")
    ax.set_ylabel("Kendall's Tau")
    ax.set_title("Feature Importance Consistency by User Cluster")
    ax.legend()
    for bar, val in zip(bars, taus):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("data/raw/fig5b_cluster_tau.png", dpi=150)
    plt.close()

print("\n✅ User clustering complete.")
print("   → data/raw/fig5a_cluster_profiles.png")
print("   → data/raw/fig5b_cluster_tau.png")