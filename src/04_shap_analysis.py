"""
Step 4: SHAP Analysis on Drift vs Stable periods
Dissertation: Predicting User Engagement on a Music Streaming Platform
Author: Tianlan Xu (25332222)
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import kendalltau

FEATURE_COLS = [
    "morning_ratio", "afternoon_ratio", "evening_ratio", "night_ratio",
    "n_scrobbles", "days_active", "daily_rate",
    "n_unique_artists", "n_unique_tracks", "artist_diversity", "track_diversity"
]

# ── 1. Load labelled features ──────────────────────────────────────────────────
print("Loading labelled features...")
df = pd.read_csv("data/raw/features_labelled.csv")

drift_df  = df[df["is_drift"] == 1].copy()
stable_df = df[df["is_drift"] == 0].copy()

print(f"Drift chunks:  {len(drift_df)}")
print(f"Stable chunks: {len(stable_df)}")

# ── 2. Train Random Forest on each group separately ────────────────────────────
# Target: predict whether daily_rate is above median (high vs low engagement)
threshold = df["daily_rate"].median()
df["high_engagement"] = (df["daily_rate"] > threshold).astype(int)
drift_df  = df[df["is_drift"] == 1].copy()
stable_df = df[df["is_drift"] == 0].copy()

def train_and_shap(data, label):
    X = data[FEATURE_COLS].drop(columns=["daily_rate"])
    y = data["high_engagement"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"\n[{label}] ROC-AUC (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values is a list; take class=1
    sv = shap_values
    if hasattr(sv, '__len__') and not isinstance(sv, np.ndarray):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs_shap = pd.Series(
        np.abs(sv).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False)

    print(f"[{label}] Feature importance ranking (SHAP):")
    print(mean_abs_shap.round(4))

    # Save SHAP bar plot
    plt.figure(figsize=(8, 5))
    mean_abs_shap.plot(kind="barh", color="steelblue")
    plt.gca().invert_yaxis()
    plt.title(f"Mean |SHAP| - {label} periods")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(f"data/raw/shap_{label.lower()}.png", dpi=150)
    plt.close()
    print(f"[{label}] SHAP plot saved.")

    return mean_abs_shap

feat_drift  = train_and_shap(drift_df,  "Drift")
feat_stable = train_and_shap(stable_df, "Stable")

# ── 3. Kendall's Tau comparison ────────────────────────────────────────────────
print("\n=== Kendall's Tau Comparison ===")

# Align rankings
all_features = feat_drift.index.tolist()
rank_drift  = feat_drift[all_features].rank(ascending=False)
rank_stable = feat_stable[all_features].rank(ascending=False)

tau, pvalue = kendalltau(rank_drift, rank_stable)
print(f"Kendall's Tau: {tau:.4f}")
print(f"P-value:       {pvalue:.4f}")

if tau > 0.7:
    print("→ High consistency: model explains engagement similarly in both periods.")
elif tau > 0.4:
    print("→ Moderate consistency: some features shift in importance across periods.")
else:
    print("→ Low consistency: feature importance differs substantially between periods.")

# ── 4. Save ranking comparison ─────────────────────────────────────────────────
comparison = pd.DataFrame({
    "shap_drift" : feat_drift[all_features],
    "shap_stable": feat_stable[all_features],
    "rank_drift" : rank_drift,
    "rank_stable": rank_stable,
}).round(4)

comparison.to_csv("data/raw/shap_comparison.csv")
print("\n✅ SHAP analysis complete.")
print("   → data/raw/shap_drift.png")
print("   → data/raw/shap_stable.png")
print("   → data/raw/shap_comparison.csv")