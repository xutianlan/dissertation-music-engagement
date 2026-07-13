"""
Step 6: Robustness Check - One-week window
Dissertation: Predicting User Engagement on a Music Streaming Platform
Author: Tianlan Xu (25332222)
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau

FEATURE_COLS = [
    "morning_ratio", "afternoon_ratio", "evening_ratio", "night_ratio",
    "n_scrobbles", "days_active", "daily_rate",
    "n_unique_artists", "n_unique_tracks", "artist_diversity", "track_diversity"
]

def run_pipeline(df, window_days, label):
    print(f"\n{'='*50}")
    print(f"Window: {label} ({window_days} days)")
    print(f"{'='*50}")

    # Filter users with enough history
    span = df.groupby("user_id")["timestamp"].agg(["min","max"])
    span["span_days"] = (span["max"]-span["min"]).dt.total_seconds()/86400
    valid = span[span["span_days"] >= window_days * 4].index
    df2 = df[df["user_id"].isin(valid)].copy()
    print(f"Valid users: {df2['user_id'].nunique()}")

    # Assign chunks
    def assign_chunks(group):
        group = group.sort_values("timestamp").copy()
        start = group["timestamp"].min()
        group["chunk_id"] = ((group["timestamp"]-start).dt.total_seconds()
                             / (window_days*86400)).astype(int)
        return group

    df2 = df2.groupby("user_id", group_keys=False).apply(assign_chunks)

    # Compute features
    def compute_features(group):
        hours = group["timestamp"].dt.hour
        total = len(group)
        days_active = group["timestamp"].dt.date.nunique()
        n_unique_artists = group["artist_id"].nunique()
        n_unique_tracks  = group["track_id"].nunique()
        return pd.Series({
            "morning_ratio"   : ((hours>=6)&(hours<12)).sum()/total,
            "afternoon_ratio" : ((hours>=12)&(hours<18)).sum()/total,
            "evening_ratio"   : ((hours>=18)&(hours<24)).sum()/total,
            "night_ratio"     : ((hours>=0)&(hours<6)).sum()/total,
            "n_scrobbles"     : total,
            "days_active"     : days_active,
            "daily_rate"      : total/max(days_active,1),
            "n_unique_artists": n_unique_artists,
            "n_unique_tracks" : n_unique_tracks,
            "artist_diversity": n_unique_artists/total,
            "track_diversity" : n_unique_tracks/total,
        })

    features = (df2.groupby(["user_id","chunk_id"])
                   .apply(compute_features)
                   .reset_index())
    print(f"Total chunks: {len(features)}")

    # Cosine distances
    records = []
    for user_id, group in features.groupby("user_id"):
        group = group.sort_values("chunk_id").reset_index(drop=True)
        if len(group) < 2:
            continue
        X = StandardScaler().fit_transform(group[FEATURE_COLS].values)
        for i in range(len(X)-1):
            dist = cosine_distances(X[i].reshape(1,-1), X[i+1].reshape(1,-1))[0][0]
            records.append({"user_id":user_id,
                            "chunk_to":group.loc[i+1,"chunk_id"],
                            "cosine_dist":dist})

    distances = pd.DataFrame(records)
    cutoff = distances["cosine_dist"].quantile(0.75)
    distances["is_drift"] = (distances["cosine_dist"] >= cutoff).astype(int)
    print(f"Cutoff: {cutoff:.4f} | Drift: {distances['is_drift'].sum()} | Stable: {(distances['is_drift']==0).sum()}")

    # Merge labels
    features_labelled = features.merge(
        distances[["user_id","chunk_to","is_drift"]].rename(columns={"chunk_to":"chunk_id"}),
        on=["user_id","chunk_id"], how="left"
    )
    features_labelled["is_drift"] = features_labelled["is_drift"].fillna(0).astype(int)

    threshold = features_labelled["daily_rate"].median()
    features_labelled["high_engagement"] = (features_labelled["daily_rate"] > threshold).astype(int)

    # SHAP per group
    def get_shap_ranking(data, name):
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
        ranking = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
        return ranking

    drift_df  = features_labelled[features_labelled["is_drift"]==1]
    stable_df = features_labelled[features_labelled["is_drift"]==0]

    feat_drift  = get_shap_ranking(drift_df,  "Drift")
    feat_stable = get_shap_ranking(stable_df, "Stable")

    all_features = feat_drift.index.tolist()
    tau, pvalue = kendalltau(
        feat_drift[all_features].rank(ascending=False),
        feat_stable[all_features].rank(ascending=False)
    )
    print(f"Kendall's Tau: {tau:.4f}  |  P-value: {pvalue:.4f}")
    return tau

# ── Main ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("data/raw/lastfm-dataset-50.snappy.parquet")

tau_2week = 0.9111  # from main analysis
tau_1week = run_pipeline(df, window_days=7,  label="One-week window")
tau_3week = run_pipeline(df, window_days=21, label="Three-week window")

print("\n" + "="*50)
print("ROBUSTNESS CHECK SUMMARY")
print("="*50)
print(f"Two-week window  (main): Kendall's Tau = {tau_2week:.4f}")
print(f"One-week window        : Kendall's Tau = {tau_1week:.4f}")
print(f"Three-week window      : Kendall's Tau = {tau_3week:.4f}")
print("\n✅ Robustness check complete.")