"""
Step 3: Drift Detection - Cosine Distance between consecutive chunks
Dissertation: Predicting User Engagement on a Music Streaming Platform
Author: Tianlan Xu (25332222)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances

# ── 1. Load features ───────────────────────────────────────────────────────────
print("Loading features...")
features = pd.read_csv("data/raw/features_per_chunk.csv")

FEATURE_COLS = [
    "morning_ratio", "afternoon_ratio", "evening_ratio", "night_ratio",
    "n_scrobbles", "days_active", "daily_rate",
    "n_unique_artists", "n_unique_tracks", "artist_diversity", "track_diversity"
]

# ── 2. Compute cosine distance between consecutive chunks per user ──────────────
print("Computing cosine distances...")

records = []

for user_id, group in features.groupby("user_id"):
    group = group.sort_values("chunk_id").reset_index(drop=True)

    if len(group) < 2:
        continue

    # Normalise features
    X = group[FEATURE_COLS].values
    X = StandardScaler().fit_transform(X)

    # Distance between chunk[i] and chunk[i+1]
    for i in range(len(X) - 1):
        dist = cosine_distances(X[i].reshape(1, -1), X[i+1].reshape(1, -1))[0][0]
        records.append({
            "user_id"     : user_id,
            "chunk_from"  : group.loc[i,   "chunk_id"],
            "chunk_to"    : group.loc[i+1, "chunk_id"],
            "cosine_dist" : dist,
        })

distances = pd.DataFrame(records)
print(f"Total consecutive chunk pairs: {len(distances)}")

# ── 3. Label drift vs stable using upper quartile cutoff ──────────────────────
cutoff = distances["cosine_dist"].quantile(0.75)
print(f"\nUpper quartile cutoff: {cutoff:.4f}")

distances["is_drift"] = (distances["cosine_dist"] >= cutoff).astype(int)

print(f"Drift periods:  {distances['is_drift'].sum()}")
print(f"Stable periods: {(distances['is_drift'] == 0).sum()}")

# ── 4. Merge label back onto features ─────────────────────────────────────────
# Label chunk[i+1] with the drift flag from transition i→i+1
distances_to_merge = distances[["user_id", "chunk_to", "is_drift"]].rename(
    columns={"chunk_to": "chunk_id"}
)
features_labelled = features.merge(distances_to_merge,
                                   on=["user_id", "chunk_id"],
                                   how="left")
features_labelled["is_drift"] = features_labelled["is_drift"].fillna(0).astype(int)

print(f"\nLabelled feature table shape: {features_labelled.shape}")
print(features_labelled["is_drift"].value_counts())

# ── 5. Save ────────────────────────────────────────────────────────────────────
distances.to_csv("data/raw/cosine_distances.csv", index=False)
features_labelled.to_csv("data/raw/features_labelled.csv", index=False)

print("\n✅ Drift detection complete.")
print("   → data/raw/cosine_distances.csv")
print("   → data/raw/features_labelled.csv")