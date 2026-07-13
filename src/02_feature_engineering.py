"""
Step 2: Feature Engineering - Sliding Window + Behavioral Features
Dissertation: Predicting User Engagement on a Music Streaming Platform
Author: Tianlan Xu (25332222)
"""

import pandas as pd
import numpy as np

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("data/raw/lastfm-dataset-50.snappy.parquet")

# Drop users with fewer than 4 two-week chunks
span = df.groupby("user_id")["timestamp"].agg(["min", "max"])
span["span_days"] = (span["max"] - span["min"]).dt.total_seconds() / 86400
valid_users = span[span["span_days"] >= 56].index  # 56 days = 4 chunks
df = df[df["user_id"].isin(valid_users)].copy()
print(f"Users after filtering: {df['user_id'].nunique()}")

# ── 2. Assign each scrobble to a two-week chunk ────────────────────────────────
def assign_chunks(group):
    group = group.sort_values("timestamp").copy()
    start = group["timestamp"].min()
    group["chunk_id"] = ((group["timestamp"] - start).dt.total_seconds()
                         / (14 * 86400)).astype(int)
    return group

print("Assigning two-week chunks...")
df = df.groupby("user_id", group_keys=False).apply(assign_chunks)

# ── 3. Compute behavioral features per user per chunk ──────────────────────────
print("Computing behavioral features...")

def compute_features(group):
    # Temporal features
    hours = group["timestamp"].dt.hour
    total = len(group)

    morning   = ((hours >= 6)  & (hours < 12)).sum() / total   # 06-12
    afternoon = ((hours >= 12) & (hours < 18)).sum() / total   # 12-18
    evening   = ((hours >= 18) & (hours < 24)).sum() / total   # 18-24
    night     = ((hours >= 0)  & (hours < 6)).sum()  / total   # 00-06

    # Intensity features
    n_scrobbles = total
    days_active = group["timestamp"].dt.date.nunique()
    daily_rate  = n_scrobbles / max(days_active, 1)

    # Diversity features
    n_unique_artists = group["artist_id"].nunique()
    n_unique_tracks  = group["track_id"].nunique()
    artist_diversity = n_unique_artists / total
    track_diversity  = n_unique_tracks  / total

    return pd.Series({
        "morning_ratio"    : morning,
        "afternoon_ratio"  : afternoon,
        "evening_ratio"    : evening,
        "night_ratio"      : night,
        "n_scrobbles"      : n_scrobbles,
        "days_active"      : days_active,
        "daily_rate"       : daily_rate,
        "n_unique_artists" : n_unique_artists,
        "n_unique_tracks"  : n_unique_tracks,
        "artist_diversity" : artist_diversity,
        "track_diversity"  : track_diversity,
    })

features = (df.groupby(["user_id", "chunk_id"])
              .apply(compute_features)
              .reset_index())

print(f"Feature table shape: {features.shape}")
print(features.head(10))

# ── 4. Save ────────────────────────────────────────────────────────────────────
features.to_csv("data/raw/features_per_chunk.csv", index=False)
print("\n✅ Features saved to data/raw/features_per_chunk.csv")