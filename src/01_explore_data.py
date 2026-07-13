"""
Step 1: Data Exploration
Dissertation: Predicting User Engagement on a Music Streaming Platform
Author: Tianlan Xu (25332222)
"""

import pandas as pd

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("data/raw/lastfm-dataset-50.snappy.parquet")

# ── 2. Basic info ─────────────────────────────────────────────────────────────
print("\n=== Dataset Shape ===")
print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")

print("\n=== Column Names & Types ===")
print(df.dtypes)

print("\n=== First 5 Rows ===")
print(df.head())

# ── 3. Per-user summary ───────────────────────────────────────────────────────
print("\n=== Number of Users ===")
print(df["user_id"].nunique())

summary = df.groupby("user_id")["timestamp"].agg(["min", "max", "count"])
summary.columns = ["first_listen", "last_listen", "n_scrobbles"]
summary["span_days"] = (
    summary["last_listen"] - summary["first_listen"]
).dt.total_seconds() / 86400
summary["n_weeks"] = summary["span_days"] / 7
summary["n_2week_chunks"] = (summary["span_days"] / 14).astype(int)

print("\n=== Per-User Listening History Summary ===")
print(summary.describe().round(1))

print("\n=== Users with fewer than 4 two-week chunks (may be dropped) ===")
low = summary[summary["n_2week_chunks"] < 4]
print(f"{len(low)} users out of {len(summary)}")
print(low[["n_scrobbles", "span_days", "n_2week_chunks"]])

# ── 4. Overall date range ─────────────────────────────────────────────────────
print("\n=== Overall Date Range ===")
print(f"Earliest: {df['timestamp'].min()}")
print(f"Latest:   {df['timestamp'].max()}")

print("\n✅ Exploration complete.")