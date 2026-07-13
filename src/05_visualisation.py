"""
Step 5: Visualisation
Dissertation: Predicting User Engagement on a Music Streaming Platform
Author: Tianlan Xu (25332222)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 0. Setup ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"  : "serif",
    "font.size"    : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# ── 1. Cosine distance distribution ───────────────────────────────────────────
print("Plot 1: Cosine distance distribution...")
distances = pd.read_csv("data/raw/cosine_distances.csv")
cutoff = distances["cosine_dist"].quantile(0.75)

fig, ax = plt.subplots(figsize=(8, 4))
stable = distances[distances["cosine_dist"] < cutoff]["cosine_dist"]
drift  = distances[distances["cosine_dist"] >= cutoff]["cosine_dist"]

ax.hist(stable, bins=40, color="#4C72B0", alpha=0.7, label="Stable")
ax.hist(drift,  bins=40, color="#DD8452", alpha=0.7, label="Drift")
ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.5,
           label=f"75th percentile cutoff ({cutoff:.3f})")
ax.set_xlabel("Cosine Distance (consecutive chunks)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Cosine Distances Between Consecutive Two-Week Chunks")
ax.legend()
plt.tight_layout()
plt.savefig("data/raw/fig1_cosine_distribution.png", dpi=150)
plt.close()
print("  → fig1_cosine_distribution.png")

# ── 2. SHAP comparison bar chart ───────────────────────────────────────────────
print("Plot 2: SHAP feature importance comparison...")
comp = pd.read_csv("data/raw/shap_comparison.csv", index_col=0)
comp = comp.sort_values("shap_drift", ascending=False)

x = np.arange(len(comp))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, comp["shap_drift"],  width, label="Drift",  color="#DD8452", alpha=0.85)
ax.bar(x + width/2, comp["shap_stable"], width, label="Stable", color="#4C72B0", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(comp.index, rotation=35, ha="right")
ax.set_ylabel("Mean |SHAP value|")
ax.set_title("Feature Importance (SHAP) — Drift vs Stable Periods")
ax.legend()
plt.tight_layout()
plt.savefig("data/raw/fig2_shap_comparison.png", dpi=150)
plt.close()
print("  → fig2_shap_comparison.png")

# ── 3. Feature importance rank comparison ──────────────────────────────────────
print("Plot 3: Rank comparison (bump chart)...")
comp["rank_drift"]  = comp["shap_drift"].rank(ascending=False).astype(int)
comp["rank_stable"] = comp["shap_stable"].rank(ascending=False).astype(int)
comp_sorted = comp.sort_values("rank_drift")

fig, ax = plt.subplots(figsize=(7, 6))
for feat in comp_sorted.index:
    rd = comp_sorted.loc[feat, "rank_drift"]
    rs = comp_sorted.loc[feat, "rank_stable"]
    color = "#DD8452" if abs(rd - rs) >= 2 else "#4C72B0"
    ax.plot([0, 1], [rd, rs], color=color, linewidth=1.8, alpha=0.8)
    ax.text(-0.05, rd, feat, ha="right", va="center", fontsize=9)
    ax.text(1.05,  rs, feat, ha="left",  va="center", fontsize=9)

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(len(comp) + 0.5, 0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Drift", "Stable"], fontsize=12)
ax.set_ylabel("Rank (1 = most important)")
ax.set_title("Feature Importance Rank Shift: Drift vs Stable")
orange = mpatches.Patch(color="#DD8452", label="Rank shift ≥ 2")
blue   = mpatches.Patch(color="#4C72B0", label="Rank shift < 2")
ax.legend(handles=[orange, blue], loc="lower right")
plt.tight_layout()
plt.savefig("data/raw/fig3_rank_comparison.png", dpi=150)
plt.close()
print("  → fig3_rank_comparison.png")

print("\n✅ All visualisations saved to data/raw/")