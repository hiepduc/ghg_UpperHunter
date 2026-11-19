#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/2026-04/MCMC_13mine_daily_Apr2026_MH"
)

INPUT = BASE / (
    "FLEXPART_Gibbs_vs_MH_scaling_factor_comparison_Apr2026.csv"
)

OUT = BASE / "Gibbs_vs_MH_plots"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

print("=" * 90)
print("GIBBS vs MH PLOT GENERATION")
print("=" * 90)

print(f"\nInput:")
print(INPUT)

print(f"\nParameters: {len(df)}")

print(f"\nOutput directory:")
print(OUT)


# ============================================================
# PLOT 1
# POSTERIOR MEAN: GIBBS vs MH
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(
    df["Gibbs_mean"],
    df["MH_mean"],
    s=25,
    alpha=0.7
)

lo = min(
    df["Gibbs_mean"].min(),
    df["MH_mean"].min()
)

hi = max(
    df["Gibbs_mean"].max(),
    df["MH_mean"].max()
)

ax.plot(
    [lo, hi],
    [lo, hi],
    linestyle="--"
)

ax.set_xlabel(
    "Gibbs posterior mean scaling factor"
)

ax.set_ylabel(
    "MH posterior mean scaling factor"
)

ax.set_title(
    "Gibbs vs Metropolis-Hastings Posterior Means"
)

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUT / "01_Gibbs_vs_MH_posterior_means.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# PLOT 2
# MH - GIBBS DIFFERENCE FOR ALL 351 PARAMETERS
# ============================================================

plot_df = df.sort_values(
    "mean_difference_MH_minus_Gibbs"
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 7))

x = np.arange(len(plot_df))

ax.scatter(
    x,
    plot_df["mean_difference_MH_minus_Gibbs"],
    s=14
)

ax.axhline(
    0,
    linestyle="--"
)

ax.axhline(
    0.01,
    linestyle=":",
    alpha=0.7
)

ax.axhline(
    -0.01,
    linestyle=":",
    alpha=0.7
)

ax.set_xlabel(
    "Source-day parameters sorted by MH − Gibbs"
)

ax.set_ylabel(
    "MH − Gibbs posterior mean"
)

ax.set_title(
    "Difference Between MH and Gibbs Posterior Means"
)

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUT / "02_MH_minus_Gibbs_parameter_difference.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# PLOT 3
# MINE-LEVEL POSTERIOR MEANS
# ============================================================

mine = (
    df
    .groupby("mine", sort=False)
    .agg(
        Gibbs=("Gibbs_mean", "mean"),
        MH=("MH_mean", "mean")
    )
    .reset_index()
)

x = np.arange(len(mine))

width = 0.38

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    mine["Gibbs"],
    width,
    label="Gibbs"
)

ax.bar(
    x + width / 2,
    mine["MH"],
    width,
    label="Metropolis-Hastings"
)

ax.set_xticks(x)

ax.set_xticklabels(
    mine["mine"]
)

ax.set_xlabel(
    "Mine"
)

ax.set_ylabel(
    "Mean daily scaling factor"
)

ax.set_title(
    "Mine-level Gibbs vs MH Posterior Scaling Factors"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT / "03_mine_level_Gibbs_vs_MH.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# PLOT 4
# MINE-LEVEL 95% CREDIBLE INTERVALS
# ============================================================

mine_ci = (
    df
    .groupby("mine", sort=False)
    .agg(
        Gibbs_mean=("Gibbs_mean", "mean"),
        Gibbs_q025=("Gibbs_q025", "mean"),
        Gibbs_q975=("Gibbs_q975", "mean"),

        MH_mean=("MH_mean", "mean"),
        MH_q025=("MH_q025", "mean"),
        MH_q975=("MH_q975", "mean")
    )
    .reset_index()
)

x = np.arange(len(mine_ci))

fig, ax = plt.subplots(figsize=(12, 7))

ax.errorbar(
    x - 0.12,
    mine_ci["Gibbs_mean"],
    yerr=[
        mine_ci["Gibbs_mean"] - mine_ci["Gibbs_q025"],
        mine_ci["Gibbs_q975"] - mine_ci["Gibbs_mean"]
    ],
    fmt="o",
    capsize=4,
    label="Gibbs 95% CI"
)

ax.errorbar(
    x + 0.12,
    mine_ci["MH_mean"],
    yerr=[
        mine_ci["MH_mean"] - mine_ci["MH_q025"],
        mine_ci["MH_q975"] - mine_ci["MH_mean"]
    ],
    fmt="o",
    capsize=4,
    label="MH 95% CI"
)

ax.axhline(
    1.0,
    linestyle="--",
    alpha=0.7,
    label="Prior factor = 1"
)

ax.set_xticks(x)

ax.set_xticklabels(
    mine_ci["mine"]
)

ax.set_xlabel(
    "Mine"
)

ax.set_ylabel(
    "Posterior scaling factor"
)

ax.set_title(
    "Mine-level Posterior Means and 95% Credible Intervals"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT / "04_mine_level_credible_intervals.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# PLOT 5
# DAILY MEAN FACTOR
# ============================================================

daily = (
    df
    .groupby("date")
    .agg(
        Gibbs=("Gibbs_mean", "mean"),
        MH=("MH_mean", "mean")
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    daily["date"],
    daily["Gibbs"],
    marker="o",
    label="Gibbs"
)

ax.plot(
    daily["date"],
    daily["MH"],
    marker="o",
    label="Metropolis-Hastings"
)

ax.axhline(
    1.0,
    linestyle="--",
    alpha=0.7,
    label="Prior factor = 1"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Mean scaling factor across 13 mines"
)

ax.set_title(
    "Daily Mean Posterior Scaling Factor"
)

ax.legend()

ax.grid(alpha=0.25)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    OUT / "05_daily_Gibbs_vs_MH.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# PLOT 6
# DISTRIBUTION OF MH - GIBBS DIFFERENCES
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(
    df["mean_difference_MH_minus_Gibbs"],
    bins=30
)

ax.axvline(
    0,
    linestyle="--"
)

ax.set_xlabel(
    "MH − Gibbs posterior mean"
)

ax.set_ylabel(
    "Number of source-day parameters"
)

ax.set_title(
    "Distribution of MH − Gibbs Posterior Mean Differences"
)

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUT / "06_distribution_MH_minus_Gibbs.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# PRINT OUTPUT
# ============================================================

print()
print("=" * 90)
print("PLOTS CREATED")
print("=" * 90)

for filename in sorted(OUT.glob("*.png")):
    print(filename)

print()
print("=" * 90)
print("DONE")
print("=" * 90)

