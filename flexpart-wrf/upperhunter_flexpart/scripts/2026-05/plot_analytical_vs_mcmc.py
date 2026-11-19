#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"

ANALYTICAL = os.path.join(
    BASE,
    "inversion_results/2026-05",
    "truncated_gaussian_analytical_posterior.csv"
)

MCMC = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-05",
    "MCMC_13mine_daily_May2026",
    "FLEXPART_MCMC_posterior_daily_factors_May2026.csv"
)

OUTDIR = os.path.join(
    BASE,
    "inversion_results/2026-05",
    "analytical_vs_mcmc_plots"
)

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 90)
print("ANALYTICAL vs FLEXPART MCMC POSTERIOR SCALING FACTORS")
print("=" * 90)

# ---------------------------------------------------------------------
# Read analytical truncated-Gaussian posterior
# ---------------------------------------------------------------------

ana = pd.read_csv(ANALYTICAL)

print("\nAnalytical columns:")
print(ana.columns.tolist())
print("Shape:", ana.shape)

# Expected columns include parameter, mean, sd, q025, q975.
# Detect the mean column robustly.
if "parameter" not in ana.columns:
    raise ValueError("Analytical file does not contain 'parameter' column.")

if "truncated_posterior_mean" in ana.columns:
    ana_mean_col = "truncated_posterior_mean"
elif "posterior_mean" in ana.columns:
    ana_mean_col = "posterior_mean"
elif "mean" in ana.columns:
    ana_mean_col = "mean"
else:
    raise ValueError(
        "Could not identify analytical posterior mean column."
    )

if "truncated_posterior_sd" in ana.columns:
    ana_sd_col = "truncated_posterior_sd"
elif "posterior_sd" in ana.columns:
    ana_sd_col = "posterior_sd"
elif "sd" in ana.columns:
    ana_sd_col = "sd"
else:
    ana_sd_col = None

ana2 = ana[["parameter", ana_mean_col]].copy()
ana2.columns = ["parameter", "analytical_mean"]

if ana_sd_col is not None:
    ana2["analytical_sd"] = ana[ana_sd_col].values

# ---------------------------------------------------------------------
# Read MCMC posterior
# ---------------------------------------------------------------------

mcmc = pd.read_csv(MCMC)

print("\nMCMC columns:")
print(mcmc.columns.tolist())
print("Shape:", mcmc.shape)

if "parameter" not in mcmc.columns:
    raise ValueError("MCMC file does not contain 'parameter' column.")

if "posterior_mean" not in mcmc.columns:
    raise ValueError(
        "MCMC file does not contain 'posterior_mean' column."
    )

mcmc2 = mcmc[
    ["parameter", "posterior_mean"]
].copy()

mcmc2.columns = ["parameter", "mcmc_mean"]

if "posterior_sd" in mcmc.columns:
    mcmc2["mcmc_sd"] = mcmc["posterior_sd"].values

# ---------------------------------------------------------------------
# Align exactly by parameter name
# ---------------------------------------------------------------------

df = pd.merge(
    ana2,
    mcmc2,
    on="parameter",
    how="inner"
)

print("\nMatched parameters:", len(df))

if len(df) != 403:
    raise ValueError(
        f"Expected 403 matched parameters, found {len(df)}"
    )

# Preserve parameter order from analytical solution
df["_order"] = np.arange(len(df))
df = df.sort_values("_order").drop(columns="_order")

df["difference"] = (
    df["mcmc_mean"] - df["analytical_mean"]
)

df["abs_difference"] = df["difference"].abs()

# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

x = df["analytical_mean"].to_numpy()
y = df["mcmc_mean"].to_numpy()

mae = np.mean(np.abs(y - x))
rmse = np.sqrt(np.mean((y - x) ** 2))
bias = np.mean(y - x)
corr = np.corrcoef(x, y)[0, 1]
max_abs = np.max(np.abs(y - x))

print("\nComparison statistics")
print("-" * 50)
print(f"Parameters        : {len(df)}")
print(f"Analytical range  : {x.min():.6f} to {x.max():.6f}")
print(f"MCMC range        : {y.min():.6f} to {y.max():.6f}")
print(f"Mean difference   : {bias:.8f}")
print(f"MAE               : {mae:.8f}")
print(f"RMSE              : {rmse:.8f}")
print(f"Correlation       : {corr:.8f}")
print(f"Max abs difference: {max_abs:.8f}")

# ---------------------------------------------------------------------
# Plot 1: Analytical vs MCMC scatter
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    x,
    y,
    s=24,
    alpha=0.65,
    edgecolors="none"
)

lo = min(x.min(), y.min())
hi = max(x.max(), y.max())

margin = 0.05 * (hi - lo)

ax.plot(
    [lo - margin, hi + margin],
    [lo - margin, hi + margin],
    linestyle="--",
    linewidth=1.5,
    label="1:1 line"
)

ax.set_xlim(lo - margin, hi + margin)
ax.set_ylim(lo - margin, hi + margin)

ax.set_xlabel("Analytical posterior mean scaling factor")
ax.set_ylabel("FLEXPART MCMC posterior mean scaling factor")

ax.set_title(
    "Analytical vs FLEXPART MCMC posterior scaling factors\n"
    "13 mines × 27 days = 351 parameters"
)

ax.text(
    0.04,
    0.96,
    f"n = {len(df)}\n"
    f"r = {corr:.6f}\n"
    f"RMSE = {rmse:.6f}\n"
    f"MAE = {mae:.6f}",
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()

scatter_file = os.path.join(
    OUTDIR,
    "analytical_vs_mcmc_351_parameters.png"
)

fig.savefig(
    scatter_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ---------------------------------------------------------------------
# Plot 2: Difference MCMC - analytical
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

ax.axhline(
    0,
    linestyle="--",
    linewidth=1.2
)

ax.plot(
    np.arange(1, len(df) + 1),
    df["difference"],
    marker=".",
    linestyle="none",
    markersize=5,
    alpha=0.7
)

ax.set_xlabel("Parameter index")
ax.set_ylabel(
    "MCMC posterior mean − analytical posterior mean"
)

ax.set_title(
    "Difference between FLEXPART MCMC and truncated analytical posterior"
)

ax.grid(alpha=0.25)

fig.tight_layout()

difference_file = os.path.join(
    OUTDIR,
    "mcmc_minus_analytical_351_parameters.png"
)

fig.savefig(
    difference_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ---------------------------------------------------------------------
# Add mine and date columns
# ---------------------------------------------------------------------

parts = df["parameter"].str.extract(
    r"^(OC\d+)_(\d{8})$"
)

df["mine"] = parts[0]
df["date"] = pd.to_datetime(
    parts[1],
    format="%Y%m%d"
)

# ---------------------------------------------------------------------
# Plot 3: Mine-level posterior means
# ---------------------------------------------------------------------

mine_order = [f"OC{i}" for i in range(1, 14)]

mine = (
    df.groupby("mine")
    .agg(
        analytical_mean=("analytical_mean", "mean"),
        mcmc_mean=("mcmc_mean", "mean")
    )
    .reindex(mine_order)
)

fig, ax = plt.subplots(figsize=(13, 6))

positions = np.arange(len(mine))

width = 0.38

ax.bar(
    positions - width / 2,
    mine["analytical_mean"],
    width=width,
    label="Analytical"
)

ax.bar(
    positions + width / 2,
    mine["mcmc_mean"],
    width=width,
    label="MCMC"
)

ax.set_xticks(positions)
ax.set_xticklabels(mine_order)

ax.set_xlabel("Mine")
ax.set_ylabel("Mean daily scaling factor")

ax.set_title(
    "Mine-level mean posterior scaling factors\n"
    "Analytical vs FLEXPART MCMC"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

mine_file = os.path.join(
    OUTDIR,
    "analytical_vs_mcmc_mine_level.png"
)

fig.savefig(
    mine_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ---------------------------------------------------------------------
# Save comparison table
# ---------------------------------------------------------------------

comparison_file = os.path.join(
    OUTDIR,
    "analytical_vs_mcmc_351_parameters.csv"
)

df.to_csv(
    comparison_file,
    index=False
)

print("\nSaved plots:")
print(" ", scatter_file)
print(" ", difference_file)
print(" ", mine_file)

print("\nSaved comparison table:")
print(" ", comparison_file)

print("\nDone.")
