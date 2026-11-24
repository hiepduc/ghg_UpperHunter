#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"

ANALYTICAL_FILE = os.path.join(
    BASE,
    "inversion_results/2026-07",
    "truncated_gaussian_analytical_posterior.csv"
)

MCMC_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-07",
    "MCMC_13mine_daily_Jul2026",
    "FLEXPART_MCMC_posterior_daily_factors_Jul2026.csv"
)

OUTDIR = os.path.join(
    BASE,
    "inversion_results/2026-07",
    "analytical_vs_mcmc_plots",
    "mine_daily"
)

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 90)
print("DAILY POSTERIOR SCALING FACTORS")
print("ANALYTICAL vs FLEXPART MCMC")
print("13 mines × 31 days")
print("=" * 90)

# ============================================================
# READ ANALYTICAL POSTERIOR
# ============================================================

ana = pd.read_csv(ANALYTICAL_FILE)

print("\nAnalytical file:")
print(ANALYTICAL_FILE)
print("Shape:", ana.shape)

required_ana = [
    "parameter",
    "mine",
    "date",
    "truncated_posterior_mean",
    "truncated_posterior_sd",
    "truncated_q025",
    "truncated_q975"
]

missing = [c for c in required_ana if c not in ana.columns]

if missing:
    raise ValueError(
        "Missing analytical columns: " + str(missing)
    )

ana = ana[
    [
        "parameter",
        "mine",
        "date",
        "truncated_posterior_mean",
        "truncated_posterior_sd",
        "truncated_q025",
        "truncated_q975"
    ]
].copy()

ana = ana.rename(
    columns={
        "truncated_posterior_mean": "analytical_mean",
        "truncated_posterior_sd": "analytical_sd",
        "truncated_q025": "analytical_q025",
        "truncated_q975": "analytical_q975"
    }
)

ana["date"] = pd.to_datetime(ana["date"])

# ============================================================
# READ MCMC POSTERIOR
# ============================================================

mcmc = pd.read_csv(MCMC_FILE)

print("\nMCMC file:")
print(MCMC_FILE)
print("Shape:", mcmc.shape)

required_mcmc = [
    "parameter",
    "mine",
    "date",
    "posterior_mean",
    "posterior_sd",
    "q025",
    "q975"
]

missing = [c for c in required_mcmc if c not in mcmc.columns]

if missing:
    raise ValueError(
        "Missing MCMC columns: " + str(missing)
    )

mcmc = mcmc[
    [
        "parameter",
        "mine",
        "date",
        "posterior_mean",
        "posterior_sd",
        "q025",
        "q975"
    ]
].copy()

mcmc = mcmc.rename(
    columns={
        "posterior_mean": "mcmc_mean",
        "posterior_sd": "mcmc_sd",
        "q025": "mcmc_q025",
        "q975": "mcmc_q975"
    }
)

mcmc["date"] = pd.to_datetime(mcmc["date"])

# ============================================================
# MERGE
# ============================================================

df = pd.merge(
    ana,
    mcmc,
    on=["parameter", "mine", "date"],
    how="inner"
)

print("\nMatched parameters:", len(df))

if len(df) != 403:
    raise ValueError(
        f"Expected 403 matched parameters, got {len(df)}"
    )

df = df.sort_values(
    ["mine", "date"]
).reset_index(drop=True)

df["difference"] = (
    df["mcmc_mean"] - df["analytical_mean"]
)

# ============================================================
# OVERALL STATISTICS
# ============================================================

mae = np.mean(np.abs(df["difference"]))
rmse = np.sqrt(np.mean(df["difference"] ** 2))
bias = np.mean(df["difference"])
corr = np.corrcoef(
    df["analytical_mean"],
    df["mcmc_mean"]
)[0, 1]

print("\nOverall comparison:")
print("-" * 60)
print(f"Parameters        : {len(df)}")
print(f"Correlation       : {corr:.8f}")
print(f"Mean difference   : {bias:.8f}")
print(f"MAE               : {mae:.8f}")
print(f"RMSE              : {rmse:.8f}")

# ============================================================
# 13 INDIVIDUAL MINE PLOTS
# ============================================================

mine_order = [f"OC{i}" for i in range(1, 14)]

print("\nCreating individual mine plots...")

for mine in mine_order:

    sub = df[df["mine"] == mine].sort_values("date")

    if len(sub) == 0:
        print(f"WARNING: no data for {mine}")
        continue

    fig, ax = plt.subplots(figsize=(11, 6))

    dates = sub["date"]

    # --------------------------------------------------------
    # Analytical
    # --------------------------------------------------------

    ax.plot(
        dates,
        sub["analytical_mean"],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Analytical posterior mean"
    )

    ax.fill_between(
        dates,
        sub["analytical_q025"],
        sub["analytical_q975"],
        alpha=0.18,
        label="Analytical 95% interval"
    )

    # --------------------------------------------------------
    # MCMC
    # --------------------------------------------------------

    ax.plot(
        dates,
        sub["mcmc_mean"],
        marker="s",
        linewidth=2,
        markersize=4,
        linestyle="--",
        label="FLEXPART MCMC posterior mean"
    )

    ax.fill_between(
        dates,
        sub["mcmc_q025"],
        sub["mcmc_q975"],
        alpha=0.12,
        label="MCMC 95% interval"
    )

    # --------------------------------------------------------
    # Prior
    # --------------------------------------------------------

    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1.5,
        label="Prior scaling factor = 1"
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        f"{mine}: Daily posterior emission scaling factor\n"
        "July 2026 — Analytical vs FLEXPART MCMC"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Emission scaling factor")

    ax.set_xlim(
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-07-31")
    )

    ax.grid(alpha=0.25)

    ax.legend(
        loc="best",
        fontsize=9
    )

    fig.autofmt_xdate()

    fig.tight_layout()

    outfile = os.path.join(
        OUTDIR,
        f"{mine}_daily_analytical_vs_mcmc.png"
    )

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(f"  {mine}: {outfile}")

# ============================================================
# CREATE COMBINED 13-MINE FIGURE
# ============================================================

fig, axes = plt.subplots(
    13,
    1,
    figsize=(14, 34),
    sharex=True
)

for ax, mine in zip(axes, mine_order):

    sub = df[df["mine"] == mine].sort_values("date")

    dates = sub["date"]

    ax.plot(
        dates,
        sub["analytical_mean"],
        marker="o",
        linewidth=1.6,
        markersize=3,
        label="Analytical"
    )

    ax.plot(
        dates,
        sub["mcmc_mean"],
        marker="s",
        linewidth=1.6,
        markersize=3,
        linestyle="--",
        label="MCMC"
    )

    ax.fill_between(
        dates,
        sub["analytical_q025"],
        sub["analytical_q975"],
        alpha=0.12
    )

    ax.fill_between(
        dates,
        sub["mcmc_q025"],
        sub["mcmc_q975"],
        alpha=0.08
    )

    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1.0
    )

    ax.set_ylabel(
        f"{mine}\nScaling",
        rotation=0,
        labelpad=35,
        va="center"
    )

    ax.grid(alpha=0.20)

    ax.legend(
        loc="upper right",
        fontsize=8
    )

axes[-1].set_xlabel("Date")

fig.suptitle(
    "Upper Hunter CH₄ inverse modelling\n"
    "Daily posterior scaling factors: Analytical vs FLEXPART MCMC\n"
    "13 mines × 31 days — July 2026",
    fontsize=16,
    y=0.995
)

fig.tight_layout(
    rect=[0, 0, 1, 0.985]
)

combined_file = os.path.join(
    OUTDIR,
    "ALL_13_MINES_daily_analytical_vs_mcmc.png"
)

fig.savefig(
    combined_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ============================================================
# SAVE DAILY COMPARISON TABLE
# ============================================================

comparison_file = os.path.join(
    OUTDIR,
    "daily_analytical_vs_mcmc_comparison.csv"
)

df.to_csv(
    comparison_file,
    index=False
)

# ============================================================
# MINE-LEVEL SUMMARY OF DIFFERENCES
# ============================================================

summary = (
    df.groupby("mine")
    .agg(
        n=("parameter", "count"),
        analytical_mean=("analytical_mean", "mean"),
        mcmc_mean=("mcmc_mean", "mean"),
        mean_difference=("difference", "mean"),
        MAE=("difference", lambda x: np.mean(np.abs(x))),
        RMSE=("difference", lambda x: np.sqrt(np.mean(x ** 2))),
        max_abs_difference=(
            "difference",
            lambda x: np.max(np.abs(x))
        )
    )
    .reindex(mine_order)
)

summary_file = os.path.join(
    OUTDIR,
    "mine_comparison_statistics.csv"
)

summary.to_csv(summary_file)

print("\nSaved combined figure:")
print(" ", combined_file)

print("\nSaved daily comparison table:")
print(" ", comparison_file)

print("\nSaved mine statistics:")
print(" ", summary_file)

print("\nMine comparison statistics:")
print(summary.to_string(float_format=lambda x: f"{x:.6f}"))

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
