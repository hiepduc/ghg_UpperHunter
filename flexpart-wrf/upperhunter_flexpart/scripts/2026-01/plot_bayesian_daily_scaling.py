#!/usr/bin/env python3

"""
===============================================================================
PLOT BAYESIAN DAILY MINE SCALING FACTORS
Upper Hunter CH4 inverse modelling
===============================================================================

Reads:
    hmatrix/daily_20260105_20260131/bayesian_analytical/
        posterior_scaling_factors.csv

Actual input columns:
    parameter
    prior_scaling
    posterior_mean
    posterior_sd
    posterior_lower_95
    posterior_upper_95
    prior_emission_kg_s
    posterior_emission_kg_s
    posterior_emission_lower_kg_s
    posterior_emission_upper_kg_s

Parameter format:
    OC1_20260105
    OC2_20260105
    ...
    OC13_20260131

Produces:
    daily_scaling_plots/
        daily_scaling_factors_long.csv
        daily_scaling_factors_wide.csv
        daily_scaling_summary.csv
        all_13_mines_daily_scaling.png
        individual_mine_daily_scaling.png
        individual_mine_daily_scaling_with_95CI.png
===============================================================================
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

INPUT = (
    BASE
    / "hmatrix/daily_20260105_20260131/bayesian_analytical"
    / "posterior_scaling_factors.csv"
)

OUTPUT = (
    BASE
    / "hmatrix/daily_20260105_20260131/bayesian_analytical"
    / "daily_scaling_plots"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# READ DATA
# =============================================================================

print("=" * 90)
print("BAYESIAN DAILY MINE SCALING FACTORS")
print("=" * 90)

print()
print("Input file:")
print(INPUT)

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT}")

df = pd.read_csv(INPUT)

print()
print("Columns:")
print(df.columns.tolist())

print()
print(f"Rows: {len(df)}")


# =============================================================================
# CHECK REQUIRED COLUMNS
# =============================================================================

required = [
    "parameter",
    "prior_scaling",
    "posterior_mean",
    "posterior_sd",
    "posterior_lower_95",
    "posterior_upper_95",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# =============================================================================
# PARSE PARAMETER NAME
# =============================================================================

print()
print("Parsing parameter names...")

pattern = re.compile(r"^(OC\d+)_(\d{8})$")


def parse_parameter(parameter):
    parameter = str(parameter)

    match = pattern.match(parameter)

    if match is None:
        raise ValueError(
            f"Cannot parse parameter name '{parameter}'. "
            "Expected format such as OC1_20260105."
        )

    mine = match.group(1)
    date = pd.to_datetime(match.group(2), format="%Y%m%d")

    return mine, date


parsed = df["parameter"].apply(parse_parameter)

df["mine"] = parsed.apply(lambda x: x[0])
df["date"] = parsed.apply(lambda x: x[1])


# =============================================================================
# SORT
# =============================================================================

mine_order = [f"OC{i}" for i in range(1, 14)]

df["mine"] = pd.Categorical(
    df["mine"],
    categories=mine_order,
    ordered=True,
)

df = df.sort_values(["date", "mine"]).reset_index(drop=True)


# =============================================================================
# PRINT BASIC INFORMATION
# =============================================================================

print()
print("Mines:")
print(df["mine"].unique().tolist())

print()
print("Date range:")
print(
    df["date"].min().strftime("%Y-%m-%d"),
    "to",
    df["date"].max().strftime("%Y-%m-%d"),
)

print()
print("Number of dates:")
print(df["date"].nunique())


# =============================================================================
# CREATE LONG-FORM OUTPUT
# =============================================================================

long_columns = [
    "date",
    "mine",
    "parameter",
    "prior_scaling",
    "posterior_mean",
    "posterior_sd",
    "posterior_lower_95",
    "posterior_upper_95",
    "prior_emission_kg_s",
    "posterior_emission_kg_s",
    "posterior_emission_lower_kg_s",
    "posterior_emission_upper_kg_s",
]

long_df = df[long_columns].copy()

long_file = OUTPUT / "daily_scaling_factors_long.csv"

long_df.to_csv(long_file, index=False)

print()
print("Saved:")
print(long_file)


# =============================================================================
# CREATE WIDE TABLE
# =============================================================================

wide_mean = df.pivot(
    index="date",
    columns="mine",
    values="posterior_mean",
)

wide_mean = wide_mean.reindex(columns=mine_order)

wide_file = OUTPUT / "daily_scaling_factors_wide.csv"

wide_mean.to_csv(wide_file)

print()
print("Saved:")
print(wide_file)


# =============================================================================
# PRINT COMPLETE DAILY TABLE
# =============================================================================

print()
print("=" * 90)
print("DAILY POSTERIOR SCALING FACTORS")
print("=" * 90)

display_df = wide_mean.copy()

display_df.index = display_df.index.strftime("%Y-%m-%d")

print(
    display_df.to_string(
        float_format=lambda x: f"{x:10.4f}"
    )
)


# =============================================================================
# SUMMARY BY MINE
# =============================================================================

summary_rows = []

for mine in mine_order:

    sub = df[df["mine"] == mine]

    if len(sub) == 0:
        continue

    summary_rows.append(
        {
            "mine": mine,
            "n_days": len(sub),
            "mean_scaling": sub["posterior_mean"].mean(),
            "median_scaling": sub["posterior_mean"].median(),
            "min_scaling": sub["posterior_mean"].min(),
            "max_scaling": sub["posterior_mean"].max(),
            "mean_sd": sub["posterior_sd"].mean(),
            "min_lower95": sub["posterior_lower_95"].min(),
            "max_upper95": sub["posterior_upper_95"].max(),
            "prior_emission_kg_s": sub["prior_emission_kg_s"].iloc[0],
            "mean_posterior_emission_kg_s": sub["posterior_emission_kg_s"].mean(),
        }
    )

summary_df = pd.DataFrame(summary_rows)

summary_file = OUTPUT / "daily_scaling_summary.csv"

summary_df.to_csv(summary_file, index=False)

print()
print("=" * 90)
print("SUMMARY BY MINE")
print("=" * 90)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:12.5f}",
    )
)

print()
print("Saved:")
print(summary_file)


# =============================================================================
# PLOT 1: ALL 13 MINES
# =============================================================================

fig, ax = plt.subplots(figsize=(16, 9))

for mine in mine_order:

    sub = df[df["mine"] == mine]

    if len(sub) == 0:
        continue

    ax.plot(
        sub["date"],
        sub["posterior_mean"],
        linewidth=1.5,
        marker="o",
        markersize=3,
        label=mine,
    )


ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="Prior scaling = 1",
)

ax.axhline(
    0.0,
    linestyle=":",
    linewidth=1.2,
)

ax.set_title(
    "Upper Hunter CH4 — Daily Bayesian Mine Scaling Factors\n"
    "January 2026",
    fontsize=16,
)

ax.set_xlabel("Date")
ax.set_ylabel("Posterior scaling factor")

ax.grid(True, alpha=0.3)

ax.legend(
    ncol=3,
    fontsize=9,
    loc="upper left",
)

fig.autofmt_xdate()

fig.tight_layout()

file_all = OUTPUT / "all_13_mines_daily_scaling.png"

fig.savefig(
    file_all,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print()
print("Saved:")
print(file_all)


# =============================================================================
# PLOT 2: INDIVIDUAL MINES
# =============================================================================

fig, axes = plt.subplots(
    7,
    2,
    figsize=(16, 24),
    sharex=True,
)

axes = axes.flatten()

for i, mine in enumerate(mine_order):

    ax = axes[i]

    sub = df[df["mine"] == mine]

    ax.plot(
        sub["date"],
        sub["posterior_mean"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.axhline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_title(mine)

    ax.set_ylabel("Scaling")

    ax.grid(True, alpha=0.3)


# Hide unused 14th panel
axes[-1].axis("off")

fig.suptitle(
    "Upper Hunter CH4 — Daily Bayesian Mine Scaling Factors\n"
    "January 2026",
    fontsize=18,
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.98])

file_individual = OUTPUT / "individual_mine_daily_scaling.png"

fig.savefig(
    file_individual,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print()
print("Saved:")
print(file_individual)


# =============================================================================
# PLOT 3: INDIVIDUAL MINES WITH 95% CI
# =============================================================================

fig, axes = plt.subplots(
    7,
    2,
    figsize=(16, 24),
    sharex=True,
)

axes = axes.flatten()

for i, mine in enumerate(mine_order):

    ax = axes[i]

    sub = df[df["mine"] == mine]

    x = sub["date"]

    mean = sub["posterior_mean"].to_numpy()
    lower = sub["posterior_lower_95"].to_numpy()
    upper = sub["posterior_upper_95"].to_numpy()

    ax.plot(
        x,
        mean,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Posterior mean",
    )

    ax.fill_between(
        x,
        lower,
        upper,
        alpha=0.20,
        label="95% CI",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="Prior = 1",
    )

    ax.axhline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_title(mine)

    ax.set_ylabel("Scaling")

    ax.grid(True, alpha=0.3)


axes[-1].axis("off")

fig.suptitle(
    "Upper Hunter CH4 — Daily Bayesian Mine Scaling Factors\n"
    "Posterior mean and 95% credible interval — January 2026",
    fontsize=18,
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.98])

file_ci = OUTPUT / "individual_mine_daily_scaling_with_95CI.png"

fig.savefig(
    file_ci,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print()
print("Saved:")
print(file_ci)


# =============================================================================
# FINISHED
# =============================================================================

print()
print("=" * 90)
print("DONE")
print("=" * 90)

print()
print("Output directory:")
print(OUTPUT)

print()
print("Files:")
for f in sorted(OUTPUT.iterdir()):
    print(" ", f.name)

