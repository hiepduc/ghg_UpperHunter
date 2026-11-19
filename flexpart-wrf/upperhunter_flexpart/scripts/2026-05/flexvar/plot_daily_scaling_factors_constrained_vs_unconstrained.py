#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart"

UNCONSTRAINED_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_unconstrained"
)

CONSTRAINED_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_constrained"
)

UNCONSTRAINED_POSTERIOR = (
    UNCONSTRAINED_DIR +
    "/4dvar_posterior_scaling_factors.csv"
)

CONSTRAINED_POSTERIOR = (
    CONSTRAINED_DIR +
    "/4dvar_posterior_scaling_factors.csv"
)

OUTPUT_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_daily_scaling_comparison"
)

INDIVIDUAL_DIR = OUTPUT_DIR + "/individual_mines"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MINES = [f"OC{i}" for i in range(1, 14)]

DATES = pd.date_range(
    start="2026-01-05",
    end="2026-01-31",
    freq="D",
)

N_MINES = 13
N_DAYS = 27
N_PARAMETERS = N_MINES * N_DAYS

PRIOR_SCALING = 1.0


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("4D-VAR DAILY SCALING FACTORS")
print("CONSTRAINED vs UNCONSTRAINED")
print("=" * 90)

print()
print("Unconstrained:")
print(UNCONSTRAINED_POSTERIOR)

print()
print("Constrained:")
print(CONSTRAINED_POSTERIOR)

print()
print("Output:")
print(OUTPUT_DIR)


# ============================================================
# READ POSTERIOR FILES
# ============================================================

print()
print("=" * 90)
print("READING POSTERIOR FILES")
print("=" * 90)

unc = pd.read_csv(UNCONSTRAINED_POSTERIOR)
con = pd.read_csv(CONSTRAINED_POSTERIOR)

print()
print("Unconstrained shape:", unc.shape)
print("Constrained shape  :", con.shape)

print()
print("Unconstrained columns:")
print(list(unc.columns))

print()
print("Constrained columns:")
print(list(con.columns))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "parameter",
    "posterior_scaling",
}

for name, df in [
    ("unconstrained", unc),
    ("constrained", con),
]:
    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name} posterior is missing columns: {missing}"
        )


# ============================================================
# CHECK PARAMETER COUNTS
# ============================================================

if len(unc) != N_PARAMETERS:
    raise RuntimeError(
        f"Expected {N_PARAMETERS} unconstrained parameters, "
        f"found {len(unc)}"
    )

if len(con) != N_PARAMETERS:
    raise RuntimeError(
        f"Expected {N_PARAMETERS} constrained parameters, "
        f"found {len(con)}"
    )


# ============================================================
# CHECK PARAMETER ORDER
# ============================================================

if not np.array_equal(
    unc["parameter"].astype(str).to_numpy(),
    con["parameter"].astype(str).to_numpy(),
):
    raise RuntimeError(
        "Parameter ordering differs between constrained "
        "and unconstrained posterior files."
    )


parameters = unc["parameter"].astype(str).to_numpy()

print()
print("Number of parameters:", len(parameters))

print()
print("First 13 parameters:")
print(parameters[:13])

print()
print("Last 13 parameters:")
print(parameters[-13:])


# ============================================================
# EXTRACT SCALING FACTORS
# ============================================================

unc_values = unc["posterior_scaling"].to_numpy(dtype=float)
con_values = con["posterior_scaling"].to_numpy(dtype=float)

print()
print("=" * 90)
print("POSTERIOR SCALING FACTOR CHECK")
print("=" * 90)

print()
print("Unconstrained:")
print("  min    :", unc_values.min())
print("  max    :", unc_values.max())
print("  mean   :", unc_values.mean())
print("  median :", np.median(unc_values))

print()
print("Constrained:")
print("  min    :", con_values.min())
print("  max    :", con_values.max())
print("  mean   :", con_values.mean())
print("  median :", np.median(con_values))

below_zero = np.sum(con_values < 0)
above_three = np.sum(con_values > 3)

print()
print("Constrained below 0:", below_zero)
print("Constrained above 3:", above_three)

if below_zero > 0 or above_three > 0:
    raise RuntimeError(
        "Constrained posterior contains factors outside [0, 3]."
    )


# ============================================================
# BUILD DAILY DATAFRAME
#
# Parameter ordering is:
#
# OC1_20260105
# OC2_20260105
# ...
# OC13_20260105
# OC1_20260106
# ...
# ============================================================

rows = []

for day_index, date in enumerate(DATES):

    date_string = date.strftime("%Y%m%d")

    for mine_index, mine in enumerate(MINES):

        parameter_index = (
            day_index * N_MINES
            + mine_index
        )

        expected_parameter = (
            f"{mine}_{date_string}"
        )

        actual_parameter = parameters[parameter_index]

        if actual_parameter != expected_parameter:
            raise RuntimeError(
                "Parameter ordering mismatch:\n"
                f"Expected: {expected_parameter}\n"
                f"Found:    {actual_parameter}\n"
                f"Index:    {parameter_index}"
            )

        rows.append(
            {
                "date": date,
                "mine": mine,
                "parameter": actual_parameter,
                "unconstrained": unc_values[parameter_index],
                "constrained": con_values[parameter_index],
                "prior": PRIOR_SCALING,
            }
        )

daily = pd.DataFrame(rows)

print()
print("=" * 90)
print("DAILY DATAFRAME")
print("=" * 90)

print()
print("Shape:", daily.shape)

print()
print(daily.head(15).to_string(index=False))


# ============================================================
# SAVE LONG-FORM DAILY DATA
# ============================================================

daily_csv = (
    OUTPUT_DIR +
    "/daily_scaling_factors_constrained_vs_unconstrained.csv"
)

daily.to_csv(
    daily_csv,
    index=False,
)

print()
print("Saved:")
print(daily_csv)


# ============================================================
# SAVE WIDE DAILY DATA
# ============================================================

wide = daily.pivot(
    index="date",
    columns="mine",
    values=[
        "unconstrained",
        "constrained",
    ],
)

wide_csv = (
    OUTPUT_DIR +
    "/daily_scaling_factors_wide.csv"
)

wide.to_csv(wide_csv)

print()
print("Saved:")
print(wide_csv)


# ============================================================
# INDIVIDUAL MINE PLOTS
# ============================================================

print()
print("=" * 90)
print("CREATING INDIVIDUAL MINE PLOTS")
print("=" * 90)

for mine in MINES:

    df = daily[
        daily["mine"] == mine
    ].sort_values("date")

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    ax.plot(
        df["date"],
        df["unconstrained"],
        "-o",
        linewidth=1.8,
        markersize=4,
        label="4D-Var unconstrained",
    )

    ax.plot(
        df["date"],
        df["constrained"],
        "-o",
        linewidth=1.8,
        markersize=4,
        label="4D-Var constrained",
    )

    ax.axhline(
        PRIOR_SCALING,
        linestyle="--",
        linewidth=1.3,
        label="Prior = 1.0",
    )

    ax.set_title(
        f"{mine} daily posterior scaling factor — January 2026"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Scaling factor")

    ax.set_xlim(
        DATES.min(),
        DATES.max(),
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    outfile = (
        INDIVIDUAL_DIR +
        f"/{mine}_daily_scaling_factors.png"
    )

    plt.savefig(
        outfile,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("Saved:", outfile)


# ============================================================
# 13-MINE OVERVIEW
# ============================================================

print()
print("=" * 90)
print("CREATING 13-MINE OVERVIEW")
print("=" * 90)

fig, axes = plt.subplots(
    7,
    2,
    figsize=(15, 24),
    sharex=True,
)

axes = axes.flatten()

for i, mine in enumerate(MINES):

    ax = axes[i]

    df = daily[
        daily["mine"] == mine
    ].sort_values("date")

    ax.plot(
        df["date"],
        df["unconstrained"],
        "-o",
        linewidth=1.3,
        markersize=2.8,
        label="Unconstrained",
    )

    ax.plot(
        df["date"],
        df["constrained"],
        "-o",
        linewidth=1.3,
        markersize=2.8,
        label="Constrained",
    )

    ax.axhline(
        PRIOR_SCALING,
        linestyle="--",
        linewidth=1.0,
        label="Prior = 1",
    )

    ax.set_title(
        mine,
        fontsize=11,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Scaling factor"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.set_xlim(
        DATES.min(),
        DATES.max(),
    )

# Hide unused 14th subplot
for i in range(N_MINES, len(axes)):
    axes[i].axis("off")

# Put legend only on first panel
axes[0].legend(
    fontsize=8,
    loc="best",
)

fig.suptitle(
    "Upper Hunter CH4 — Daily posterior scaling factors\n"
    "4D-Var constrained vs unconstrained — January 2026",
    fontsize=16,
    fontweight="bold",
)

fig.supxlabel(
    "Date",
    fontsize=12,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

overview_file = (
    OUTPUT_DIR +
    "/ALL_13_MINES_daily_scaling_factors.png"
)

plt.savefig(
    overview_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()
plt.close()

print()
print("Saved:")
print(overview_file)


# ============================================================
# COMBINED OVERLAY — ALL MINES
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 7)
)

for mine in MINES:

    df = daily[
        daily["mine"] == mine
    ].sort_values("date")

    ax.plot(
        df["date"],
        df["constrained"],
        "-",
        linewidth=1.0,
        alpha=0.65,
        label=mine,
    )

ax.axhline(
    PRIOR_SCALING,
    linestyle="--",
    linewidth=1.5,
    label="Prior = 1.0",
)

ax.set_title(
    "Constrained 4D-Var daily scaling factors — all 13 mines"
)

ax.set_xlabel("Date")
ax.set_ylabel("Scaling factor")

ax.set_xlim(
    DATES.min(),
    DATES.max(),
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    ncol=2,
    fontsize=8,
)

fig.autofmt_xdate()

plt.tight_layout()

all_constrained_file = (
    OUTPUT_DIR +
    "/ALL_13_MINES_constrained_daily_scaling_factors.png"
)

plt.savefig(
    all_constrained_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()
plt.close()

print()
print("Saved:")
print(all_constrained_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 90)
print("COMPLETED")
print("=" * 90)

print()
print("Individual plots:")
print(INDIVIDUAL_DIR)

print()
print("Overview:")
print(overview_file)

print()
print("Daily CSV:")
print(daily_csv)

print()
print("Wide CSV:")
print(wide_csv)

print()
print("Constrained all-mine plot:")
print(all_constrained_file)

print()
print("=" * 90)

