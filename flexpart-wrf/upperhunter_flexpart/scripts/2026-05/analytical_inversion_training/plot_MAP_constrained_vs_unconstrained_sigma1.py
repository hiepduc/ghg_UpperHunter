
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare unconstrained and constrained MAP emission scaling factors
for the 13 Upper Hunter open-cut coal mines.

Period: 1-31 May 2026
Parameters: 13 mines x 31 days = 403 scaling factors

Plots:
    - One panel per mine
    - Unconstrained MAP
    - Constrained MAP (s >= 0)
    - Prior scaling factor = 1

Input files:
    analytical_posterior_scaling_factors_sigma1.csv
    analytical_posterior_scaling_factors_sigma1_constrained.csv

Output:
    MAP_constrained_vs_unconstrained_May2026.png
    MAP_constrained_vs_unconstrained_May2026.pdf
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# SETTINGS
# =============================================================================

BASE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "flexpart-wrf/upperhunter_flexpart"
)

RESULT_DIR = os.path.join(
    BASE,
    "inversion_results",
    "2026-05"
)

UNCONSTRAINED_FILE = os.path.join(
    RESULT_DIR,
    "analytical_posterior_scaling_factors_sigma1.csv"
)

CONSTRAINED_FILE = os.path.join(
    RESULT_DIR,
    "analytical_posterior_scaling_factors_sigma1_constrained.csv"
)

OUTPUT_PNG = os.path.join(
    RESULT_DIR,
    "MAP_constrained_vs_unconstrained_May2026.png"
)

OUTPUT_PDF = os.path.join(
    RESULT_DIR,
    "MAP_constrained_vs_unconstrained_May2026.pdf"
)

MINES = [f"OC{i}" for i in range(1, 14)]

PRIOR_SCALING = 1.0

START_DATE = "2026-05-01"
END_DATE = "2026-05-31"


# =============================================================================
# READ AND STANDARDISE INPUT FILES
# =============================================================================

def read_map_file(filename, label):

    print("\nReading:", label)
    print(filename)

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Cannot find {label} file:\n{filename}"
        )

    df = pd.read_csv(filename)

    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    required = ["parameter", "posterior_scaling"]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"{label} file is missing column: {col}"
            )

    df = df[["parameter", "posterior_scaling"]].copy()

    df["parameter"] = df["parameter"].astype(str)

    df["posterior_scaling"] = pd.to_numeric(
        df["posterior_scaling"],
        errors="coerce"
    )

    if df["posterior_scaling"].isna().any():
        raise RuntimeError(
            f"Invalid posterior_scaling values in {label} file."
        )

    if df["parameter"].duplicated().any():
        raise RuntimeError(
            f"Duplicate parameter names in {label} file."
        )

    # Extract mine and date from names such as OC1_20260305
    extracted = df["parameter"].str.extract(
        r"^(OC\d+)_(\d{8})$"
    )

    if extracted.isna().any().any():
        bad = df.loc[
            extracted.isna().any(axis=1),
            "parameter"
        ].tolist()

        raise RuntimeError(
            f"Could not interpret parameter names in {label}:\n"
            f"{bad[:10]}"
        )

    df["mine"] = extracted[0]
    df["date"] = pd.to_datetime(
        extracted[1],
        format="%Y%m%d"
    )

    return df


unconstrained = read_map_file(
    UNCONSTRAINED_FILE,
    "UNCONSTRAINED MAP"
)

constrained = read_map_file(
    CONSTRAINED_FILE,
    "CONSTRAINED MAP"
)


# =============================================================================
# CHECK PARAMETER CONSISTENCY
# =============================================================================

print("\n" + "=" * 80)
print("CHECKING PARAMETER CONSISTENCY")
print("=" * 80)

if len(unconstrained) != 403:
    raise RuntimeError(
        f"Expected 403 unconstrained parameters, got {len(unconstrained)}"
    )

if len(constrained) != 403:
    raise RuntimeError(
        f"Expected 403 constrained parameters, got {len(constrained)}"
    )

u_parameters = set(unconstrained["parameter"])
c_parameters = set(constrained["parameter"])

if u_parameters != c_parameters:

    missing_constrained = sorted(
        u_parameters - c_parameters
    )

    missing_unconstrained = sorted(
        c_parameters - u_parameters
    )

    raise RuntimeError(
        "The two MAP files do not contain identical parameters.\n"
        f"Missing from constrained: {missing_constrained[:10]}\n"
        f"Missing from unconstrained: {missing_unconstrained[:10]}"
    )

# Reorder constrained parameters to match unconstrained
constrained = constrained.set_index(
    "parameter"
).loc[
    unconstrained["parameter"]
].reset_index()

print("Unconstrained parameters:", len(unconstrained))
print("Constrained parameters:", len(constrained))
print("Parameter ordering: MATCHED")


# =============================================================================
# BUILD DAILY MINE DATA
# =============================================================================

unconstrained_wide = unconstrained.pivot(
    index="date",
    columns="mine",
    values="posterior_scaling"
)

constrained_wide = constrained.pivot(
    index="date",
    columns="mine",
    values="posterior_scaling"
)

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

unconstrained_wide = unconstrained_wide.reindex(
    index=dates,
    columns=MINES
)

constrained_wide = constrained_wide.reindex(
    index=dates,
    columns=MINES
)

if unconstrained_wide.isna().any().any():
    raise RuntimeError(
        "Missing values in unconstrained MAP data."
    )

if constrained_wide.isna().any().any():
    raise RuntimeError(
        "Missing values in constrained MAP data."
    )

print("\nDaily data shape:", unconstrained_wide.shape)


# =============================================================================
# CHECK NON-NEGATIVITY OF CONSTRAINED SOLUTION
# =============================================================================

minimum_constrained = constrained_wide.to_numpy().min()

print("\nMinimum constrained scaling factor:", minimum_constrained)

if minimum_constrained < -1e-8:
    raise RuntimeError(
        "The constrained MAP file contains negative scaling factors."
    )


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("MAP COMPARISON SUMMARY")
print("=" * 80)

for mine in MINES:

    u = unconstrained_wide[mine].to_numpy()
    c = constrained_wide[mine].to_numpy()

    print(
        f"{mine:5s} | "
        f"Unconstrained mean = {np.mean(u):9.4f} | "
        f"Constrained mean = {np.mean(c):9.4f} | "
        f"Difference = {np.mean(c-u):9.4f} | "
        f"Minimum unconstrained = {np.min(u):9.4f}"
    )


# =============================================================================
# PLOT: 13 PANELS
# =============================================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, axes = plt.subplots(
    nrows=4,
    ncols=4,
    figsize=(18, 13),
    sharex=True
)

axes = axes.flatten()

for i, mine in enumerate(MINES):

    ax = axes[i]

    u = unconstrained_wide[mine]
    c = constrained_wide[mine]

    # Unconstrained MAP
    ax.plot(
        dates,
        u.values,
        color="blue",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Unconstrained MAP"
    )

    # Constrained MAP
    ax.plot(
        dates,
        c.values,
        color="red",
        linewidth=1.8,
        marker="s",
        markersize=3,
        linestyle="--",
        label="Constrained MAP"
    )

    # Prior scaling factor
    ax.axhline(
        PRIOR_SCALING,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Prior = 1"
    )

    ax.set_title(mine, fontweight="bold")

    ax.set_ylabel("Emission scaling factor")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=5)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d Feb")
    )

    ax.set_xlim(
        dates[0],
        dates[-1]
    )

    # Ensure the y-axis includes the prior reference
    y_min = min(
        float(u.min()),
        float(c.min()),
        PRIOR_SCALING
    )

    y_max = max(
        float(u.max()),
        float(c.max()),
        PRIOR_SCALING
    )

    margin = max(
        0.08 * (y_max - y_min),
        0.05
    )

    ax.set_ylim(
        y_min - margin,
        y_max + margin
    )


# Remove the unused 14th, 15th and 16th panels
for j in range(len(MINES), len(axes)):
    fig.delaxes(axes[j])


# =============================================================================
# COMMON LEGEND
# =============================================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=3,
    frameon=True
)


# =============================================================================
# FIGURE TITLE AND LAYOUT
# =============================================================================

fig.suptitle(
    "Upper Hunter FLEXPART-WRF: Daily Mine Emission Scaling Factors\n"
    "Constrained MAP versus Unconstrained MAP | 1-31 May 2026",
    fontsize=15,
    fontweight="bold",
    y=0.995
)

fig.supxlabel(
    "Date (May 2026)",
    fontsize=12
)

plt.tight_layout(
    rect=[0.02, 0.04, 0.98, 0.91]
)


# =============================================================================
# SAVE
# =============================================================================

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

print("\n" + "=" * 80)
print("PLOTS SAVED")
print("=" * 80)

print("PNG:", OUTPUT_PNG)
print("PDF:", OUTPUT_PDF)

plt.show()

print("\nDONE.")

