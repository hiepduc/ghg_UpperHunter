#!/usr/bin/env python3

"""
Plot posterior daily emission scaling factors for the 13 Upper Hunter
open-cut mines for February 2026.

Input:
    Bayesian_daily_inversion_feb2026/daily_scaling_factors_Feb2026.csv

For each mine:
    - posterior mean scaling factor
    - 95% credible interval
    - prior/current inventory = 1.0

Output:
    posterior_daily_scaling_factors_Feb2026.png
    posterior_daily_scaling_factors_Feb2026.pdf
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-02"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "Bayesian_daily_inversion_Feb2026",
    "daily_scaling_factors_Feb2026.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Bayesian_daily_inversion_Feb2026"
)

PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "posterior_daily_scaling_factors_Feb2026.png"
)

PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "posterior_daily_scaling_factors_Feb2026.pdf"
)


# ============================================================
# MINE LIST
# ============================================================

MINES = [f"OC{i}" for i in range(1, 14)]


# ============================================================
# READ DATA
# ============================================================

print("=" * 80)
print("Reading posterior daily scaling factors")
print("=" * 80)

print(f"Input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"Number of days: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["date"]

for mine in MINES:
    required_columns.extend([
        mine,
        f"{mine}_std",
        f"{mine}_lower95",
        f"{mine}_upper95"
    ])

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise RuntimeError(
        "The following required columns are missing from the CSV:\n"
        + "\n".join(missing)
    )


# ============================================================
# FIND COMMON Y LIMIT
# ============================================================

all_lower = []
all_upper = []

for mine in MINES:
    all_lower.extend(df[f"{mine}_lower95"].values)
    all_upper.extend(df[f"{mine}_upper95"].values)

all_lower = np.asarray(all_lower, dtype=float)
all_upper = np.asarray(all_upper, dtype=float)

ymin_data = np.nanmin(all_lower)
ymax_data = np.nanmax(all_upper)

# Add some margin
yrange = ymax_data - ymin_data

if yrange <= 0:
    yrange = 1.0

ymin = min(0.0, ymin_data - 0.05 * yrange)
ymax = ymax_data + 0.05 * yrange

# Make sure 1.0 is comfortably inside the plot
ymin = min(ymin, 0.8)
ymax = max(ymax, 1.2)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(17, 13),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


# ============================================================
# PLOT EACH MINE
# ============================================================

for i, mine in enumerate(MINES):

    ax = axes[i]

    mean = df[mine].to_numpy(dtype=float)
    lower = df[f"{mine}_lower95"].to_numpy(dtype=float)
    upper = df[f"{mine}_upper95"].to_numpy(dtype=float)

    # 95% credible interval
    ax.fill_between(
        df["date"],
        lower,
        upper,
        alpha=0.25,
        label="95% credible interval"
    )

    # Posterior mean
    ax.plot(
        df["date"],
        mean,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Posterior mean"
    )

    # Prior / current inventory
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Prior = 1.0"
    )

    ax.set_title(
        mine,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylim(ymin, ymax)

    ax.grid(
        True,
        alpha=0.3,
        linewidth=0.7
    )

    ax.tick_params(
        axis="both",
        labelsize=9
    )

    # Only show y-axis label on left column
    if i % 4 == 0:
        ax.set_ylabel(
            "Emission scaling factor",
            fontsize=10
        )


# ============================================================
# REMOVE UNUSED 3 PANELS
# ============================================================

for i in range(13, 16):
    axes[i].axis("off")


# ============================================================
# X AXIS
# ============================================================

# February 2026: show every 5 days
locator = mdates.DayLocator(interval=5)
formatter = mdates.DateFormatter("%d Feb")

for ax in axes[:13]:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

# Bottom row containing actual plots
for ax in axes[8:13]:
    ax.tick_params(
        axis="x",
        rotation=45
    )


# ============================================================
# COMMON TITLE
# ============================================================

fig.suptitle(
    "Upper Hunter CH$_4$ Inverse Modelling\n"
    "Posterior Daily Emission Scaling Factors — February 2026",
    fontsize=17,
    fontweight="bold",
    y=0.985
)


# ============================================================
# COMMON X LABEL
# ============================================================

fig.supxlabel(
    "Date (February 2026)",
    fontsize=12
)


# ============================================================
# COMMON LEGEND
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=3,
    fontsize=10,
    frameon=True
)


# ============================================================
# LAYOUT
# ============================================================

plt.subplots_adjust(
    left=0.065,
    right=0.985,
    bottom=0.075,
    top=0.88,
    hspace=0.35,
    wspace=0.12
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    PNG_FILE,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    PDF_FILE,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 80)
print("PLOT COMPLETE")
print("=" * 80)

print(f"PNG: {PNG_FILE}")
print(f"PDF: {PDF_FILE}")
print()

print("Posterior scaling-factor ranges:")
print()

for mine in MINES:

    mean = df[mine].to_numpy(dtype=float)
    lower = df[f"{mine}_lower95"].to_numpy(dtype=float)
    upper = df[f"{mine}_upper95"].to_numpy(dtype=float)

    print(
        f"{mine:5s}  "
        f"mean range = {np.nanmin(mean):.3f} - {np.nanmax(mean):.3f}   "
        f"95% range = {np.nanmin(lower):.3f} - {np.nanmax(upper):.3f}"
    )

print("=" * 80)

