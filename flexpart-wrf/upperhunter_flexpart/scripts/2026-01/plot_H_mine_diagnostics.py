#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

HFILE = (
    BASE
    / "hmatrix/daily_20260105_20260131"
    / "H_daily_13mine_1124x351_ppm.npy"
)

OUTDIR = (
    BASE
    / "hmatrix/daily_20260105_20260131"
    / "H_identifiability"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# READ H
# ================================================================

H = np.load(HFILE)

print("H shape:", H.shape)

if H.shape != (1124, 351):
    raise ValueError(f"Unexpected H shape: {H.shape}")


MINES = [f"OC{i}" for i in range(1, 14)]

N_MINE = 13
N_DAY = 27


# ================================================================
# BUILD AGGREGATED MINE H
# ================================================================
#
# H columns:
#
# OC1_day1, OC2_day1, ... OC13_day1,
# OC1_day2, OC2_day2, ... OC13_day2,
# ...
#
# For the mine-level correlation we sum the 27 daily
# sensitivities belonging to each mine.
#
# This gives:
#
# H_mine = 1124 observations x 13 mines
#
# ================================================================

H_mine = np.zeros((H.shape[0], N_MINE))

for m in range(N_MINE):

    mine_columns = H[:, m::N_MINE]

    print(
        f"{MINES[m]}: "
        f"{mine_columns.shape[1]} daily columns"
    )

    H_mine[:, m] = np.sum(mine_columns, axis=1)


# ================================================================
# MINE-TO-MINE CORRELATION
# ================================================================

corr = np.corrcoef(H_mine, rowvar=False)

corr_df = pd.DataFrame(
    corr,
    index=MINES,
    columns=MINES
)

corr_csv = OUTDIR / "mine_H_correlation_for_plot.csv"
corr_df.to_csv(corr_csv)


# ================================================================
# PLOT 1: CORRELATION HEATMAP
# ================================================================

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    corr,
    vmin=-1,
    vmax=1,
    aspect="equal"
)

ax.set_xticks(
    np.arange(N_MINE),
    MINES,
    rotation=45,
    ha="right"
)

ax.set_yticks(
    np.arange(N_MINE),
    MINES
)

ax.set_xlabel("Mine")
ax.set_ylabel("Mine")

ax.set_title(
    "Mine-to-mine correlation of aggregated H"
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Pearson correlation")

# Print correlation value inside every cell
for i in range(N_MINE):
    for j in range(N_MINE):

        ax.text(
            j,
            i,
            f"{corr[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

fig.tight_layout()

heatmap_file = (
    OUTDIR / "mine_H_correlation_heatmap.png"
)

fig.savefig(
    heatmap_file,
    dpi=250,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# PREPARE H VALUES FOR BOXPLOTS
# ================================================================
#
# Each mine:
#
# 27 daily columns x 1124 observations
#
# = 30,348 H values per mine.
#
# ZEROS ARE RETAINED in the first boxplot because zero sensitivity
# is physically meaningful.
# ================================================================

mine_values = []

for m in range(N_MINE):

    values = H[:, m::N_MINE].ravel()

    mine_values.append(values)

    print(
        f"{MINES[m]:5s}: "
        f"N={len(values):6d}, "
        f"zero={np.sum(values == 0):6d}, "
        f"positive={np.sum(values > 0):6d}, "
        f"max={np.max(values):.6e}"
    )


# ================================================================
# PLOT 2: BOXPLOT INCLUDING ZERO H
# ================================================================

fig, ax = plt.subplots(figsize=(12, 7))

ax.boxplot(
    mine_values,
    labels=MINES,
    showfliers=False
)

ax.set_xlabel("Mine")

ax.set_ylabel(
    "H (ppmv per unit daily scaling factor)"
)

ax.set_title(
    "Distribution of H by mine\n"
    "27 source-days × 1124 observations; zeros retained"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

box_file = (
    OUTDIR / "mine_H_boxplots.png"
)

fig.savefig(
    box_file,
    dpi=250,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# PLOT 3: LOG-SCALE BOXPLOT
# ================================================================
#
# H = 0 cannot be displayed on logarithmic scale.
#
# Therefore ONLY positive H values are retained.
#
# This plot is useful for comparing the magnitude of sensitivity
# between mines.
# ================================================================

positive_values = []

for values in mine_values:

    positive = values[values > 0]

    positive_values.append(positive)


fig, ax = plt.subplots(figsize=(12, 7))

ax.boxplot(
    positive_values,
    labels=MINES,
    showfliers=False
)

ax.set_yscale("log")

ax.set_xlabel("Mine")

ax.set_ylabel(
    "Positive H (ppmv per unit daily scaling factor)"
)

ax.set_title(
    "Positive H distribution by mine\n"
    "Log scale; zero-sensitivity values excluded"
)

ax.grid(
    axis="y",
    alpha=0.25,
    which="both"
)

fig.tight_layout()

box_log_file = (
    OUTDIR / "mine_H_boxplots_log.png"
)

fig.savefig(
    box_log_file,
    dpi=250,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# SUMMARY
# ================================================================

print()
print("=" * 80)
print("PLOTS CREATED")
print("=" * 80)

print()
print("Correlation heatmap:")
print(heatmap_file)

print()
print("Boxplot including zeros:")
print(box_file)

print()
print("Log-scale boxplot, positive H only:")
print(box_log_file)

print()
print("Correlation CSV:")
print(corr_csv)

print()
print("=" * 80)

