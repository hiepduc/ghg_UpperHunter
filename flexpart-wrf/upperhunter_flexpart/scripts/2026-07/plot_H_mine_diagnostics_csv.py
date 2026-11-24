
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H-matrix diagnostics using the FLEXPART-WRF CSV H matrix.

Input:
    H_flexpart_0_10m_13mine_Jul2026.csv
    H_flexpart_metadata_Jul2026.csv

Expected H matrix:
    1827 observations x 403 mine-day sensitivity columns

Diagnostics:
    1. Mine-to-mine correlation heatmap using aggregated daily H
    2. Boxplot of H including zero sensitivities
    3. Log-scale boxplot of positive H only
    4. Summary statistics for each mine
    5. Observation-level sensitivity diagnostics

All plots and diagnostic CSV files are saved to H_identifiability.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib

# Suitable for HPC systems without a graphical display.
#matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

BASE = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "flexpart-wrf/upperhunter_flexpart"
)

HFILE = (
    BASE
    / "H_matrix_flexpart"
    / "2026-07"
    / "H_flexpart_0_10m_13mine_Jul2026.csv"
)

METAFILE = (
    BASE
    / "H_matrix_flexpart"
    / "2026-07"
    / "H_flexpart_metadata_Jul2026.csv"
)

OUTDIR = (
    BASE
    / "H_matrix_flexpart"
    / "2026-07"
    / "H_identifiability"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

N_MINE = 13
N_DAY = 31
N_OBS_EXPECTED = 1827
N_H_EXPECTED = N_MINE * N_DAY

MINES = [f"OC{i}" for i in range(1, N_MINE + 1)]


# ================================================================
# READ CSV H MATRIX
# ================================================================

print("=" * 80)
print("READING CSV H MATRIX")
print("=" * 80)

if not HFILE.exists():
    raise FileNotFoundError(f"H matrix not found: {HFILE}")

Hdf = pd.read_csv(HFILE)

print("H file:", HFILE)
print("Full CSV shape:", Hdf.shape)
print("CSV columns:", Hdf.columns.tolist()[:10])

# Select only mine-day sensitivity columns.
# Examples: OC1_20260105, OC13_20260131.
pattern = re.compile(r"^OC(?:[1-9]|1[0-3])_\d{8}$")

hcols = [c for c in Hdf.columns if pattern.fullmatch(c)]

print("Number of H sensitivity columns:", len(hcols))

if len(hcols) != N_H_EXPECTED:
    raise ValueError(
        f"Expected {N_H_EXPECTED} H columns, found {len(hcols)}.\n"
        "Check the CSV column names and the mine/day naming convention."
    )

if len(Hdf) != N_OBS_EXPECTED:
    #raise ValueError(
    #    f"Expected {N_OBS_EXPECTED} observations, found {len(Hdf)}."
    #)
    print(
        f"Expected {N_OBS_EXPECTED} observations, found {len(Hdf)}."
    )

# Convert the selected H columns to numerical values.
Hdf[hcols] = Hdf[hcols].apply(pd.to_numeric, errors="raise")
H = Hdf[hcols].to_numpy(dtype=float)

if not np.isfinite(H).all():
    raise ValueError("H matrix contains NaN or infinite values.")

if (H < 0).any():
    raise ValueError("H matrix contains negative values.")

print("Numerical H shape:", H.shape)
print("Finite:", np.isfinite(H).all())
print("Minimum H:", H.min())
print("Maximum H:", H.max())
print("Non-zero elements:", np.count_nonzero(H))
print("Zero elements:", np.count_nonzero(H == 0))


# ================================================================
# VERIFY THE DAILY COLUMNS FOR EACH MINE
# ================================================================

print("\n" + "=" * 80)
print("VERIFYING MINE-DAY COLUMNS")
print("=" * 80)

mine_columns = {}

for mine in MINES:
    # The underscore avoids accidentally selecting OC10 when selecting OC1.
    cols = [c for c in hcols if c.startswith(mine + "_")]

    if len(cols) != N_DAY:
        raise ValueError(
            f"{mine}: expected {N_DAY} daily columns, found {len(cols)}."
        )

    mine_columns[mine] = cols

    dates = [c.split("_", 1)[1] for c in cols]

    print(
        f"{mine:5s}: {len(cols):2d} daily columns, "
        f"{dates[0]} to {dates[-1]}"
    )

print("\nAll mines have the expected number of daily columns.")


# ================================================================
# AGGREGATE DAILY H FOR EACH MINE
# ================================================================
#
# H_mine[:, m] = sum of the 28 daily H sensitivities for mine m.
#
# Shape:
#     1827 observations x 13 mines
#
# This is useful for examining whether the mines have similar
# overall sensitivity patterns across the observations.
# ================================================================

H_mine_df = pd.DataFrame(index=Hdf.index)

for mine in MINES:
    H_mine_df[mine] = Hdf[mine_columns[mine]].sum(axis=1)

H_mine = H_mine_df.to_numpy(dtype=float)

print("\nAggregated mine H shape:", H_mine.shape)

aggregated_file = OUTDIR / "H_aggregated_by_mine.csv"
H_mine_df.to_csv(aggregated_file, index=False)


# ================================================================
# MINE-TO-MINE CORRELATION
# ================================================================

# pandas correlation returns NaN for a column with zero variance.
# Such columns must be investigated rather than assigned an
# artificial correlation.
corr_df = H_mine_df.corr(method="pearson")
corr = corr_df.to_numpy()

corr_csv = OUTDIR / "mine_H_correlation_for_plot.csv"
corr_df.to_csv(corr_csv)

print("\n" + "=" * 80)
print("MINE-TO-MINE CORRELATION")
print("=" * 80)

print(corr_df.round(3).to_string())

constant_mines = [
    mine for mine in MINES
    if H_mine_df[mine].std() == 0
]

if constant_mines:
    print(
        "\nWARNING: Zero-variance aggregated H for:",
        ", ".join(constant_mines)
    )
    print("Correlations involving these mines may be NaN.")


# ================================================================
# PLOT 1: MINE CORRELATION HEATMAP
# ================================================================

fig, ax = plt.subplots(figsize=(11, 9))

im = ax.imshow(
    corr,
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
    aspect="equal"
)

ax.set_xticks(np.arange(N_MINE))
ax.set_xticklabels(MINES, rotation=45, ha="right")

ax.set_yticks(np.arange(N_MINE))
ax.set_yticklabels(MINES)

ax.set_xlabel("Mine")
ax.set_ylabel("Mine")
ax.set_title(
    "Mine-to-mine correlation of aggregated H\n"
    "Daily sensitivities summed across 28 days"
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Pearson correlation")

for i in range(N_MINE):
    for j in range(N_MINE):
        value = corr[i, j]

        label = "NaN" if not np.isfinite(value) else f"{value:.2f}"

        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=8
        )

fig.tight_layout()

heatmap_file = OUTDIR / "mine_H_correlation_heatmap.png"
fig.savefig(heatmap_file, dpi=250, bbox_inches="tight")
plt.show()
plt.close(fig)


# ================================================================
# PREPARE H VALUES FOR BOXPLOTS
# ================================================================
#
# Each mine has:
#     1827 observations x 31 daily sensitivity columns
#     = 56,637 values
#
# Zeros are retained in the linear-scale boxplot.
# ================================================================

mine_values = {}

print("\n" + "=" * 80)
print("H DISTRIBUTION BY MINE for July 2026")
print("=" * 80)

summary_rows = []

for mine in MINES:
    values = Hdf[mine_columns[mine]].to_numpy(dtype=float).ravel()
    positive = values[values > 0]

    mine_values[mine] = values

    summary_rows.append({
        "mine": mine,
        "n_values": len(values),
        "n_zero": int(np.count_nonzero(values == 0)),
        "n_positive": int(np.count_nonzero(values > 0)),
        "fraction_zero": float(np.mean(values == 0)),
        "mean_all": float(np.mean(values)),
        "median_all": float(np.median(values)),
        "max_all": float(np.max(values)),
        "mean_positive": (
            float(np.mean(positive)) if len(positive) else np.nan
        ),
        "median_positive": (
            float(np.median(positive)) if len(positive) else np.nan
        ),
        "max_positive": (
            float(np.max(positive)) if len(positive) else np.nan
        ),
    })

    print(
        f"{mine:5s}: "
        f"N={len(values):6d}, "
        f"zero={np.count_nonzero(values == 0):6d}, "
        f"positive={len(positive):6d}, "
        f"fraction_zero={np.mean(values == 0):.3f}, "
        f"max={np.max(values):.6e}"
    )

summary_df = pd.DataFrame(summary_rows)
summary_file = OUTDIR / "H_mine_summary_statistics.csv"
summary_df.to_csv(summary_file, index=False)


# ================================================================
# PLOT 2: BOXPLOT INCLUDING ZERO H
# ================================================================

fig, ax = plt.subplots(figsize=(13, 7))

ax.boxplot(
    [mine_values[mine] for mine in MINES],
    labels=MINES,
    showfliers=False
)

ax.set_xlabel("Mine")
ax.set_ylabel("H (ppmv per unit daily scaling factor)")
ax.set_title(
    "Distribution of H by mine\n"
    "31 source-days × 1827 observations; zeros retained"
)

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

box_file = OUTDIR / "mine_H_boxplots.png"
fig.savefig(box_file, dpi=250, bbox_inches="tight")
plt.show()
plt.close(fig)


# ================================================================
# PLOT 3: LINEAR-SCALE BOXPLOT OF POSITIVE H
# ================================================================
#
# Only positive H values are retained.
# This avoids the large number of zeros compressing the boxes.
# The vertical axis remains linear.
# ================================================================

positive_values = []

for mine in MINES:
    values = mine_values[mine]
    positive = values[values > 0]
    positive_values.append(positive)

    print(
        f"{mine:5s}: positive median="
        f"{np.median(positive):.6e}, "
        f"positive mean={np.mean(positive):.6e}"
        if len(positive) > 0
        else f"{mine:5s}: no positive H values"
    )

fig, ax = plt.subplots(figsize=(13, 7))

ax.boxplot(
    positive_values,
    labels=MINES,
    showfliers=False
)

ax.set_xlabel("Mine")
ax.set_ylabel(
    "Positive H (ppmv per unit daily scaling factor)"
)
ax.set_title(
    "Positive H distribution by mine for July 2026\n"
    "Linear scale; zero-sensitivity values excluded"
)

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

box_positive_linear_file = (
    OUTDIR / "mine_H_boxplots_positive_linear.png"
)

fig.savefig(
    box_positive_linear_file,
    dpi=250,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# PLOT 4: LOG-SCALE BOXPLOT OF POSITIVE H
# ================================================================
#
# The same positive values are shown on a logarithmic scale.
# This is useful when mine sensitivities differ greatly in magnitude.
# ================================================================

plot_values = [
    values if len(values) else np.array([np.nan])
    for values in positive_values
]

fig, ax = plt.subplots(figsize=(13, 7))

ax.boxplot(
    plot_values,
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

ax.grid(axis="y", alpha=0.25, which="both")

fig.tight_layout()

box_positive_log_file = (
    OUTDIR / "mine_H_boxplots_positive_log.png"
)

fig.savefig(
    box_positive_log_file,
    dpi=250,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# OBSERVATION-LEVEL SENSITIVITY DIAGNOSTICS
# ================================================================

obs_summary = Hdf.drop(columns=hcols).copy()

obs_summary["n_nonzero_H"] = np.count_nonzero(H, axis=1)
obs_summary["max_H"] = H.max(axis=1)
obs_summary["sum_H"] = H.sum(axis=1)
obs_summary["zero_sensitivity"] = np.all(H == 0, axis=1)

obs_file = OUTDIR / "H_observation_sensitivity_diagnostics.csv"
obs_summary.to_csv(obs_file, index=False)

n_zero_rows = int(obs_summary["zero_sensitivity"].sum())

print("\n" + "=" * 80)
print("OBSERVATION-LEVEL SUMMARY")
print("=" * 80)

print("Observations:", len(H))
print("Rows with any non-zero H:", len(H) - n_zero_rows)
print("Rows with all-zero H:", n_zero_rows)
print("Fraction all-zero:", n_zero_rows / len(H))

print("\nFirst 20 all-zero observation rows:")
print(
    obs_summary.loc[
        obs_summary["zero_sensitivity"]
    ].head(20).to_string(index=True)
)


# ================================================================
# OPTIONAL: VERIFY SEPARATE METADATA FILE
# ================================================================

if METAFILE.exists():
    meta = pd.read_csv(METAFILE)

    print("\nSeparate metadata file:", METAFILE)
    print("Metadata shape:", meta.shape)
    print("Metadata columns:", meta.columns.tolist())

    if len(meta) != len(Hdf):
        print(
            "WARNING: Metadata row count differs from H matrix row count."
        )
    else:
        print("Metadata row count matches H matrix.")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("H MATRIX DIAGNOSTICS COMPLETE")
print("=" * 80)

print("\nOutput directory:")
print(OUTDIR)

print("\nCreated files:")
for path in [
    heatmap_file,
    box_file,
    box_positive_linear_file,
    box_positive_log_file,
    corr_csv,
    aggregated_file,
    summary_file,
    obs_file,
]:
    print(path)

print("\nDone.")


