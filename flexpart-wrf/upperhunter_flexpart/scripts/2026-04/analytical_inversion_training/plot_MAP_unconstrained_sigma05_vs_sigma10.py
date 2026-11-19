
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare unconstrained MAP scaling factors for two prior uncertainties.

Upper Hunter FLEXPART-WRF
Period: 1-30 April 2026
13 mines x 30 days = 390 parameters

Comparison:
    Prior relative error = 0.5 (50%)
    Prior relative error = 1.0 (100%)

Both solutions are UNCONSTRAINED MAP estimates.

Outputs:
    MAP_unconstrained_sigma05_vs_sigma10_Apr2026.png
    MAP_unconstrained_sigma05_vs_sigma10_Apr2026.pdf
"""

import os
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
    "2026-04"
)

# Change these filenames if your actual output names differ.
FILE_SIGMA05 = os.path.join(
    RESULT_DIR,
    "analytical_posterior_scaling_factors.csv"
)

FILE_SIGMA10 = os.path.join(
    RESULT_DIR,
    "analytical_posterior_scaling_factors_sigma1.csv"
)

OUTPUT_PNG = os.path.join(
    RESULT_DIR,
    "MAP_unconstrained_sigma05_vs_sigma10_Apr2026.png"
)

OUTPUT_PDF = os.path.join(
    RESULT_DIR,
    "MAP_unconstrained_sigma05_vs_sigma10_Apr2026.pdf"
)

MINES = [f"OC{i}" for i in range(1, 14)]

PRIOR_SCALING = 1.0

START_DATE = "2026-04-01"
END_DATE = "2026-04-30"


# =============================================================================
# READ MAP FILE
# =============================================================================

def read_map_file(filename, label):

    print("\nReading:", label)
    print("File:", filename)

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Cannot find {label} file:\n{filename}"
        )

    df = pd.read_csv(filename)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    required = ["parameter", "posterior_scaling"]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing column '{col}' in {label} file."
            )

    df = df[required].copy()

    df["parameter"] = df["parameter"].astype(str)

    df["posterior_scaling"] = pd.to_numeric(
        df["posterior_scaling"],
        errors="coerce"
    )

    if df["posterior_scaling"].isna().any():
        raise RuntimeError(
            f"Invalid scaling factors in {label} file."
        )

    if df["parameter"].duplicated().any():
        raise RuntimeError(
            f"Duplicate parameters in {label} file."
        )

    extracted = df["parameter"].str.extract(
        r"^(OC\d+)_(\d{8})$"
    )

    if extracted.isna().any().any():

        bad = df.loc[
            extracted.isna().any(axis=1),
            "parameter"
        ].tolist()

        raise RuntimeError(
            f"Invalid parameter names in {label}:\n{bad[:10]}"
        )

    df["mine"] = extracted[0]

    df["date"] = pd.to_datetime(
        extracted[1],
        format="%Y%m%d"
    )

    return df


# =============================================================================
# LOAD BOTH RESULTS
# =============================================================================

df05 = read_map_file(
    FILE_SIGMA05,
    "UNCONSTRAINED MAP sigma = 0.5"
)

df10 = read_map_file(
    FILE_SIGMA10,
    "UNCONSTRAINED MAP sigma = 1.0"
)


# =============================================================================
# CHECK PARAMETERS
# =============================================================================

print("\n" + "=" * 80)
print("PARAMETER CONSISTENCY")
print("=" * 80)

if len(df05) != 390 or len(df10) != 390:
    raise RuntimeError(
        "Both files must contain exactly 390 parameters."
    )

if set(df05["parameter"]) != set(df10["parameter"]):
    raise RuntimeError(
        "The two files do not contain identical parameters."
    )

print("Number of parameters:", len(df05))
print("Parameter names: MATCHED")


# =============================================================================
# CREATE DAILY DATA
# =============================================================================

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

wide05 = df05.pivot(
    index="date",
    columns="mine",
    values="posterior_scaling"
).reindex(
    index=dates,
    columns=MINES
)

wide10 = df10.pivot(
    index="date",
    columns="mine",
    values="posterior_scaling"
).reindex(
    index=dates,
    columns=MINES
)

if wide05.isna().any().any():
    raise RuntimeError("Missing values in sigma = 0.5 results.")

if wide10.isna().any().any():
    raise RuntimeError("Missing values in sigma = 1.0 results.")

print("Daily data shape:", wide05.shape)


# =============================================================================
# NUMERICAL COMPARISON
# =============================================================================

print("\n" + "=" * 100)
print("UNCONSTRAINED MAP: PRIOR UNCERTAINTY COMPARISON")
print("=" * 100)

summary = []

for mine in MINES:

    s05 = wide05[mine].to_numpy()
    s10 = wide10[mine].to_numpy()

    difference = s10 - s05

    summary.append({
        "mine": mine,
        "mean_sigma05": np.mean(s05),
        "mean_sigma10": np.mean(s10),
        "mean_difference": np.mean(difference),
        "mean_absolute_difference": np.mean(np.abs(difference)),
        "minimum_sigma05": np.min(s05),
        "minimum_sigma10": np.min(s10),
        "maximum_sigma05": np.max(s05),
        "maximum_sigma10": np.max(s10),
    })

    print(
        f"{mine:5s} | "
        f"Mean sigma=0.5: {np.mean(s05):9.4f} | "
        f"Mean sigma=1.0: {np.mean(s10):9.4f} | "
        f"Mean difference: {np.mean(difference):9.4f} | "
        f"MAE: {np.mean(np.abs(difference)):9.4f}"
    )

summary_df = pd.DataFrame(summary)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "MAP_unconstrained_sigma05_vs_sigma10_summary.csv"
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\nSummary saved:", SUMMARY_FILE)


# =============================================================================
# PLOT 13 MINES
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

    s05 = wide05[mine]
    s10 = wide10[mine]

    # Prior relative error = 0.5
    ax.plot(
        dates,
        s05.values,
        color="blue",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label=r"$\sigma_{prior}=0.5$"
    )

    # Prior relative error = 1.0
    ax.plot(
        dates,
        s10.values,
        color="red",
        linewidth=1.8,
        marker="s",
        markersize=3,
        linestyle="--",
        label=r"$\sigma_{prior}=1.0$"
    )

    # Prior scaling factor
    ax.axhline(
        PRIOR_SCALING,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Prior = 1.0"
    )

    ax.set_title(
        mine,
        fontweight="bold"
    )

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

    # Individual y-axis limits for each mine
    ymin = min(
        s05.min(),
        s10.min(),
        PRIOR_SCALING
    )

    ymax = max(
        s05.max(),
        s10.max(),
        PRIOR_SCALING
    )

    margin = max(
        0.08 * (ymax - ymin),
        0.05
    )

    ax.set_ylim(
        ymin - margin,
        ymax + margin
    )


# Remove unused panels
for j in range(len(MINES), len(axes)):
    fig.delaxes(axes[j])


# =============================================================================
# LEGEND
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
# TITLE AND LAYOUT
# =============================================================================

fig.suptitle(
    "Upper Hunter FLEXPART-WRF: Unconstrained MAP Emission Scaling Factors\n"
    "Effect of Prior Relative Error: 0.5 versus 1.0 | "
    "1-30 April 2026",
    fontsize=15,
    fontweight="bold",
    y=0.995
)

fig.supxlabel(
    "Date (April 2026)",
    fontsize=12
)

plt.tight_layout(
    rect=[0.02, 0.04, 0.98, 0.91]
)


# =============================================================================
# SAVE FIGURES
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
print("FIGURES SAVED")
print("=" * 80)

print("PNG:", OUTPUT_PNG)
print("PDF:", OUTPUT_PDF)

plt.show()

print("\nDONE.")

