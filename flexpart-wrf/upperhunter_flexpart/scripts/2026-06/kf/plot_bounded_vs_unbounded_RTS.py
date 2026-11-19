#!/usr/bin/env python3

"""
Compare bounded projected RTS-KF against unbounded RTS-KF
for the 13 Upper Hunter coal mines, January 2026.

Bounded RTS:
    RTS_projected_bounded_full_footprint_13mine_Jan2026/
    daily_projected_bounded_RTS_scaling_factors.csv

Unbounded RTS:
    KF_full_footprint_13mine_Jan2026/
    RTS_full_footprint_daily_scaling_factors.csv

Outputs:
    - Combined 4x4 figure for all 13 mines
    - Individual figure for each mine
    - CSV containing comparison statistics
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# PATHS
# ============================================================

BASE = (
    "/home/duch/flexpart-wrf/"
    "upperhunter_flexpart/H_matrix_flexpart"
)

BOUNDED_FILE = os.path.join(
    BASE,
    "RTS_projected_bounded_full_footprint_13mine_Jan2026",
    "daily_projected_bounded_RTS_scaling_factors.csv",
)

UNBOUNDED_FILE = os.path.join(
    BASE,
    "KF_full_footprint_13mine_Jan2026",
    "RTS_full_footprint_daily_scaling_factors.csv",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "RTS_projected_bounded_full_footprint_13mine_Jan2026",
    "bounded_vs_unbounded_RTS",
)

INDIVIDUAL_DIR = os.path.join(
    OUTPUT_DIR,
    "individual_mine_timeseries",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)


# ============================================================
# READ DATA
# ============================================================

print("=" * 80)
print("READING BOUNDED RTS")
print("=" * 80)

bounded = pd.read_csv(BOUNDED_FILE)

print("File:")
print(BOUNDED_FILE)
print("Shape:", bounded.shape)
print("Columns:")
print(list(bounded.columns))


print()
print("=" * 80)
print("READING UNBOUNDED RTS")
print("=" * 80)

unbounded = pd.read_csv(UNBOUNDED_FILE)

print("File:")
print(UNBOUNDED_FILE)
print("Shape:", unbounded.shape)
print("Columns:")
print(list(unbounded.columns))


# ============================================================
# IDENTIFY DATE COLUMN
# ============================================================

def find_date_column(df, name):
    candidates = [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "time",
        "Time",
        "day",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not identify date column in {name}. "
        f"Available columns: {list(df.columns)}"
    )


bounded_date_col = find_date_column(bounded, "bounded RTS")
unbounded_date_col = find_date_column(unbounded, "unbounded RTS")

bounded[bounded_date_col] = pd.to_datetime(
    bounded[bounded_date_col]
)

unbounded[unbounded_date_col] = pd.to_datetime(
    unbounded[unbounded_date_col]
)


# ============================================================
# MINE COLUMNS
# ============================================================

MINES = [f"OC{i}" for i in range(1, 14)]

missing_bounded = [
    mine for mine in MINES
    if mine not in bounded.columns
]

missing_unbounded = [
    mine for mine in MINES
    if mine not in unbounded.columns
]

if missing_bounded:
    raise ValueError(
        "Missing mine columns in bounded RTS: "
        + str(missing_bounded)
    )

if missing_unbounded:
    raise ValueError(
        "Missing mine columns in unbounded RTS: "
        + str(missing_unbounded)
    )


# ============================================================
# SORT AND ALIGN DATES
# ============================================================

bounded = bounded.sort_values(
    bounded_date_col
).reset_index(drop=True)

unbounded = unbounded.sort_values(
    unbounded_date_col
).reset_index(drop=True)


bounded = bounded.rename(
    columns={bounded_date_col: "date"}
)

unbounded = unbounded.rename(
    columns={unbounded_date_col: "date"}
)


# Keep only common dates
common_dates = sorted(
    set(bounded["date"]).intersection(
        set(unbounded["date"])
    )
)

if len(common_dates) == 0:
    raise ValueError(
        "No common dates between bounded and unbounded RTS."
    )

bounded = bounded[
    bounded["date"].isin(common_dates)
].copy()

unbounded = unbounded[
    unbounded["date"].isin(common_dates)
].copy()


bounded = bounded.sort_values("date").reset_index(drop=True)
unbounded = unbounded.sort_values("date").reset_index(drop=True)


print()
print("=" * 80)
print("DATA ALIGNMENT")
print("=" * 80)

print("Number of common days:", len(common_dates))
print("First date:", common_dates[0])
print("Last date :", common_dates[-1])

print()
print("Bounded RTS range:")
print(
    bounded[MINES].to_numpy().min(),
    bounded[MINES].to_numpy().max()
)

print("Unbounded RTS range:")
print(
    unbounded[MINES].to_numpy().min(),
    unbounded[MINES].to_numpy().max()
)


# ============================================================
# COMPARISON STATISTICS
# ============================================================

statistics = []

for mine in MINES:

    b = bounded[mine].to_numpy(dtype=float)
    u = unbounded[mine].to_numpy(dtype=float)

    diff = b - u

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    max_abs = np.max(np.abs(diff))
    mean_diff = np.mean(diff)

    if np.std(b) > 0 and np.std(u) > 0:
        corr = np.corrcoef(b, u)[0, 1]
    else:
        corr = np.nan

    statistics.append(
        {
            "mine": mine,
            "RMSE_bounded_minus_unbounded": rmse,
            "MAE_bounded_minus_unbounded": mae,
            "max_abs_difference": max_abs,
            "mean_difference_bounded_minus_unbounded": mean_diff,
            "correlation": corr,
            "bounded_min": np.min(b),
            "bounded_max": np.max(b),
            "unbounded_min": np.min(u),
            "unbounded_max": np.max(u),
        }
    )


stats_df = pd.DataFrame(statistics)

stats_file = os.path.join(
    OUTPUT_DIR,
    "bounded_vs_unbounded_RTS_statistics.csv",
)

stats_df.to_csv(
    stats_file,
    index=False,
)

print()
print("=" * 80)
print("COMPARISON STATISTICS")
print("=" * 80)

print(
    stats_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print("Statistics saved to:")
print(stats_file)


# ============================================================
# OVERALL STATISTICS
# ============================================================

B = bounded[MINES].to_numpy(dtype=float)
U = unbounded[MINES].to_numpy(dtype=float)

D = B - U

overall_rmse = np.sqrt(np.mean(D ** 2))
overall_mae = np.mean(np.abs(D))
overall_max = np.max(np.abs(D))
overall_mean = np.mean(D)

if np.std(B.ravel()) > 0 and np.std(U.ravel()) > 0:
    overall_corr = np.corrcoef(
        B.ravel(),
        U.ravel()
    )[0, 1]
else:
    overall_corr = np.nan


print()
print("=" * 80)
print("OVERALL 351-PARAMETER COMPARISON")
print("=" * 80)

print(f"Number of parameters : {B.size}")
print(f"RMSE                  : {overall_rmse:.10f}")
print(f"MAE                   : {overall_mae:.10f}")
print(f"Maximum absolute diff : {overall_max:.10f}")
print(f"Mean difference       : {overall_mean:.10f}")
print(f"Correlation           : {overall_corr:.10f}")

for threshold in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:

    n = np.sum(np.abs(D) > threshold)

    print(
        f"|difference| > {threshold:<5} : "
        f"{n:3d} / {D.size}"
    )


# ============================================================
# COMBINED 4 x 4 FIGURE
# ============================================================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(17, 13),
    sharex=True,
    sharey=True,
)

axes = axes.ravel()

for i, mine in enumerate(MINES):

    ax = axes[i]

    ax.plot(
        bounded["date"],
        bounded[mine],
        linestyle="-",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Bounded RTS",
    )

    ax.plot(
        unbounded["date"],
        unbounded[mine],
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=3.0,
        label="Unbounded RTS",
    )

    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1.2,
        label="Prior = 1"
        if i == 0 else None,
    )

    ax.set_title(
        mine,
        fontsize=9,
    )

    ax.set_ylim(
        -0.1,
        5.6,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=4)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    ax.tick_params(
        axis="x",
        labelrotation=0,
        labelsize=8,
    )

    ax.tick_params(
        axis="y",
        labelsize=8,
    )


# Hide unused 16th panel
for j in range(len(MINES), len(axes)):
    axes[j].axis("off")


# Put common labels on figure
fig.supxlabel(
    "Date — January 2026",
    fontsize=12,
)

fig.supylabel(
    "Daily scaling factor",
    fontsize=12,
)

fig.suptitle(
    "Bounded RTS-KF vs Unbounded RTS-KF — Upper Hunter CH₄",
    fontsize=15,
    y=0.995,
)


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=3,
    frameon=True,
)


fig.tight_layout(
    rect=[0.03, 0.04, 0.99, 0.94]
)


combined_png = os.path.join(
    OUTPUT_DIR,
    "bounded_vs_unbounded_RTS_all_mines.png",
)

combined_pdf = os.path.join(
    OUTPUT_DIR,
    "bounded_vs_unbounded_RTS_all_mines.pdf",
)

fig.savefig(
    combined_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    combined_pdf,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


print()
print("Combined figure:")
print(combined_png)
print(combined_pdf)


# ============================================================
# INDIVIDUAL MINE FIGURES
# ============================================================

print()
print("=" * 80)
print("CREATING INDIVIDUAL MINE PLOTS")
print("=" * 80)

for mine in MINES:

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    ax.plot(
        bounded["date"],
        bounded[mine],
        linestyle="-",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Bounded RTS",
    )

    ax.plot(
        unbounded["date"],
        unbounded[mine],
        linestyle="--",
        linewidth=2.2,
        marker="s",
        markersize=4,
        label="Unbounded RTS",
    )

    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1.4,
        label="Prior = 1",
    )

    ax.set_title(
        f"{mine} — Bounded RTS vs Unbounded RTS, January 2026",
        fontsize=13,
    )

    ax.set_xlabel(
        "Date",
        fontsize=11,
    )

    ax.set_ylabel(
        "Daily scaling factor",
        fontsize=11,
    )

    ax.set_ylim(
        -0.1,
        5.6,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=4)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    ax.tick_params(
        axis="x",
        labelrotation=0,
        labelsize=9,
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    # Add numerical summary to the plot
    row = stats_df[
        stats_df["mine"] == mine
    ].iloc[0]

    summary = (
        f"RMSE = {row['RMSE_bounded_minus_unbounded']:.4f}\n"
        f"MAE = {row['MAE_bounded_minus_unbounded']:.4f}\n"
        f"Max |Δ| = {row['max_abs_difference']:.4f}\n"
        f"r = {row['correlation']:.4f}"
    )

    ax.text(
        0.99,
        0.97,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )

    fig.tight_layout()

    png_file = os.path.join(
        INDIVIDUAL_DIR,
        f"{mine}_bounded_vs_unbounded_RTS.png",
    )

    pdf_file = os.path.join(
        INDIVIDUAL_DIR,
        f"{mine}_bounded_vs_unbounded_RTS.pdf",
    )

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"{mine}:")
    print(f"  {png_file}")
    print(f"  {pdf_file}")


# ============================================================
# SAVE A DAILY DIFFERENCE TABLE
# ============================================================

difference_rows = []

for i, date in enumerate(common_dates):

    row = {
        "date": date,
    }

    for mine in MINES:
        row[f"{mine}_bounded_RTS"] = bounded.loc[
            bounded["date"] == date,
            mine,
        ].iloc[0]

        row[f"{mine}_unbounded_RTS"] = unbounded.loc[
            unbounded["date"] == date,
            mine,
        ].iloc[0]

        row[f"{mine}_difference"] = (
            row[f"{mine}_bounded_RTS"]
            - row[f"{mine}_unbounded_RTS"]
        )

    difference_rows.append(row)


difference_df = pd.DataFrame(
    difference_rows
)

difference_file = os.path.join(
    OUTPUT_DIR,
    "bounded_vs_unbounded_RTS_daily_difference.csv",
)

difference_df.to_csv(
    difference_file,
    index=False,
)

print()
print("Daily difference table:")
print(difference_file)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print("Output directory:")
print(OUTPUT_DIR)

print()
print("Main figure:")
print(combined_png)

print()
print("Individual figures:")
print(INDIVIDUAL_DIR)

print()
print("Statistics:")
print(stats_file)

print()
print("Daily differences:")
print(difference_file)

