#!/usr/bin/env python3
"""
Plot bounded KF, projected bounded RTS, and bounded batch MAP
daily scaling factors for 13 Upper Hunter open-cut mines.

A horizontal dotted line marks the prior scaling factor of 1.0.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart/H_matrix_flexpart"

KF_FILE = (
    BASE
    / "KF_bounded_full_footprint_13mine_Jan2026"
    / "daily_bounded_full_footprint_KF_scaling_factors.csv"
)

RTS_FILE = (
    BASE
    / "RTS_projected_bounded_full_footprint_13mine_Jan2026"
    / "daily_projected_bounded_RTS_scaling_factors.csv"
)

MAP_FILE = (
    BASE
    / "Global_constrained_smoother_13mine_Jan2026"
    / "daily_global_constrained_smoother_scaling_factors.csv"
)

OUT_DIR = (
    BASE
    / "RTS_projected_bounded_full_footprint_13mine_Jan2026"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_FILE = OUT_DIR / "bounded_KF_RTS_MAP_timeseries.png"
PDF_FILE = OUT_DIR / "bounded_KF_RTS_MAP_timeseries.pdf"

PRIOR = 1.0
LOWER_BOUND = 0.0
UPPER_BOUND = 5.0

# ------------------------------------------------------------
# Read data
# ------------------------------------------------------------
def read_scaling_factors(filename, label):
    df = pd.read_csv(filename)

    # Standardise date-column name.
    date_candidates = ["date", "Date", "day", "Day"]
    date_col = next(
        (c for c in date_candidates if c in df.columns),
        None,
    )
    if date_col is None:
        raise ValueError(
            f"No date column found in {filename}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})

    # Accept either wide-format mine columns (OC1, OC2, ...)
    # or long-format columns (mine, scaling_factor).
    if {"mine", "scaling_factor"}.issubset(df.columns):
        df = df.pivot(
            index="date",
            columns="mine",
            values="scaling_factor",
        ).reset_index()
    elif {"mine", "scaling_factor_mean"}.issubset(df.columns):
        df = df.pivot(
            index="date",
            columns="mine",
            values="scaling_factor_mean",
        ).reset_index()

    mine_cols = [
        c for c in df.columns
        if isinstance(c, str)
        and c.startswith("OC")
        and c[2:].isdigit()
    ]

    if not mine_cols:
        raise ValueError(
            f"Could not identify OC1–OC13 columns in {filename}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[["date"] + sorted(
        mine_cols, key=lambda x: int(x[2:])
    )]

    return df, label


kf, _ = read_scaling_factors(KF_FILE, "Bounded KF")
rts, _ = read_scaling_factors(RTS_FILE, "Projected bounded RTS")
map_df, _ = read_scaling_factors(MAP_FILE, "Bounded batch MAP")

# Rename columns so the three methods can be merged without ambiguity.
def rename_mines(df, suffix):
    return df.rename(
        columns={
            c: f"{c}_{suffix}"
            for c in df.columns
            if c != "date"
        }
    )

kf = rename_mines(kf, "KF")
rts = rename_mines(rts, "RTS")
map_df = rename_mines(map_df, "MAP")

df = kf.merge(rts, on="date", how="inner")
df = df.merge(map_df, on="date", how="inner")
df = df.sort_values("date")

print(f"Dates: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ------------------------------------------------------------
# Plot: 13 mine panels
# ------------------------------------------------------------
mines = [f"OC{i}" for i in range(1, 14)]

fig, axes = plt.subplots(
    4, 4,
    figsize=(19, 14),
    sharex=True,
    sharey=True,
)

axes = axes.flatten()

for i, mine in enumerate(mines):
    ax = axes[i]

    ax.plot(
        df["date"],
        df[f"{mine}_KF"],
        color="tab:blue",
        linewidth=1.7,
        marker="o",
        markersize=2.5,
        label="Bounded KF",
    )

    ax.plot(
        df["date"],
        df[f"{mine}_RTS"],
        color="tab:orange",
        linewidth=1.7,
        linestyle="--",
        marker="s",
        markersize=2.5,
        label="Projected bounded RTS",
    )

    ax.plot(
        df["date"],
        df[f"{mine}_MAP"],
        color="tab:green",
        linewidth=1.7,
        linestyle="-.",
        marker="^",
        markersize=2.5,
        label="Bounded batch MAP",
    )

    # Prior scaling factor = 1.0, shown as a broken horizontal line.
    ax.axhline(
        PRIOR,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="Prior = 1.0",
    )

    ax.set_title(mine)
    ax.set_ylim(LOWER_BOUND - 0.1, UPPER_BOUND + 0.1)
    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.grid(True, alpha=0.25)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d Jan"))

    if i % 4 == 0:
        ax.set_ylabel("Daily scaling factor")

    if i // 4 == 3:
        ax.set_xlabel("Date")

# Remove the unused 14th, 15th and 16th panels.
for j in range(len(mines), len(axes)):
    fig.delaxes(axes[j])

# One shared legend for all panels.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.965),
)

fig.suptitle(
    "Upper Hunter daily mine scaling factors: January 2026",
    fontsize=16,
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.93])

fig.savefig(PNG_FILE, dpi=300, bbox_inches="tight")
fig.savefig(PDF_FILE, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"Saved PNG: {PNG_FILE}")
print(f"Saved PDF: {PDF_FILE}")


