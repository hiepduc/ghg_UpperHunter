#!/usr/bin/env python3
"""Plot hourly and daily bounded KF, projected RTS, and bounded batch MAP."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = (
    Path.home()
    / "flexpart-wrf/upperhunter_flexpart/H_matrix_flexpart"
)

KF_DIR = BASE / "KF_bounded_full_footprint_13mine_Jan2026"

KF_DAILY_FILE = (
    KF_DIR / "daily_bounded_full_footprint_KF_scaling_factors.csv"
)
KF_HOURLY_FILE = (
    KF_DIR / "hourly_bounded_full_footprint_KF_results.csv"
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

OUT_DIR = BASE / "RTS_projected_bounded_full_footprint_13mine_Jan2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_FILE = OUT_DIR / "bounded_hourly_KF_daily_KF_RTS_MAP_timeseries.png"
PDF_FILE = OUT_DIR / "bounded_hourly_KF_daily_KF_RTS_MAP_timeseries.pdf"

PRIOR = 1.0
MIN_BOUND = 0.0
MAX_BOUND = 5.0
MINES = [f"OC{i}" for i in range(1, 14)]


# ------------------------------------------------------------
# Read daily scaling-factor CSVs
# ------------------------------------------------------------
def read_daily_file(filename, method):
    df = pd.read_csv(filename)

    if "date" not in df.columns:
        raise ValueError(
            f"No 'date' column in {filename}. "
            f"Columns: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"])

    missing = [m for m in MINES if m not in df.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing mine columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[["date"] + MINES].copy()

    return df.rename(
        columns={m: f"{m}_{method}" for m in MINES}
    )


kf_daily = read_daily_file(KF_DAILY_FILE, "KF")
rts_daily = read_daily_file(RTS_FILE, "RTS")
map_daily = read_daily_file(MAP_FILE, "MAP")

daily = (
    kf_daily
    .merge(rts_daily, on="date", how="inner")
    .merge(map_daily, on="date", how="inner")
    .sort_values("date")
)


# ------------------------------------------------------------
# Read hourly bounded KF results
# ------------------------------------------------------------
hourly = pd.read_csv(KF_HOURLY_FILE)
hourly["time_aest"] = pd.to_datetime(hourly["time_aest"])

# Plot the current-day state after each observation update.
hourly_cols = [f"current_{mine}" for mine in MINES]
missing = [col for col in hourly_cols if col not in hourly.columns]

if missing:
    raise ValueError(
        f"Missing hourly state columns: {missing}"
    )

# Keep the last update at each timestamp, so simultaneous receptor
# updates do not produce duplicate timestamps in the plotted series.
hourly = (
    hourly.sort_values("time_aest")
    .drop_duplicates(subset="time_aest", keep="last")
)

print(f"Daily dates: {len(daily)}")
print(f"Hourly state timestamps: {len(hourly)}")
print(
    f"Hourly time range: {hourly['time_aest'].min()} "
    f"to {hourly['time_aest'].max()}"
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(
    4,
    4,
    figsize=(21, 16),
    sharex=True,
    sharey=True,
)
axes = axes.flatten()

for i, mine in enumerate(MINES):
    ax = axes[i]

    # Hourly bounded KF
    ax.plot(
        hourly["time_aest"],
        hourly[f"current_{mine}"],
        color="tab:purple",
        linewidth=0.8,
        alpha=0.55,
        label="Hourly bounded KF",
        zorder=1,
    )

    # Daily bounded KF
    ax.plot(
        daily["date"],
        daily[f"{mine}_KF"],
        color="tab:blue",
        linewidth=1.6,
        marker="o",
        markersize=2.5,
        label="Daily bounded KF",
        zorder=3,
    )

    # Projected bounded RTS
    ax.plot(
        daily["date"],
        daily[f"{mine}_RTS"],
        color="tab:orange",
        linewidth=1.7,
        linestyle="--",
        marker="s",
        markersize=2.5,
        label="Projected bounded RTS",
        zorder=4,
    )

    # Global constrained batch MAP
    ax.plot(
        daily["date"],
        daily[f"{mine}_MAP"],
        color="tab:green",
        linewidth=1.7,
        linestyle="-.",
        marker="^",
        markersize=2.5,
        label="Bounded batch MAP",
        zorder=5,
    )

    # Prior scaling factor = 1
    ax.axhline(
        PRIOR,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="Prior = 1.0",
        zorder=2,
    )

    ax.set_title(mine, fontsize = 9)
    ax.set_ylim(MIN_BOUND - 0.1, MAX_BOUND + 0.1)
    ax.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # Date and time tick labels
    # --------------------------------------------------------
    # One tick every 120 hours; show date and time on two lines.
    # Show date and time every 5 days, horizontally
    ax.xaxis.set_major_locator(
        mdates.HourLocator(interval=120)
    )
    ax.xaxis.set_major_formatter(
        #mdates.DateFormatter("%d %b %H:%M")
        mdates.DateFormatter("%d %b %H")
    )

    # Explicitly enable labels despite sharex=True.
    ax.tick_params(
        axis="x",
        which="major",
        labelbottom=True,
        labelsize=8,
        labelrotation=0,
        pad=3,
    )

    if i % 4 == 0:
        ax.set_ylabel("Scaling factor")

# Remove the unused 14th–16th panels.
for j in range(len(MINES), len(axes)):
    fig.delaxes(axes[j])

# Explicitly enable date/time labels on all remaining panels.
for ax in fig.axes:
    ax.tick_params(axis="x", labelbottom=True)

# Shared x-axis label, with space reserved below the panels.
fig.supxlabel(
    "Date and time (AEST)",
    fontsize=11,
    y=0.025,
)

# ------------------------------------------------------------
# Shared legend and title
# ------------------------------------------------------------
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.965),
)

fig.suptitle(
    "Upper Hunter mine scaling factors — January 2026",
    fontsize=16,
    y=0.995,
)

# Leave room for the title, legend, and date/time labels.
fig.tight_layout(
    rect=[0, 0.055, 1, 0.925],
    h_pad=1.5,
)

# ------------------------------------------------------------
# Save and display
# ------------------------------------------------------------
fig.savefig(
    PNG_FILE,
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    PDF_FILE,
    bbox_inches="tight",
)

print(f"Saved PNG: {PNG_FILE}")
print(f"Saved PDF: {PDF_FILE}")
plt.show()
plt.close(fig)

# ------------------------------------------------------------
# Individual plots: one figure per mine
# ------------------------------------------------------------
INDIVIDUAL_DIR = OUT_DIR / "individual_mine_timeseries"
INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

for mine in MINES:
    fig_one, ax = plt.subplots(figsize=(12, 5.5))

    # Hourly bounded KF
    ax.plot(
        hourly["time_aest"],
        hourly[f"current_{mine}"],
        color="tab:purple",
        linewidth=0.9,
        alpha=0.65,
        label="Hourly bounded KF",
        zorder=1,
    )

    # Daily bounded KF
    ax.plot(
        daily["date"],
        daily[f"{mine}_KF"],
        color="tab:blue",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Daily bounded KF",
        zorder=3,
    )

    # Projected bounded RTS
    ax.plot(
        daily["date"],
        daily[f"{mine}_RTS"],
        color="tab:orange",
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Projected bounded RTS",
        zorder=4,
    )

    # Global constrained batch MAP
    ax.plot(
        daily["date"],
        daily[f"{mine}_MAP"],
        color="tab:green",
        linewidth=1.8,
        linestyle="-.",
        marker="^",
        markersize=4,
        label="Bounded batch MAP",
        zorder=5,
    )

    # Prior scaling factor
    ax.axhline(
        PRIOR,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="Prior = 1.0",
        zorder=2,
    )

    ax.set_title(
        f"{mine} — Daily scaling factor, January 2026",
        fontsize=12,
    )
    ax.set_xlabel("Date and time (AEST)", fontsize=10)
    ax.set_ylabel("Scaling factor", fontsize=10)
    ax.set_ylim(MIN_BOUND - 0.1, MAX_BOUND + 0.1)
    ax.grid(True, alpha=0.25)

    # Date/time labels horizontally, every 4 days
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=96))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b %H:%M")
    )
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()

    png_file = INDIVIDUAL_DIR / f"{mine}_scaling_factors_Jan2026.png"
    pdf_file = INDIVIDUAL_DIR / f"{mine}_scaling_factors_Jan2026.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved individual plots for {mine}:")
    print(f"  {png_file}")
    print(f"  {pdf_file}")

print(f"\nAll individual mine plots saved in: {INDIVIDUAL_DIR}")

