#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

BASE = os.path.expanduser(
    "~/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/KF_full_footprint_13mine_Jan2026"
)

HOURLY_FILE = os.path.join(
    BASE,
    "hourly_full_footprint_KF_results.csv"
)

DAILY_KF_FILE = os.path.join(
    BASE,
    "daily_full_footprint_KF_scaling_factors.csv"
)

RTS_FILE = os.path.join(
    BASE,
    "daily_RTS_smoothed_scaling_factors.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "hourly_KF_scaling_factor_plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD HOURLY KF RESULTS
# ============================================================

print("=" * 80)
print("LOADING HOURLY KF RESULTS")
print("=" * 80)

hourly = pd.read_csv(HOURLY_FILE)

print("Hourly shape:", hourly.shape)
print("Columns:")
print(hourly.columns.tolist())


# ------------------------------------------------------------
# Convert timestamp
# ------------------------------------------------------------

hourly["time_aest"] = pd.to_datetime(
    hourly["time_aest"],
    errors="coerce"
)

hourly = hourly.dropna(
    subset=["time_aest"]
).sort_values(
    ["time_aest"]
).reset_index(drop=True)


# ============================================================
# 2. EXTRACT HOURLY POSTERIOR STATE
# ============================================================
#
# There can be multiple receptor observations at the same
# timestamp.
#
# The KF state is updated sequentially for each observation.
# Therefore, the LAST row at each timestamp represents the
# state after all observations available at that hour have
# been assimilated.
# ============================================================

mine_names = [
    f"OC{i}" for i in range(1, 14)
]

hourly_cols = [
    f"current_{mine}" for mine in mine_names
]

missing = [
    col for col in hourly_cols
    if col not in hourly.columns
]

if missing:
    raise RuntimeError(
        "Missing hourly KF columns:\n"
        + "\n".join(missing)
    )


hourly_by_hour = (
    hourly
    .sort_values("time_aest")
    .groupby("time_aest", as_index=False)
    .last()
)

hourly_scaling = hourly_by_hour[
    ["time_aest"] + hourly_cols
].copy()


# Rename current_OC1 -> OC1 etc.
hourly_scaling = hourly_scaling.rename(
    columns={
        f"current_{mine}": mine
        for mine in mine_names
    }
)

print()
print("Unique hourly timestamps:", len(hourly_scaling))
print(
    "Time range:",
    hourly_scaling["time_aest"].min(),
    "to",
    hourly_scaling["time_aest"].max()
)


# ============================================================
# 3. SAVE CLEANED HOURLY DATA
# ============================================================

clean_hourly_file = os.path.join(
    OUTPUT_DIR,
    "hourly_KF_posterior_scaling_factors_by_hour.csv"
)

hourly_scaling.to_csv(
    clean_hourly_file,
    index=False
)

print()
print("Saved:")
print(clean_hourly_file)


# ============================================================
# 4. LOAD DAILY KF RESULTS
# ============================================================

print()
print("=" * 80)
print("LOADING DAILY KF RESULTS")
print("=" * 80)

daily_kf = pd.read_csv(DAILY_KF_FILE)

print("Daily KF shape:", daily_kf.shape)
print("Columns:", daily_kf.columns.tolist())


# ------------------------------------------------------------
# Automatically identify date column
# ------------------------------------------------------------

possible_date_cols = [
    "date",
    "Date",
    "day",
    "time_aest",
    "datetime"
]

daily_kf_date_col = None

for col in possible_date_cols:
    if col in daily_kf.columns:
        daily_kf_date_col = col
        break

if daily_kf_date_col is None:
    raise RuntimeError(
        "Could not identify date column in daily KF file."
    )

daily_kf["date_plot"] = pd.to_datetime(
    daily_kf[daily_kf_date_col],
    errors="coerce"
)

daily_kf = daily_kf.dropna(
    subset=["date_plot"]
).copy()

daily_kf["date_plot"] = daily_kf[
    "date_plot"
].dt.normalize()


# ============================================================
# 5. LOAD RTS RESULTS
# ============================================================

print()
print("=" * 80)
print("LOADING RTS RESULTS")
print("=" * 80)

rts = pd.read_csv(RTS_FILE)

print("RTS shape:", rts.shape)
print("Columns:", rts.columns.tolist())


rts_date_col = None

for col in possible_date_cols:
    if col in rts.columns:
        rts_date_col = col
        break

if rts_date_col is None:
    raise RuntimeError(
        "Could not identify date column in RTS file."
    )

rts["date_plot"] = pd.to_datetime(
    rts[rts_date_col],
    errors="coerce"
)

rts = rts.dropna(
    subset=["date_plot"]
).copy()

rts["date_plot"] = rts[
    "date_plot"
].dt.normalize()


# ============================================================
# 6. CHECK MINE COLUMNS
# ============================================================

for mine in mine_names:

    if mine not in daily_kf.columns:
        raise RuntimeError(
            f"{mine} not found in daily KF file."
        )

    if mine not in rts.columns:
        raise RuntimeError(
            f"{mine} not found in RTS file."
        )

# ============================================================
# 7. PLOT EACH MINE
# ============================================================

print()
print("=" * 80)
print("CREATING MINE-BY-MINE LINE PLOTS")
print("=" * 80)

summary_rows = []


for mine in mine_names:

    print(f"Plotting {mine} ...")

    # --------------------------------------------------------
    # Hourly KF
    # --------------------------------------------------------

    h_time = hourly_scaling["time_aest"]
    h_value = hourly_scaling[mine].astype(float)

    # --------------------------------------------------------
    # Daily KF
    # --------------------------------------------------------

    d_time = daily_kf["date_plot"]
    d_value = daily_kf[mine].astype(float)

    # --------------------------------------------------------
    # RTS / Batch MAP
    # --------------------------------------------------------

    r_time = rts["date_plot"]
    r_value = rts[mine].astype(float)

    # --------------------------------------------------------
    # Create plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    # ========================================================
    # HOURLY KF POSTERIOR
    # ========================================================

    ax.plot(
        h_time,
        h_value,
        linewidth=1.2,
        label="Hourly KF posterior"
    )

    # ========================================================
    # DAILY KF POSTERIOR
    # ========================================================

    ax.plot(
        d_time,
        d_value,
        linewidth=2.0,
        label="Daily KF posterior"
    )

    # ========================================================
    # RTS / BATCH MAP
    # ========================================================

    ax.plot(
        r_time,
        r_value,
        linewidth=2.0,
        label="RTS / Batch MAP"
    )

    # ========================================================
    # PRIOR
    # ========================================================

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="Prior = 1.0"
    )

    # ========================================================
    # AXIS LABELS
    # ========================================================

    ax.set_xlabel(
        "Time (AEST)"
    )

    ax.set_ylabel(
        "Mine emission scaling factor"
    )

    ax.set_title(
        f"{mine} — Hourly KF, Daily KF and RTS/Batch MAP"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.autofmt_xdate()

    plt.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    output_png = os.path.join(
        OUTPUT_DIR,
        f"{mine}_hourly_KF_daily_KF_RTS.png"
    )

    fig.savefig(
        output_png,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    rts_interp = pd.Series(
        r_value.values,
        index=r_time
    ).reindex(
        h_time
    ).interpolate(
        method="time"
    ).ffill().bfill()

    diff = h_value.values - rts_interp.values

    summary_rows.append(
        {
            "mine": mine,
            "hourly_n": len(h_value),
            "hourly_min": h_value.min(),
            "hourly_max": h_value.max(),
            "hourly_mean": h_value.mean(),
            "RTS_mean": r_value.mean(),
            "hourly_minus_RTS_RMSE": np.sqrt(
                np.mean(diff ** 2)
            ),
            "hourly_minus_RTS_MAE": np.mean(
                np.abs(diff)
            ),
            "final_hourly_KF": h_value.iloc[-1],
            "final_daily_KF": d_value.iloc[-1],
            "final_RTS": r_value.iloc[-1],
        }
    )


# ============================================================
# 8. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "hourly_KF_vs_RTS_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 9. FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Created:")
print(
    "  hourly_KF_posterior_scaling_factors_by_hour.csv"
)

for mine in mine_names:
    print(
        f"  {mine}_hourly_KF_daily_KF_RTS.png"
    )

print(
    "  hourly_KF_vs_RTS_summary.csv"
)

print()
print(summary.to_string(index=False))

