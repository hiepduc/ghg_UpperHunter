#!/usr/bin/env python3
"""
validate_flexpart_prior_3layer_revised.py

Validate FLEXPART-WRF 13-mine prior methane prediction against
Picarro observations at Singleton, Muswellbrook and Merriwa.

H matrix:
    H_flexpart_0_10m_13mine_May2026.csv

Interpretation:
    H[row, mine_day] = ppmv contribution from that mine/day using
                       the PRIOR emission for that mine/day.

Therefore:
    prior_enhancement = H @ ones(403)

NOT:
    H @ prior_emissions

Total CH4:
    prior_total = BACKGROUND_PPM + prior_enhancement

Plot:
    FLEXPART prior = continuous line
    Picarro = line + markers
    Missing observations are retained as NaN and are NOT interpolated.

Time:
    FLEXPART UTC -> AEST = UTC + 10 hours
    Picarro timestamps are already AEST.

Period:
    2026-05-01 through 2026-05-31
"""

# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================================
# USER SETTINGS
# ============================================================================

# ---------------------------------------------------------------------------
# Main directories
# ---------------------------------------------------------------------------

BASE_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"
)

FLEX_ROOT = BASE_DIR / "run/2026-05" / "hourly_obs_3layer"

H_DIR = BASE_DIR / "H_matrix_flexpart" / "2026-05"

H_FILE = H_DIR / "H_flexpart_0_10m_13mine_May2026.csv"

META_FILE = H_DIR / "H_flexpart_metadata_May2026.csv"

PLOT_DIR = H_DIR / "plots"

# ---------------------------------------------------------------------------
# Picarro observations
# ---------------------------------------------------------------------------

PICARRO_FILE = Path(
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

START_DATE = pd.Timestamp("2026-05-01")
END_DATE = pd.Timestamp("2026-05-31")

# ---------------------------------------------------------------------------
# Background methane
# ---------------------------------------------------------------------------

BACKGROUND_PPM = 1.85

# ---------------------------------------------------------------------------
# Receptors
# ---------------------------------------------------------------------------

RECEPTORS = {
    "singleton": {
        "lat": -32.57123,
        "lon": 151.16132,
        "obs_column": "Singleton cleaned",
    },

    "muswellbrook": {
        "lat": -32.26202,
        "lon": 150.89297,
        "obs_column": "Muswellbrook cleaned",
    },

    "merriwa": {
        "lat": -32.14097,
        "lon": 150.35767,
        "obs_column": "Merriwa cleaned",
    },
}

# ---------------------------------------------------------------------------
# Mines
# ---------------------------------------------------------------------------

MINES = [f"OC{i}" for i in range(1, 14)]

# Expected number of days
DATES = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

EXPECTED_H_COLUMNS = [
    f"{mine}_{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINES
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def find_time_column(df):
    """
    Identify likely timestamp column in the Picarro CSV.
    """

    candidates = [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "datetime",
        "Datetime",
        "DateTime",
        "date_time",
        "Date",
        "date",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Try columns containing time/date/datetime
    for col in df.columns:
        c = str(col).lower()

        if (
            "time" in c
            or "date" in c
            or "datetime" in c
        ):
            return col

    raise ValueError(
        "Could not identify the timestamp column in the Picarro CSV.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def extract_receptor_from_path(path):
    """
    Determine receptor from FLEXPART directory path.
    """

    parts_lower = [p.lower() for p in path.parts]

    for receptor in RECEPTORS:
        if receptor.lower() in parts_lower:
            return receptor

    return None


def extract_case_datetime(path):
    """
    Extract FLEXPART case/release datetime from directory name.

    Expected examples:
        20260104_140000
        20260110_020000

    Returns naive UTC Timestamp.
    """

    for part in reversed(path.parts):
        match = re.fullmatch(
            r"(\d{8})_(\d{6})",
            part
        )

        if match:
            date_part = match.group(1)
            time_part = match.group(2)

            return pd.to_datetime(
                date_part + time_part,
                format="%Y%m%d%H%M%S"
            )

    return None


def calculate_metrics(obs, model):
    """
    Calculate standard validation metrics.
    """

    mask = (
        np.isfinite(obs)
        & np.isfinite(model)
    )

    obs = np.asarray(obs)[mask]
    model = np.asarray(model)[mask]

    n = len(obs)

    if n == 0:
        return {
            "N": 0,
            "R": np.nan,
            "R2": np.nan,
            "RMSE_ppm": np.nan,
            "MAE_ppm": np.nan,
            "Bias_ppm": np.nan,
            "FAC2": np.nan,
        }

    error = model - obs

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    mae = np.mean(
        np.abs(error)
    )

    bias = np.mean(error)

    if n >= 2:
        r = np.corrcoef(obs, model)[0, 1]
        r2 = r ** 2
    else:
        r = np.nan
        r2 = np.nan

    # FAC2:
    # model/obs between 0.5 and 2.0
    valid_fac2 = (
        np.isfinite(obs)
        & np.isfinite(model)
        & (obs > 0)
    )

    if np.any(valid_fac2):

        ratio = (
            model[valid_fac2]
            / obs[valid_fac2]
        )

        fac2 = np.mean(
            (ratio >= 0.5)
            & (ratio <= 2.0)
        )

    else:
        fac2 = np.nan

    return {
        "N": n,
        "R": r,
        "R2": r2,
        "RMSE_ppm": rmse,
        "MAE_ppm": mae,
        "Bias_ppm": bias,
        "FAC2": fac2,
    }


# ============================================================================
# START
# ============================================================================

print_header(
    "FLEXPART-WRF 13-MINE PRIOR VALIDATION"
)

print(f"Base directory : {BASE_DIR}")
print(f"FLEXPART root  : {FLEX_ROOT}")
print(f"H matrix       : {H_FILE}")
print(f"Picarro file   : {PICARRO_FILE}")
print(f"Period         : {START_DATE.date()} to {END_DATE.date()}")
print(f"Background     : {BACKGROUND_PPM:.2f} ppmv")
print(f"Number of mines: {len(MINES)}")
print(f"Number of days : {len(DATES)}")
print(f"Expected H cols: {len(EXPECTED_H_COLUMNS)}")


# ============================================================================
# CHECK FILES
# ============================================================================

print_header("CHECKING INPUT FILES")

if not H_FILE.exists():
    raise FileNotFoundError(
        f"H matrix not found:\n{H_FILE}"
    )

if not PICARRO_FILE.exists():
    raise FileNotFoundError(
        f"Picarro file not found:\n{PICARRO_FILE}"
    )

if not FLEX_ROOT.exists():
    raise FileNotFoundError(
        f"FLEXPART root not found:\n{FLEX_ROOT}"
    )

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("All required input paths exist.")


# ============================================================================
# READ H MATRIX
# ============================================================================

print_header("READING H MATRIX")

H_df = pd.read_csv(H_FILE)

print(f"H shape: {H_df.shape}")

print("\nH columns:")
print(H_df.columns.tolist())


# ---------------------------------------------------------------------------
# Identify mine/day columns
# ---------------------------------------------------------------------------

mine_day_pattern = re.compile(
    r"^OC\d+_\d{8}$"
)

mine_day_cols = [
    col
    for col in H_df.columns
    if mine_day_pattern.match(str(col))
]

print(
    f"\nMine/day H columns found: "
    f"{len(mine_day_cols)}"
)

if len(mine_day_cols) != 403:

    raise ValueError(
        f"Expected 403 mine/day columns, "
        f"but found {len(mine_day_cols)}."
    )


# ---------------------------------------------------------------------------
# Check expected order
# ---------------------------------------------------------------------------

missing_h_cols = [
    col
    for col in EXPECTED_H_COLUMNS
    if col not in mine_day_cols
]

extra_h_cols = [
    col
    for col in mine_day_cols
    if col not in EXPECTED_H_COLUMNS
]

if missing_h_cols:
    print("\nWARNING: Missing expected H columns:")
    print(missing_h_cols)

if extra_h_cols:
    print("\nWARNING: Extra H columns:")
    print(extra_h_cols)

if missing_h_cols or extra_h_cols:
    raise ValueError(
        "H matrix columns do not match the expected "
        "13 mines × 27 days structure."
    )

# Force the correct order:
mine_day_cols = EXPECTED_H_COLUMNS.copy()

print("\nH columns are in expected order:")
print(
    f"13 mines × 27 days = "
    f"{len(mine_day_cols)} columns"
)


# ============================================================================
# H MATRIX STATISTICS
# ============================================================================

print_header("H MATRIX STATISTICS")

H_values = H_df[mine_day_cols].to_numpy(
    dtype=float
)

print(
    f"H shape                  : {H_values.shape}"
)

print(
    f"H minimum                : "
    f"{np.nanmin(H_values):.6e} ppmv"
)

print(
    f"H maximum                : "
    f"{np.nanmax(H_values):.6e} ppmv"
)

nonzero_H = H_values[
    np.isfinite(H_values)
    & (H_values > 0)
]

if len(nonzero_H) > 0:

    print(
        f"H median non-zero       : "
        f"{np.median(nonzero_H):.6e} ppmv"
    )

    print(
        f"H non-zero count        : "
        f"{len(nonzero_H)}"
    )

print(
    "\nH units: ppmv per unit daily scaling factor"
)


# ============================================================================
# BUILD FLEXPART ROW METADATA
# ============================================================================

print_header(
    "SCANNING FLEXPART OUTPUT FILES"
)

flx_files = sorted(
    FLEX_ROOT.glob(
        "*/*/flxout_d01_*.nc"
    )
)

print(
    f"FLEXPART files found: "
    f"{len(flx_files)}"
)

if len(flx_files) == 0:
    raise FileNotFoundError(
        "No flxout_d01_*.nc files found."
    )


# ============================================================================
# BUILD RECEPTOR/TIME TABLE
# ============================================================================

records = []

for flx_file in flx_files:

    receptor = extract_receptor_from_path(
        flx_file
    )

    if receptor is None:
        continue

    case_time_utc = extract_case_datetime(
        flx_file.parent
    )

    if case_time_utc is None:
        warnings.warn(
            f"Could not extract datetime from:\n"
            f"{flx_file}"
        )
        continue

    # UTC -> AEST
    time_aest = (
        case_time_utc
        + pd.Timedelta(hours=10)
    )

    # Restrict to desired validation period
    if (
        time_aest < START_DATE
        or time_aest >= END_DATE + pd.Timedelta(days=1)
    ):
        continue

    records.append(
        {
            "receptor": receptor,
            "time_utc": case_time_utc,
            "time_aest": time_aest,
            "flx_file": str(flx_file),
        }
    )


meta = pd.DataFrame(records)

if meta.empty:
    raise RuntimeError(
        "No FLEXPART files fall within the requested "
        "May 1–31, 2026 period."
    )


meta = meta.sort_values(
    ["receptor", "time_aest"]
).reset_index(drop=True)


print(
    f"\nFLEXPART rows in validation period: "
    f"{len(meta)}"
)

print("\nRows by receptor:")
print(
    meta["receptor"]
    .value_counts()
    .sort_index()
)


print("\nFirst/last times by receptor:")

for receptor in RECEPTORS:

    sub = meta[
        meta["receptor"] == receptor
    ]

    if len(sub) == 0:
        print(
            f"{receptor:15s}: NO DATA"
        )
    else:
        print(
            f"{receptor:15s}: "
            f"{sub['time_aest'].min()} "
            f"to "
            f"{sub['time_aest'].max()} "
            f"({len(sub)} rows)"
        )


# ============================================================================
# CALCULATE PRIOR FLEXPART CH4
# ============================================================================

print_header(
    "CALCULATING FLEXPART PRIOR CH4"
)

# The prior daily scaling factor is 1 for every mine/day.
#
# IMPORTANT:
# H already contains the prior emissions.
#
# Therefore:
#
#     prior_enhancement = H @ ones(403)
#
# NOT:
#
#     H @ prior_emission
#
# because that would multiply the prior emissions twice.

prior_factors = np.ones(
    len(mine_day_cols),
    dtype=float
)

meta["prior_enhancement_ppm"] = (
    H_values.sum(axis=1)
)

meta["prior_total_ppm"] = (
    BACKGROUND_PPM
    + meta["prior_enhancement_ppm"]
)


# ============================================================================
# PRIOR STATISTICS
# ============================================================================

print_header("FLEXPART PRIOR STATISTICS")

print(
    f"Global prior enhancement mean : "
    f"{meta['prior_enhancement_ppm'].mean():.6f} ppm"
)

print(
    f"Global prior enhancement max  : "
    f"{meta['prior_enhancement_ppm'].max():.6f} ppm"
)

print(
    f"Global prior total mean       : "
    f"{meta['prior_total_ppm'].mean():.6f} ppm"
)

print(
    f"Global prior total max        : "
    f"{meta['prior_total_ppm'].max():.6f} ppm"
)


print("\nPrior statistics by receptor:")

for receptor in RECEPTORS:

    sub = meta[
        meta["receptor"] == receptor
    ]

    if len(sub) == 0:
        continue

    print(
        f"\n{receptor.upper()}"
    )

    print(
        f"  N                         : "
        f"{len(sub)}"
    )

    print(
        f"  Enhancement mean          : "
        f"{sub['prior_enhancement_ppm'].mean():.6f} ppm"
    )

    print(
        f"  Enhancement max           : "
        f"{sub['prior_enhancement_ppm'].max():.6f} ppm"
    )

    print(
        f"  Total CH4 mean            : "
        f"{sub['prior_total_ppm'].mean():.6f} ppm"
    )

    print(
        f"  Total CH4 max             : "
        f"{sub['prior_total_ppm'].max():.6f} ppm"
    )

# ============================================================
# READ PICARRO OBSERVATIONS
# ============================================================

print("\n" + "=" * 90)
print("READING PICARRO OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(PICARRO_FILE)

# ------------------------------------------------------------
# Use the dedicated Datetime column.
#
# CSV format:
#   1/01/2026 1:00
#
# This is DD/MM/YYYY HH:MM.
# ------------------------------------------------------------

obs["time_aest"] = pd.to_datetime(
    obs["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

print("\nPicarro datetime parsing:")
print("  Total rows       :", len(obs))
print("  Valid timestamps :", obs["time_aest"].notna().sum())
print("  Invalid          :", obs["time_aest"].isna().sum())

print("\nFirst 5 parsed timestamps:")
print(obs["time_aest"].head().to_string(index=False))

print("\nFull Picarro time range:")
print("  min:", obs["time_aest"].min())
print("  max:", obs["time_aest"].max())

# ------------------------------------------------------------
# Restrict to validation period
# ------------------------------------------------------------

obs = obs[
    (obs["time_aest"] >= START_DATE) &
    (obs["time_aest"] < END_DATE + pd.Timedelta(days=1))
].copy()

print("\nPicarro rows after date filtering:")
print("  Rows:", len(obs))

print("  min:", obs["time_aest"].min())
print("  max:", obs["time_aest"].max())

# ------------------------------------------------------------
# Ensure observation columns are numeric
# ------------------------------------------------------------

for col in [
    "Singleton cleaned",
    "Muswellbrook cleaned",
    "Merriwa cleaned"
]:
    obs[col] = pd.to_numeric(obs[col], errors="coerce")

# ------------------------------------------------------------
# Round to nearest hour.
#
# Datetime values are already hourly in this CSV, but keeping
# this makes the matching robust.
# ------------------------------------------------------------

obs["time_hour"] = obs["time_aest"].dt.round("h")

# ------------------------------------------------------------
# If duplicate hourly timestamps exist, average them.
# ------------------------------------------------------------

obs_hourly = (
    obs.groupby("time_hour", as_index=False)[
        [
            "Singleton cleaned",
            "Muswellbrook cleaned",
            "Merriwa cleaned"
        ]
    ]
    .mean()
)

print("\nPicarro hourly observations:")
print("  Unique hourly times:", len(obs_hourly))
print("  min:", obs_hourly["time_hour"].min())
print("  max:", obs_hourly["time_hour"].max())

print("\nNumber of valid observations:")
for site, col in {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}.items():
    print(
        f"  {site:15s}: "
        f"{obs_hourly[col].notna().sum():4d}"
    )


# ============================================================================
# PREPARE OBSERVATIONS
# ============================================================================

print_header(
    "PREPARING HOURLY PICARRO OBSERVATIONS"
)

# Picarro timestamps are already AEST.
#
# Round to nearest hour, as used in the previous validation.

obs["time_hour"] = (
    obs["time_aest"]
    .dt.round("h")
)

# Keep only requested period

obs = obs[
    (obs["time_hour"] >= START_DATE)
    &
    (
        obs["time_hour"]
        < END_DATE + pd.Timedelta(days=1)
    )
].copy()


# ============================================================================
# CREATE HOURLY OBSERVATION DATAFRAME
# ============================================================================

obs_hourly = (
    obs
    .groupby("time_hour")
    [
        [
            RECEPTORS["singleton"]["obs_column"],
            RECEPTORS["muswellbrook"]["obs_column"],
            RECEPTORS["merriwa"]["obs_column"],
        ]
    ]
    .mean()
    .reset_index()
)

obs_hourly = obs_hourly.rename(
    columns={
        "time_hour": "time_aest"
    }
)


print(
    f"Hourly Picarro timestamps: "
    f"{len(obs_hourly)}"
)


# ============================================================================
# MERGE OBSERVATIONS WITH FLEXPART
# ============================================================================

print_header(
    "MATCHING FLEXPART WITH PICARRO"
)

validation = meta.copy()

# ---------------------------------------------------------------------------
# Add observation column for each receptor
# ---------------------------------------------------------------------------

validation["obs_ch4_ppm"] = np.nan

print("\n" + "=" * 100)
print("DEBUGGING FLEXPART / PICARRO TIME MATCHING")
print("=" * 100)

print("\nFLEXPART times:")
print(
    validation["time_aest"]
    .head(20)
    .to_string(index=False)
)

print("\nFLEXPART time dtype:")
print(validation["time_aest"].dtype)

print("\nPicarro hourly times:")
print(
    obs_hourly["time_aest"]
    .head(20)
    .to_string(index=False)
)

print("\nPicarro hourly dtype:")
print(obs_hourly["time_aest"].dtype)

print("\nExact timestamp overlap:")

flex_times = set(
    validation["time_aest"]
)

picarro_times = set(
    obs_hourly["time_aest"]
)

overlap = flex_times.intersection(
    picarro_times
)

print(
    f"FLEXPART unique times : {len(flex_times)}"
)

print(
    f"Picarro unique times  : {len(picarro_times)}"
)

print(
    f"Exact matches         : {len(overlap)}"
)

if len(overlap) > 0:
    print("\nExample matching times:")
    print(
        sorted(overlap)[:20]
    )
else:
    print("\nNO EXACT TIMESTAMP MATCHES FOUND.")

for receptor, info in RECEPTORS.items():

    mask = (
        validation["receptor"]
        == receptor
    )

    obs_col = info["obs_column"]

    obs_map = (
        obs_hourly[
            [
                "time_aest",
                obs_col
            ]
        ]
        .dropna(subset=[obs_col])
        .drop_duplicates(
            subset=["time_aest"]
        )
        .set_index("time_aest")[obs_col]
    )

    validation.loc[mask, "obs_ch4_ppm"] = (
        validation.loc[mask, "time_aest"]
        .map(obs_map)
    )


# ---------------------------------------------------------------------------
# Model - observation difference
# ---------------------------------------------------------------------------

validation["model_minus_obs_ppm"] = (
    validation["prior_total_ppm"]
    - validation["obs_ch4_ppm"]
)


# ============================================================================
# MATCHING SUMMARY
# ============================================================================

print("\nMatching summary:")

for receptor in RECEPTORS:

    sub = validation[
        validation["receptor"]
        == receptor
    ]

    n_model = len(sub)

    n_obs = (
        sub["obs_ch4_ppm"]
        .notna()
        .sum()
    )

    print(
        f"{receptor:15s}: "
        f"model={n_model:4d}, "
        f"matched observations={n_obs:4d}"
    )


# ============================================================================
# CALCULATE ENHANCEMENT OBSERVATION
# ============================================================================

validation["obs_enhancement_ppm"] = (
    validation["obs_ch4_ppm"]
    - BACKGROUND_PPM
)

validation["model_enhancement_ppm"] = (
    validation["prior_enhancement_ppm"]
)


# ============================================================================
# SAVE VALIDATION TABLE
# ============================================================================

validation_file = (
    H_DIR
    / "FLEXPART_prior_validation_May2026.csv"
)

validation.to_csv(
    validation_file,
    index=False
)

print(
    f"\nValidation table saved:\n"
    f"{validation_file}"
)


# ============================================================================
# CALCULATE METRICS
# ============================================================================

print_header(
    "VALIDATION METRICS"
)

metric_records = []


for receptor in RECEPTORS:

    sub = validation[
        validation["receptor"]
        == receptor
    ].copy()

    # ---------------------------------------------------------------
    # Total CH4
    # ---------------------------------------------------------------

    metrics_total = calculate_metrics(
        sub["obs_ch4_ppm"].to_numpy(),
        sub["prior_total_ppm"].to_numpy()
    )

    metrics_total["receptor"] = receptor
    metrics_total["quantity"] = "Total CH4"

    metric_records.append(
        metrics_total
    )

    # ---------------------------------------------------------------
    # Enhancement
    # ---------------------------------------------------------------

    metrics_enh = calculate_metrics(
        sub["obs_enhancement_ppm"].to_numpy(),
        sub["model_enhancement_ppm"].to_numpy()
    )

    metrics_enh["receptor"] = receptor
    metrics_enh["quantity"] = "CH4 enhancement"

    metric_records.append(
        metrics_enh
    )


metrics_df = pd.DataFrame(
    metric_records
)

# Put useful columns first

metrics_df = metrics_df[
    [
        "receptor",
        "quantity",
        "N",
        "R",
        "R2",
        "RMSE_ppm",
        "MAE_ppm",
        "Bias_ppm",
        "FAC2",
    ]
]


print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================================
# SAVE METRICS
# ============================================================================

metrics_file = (
    H_DIR
    / "FLEXPART_prior_validation_metrics_May2026.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

print(
    f"\nMetrics saved:\n"
    f"{metrics_file}"
)


# ============================================================================
# PLOTTING
# ============================================================================

print_header(
    "CREATING FLEXPART PRIOR VS PICARRO PLOTS"
)

print(
    "\nObservation plotting rule:"
)

print(
    "  FLEXPART = continuous line using ALL model points"
)

print(
    "  Picarro  = line + markers"
)

print(
    "  Missing Picarro values = NaN / gaps"
)

print(
    "  No interpolation is performed"
)


# ============================================================================
# PLOT STYLE SETTINGS
# ============================================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


# ============================================================================
# INDIVIDUAL RECEPTOR PLOTS
# ============================================================================

figures = []


for receptor, info in RECEPTORS.items():

    print(
        f"\nCreating plot for "
        f"{receptor.capitalize()}..."
    )

    sub = validation[
        validation["receptor"]
        == receptor
    ].copy()

    sub = sub.sort_values(
        "time_aest"
    )

    # ---------------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    # ---------------------------------------------------------------
    # FLEXPART prior
    # ---------------------------------------------------------------

    ax.plot(
        sub["time_aest"],
        sub["prior_total_ppm"],
        label="FLEXPART prior",
        linewidth=1.5,
    )

    # ---------------------------------------------------------------
    # Picarro observations
    #
    # IMPORTANT:
    # Do NOT drop NaN values here.
    #
    # Matplotlib automatically breaks the line at NaN values.
    # ---------------------------------------------------------------

    ax.plot(
        sub["time_aest"],
        sub["obs_ch4_ppm"],
        label="Picarro observation",
        linewidth=1.1,
        marker="o",
        markersize=3,
    )

    # ---------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------

    ax.set_title(
        f"{receptor.capitalize()} — "
        f"FLEXPART-WRF prior vs Picarro"
    )

    ax.set_xlabel(
        "Time (AEST)"
    )

    ax.set_ylabel(
        "CH₄ (ppmv)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )

    # ---------------------------------------------------------------
    # Date formatting
    # ---------------------------------------------------------------

    ax.xaxis.set_major_locator(
        mdates.DayLocator()
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    fig.autofmt_xdate()

    # ---------------------------------------------------------------
    # Number of observations
    # ---------------------------------------------------------------

    n_obs = (
        sub["obs_ch4_ppm"]
        .notna()
        .sum()
    )

    # ---------------------------------------------------------------
    # Add summary text
    # ---------------------------------------------------------------

    obs_mean = (
        sub["obs_ch4_ppm"]
        .mean()
    )

    model_mean = (
        sub["prior_total_ppm"]
        .mean()
    )

    text = (
        f"N obs = {n_obs}\n"
        f"Obs mean = {obs_mean:.3f} ppm"
        if np.isfinite(obs_mean)
        else f"N obs = {n_obs}\n"
        f"Obs mean = N/A"
    )

    text += (
        f"\nPrior mean = {model_mean:.3f} ppm"
    )

    ax.text(
        0.01,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output_png = (
        PLOT_DIR
        / (
            f"FLEXPART_prior_vs_Picarro_"
            f"{receptor}_May2026.png"
        )
    )

    fig.savefig(
        output_png,
        dpi=200,
        bbox_inches="tight"
    )

    print(
        f"Saved:\n{output_png}"
    )

    figures.append(fig)


# ============================================================================
# COMBINED THREE-RECEPTOR PLOT
# ============================================================================

print(
    "\nCreating combined three-receptor plot..."
)

fig_combined, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(14, 11),
    sharex=True
)


for ax, (receptor, info) in zip(
    axes,
    RECEPTORS.items()
):

    sub = validation[
        validation["receptor"]
        == receptor
    ].copy()

    sub = sub.sort_values(
        "time_aest"
    )

    # ---------------------------------------------------------------
    # FLEXPART prior
    # ---------------------------------------------------------------

    ax.plot(
        sub["time_aest"],
        sub["prior_total_ppm"],
        label="FLEXPART prior",
        linewidth=1.5,
    )

    # ---------------------------------------------------------------
    # Picarro observation
    #
    # NaN values are retained.
    # Therefore gaps remain gaps.
    # ---------------------------------------------------------------

    ax.plot(
        sub["time_aest"],
        sub["obs_ch4_ppm"],
        label="Picarro observation",
        linewidth=1.1,
        marker="o",
        markersize=3,
    )

    # ---------------------------------------------------------------
    # Axis
    # ---------------------------------------------------------------

    ax.set_ylabel(
        "CH₄ (ppmv)"
    )

    ax.set_title(
        receptor.capitalize()
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    n_obs = (
        sub["obs_ch4_ppm"]
        .notna()
        .sum()
    )

    model_mean = (
        sub["prior_total_ppm"]
        .mean()
    )

    obs_mean = (
        sub["obs_ch4_ppm"]
        .mean()
    )

    if np.isfinite(obs_mean):

        stats_text = (
            f"N = {n_obs}   "
            f"Obs mean = {obs_mean:.3f} ppm   "
            f"Prior mean = {model_mean:.3f} ppm"
        )

    else:

        stats_text = (
            f"N = {n_obs}   "
            f"Obs mean = N/A   "
            f"Prior mean = {model_mean:.3f} ppm"
        )

    ax.text(
        0.01,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )


# ============================================================================
# Combined plot x-axis
# ============================================================================

axes[-1].set_xlabel(
    "Time (AEST)"
)

axes[-1].xaxis.set_major_locator(
    mdates.DayLocator()
)

axes[-1].xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b")
)

fig_combined.suptitle(
    "FLEXPART-WRF Prior vs Picarro CH₄ — "
    "Upper Hunter, May 2026",
    fontsize=14,
)

fig_combined.tight_layout(
    rect=[0, 0, 1, 0.96]
)


# ============================================================================
# SAVE COMBINED PLOT
# ============================================================================

combined_png = (
    PLOT_DIR
    / "FLEXPART_prior_vs_Picarro_all_receptors_May2026.png"
)

fig_combined.savefig(
    combined_png,
    dpi=200,
    bbox_inches="tight"
)

print(
    "\nCombined plot saved:"
)

print(
    combined_png
)


# ============================================================================
# DISPLAY ALL FIGURES THROUGH X11
# ============================================================================

print_header(
    "DISPLAYING PLOTS THROUGH X11"
)

print(
    "Close the plot windows to allow the script to finish."
)

# All individual figures and the combined figure have been
# created and saved. Display them together.

plt.show()

# After the GUI windows are closed:

for fig in figures:
    plt.close(fig)

plt.close(fig_combined)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_header(
    "VALIDATION COMPLETE"
)

print(
    f"Validation CSV:\n"
    f"{validation_file}"
)

print(
    f"\nMetrics CSV:\n"
    f"{metrics_file}"
)

print(
    f"\nPlot directory:\n"
    f"{PLOT_DIR}"
)

print(
    "\nGenerated plots:"
)

for receptor in RECEPTORS:

    print(
        f"  {PLOT_DIR / f'FLEXPART_prior_vs_Picarro_{receptor}_May2026.png'}"
    )

print(
    f"  {combined_png}"
)

print(
    "\nFinished successfully."
)

