#!/usr/bin/env python3

"""
Validate FLEXPART-WRF prior forward model against Picarro observations.

The H matrix already contains the prior mine emissions.

Therefore:

    prior_enhancement = sum(H row)

and:

    prior_total_CH4 = BACKGROUND_PPM + prior_enhancement

The Bayesian inversion later uses:

    y = H @ f

where:

    f = 13 mines x 27 daily scaling factors

For the prior:

    f = 1 for every mine/day

so:

    H @ ones = prior enhancement

H units:

    ppmv per unit daily emission scaling factor
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

H_FILE = (
    BASE
    / "H_matrix_flexpart"
    / "H_flexpart_0_10m_13mine_Jan2026.csv"
)

FLEX_ROOT = (
    BASE
    / "run/hourly_obs_3layer"
)

OBS_FILE = (
    Path.home()
    / "flexpart-wrf/emission_UH/RH_upperHunter/tapm_output"
    / "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

BACKGROUND_PPM = 1.85

START_DATE = pd.Timestamp("2026-01-05")
END_DATE   = pd.Timestamp("2026-01-31")


# =============================================================================
# RECEPTORS
# =============================================================================

RECEPTORS = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}


# =============================================================================
# HELPERS
# =============================================================================

def parse_case_timestamp(path):

    """
    Extract FLEXPART case timestamp.

    Case directory:

        YYYYMMDD_HHMMSS

    This is the FLEXPART release/receptor time in UTC.
    """

    m = re.search(
        r"(\d{8})_(\d{6})",
        str(path)
    )

    if not m:
        raise ValueError(
            f"Cannot find timestamp in: {path}"
        )

    return pd.to_datetime(
        m.group(1) + m.group(2),
        format="%Y%m%d%H%M%S",
        utc=True,
    )


def metrics(obs, mod):

    """
    Calculate standard model-observation statistics.
    """

    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)

    mask = (
        np.isfinite(obs)
        &
        np.isfinite(mod)
    )

    obs = obs[mask]
    mod = mod[mask]

    n = len(obs)

    if n == 0:

        return {
            "N": 0,
            "R": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "FAC2": np.nan,
            "Mean_obs": np.nan,
            "Mean_model": np.nan,
        }

    diff = mod - obs

    rmse = np.sqrt(
        np.mean(diff ** 2)
    )

    mae = np.mean(
        np.abs(diff)
    )

    bias = np.mean(diff)

    mean_obs = np.mean(obs)
    mean_model = np.mean(mod)

    if n > 1:

        r = np.corrcoef(
            obs,
            mod
        )[0, 1]

        r2 = r ** 2

    else:

        r = np.nan
        r2 = np.nan

    positive = (
        (obs > 0)
        &
        (mod > 0)
    )

    if np.any(positive):

        ratio = (
            mod[positive]
            /
            obs[positive]
        )

        fac2 = np.mean(
            (ratio >= 0.5)
            &
            (ratio <= 2.0)
        )

    else:

        fac2 = np.nan

    return {

        "N": n,

        "R": r,

        "R2": r2,

        "RMSE": rmse,

        "MAE": mae,

        "Bias": bias,

        "FAC2": fac2,

        "Mean_obs": mean_obs,

        "Mean_model": mean_model,
    }


# =============================================================================
# READ H MATRIX
# =============================================================================

print("=" * 90)
print("FLEXPART-WRF PRIOR FORWARD VALIDATION")
print("=" * 90)

print("\nReading H matrix:")
print(H_FILE)

H = pd.read_csv(
    H_FILE
)

print(
    f"H shape: {H.shape}"
)


# =============================================================================
# IDENTIFY MINE/DAY COLUMNS
# =============================================================================

mine_day_cols = [

    c for c in H.columns

    if re.match(
        r"^OC\d+_\d{8}$",
        str(c)
    )
]


if len(mine_day_cols) != 351:

    raise ValueError(
        f"Expected 351 mine/day columns, "
        f"found {len(mine_day_cols)}"
    )


print(
    f"Mine/day columns: "
    f"{len(mine_day_cols)}"
)


H_values = H[
    mine_day_cols
].to_numpy(
    dtype=float
)


print(
    "\nH statistics:"
)

print(
    f"  Minimum       = "
    f"{np.nanmin(H_values):.6e} ppmv"
)

print(
    f"  Maximum       = "
    f"{np.nanmax(H_values):.6e} ppmv"
)

print(
    f"  Median nonzero = "
    f"{np.median(H_values[H_values > 0]):.6e} ppmv"
)

print(
    f"  Non-zero      = "
    f"{np.count_nonzero(H_values):,}"
)


# =============================================================================
# CHECK H COLUMN STRUCTURE
# =============================================================================

print(
    "\nChecking mine/day structure..."
)

expected_mines = [
    f"OC{i}"
    for i in range(1, 14)
]

expected_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

expected_columns = [

    f"{mine}_{date:%Y%m%d}"

    for date in expected_dates

    for mine in expected_mines
]


if mine_day_cols != expected_columns:

    print(
        "\nWARNING:"
    )

    print(
        "H column order does not exactly "
        "match expected mine/day order."
    )

    print(
        "First 20 H columns:"
    )

    print(
        mine_day_cols[:20]
    )

else:

    print(
        "H column structure confirmed:"
    )

    print(
        "13 mines × 27 days = 351 columns"
    )


# =============================================================================
# FIND FLEXPART FILES
# =============================================================================

print(
    "\nFinding FLEXPART files..."
)

flex_files = sorted(
    FLEX_ROOT.glob(
        "*/*/flxout_d01_*.nc"
    )
)

print(
    f"FLEXPART files: "
    f"{len(flex_files)}"
)


if len(flex_files) != len(H_values):

    raise ValueError(

        f"FLEXPART files "
        f"({len(flex_files)}) != "
        f"H rows ({len(H_values)})"
    )


# =============================================================================
# BUILD FLEXPART METADATA
# =============================================================================

rows = []


for i, fp in enumerate(flex_files):

    parts = fp.parts

    # Expected:
    #
    # .../hourly_obs_3layer/
    #     receptor/
    #       YYYYMMDD_HHMMSS/
    #         flxout_d01_*.nc

    receptor = parts[-3].lower()

    case = parts[-2]

    release_utc = parse_case_timestamp(
        fp
    )

    release_aest = (
        release_utc
        +
        pd.Timedelta(hours=10)
    )

    rows.append({

        "H_row": i,

        "receptor": receptor,

        "case": case,

        "release_utc": release_utc,

        "release_aest": release_aest,
    })


meta = pd.DataFrame(
    rows
)


# =============================================================================
# FLEXPART TIME DIAGNOSTICS
# =============================================================================

print(
    "\nFLEXPART observation time range:"
)

print(
    "UTC :",
    meta.release_utc.min(),
    "to",
    meta.release_utc.max()
)

print(
    "AEST:",
    meta.release_aest.min(),
    "to",
    meta.release_aest.max()
)


print(
    "\nReceptor counts:"
)

print(
    meta.receptor.value_counts()
)


# =============================================================================
# CHECK H ROW ORDER
# =============================================================================

print(
    "\nFLEXPART H-row ordering:"
)

for receptor in RECEPTORS:

    sub = meta[
        meta.receptor == receptor
    ]

    if len(sub) == 0:
        continue

    print(
        f"{receptor:15s}: "
        f"{len(sub):4d} rows"
    )

    print(
        f"    first = "
        f"{sub.release_aest.min()}"
    )

    print(
        f"    last  = "
        f"{sub.release_aest.max()}"
    )


# =============================================================================
# PRIOR FORWARD MODEL
# =============================================================================

print(
    "\nCalculating prior forward model..."
)

print(
    "Using:"
)

print(
    "    prior enhancement = sum(H row)"
)

print(
    "    prior scaling factor = 1 "
    "for every mine/day"
)


# IMPORTANT:
#
# H already contains the prior mine emissions.
#
# Therefore:
#
#     H @ ones
#
# is equivalent to:
#
#     sum(H row)
#
# We use the latter here.

prior_factors = np.ones(
    len(mine_day_cols)
)

prior_enhancement = (
    H_values
    @
    prior_factors
)


meta[
    "prior_enhancement_ppm"
] = prior_enhancement


meta[
    "prior_total_ppm"
] = (
    BACKGROUND_PPM
    +
    meta[
        "prior_enhancement_ppm"
    ]
)


# =============================================================================
# PRIOR STATISTICS
# =============================================================================

print(
    "\nPrior forward-model statistics:"
)

print(
    meta[
        [
            "prior_enhancement_ppm",
            "prior_total_ppm",
        ]
    ].describe()
)


print(
    "\nPrior enhancement range:"
)

print(
    f"Minimum = "
    f"{meta.prior_enhancement_ppm.min():.6e} ppm"
)

print(
    f"Maximum = "
    f"{meta.prior_enhancement_ppm.max():.6e} ppm"
)

print(
    f"Mean    = "
    f"{meta.prior_enhancement_ppm.mean():.6e} ppm"
)

print(
    f"Median  = "
    f"{meta.prior_enhancement_ppm.median():.6e} ppm"
)


# =============================================================================
# RECEPTOR-SPECIFIC PRIOR STATISTICS
# =============================================================================

print(
    "\nPrior enhancement by receptor:"
)

for receptor in RECEPTORS:

    sub = meta[
        meta.receptor == receptor
    ]

    if len(sub) == 0:
        continue

    print(
        f"\n{receptor.upper()}"
    )

    print(
        f"  N      = {len(sub)}"
    )

    print(
        f"  Mean   = "
        f"{sub.prior_enhancement_ppm.mean():.6f} ppm"
    )

    print(
        f"  Median = "
        f"{sub.prior_enhancement_ppm.median():.6f} ppm"
    )

    print(
        f"  Max    = "
        f"{sub.prior_enhancement_ppm.max():.6f} ppm"
    )


# =============================================================================
# READ PICARRO
# =============================================================================

print(
    "\nReading Picarro observations:"
)

print(
    OBS_FILE
)

obs = pd.read_csv(
    OBS_FILE
)

print(
    "\nObservation columns:"
)

print(
    list(obs.columns)
)


# =============================================================================
# IDENTIFY TIME COLUMN
# =============================================================================

time_candidates = [

    c

    for c in obs.columns

    if any(

        key in str(c).lower()

        for key in [

            "time",
            "date",
            "datetime",
            "timestamp",
        ]
    )
]


print(
    "\nPossible time columns:"
)

print(
    time_candidates
)


time_col = None


for c in time_candidates:

    test = pd.to_datetime(
        obs[c],
        errors="coerce"
    )

    if test.notna().sum() > 0:

        time_col = c

        break


if time_col is None:

    raise ValueError(
        "Could not identify "
        "Picarro timestamp column."
    )


print(
    f"\nUsing Picarro time column: "
    f"{time_col}"
)


obs[
    "time_aest"
] = pd.to_datetime(
    obs[time_col],
    errors="coerce"
)


# Picarro timestamps are already AEST.

if getattr(
    obs["time_aest"].dt,
    "tz",
    None
) is not None:

    obs[
        "time_aest"
    ] = (
        obs["time_aest"]
        .dt.tz_localize(None)
    )


# =============================================================================
# HOURLY OBSERVATION MATCHING
# =============================================================================

results = []


for receptor, obs_col in RECEPTORS.items():

    if obs_col not in obs.columns:

        print(
            f"\nWARNING: {obs_col} "
            f"not found. "
            f"Skipping {receptor}."
        )

        continue


    rmeta = meta[
        meta.receptor.str.lower()
        ==
        receptor
    ].copy()


    if len(rmeta) == 0:

        continue


    # ---------------------------------------------------------
    # Picarro data
    # ---------------------------------------------------------

    obs_tmp = obs[
        [
            "time_aest",
            obs_col
        ]
    ].copy()


    obs_tmp = obs_tmp.rename(
        columns={
            obs_col:
            "obs_total_ppm"
        }
    )


    # ---------------------------------------------------------
    # Round Picarro timestamps to nearest hour.
    # ---------------------------------------------------------

    obs_tmp[
        "time_hour"
    ] = (
        obs_tmp[
            "time_aest"
        ]
        .dt.round("h")
    )


    # ---------------------------------------------------------
    # Average duplicate observations
    # ---------------------------------------------------------

    obs_hourly = (

        obs_tmp

        .dropna(
            subset=[
                "time_hour",
                "obs_total_ppm"
            ]
        )

        .groupby(
            "time_hour",
            as_index=False
        )

        ["obs_total_ppm"]

        .mean()
    )


    # FLEXPART release time is already AEST
    # after UTC + 10 hours.

    rmeta[
        "obs_time_aest"
    ] = (
        rmeta[
            "release_aest"
        ]
        .dt.tz_localize(None)
    )


    # ---------------------------------------------------------
    # Merge model and observation
    # ---------------------------------------------------------

    merged = rmeta.merge(

        obs_hourly,

        left_on="obs_time_aest",

        right_on="time_hour",

        how="left",
    )


    merged[
        "obs_enhancement_ppm"
    ] = (
        merged[
            "obs_total_ppm"
        ]
        -
        BACKGROUND_PPM
    )


    merged[
        "model_minus_obs_ppm"
    ] = (
        merged[
            "prior_total_ppm"
        ]
        -
        merged[
            "obs_total_ppm"
        ]
    )


    merged[
        "receptor"
    ] = receptor


    results.append(
        merged
    )


if not results:

    raise RuntimeError(
        "No receptor observations matched."
    )


validation = pd.concat(
    results,
    ignore_index=True
)

# =============================================================================
# PLOT PRIOR CH4 VS PICARRO
# =============================================================================

import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for HPC batch execution.
# The PNG files can then be viewed/downloaded later.


print("\n")
print("=" * 90)
print("CREATING PRIOR CH4 VS PICARRO PLOTS")
print("=" * 90)


plot_dir = (
    BASE
    / "H_matrix_flexpart"
    / "plots"
)

plot_dir.mkdir(
    parents=True,
    exist_ok=True
)


for receptor in RECEPTORS:

    sub = validation[
        validation["receptor"] == receptor
    ].copy()

    if len(sub) == 0:
        print(
            f"\nNo data for {receptor}. "
            "Skipping plot."
        )
        continue

    # ---------------------------------------------------------
    # Sort by AEST time
    # ---------------------------------------------------------

    sub = sub.sort_values(
        "release_aest"
    ).copy()

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    # ---------------------------------------------------------
    # FLEXPART prior total CH4
    #
    # Plot ALL FLEXPART points.
    #
    # This is important:
    # model points are NOT removed when observations
    # are missing.
    # ---------------------------------------------------------

    ax.plot(
        sub["release_aest"],
        sub["prior_total_ppm"],
        label="FLEXPART prior",
        linewidth=1.5,
        marker="o",
        markersize=2.5,
    )

    # ---------------------------------------------------------
    # Picarro observations
    #
    # Only non-missing observations are plotted.
    # ---------------------------------------------------------

    obs_mask = (
        sub["obs_total_ppm"].notna()
    )

    if obs_mask.any():

        ax.scatter(
            sub.loc[
                obs_mask,
                "release_aest"
            ],
            sub.loc[
                obs_mask,
                "obs_total_ppm"
            ],
            label="Picarro observation",
            s=16,
            zorder=5,
        )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    ax.set_xlabel(
        "Time (AEST)"
    )

    ax.set_ylabel(
        "CH₄ (ppmv)"
    )

    ax.set_title(
        f"FLEXPART-WRF Prior CH₄ vs Picarro — "
        f"{receptor.capitalize()}"
    )

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3
    )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------

    ax.legend(
        loc="best"
    )

    # ---------------------------------------------------------
    # Statistics text
    # ---------------------------------------------------------

    matched = sub[
        "obs_total_ppm"
    ].notna()

    n_match = matched.sum()

    if n_match > 0:

        obs_mean = (
            sub.loc[
                matched,
                "obs_total_ppm"
            ]
            .mean()
        )

        model_mean = (
            sub.loc[
                matched,
                "prior_total_ppm"
            ]
            .mean()
        )

        ax.text(
            0.01,
            0.98,
            f"N = {n_match}\n"
            f"Obs mean = {obs_mean:.3f} ppm\n"
            f"Prior mean = {model_mean:.3f} ppm",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------

    fig.autofmt_xdate()

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save PNG
    # ---------------------------------------------------------

    plot_file = (
        plot_dir
        / f"FLEXPART_prior_vs_Picarro_{receptor}_Jan2026.png"
    )

    fig.savefig(
        plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(
        f"\n{receptor.upper()}:"
    )

    print(
        f"  FLEXPART points = {len(sub)}"
    )

    print(
        f"  Picarro points   = {n_match}"
    )

    print(
        f"  Plot saved       = {plot_file}"
    )


# =============================================================================
# COMBINED THREE-RECEPTOR FIGURE
# =============================================================================

print(
    "\nCreating combined three-receptor plot..."
)


fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 12),
    sharex=True
)


for ax, receptor in zip(
    axes,
    RECEPTORS
):

    sub = validation[
        validation["receptor"] == receptor
    ].copy()

    if len(sub) == 0:
        continue

    sub = sub.sort_values(
        "release_aest"
    )

    # ---------------------------------------------------------
    # FLEXPART prior
    # ---------------------------------------------------------

    ax.plot(
        sub["release_aest"],
        sub["prior_total_ppm"],
        label="FLEXPART prior",
        linewidth=1.3,
        marker="o",
        markersize=2,
    )

    # ---------------------------------------------------------
    # Picarro
    # ---------------------------------------------------------

    obs_mask = (
        sub["obs_total_ppm"].notna()
    )

    if obs_mask.any():

        ax.scatter(
            sub.loc[
                obs_mask,
                "release_aest"
            ],
            sub.loc[
                obs_mask,
                "obs_total_ppm"
            ],
            label="Picarro observation",
            s=12,
            zorder=5,
        )

    # ---------------------------------------------------------
    # Receptor label
    # ---------------------------------------------------------

    ax.set_title(
        receptor.capitalize()
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


axes[-1].set_xlabel(
    "Time (AEST)"
)

fig.suptitle(
    "FLEXPART-WRF Prior CH₄ vs Picarro Observations",
    fontsize=15
)

fig.autofmt_xdate()

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)


combined_plot_file = (
    plot_dir
    / "FLEXPART_prior_vs_Picarro_all_receptors_Jan2026.png"
)


fig.savefig(
    combined_plot_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


print(
    f"\nCombined plot saved:"
)

print(
    combined_plot_file
)


# =============================================================================
# MATCHING SUMMARY
# =============================================================================

print(
    "\n"
    + "=" * 90
)

print(
    "OBSERVATION MATCHING SUMMARY"
)

print(
    "=" * 90
)


for receptor in RECEPTORS:

    sub = validation[
        validation.receptor
        ==
        receptor
    ]

    if len(sub) == 0:
        continue

    matched = (
        sub.obs_total_ppm
        .notna()
        .sum()
    )

    print(
        f"{receptor.upper():15s}: "
        f"{matched:4d} matched / "
        f"{len(sub):4d} FLEXPART"
    )


# =============================================================================
# METRICS
# =============================================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PRIOR MODEL VS PICARRO"
)

print(
    "=" * 90
)


all_metrics = []


for receptor in RECEPTORS:

    sub = validation[
        validation.receptor
        ==
        receptor
    ].copy()


    sub = sub.dropna(
        subset=[
            "obs_total_ppm",
            "prior_total_ppm",
        ]
    )


    if len(sub) == 0:

        print(
            f"\n{receptor.upper()}: "
            "no matched observations"
        )

        continue


    # ---------------------------------------------------------
    # TOTAL CH4
    # ---------------------------------------------------------

    m_total = metrics(

        sub[
            "obs_total_ppm"
        ].values,

        sub[
            "prior_total_ppm"
        ].values,
    )


    m_total[
        "Receptor"
    ] = receptor


    m_total[
        "Quantity"
    ] = "Total CH4"


    all_metrics.append(
        m_total
    )


    # ---------------------------------------------------------
    # ENHANCEMENT
    # ---------------------------------------------------------

    m_enh = metrics(

        sub[
            "obs_enhancement_ppm"
        ].values,

        sub[
            "prior_enhancement_ppm"
        ].values,
    )


    m_enh[
        "Receptor"
    ] = receptor


    m_enh[
        "Quantity"
    ] = "CH4 enhancement"


    all_metrics.append(
        m_enh
    )


    # ---------------------------------------------------------
    # PRINT TOTAL
    # ---------------------------------------------------------

    print(
        f"\n{receptor.upper()} "
        "(TOTAL CH4)"
    )

    print(
        "-" * 60
    )


    for key in [

        "N",
        "R",
        "R2",
        "RMSE",
        "MAE",
        "Bias",
        "FAC2",
        "Mean_obs",
        "Mean_model",
    ]:

        print(
            f"{key:15s}: "
            f"{m_total[key]}"
        )


    # ---------------------------------------------------------
    # PRINT ENHANCEMENT
    # ---------------------------------------------------------

    print(
        f"\n{receptor.upper()} "
        "(CH4 ENHANCEMENT)"
    )

    print(
        "-" * 60
    )


    for key in [

        "N",
        "R",
        "R2",
        "RMSE",
        "MAE",
        "Bias",
        "FAC2",
        "Mean_obs",
        "Mean_model",
    ]:

        print(
            f"{key:15s}: "
            f"{m_enh[key]}"
        )


# =============================================================================
# ALL RECEPTORS
# =============================================================================

sub = validation.dropna(
    subset=[
        "obs_total_ppm",
        "prior_total_ppm",
    ]
)


if len(sub) > 0:

    # ---------------------------------------------------------
    # Total CH4
    # ---------------------------------------------------------

    m_total = metrics(

        sub[
            "obs_total_ppm"
        ].values,

        sub[
            "prior_total_ppm"
        ].values,
    )


    m_total[
        "Receptor"
    ] = "ALL"


    m_total[
        "Quantity"
    ] = "Total CH4"


    all_metrics.append(
        m_total
    )


    print(
        "\nALL RECEPTORS "
        "(TOTAL CH4)"
    )

    print(
        "-" * 60
    )


    for key in [

        "N",
        "R",
        "R2",
        "RMSE",
        "MAE",
        "Bias",
        "FAC2",
        "Mean_obs",
        "Mean_model",
    ]:

        print(
            f"{key:15s}: "
            f"{m_total[key]}"
        )


    # ---------------------------------------------------------
    # Enhancement
    # ---------------------------------------------------------

    m_enh = metrics(

        sub[
            "obs_enhancement_ppm"
        ].values,

        sub[
            "prior_enhancement_ppm"
        ].values,
    )


    m_enh[
        "Receptor"
    ] = "ALL"


    m_enh[
        "Quantity"
    ] = "CH4 enhancement"


    all_metrics.append(
        m_enh
    )


    print(
        "\nALL RECEPTORS "
        "(CH4 ENHANCEMENT)"
    )

    print(
        "-" * 60
    )


    for key in [

        "N",
        "R",
        "R2",
        "RMSE",
        "MAE",
        "Bias",
        "FAC2",
        "Mean_obs",
        "Mean_model",
    ]:

        print(
            f"{key:15s}: "
            f"{m_enh[key]}"
        )


# =============================================================================
# SAVE
# =============================================================================

out_dir = (
    BASE
    / "H_matrix_flexpart"
)

out_dir.mkdir(
    exist_ok=True
)


out_file = (
    out_dir
    / "FLEXPART_prior_validation_Jan2026.csv"
)


validation.to_csv(
    out_file,
    index=False
)


metrics_file = (
    out_dir
    / "FLEXPART_prior_validation_metrics_Jan2026.csv"
)


pd.DataFrame(
    all_metrics
).to_csv(
    metrics_file,
    index=False
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n"
    + "=" * 90
)

print(
    "VALIDATION COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nDetailed results:"
)

print(
    out_file
)

print(
    "\nMetrics:"
)

print(
    metrics_file
)

print(
    "\nH interpretation:"
)

print(
    "  H = ppmv response to unit daily mine scaling factor"
)

print(
    "  Prior factors = 1 for all 351 mine/day parameters"
)

print(
    "  Prior enhancement = H @ ones"
)

print(
    f"  Background = {BACKGROUND_PPM:.2f} ppm"
)

print(
    "\nNext step:"
)

print(
    "  Use the prior validation results to assess "
    "whether the FLEXPART prior emissions are "
    "consistent with Picarro before running the inversion."
)

print(
    "=" * 90
)
