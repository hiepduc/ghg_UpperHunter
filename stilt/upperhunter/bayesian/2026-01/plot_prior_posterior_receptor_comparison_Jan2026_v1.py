#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upper Hunter CH4 Bayesian Inversion
Prior vs Posterior Prediction at Three Picarro Receptors
January 2026

Prior:
    CH4_prior = BACKGROUND_PPM + sum(H_i,m)

Posterior:
    CH4_posterior = BACKGROUND_PPM
                    + sum(H_i,m * scaling_factor_m,day)

Datetime handling:
    Picarro Date: DD/MM/YYYY
    Picarro Time: HH:MM
    Time convention: AEST = UTC + 10 hours
    No daylight-saving conversion.

Observations:
    Singleton cleaned
    Muswellbrook cleaned
    Merriwa cleaned

Important:
    H-matrix timestamps define the prediction timeline.
    Observations are left-joined to the H matrix.
    No predictions are fabricated for timestamps without footprints.
    Statistics use finite observation-prediction pairs only.
    Empty observation series are handled safely.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/bayesian/2026-01"
)

H_FILE = os.path.join(
    BASE_DIR,
    "H-matrix",
    "STILT_H_matrix_mines_Jan2026.csv",
)

POSTERIOR_FILE = os.path.join(
    BASE_DIR,
    "Bayesian_daily_inversion_Jan2026",
    "daily_scaling_factors_Jan2026.csv",
)

OBS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Bayesian_daily_inversion_Jan2026",
)

BACKGROUND_PPM = 1.85

START_DATE = pd.Timestamp("2026-01-01 00:00:00")
END_DATE = pd.Timestamp("2026-02-01 00:00:00")

MINES = [f"OC{i}" for i in range(1, 14)]

SITES = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]

OBS_COLUMNS = {
    "Singleton": "Singleton cleaned",
    "Muswellbrook": "Muswellbrook cleaned",
    "Merriwa": "Merriwa cleaned",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# OUTPUT FILES
# =============================================================================

TIMESERIES_ALL_PNG = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_receptor_timeseries_Jan2026.png",
)

TIMESERIES_ALL_PDF = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_receptor_timeseries_Jan2026.pdf",
)

SCATTER_PNG = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_scatter_Jan2026.png",
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_metrics_Jan2026.csv",
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_predictions_Jan2026.csv",
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_picarro_datetime(obs):
    """
    Parse Date and Time separately using an explicit format.

    This avoids automatic datetime inference differences between pandas
    versions in stilt_env2 and base_fix.

    Example:
        Date = 01/01/2026
        Time = 01:00

    Result:
        2026-01-01 01:00:00

    The timestamps remain naive AEST wall-clock times.
    No timezone conversion is performed.
    """

    date_text = (
        obs["Date"]
        .astype("string")
        .str.strip()
    )

    time_text = (
        obs["Time"]
        .astype("string")
        .str.strip()
    )

    datetime_text = date_text + " " + time_text

    datetime_aest = pd.to_datetime(
        datetime_text,
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    n_bad = int(datetime_aest.isna().sum())

    print("\nPicarro datetime parsing")
    print("-" * 60)
    print(f"Total source rows:       {len(obs)}")
    print(f"Successfully parsed:     {len(obs) - n_bad}")
    print(f"Failed to parse:         {n_bad}")

    if n_bad:
        print("\nExamples of failed timestamps:")
        print(
            obs.loc[
                datetime_aest.isna(),
                ["Date", "Time", "Datetime"],
            ]
            .head(10)
            .to_string(index=False)
        )

    return datetime_aest


def calculate_metrics(obs_values, pred_values):
    """Calculate statistics from finite observation-prediction pairs."""

    obs_values = np.asarray(obs_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    mask = (
        np.isfinite(obs_values)
        & np.isfinite(pred_values)
    )

    obs_values = obs_values[mask]
    pred_values = pred_values[mask]

    n = len(obs_values)

    result = {
        "N": n,
        "R": np.nan,
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "FAC2": np.nan,
    }

    if n == 0:
        return result

    residual = pred_values - obs_values

    result["RMSE"] = float(
        np.sqrt(np.mean(residual ** 2))
    )

    result["MAE"] = float(
        np.mean(np.abs(residual))
    )

    if (
        n > 1
        and np.std(obs_values) > 0
        and np.std(pred_values) > 0
    ):
        r = np.corrcoef(
            obs_values,
            pred_values,
        )[0, 1]

        result["R"] = float(r)
        result["R2"] = float(r ** 2)

    positive = obs_values > 0

    if np.any(positive):
        ratio = (
            pred_values[positive]
            / obs_values[positive]
        )

        result["FAC2"] = float(
            np.mean(
                (ratio >= 0.5)
                & (ratio <= 2.0)
            )
        )

    return result


def print_metrics(label, metrics):
    """Print metrics without formatting errors for unavailable values."""

    print(f"\n{label}:")
    print(f"  N    = {metrics['N']}")

    for key, unit in [
        ("R", ""),
        ("R2", ""),
        ("RMSE", " ppm"),
        ("MAE", " ppm"),
        ("FAC2", ""),
    ]:
        value = metrics[key]

        if np.isfinite(value):
            print(f"  {key:<4} = {value:.5f}{unit}")
        else:
            print(f"  {key:<4} = unavailable")


def format_date_axis(ax):
    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )


def get_site_metrics(metrics_df, site, model):
    rows = metrics_df[
        (metrics_df["site"] == site)
        & (metrics_df["model"] == model)
    ]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

for filename in [
    H_FILE,
    POSTERIOR_FILE,
    OBS_FILE,
]:
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"Input file does not exist:\n{filename}"
        )


# =============================================================================
# READ STILT H MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("READING STILT H MATRIX")
print("=" * 80)

print(H_FILE)

H = pd.read_csv(H_FILE)

print(f"Rows: {len(H)}")
print(f"Columns: {len(H.columns)}")
print("Columns:", H.columns.tolist())

missing_h = [
    mine for mine in MINES
    if mine not in H.columns
]

if missing_h:
    raise RuntimeError(
        "Missing H-matrix mine columns: "
        + ", ".join(missing_h)
    )

if "site" not in H.columns:
    raise RuntimeError(
        "The H matrix does not contain a 'site' column."
    )

if "datetime_aest" in H.columns:
    H["datetime_aest"] = pd.to_datetime(
        H["datetime_aest"],
        errors="coerce",
    )

elif "datetime_utc" in H.columns:
    print(
        "datetime_aest not found. "
        "Converting datetime_utc to AEST by adding 10 hours."
    )

    H["datetime_utc"] = pd.to_datetime(
        H["datetime_utc"],
        errors="coerce",
    )

    H["datetime_aest"] = (
        H["datetime_utc"]
        + pd.Timedelta(hours=10)
    )

else:
    raise RuntimeError(
        "Neither datetime_aest nor datetime_utc "
        "was found in the H matrix."
    )

n_bad_h = int(H["datetime_aest"].isna().sum())

if n_bad_h:
    print(f"WARNING: removing {n_bad_h} H rows with invalid timestamps.")

    H = H.dropna(
        subset=["datetime_aest"]
    ).copy()

H["site"] = (
    H["site"]
    .astype(str)
    .str.strip()
)

# Restrict H matrix to January 2026.
H = H[
    (H["datetime_aest"] >= START_DATE)
    & (H["datetime_aest"] < END_DATE)
].copy()

# Check and remove duplicate site/time rows.
n_before = len(H)

H = H.drop_duplicates(
    subset=["site", "datetime_aest"],
    keep="first",
).copy()

if len(H) != n_before:
    print(
        "WARNING: removed "
        f"{n_before - len(H)} duplicate site/time rows."
    )

# Convert mine columns to numeric.
for mine in MINES:
    H[mine] = pd.to_numeric(
        H[mine],
        errors="coerce",
    )

print(f"\nFinal January H rows: {len(H)}")

for site in SITES:
    h_site = H[
        H["site"].str.lower() == site.lower()
    ]

    print(
        f"{site}: {len(h_site)} H rows, "
        f"{h_site['datetime_aest'].nunique()} unique timestamps"
    )


# =============================================================================
# READ POSTERIOR DAILY SCALING FACTORS
# =============================================================================

print("\n" + "=" * 80)
print("READING POSTERIOR DAILY SCALING FACTORS")
print("=" * 80)

print(POSTERIOR_FILE)

post = pd.read_csv(POSTERIOR_FILE)

if "date" not in post.columns:
    raise RuntimeError(
        "The posterior scaling-factor file has no 'date' column."
    )

post["date"] = pd.to_datetime(
    post["date"],
    errors="coerce",
).dt.normalize()

post = post.dropna(
    subset=["date"]
).copy()

missing_post = [
    mine for mine in MINES
    if mine not in post.columns
]

if missing_post:
    raise RuntimeError(
        "Missing posterior scaling-factor columns: "
        + ", ".join(missing_post)
    )

for mine in MINES:
    post[mine] = pd.to_numeric(
        post[mine],
        errors="coerce",
    )

if post["date"].duplicated().any():
    duplicate_dates = (
        post.loc[
            post["date"].duplicated(keep=False),
            "date",
        ]
        .dt.strftime("%Y-%m-%d")
        .unique()
        .tolist()
    )

    raise RuntimeError(
        "Multiple posterior rows found for these dates: "
        + ", ".join(duplicate_dates)
    )

post_lookup = post.set_index("date")

print(f"Posterior days: {len(post)}")

if not post.empty:
    print(
        "Posterior date range: "
        f"{post['date'].min().date()} to "
        f"{post['date'].max().date()}"
    )


# =============================================================================
# READ PICARRO OBSERVATIONS
# =============================================================================

print("\n" + "=" * 80)
print("READING PICARRO OBSERVATIONS")
print("=" * 80)

print(OBS_FILE)

# Keep the original CSV column names.
obs = pd.read_csv(OBS_FILE)

print(f"Source rows: {len(obs)}")
print("Source columns:")
print(obs.columns.tolist())

# Parse Date and Time separately to avoid pandas-version-dependent inference.
required_datetime_columns = ["Date", "Time"]

missing_datetime_columns = [
    col for col in required_datetime_columns
    if col not in obs.columns
]

if missing_datetime_columns:
    raise RuntimeError(
        "Missing observation date/time columns: "
        + ", ".join(missing_datetime_columns)
    )

obs["datetime_aest"] = parse_picarro_datetime(obs)

# Display sample parsing results.
print("\nFirst 10 parsed timestamps:")
print(
    obs[
        ["Date", "Time", "Datetime", "datetime_aest"]
    ]
    .head(10)
    .to_string(index=False)
)

# Keep January 2026.
obs = obs[
    (obs["datetime_aest"] >= START_DATE)
    & (obs["datetime_aest"] < END_DATE)
].copy()

print(f"\nJanuary observation rows: {len(obs)}")

# Validate cleaned observation columns.
for site, column in OBS_COLUMNS.items():

    if column not in obs.columns:
        raise RuntimeError(
            f"Missing cleaned observation column for {site}: {column}"
        )

    obs[column] = pd.to_numeric(
        obs[column],
        errors="coerce",
    )

    values = obs[column].to_numpy(dtype=float)

    print(
        f"{site}: "
        f"{np.isfinite(values).sum()} finite cleaned January observations"
    )

# Duplicate observation times could multiply rows during a merge.
if obs["datetime_aest"].duplicated().any():

    n_duplicates = int(
        obs["datetime_aest"].duplicated().sum()
    )

    print(
        f"WARNING: {n_duplicates} duplicate observation timestamps. "
        "Keeping the first row at each timestamp."
    )

    obs = obs.drop_duplicates(
        subset=["datetime_aest"],
        keep="first",
    ).copy()


# =============================================================================
# BUILD PRIOR AND POSTERIOR PREDICTIONS
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING PRIOR AND POSTERIOR PREDICTIONS")
print("=" * 80)

prediction_results = []
all_results = []

for site in SITES:

    print("\n" + "-" * 80)
    print(f"SITE: {site}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Select H rows for this site.
    # -------------------------------------------------------------------------

    h_site = H[
        H["site"].str.lower() == site.lower()
    ].copy()

    h_site = (
        h_site.sort_values("datetime_aest")
        .reset_index(drop=True)
    )

    print(f"H rows: {len(h_site)}")

    if h_site.empty:
        print(
            f"WARNING: no H rows for {site}; "
            "predictions cannot be calculated."
        )
        continue

    # -------------------------------------------------------------------------
    # Select cleaned observations.
    # -------------------------------------------------------------------------

    obs_column = OBS_COLUMNS[site]

    site_obs = obs[
        ["datetime_aest", obs_column]
    ].copy()

    site_obs = site_obs.rename(
        columns={
            obs_column: "observation"
        }
    )

    # -------------------------------------------------------------------------
    # LEFT JOIN:
    # The H matrix is the prediction timeline.
    # -------------------------------------------------------------------------

    merged = h_site[
        ["datetime_aest"] + MINES
    ].merge(
        site_obs,
        on="datetime_aest",
        how="left",
        validate="one_to_one",
    )

    merged = (
        merged.sort_values("datetime_aest")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Report observation matching.
    # -------------------------------------------------------------------------

    n_obs = int(
        np.isfinite(
            site_obs["observation"].to_numpy(dtype=float)
        ).sum()
    )

    n_matched = int(
        np.isfinite(
            merged["observation"].to_numpy(dtype=float)
        ).sum()
    )

    print(f"Finite cleaned observations in January: {n_obs}")
    print(f"Finite observations matched to H timestamps: {n_matched}")

    if n_obs > n_matched:
        print(
            f"NOTE: {n_obs - n_matched} valid observations "
            "have no matching H timestamp."
        )

    if n_obs == 0:
        print(
            "NOTE: no valid cleaned January observations. "
            "Predictions will still be saved, but metrics are unavailable."
        )

    # -------------------------------------------------------------------------
    # Prepare H matrix values.
    #
    # NaN mine responses are treated as zero contribution, as in the
    # original script. This should only be used if NaN means no contribution.
    # -------------------------------------------------------------------------

    H_values = (
        merged[MINES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # -------------------------------------------------------------------------
    # PRIOR:
    # All mine scaling factors equal 1.
    # -------------------------------------------------------------------------

    merged["prior_prediction"] = (
        BACKGROUND_PPM
        + np.sum(H_values, axis=1)
    )

    # -------------------------------------------------------------------------
    # POSTERIOR:
    # Apply daily mine-specific scaling factors.
    # -------------------------------------------------------------------------

    merged["date"] = (
        merged["datetime_aest"]
        .dt.normalize()
    )

    posterior_scaling = np.ones(
        (len(merged), len(MINES)),
        dtype=float,
    )

    missing_dates = []

    for date_value in merged["date"].drop_duplicates():

        day_mask = (
            merged["date"] == date_value
        ).to_numpy()

        if date_value not in post_lookup.index:
            missing_dates.append(date_value)
            # If factors are missing, retain prior scaling of 1.
            continue

        factors = post_lookup.loc[
            date_value,
            MINES,
        ].to_numpy(dtype=float)

        bad_factors = ~np.isfinite(factors)

        if np.any(bad_factors):

            bad_mines = [
                MINES[i]
                for i in np.where(bad_factors)[0]
            ]

            print(
                f"WARNING: missing/non-finite factors on "
                f"{date_value.date()} for {bad_mines}. "
                "Using 1 for those factors."
            )

            factors[bad_factors] = 1.0

        posterior_scaling[day_mask, :] = factors

    if missing_dates:
        print(
            "WARNING: no posterior factors for: "
            + ", ".join(
                str(d.date()) for d in missing_dates
            )
            + ". Prior scaling of 1 was used on these dates."
        )

    merged["posterior_prediction"] = (
        BACKGROUND_PPM
        + np.sum(
            H_values * posterior_scaling,
            axis=1,
        )
    )

    # -------------------------------------------------------------------------
    # Final prediction table.
    # -------------------------------------------------------------------------

    merged["site"] = site

    merged = merged[
        [
            "site",
            "datetime_aest",
            "observation",
            "prior_prediction",
            "posterior_prediction",
        ]
    ]

    # -------------------------------------------------------------------------
    # Metrics use only finite matched values.
    # -------------------------------------------------------------------------

    valid = (
        np.isfinite(
            merged["observation"].to_numpy(dtype=float)
        )
        & np.isfinite(
            merged["prior_prediction"].to_numpy(dtype=float)
        )
        & np.isfinite(
            merged["posterior_prediction"].to_numpy(dtype=float)
        )
    )

    metrics_prior = calculate_metrics(
        merged.loc[valid, "observation"],
        merged.loc[valid, "prior_prediction"],
    )

    metrics_post = calculate_metrics(
        merged.loc[valid, "observation"],
        merged.loc[valid, "posterior_prediction"],
    )

    print_metrics("Prior", metrics_prior)
    print_metrics("Posterior", metrics_post)

    # -------------------------------------------------------------------------
    # Percentage improvements.
    # -------------------------------------------------------------------------

    if (
        np.isfinite(metrics_prior["RMSE"])
        and metrics_prior["RMSE"] > 0
        and np.isfinite(metrics_post["RMSE"])
    ):
        rmse_improvement = (
            100.0
            * (
                metrics_prior["RMSE"]
                - metrics_post["RMSE"]
            )
            / metrics_prior["RMSE"]
        )

        print(f"\nRMSE improvement = {rmse_improvement:.2f}%")

    else:
        rmse_improvement = np.nan
        print("\nRMSE improvement = unavailable")

    if (
        np.isfinite(metrics_prior["MAE"])
        and metrics_prior["MAE"] > 0
        and np.isfinite(metrics_post["MAE"])
    ):
        mae_improvement = (
            100.0
            * (
                metrics_prior["MAE"]
                - metrics_post["MAE"]
            )
            / metrics_prior["MAE"]
        )

        print(f"MAE improvement  = {mae_improvement:.2f}%")

    else:
        mae_improvement = np.nan
        print("MAE improvement  = unavailable")

    all_results.append({
        "site": site,
        "model": "Prior",
        **metrics_prior,
        "RMSE_improvement_percent": rmse_improvement,
        "MAE_improvement_percent": mae_improvement,
    })

    all_results.append({
        "site": site,
        "model": "Posterior",
        **metrics_post,
        "RMSE_improvement_percent": rmse_improvement,
        "MAE_improvement_percent": mae_improvement,
    })

    prediction_results.append(merged)

    # -------------------------------------------------------------------------
    # Individual site time series.
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 5.5)
    )

    ax.plot(
        merged["datetime_aest"],
        merged["prior_prediction"],
        linewidth=1.5,
        label="Prior STILT",
    )

    ax.plot(
        merged["datetime_aest"],
        merged["posterior_prediction"],
        linewidth=1.8,
        label="Posterior STILT",
    )

    finite_obs = np.isfinite(
        merged["observation"].to_numpy(dtype=float)
    )

    if finite_obs.any():

        ax.plot(
            merged.loc[finite_obs, "datetime_aest"],
            merged.loc[finite_obs, "observation"],
            linestyle="None",
            marker="o",
            markersize=3,
            label="Picarro cleaned observation",
        )

    else:

        ax.text(
            0.5,
            0.5,
            "No valid cleaned observations\nfor January 2026",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    ax.set_title(
        f"{site} — CH$_4$ prior vs posterior prediction",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel("CH$_4$ (ppm)")
    ax.set_xlabel("Date (AEST)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    format_date_axis(ax)

    fig.autofmt_xdate()
    plt.tight_layout()

    individual_file = os.path.join(
        OUTPUT_DIR,
        f"prior_posterior_timeseries_{site}_Jan2026.png",
    )

    fig.savefig(
        individual_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Saved: {individual_file}")


# =============================================================================
# COMBINE AND SAVE PREDICTIONS
# =============================================================================

if not prediction_results:
    raise RuntimeError(
        "No predictions were generated. Check H-matrix site names."
    )

prediction_df = pd.concat(
    prediction_results,
    ignore_index=True,
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

print(f"\nSaved prediction CSV: {PREDICTIONS_FILE}")


# =============================================================================
# SAVE METRICS
# =============================================================================

metrics_df = pd.DataFrame(all_results)

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
)

print(f"Saved metrics CSV: {METRICS_FILE}")


# =============================================================================
# THREE-RECEPTOR TIME-SERIES FIGURE
# =============================================================================

print("\n" + "=" * 80)
print("CREATING THREE-RECEPTOR TIME SERIES")
print("=" * 80)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(16, 12),
    sharex=True,
)

for ax, site in zip(axes, SITES):

    data = prediction_df[
        prediction_df["site"] == site
    ].copy()

    if data.empty:

        ax.set_title(site)

        ax.text(
            0.5,
            0.5,
            "No H-matrix predictions",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

        continue

    ax.plot(
        data["datetime_aest"],
        data["prior_prediction"],
        linewidth=1.5,
        label="Prior STILT",
    )

    ax.plot(
        data["datetime_aest"],
        data["posterior_prediction"],
        linewidth=1.8,
        label="Posterior STILT",
    )

    valid_obs = np.isfinite(
        data["observation"].to_numpy(dtype=float)
    )

    if valid_obs.any():

        ax.plot(
            data.loc[valid_obs, "datetime_aest"],
            data.loc[valid_obs, "observation"],
            linestyle="None",
            marker="o",
            markersize=3,
            label="Picarro cleaned observation",
        )

    else:

        ax.text(
            0.5,
            0.5,
            "No valid cleaned observations",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_title(
        site,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel("CH$_4$ (ppm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    prior_metrics = get_site_metrics(
        metrics_df, site, "Prior"
    )

    post_metrics = get_site_metrics(
        metrics_df, site, "Posterior"
    )

    if prior_metrics is not None and post_metrics is not None:

        if prior_metrics["N"] > 0:

            annotation = (
                f"Matched N = {prior_metrics['N']}\n"
                f"Prior RMSE = {prior_metrics['RMSE']:.3f} ppm\n"
                f"Posterior RMSE = {post_metrics['RMSE']:.3f} ppm"
            )

        else:

            annotation = "No matched cleaned observations"

        ax.text(
            0.01,
            0.97,
            annotation,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
        )

axes[-1].set_xlabel("Date (AEST)")
format_date_axis(axes[-1])

fig.suptitle(
    "Upper Hunter CH$_4$ — Prior vs Posterior Bayesian Inversion\n"
    "January 2026",
    fontsize=17,
    fontweight="bold",
)

fig.autofmt_xdate()

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

fig.savefig(
    TIMESERIES_ALL_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    TIMESERIES_ALL_PDF,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(f"Saved: {TIMESERIES_ALL_PNG}")
print(f"Saved: {TIMESERIES_ALL_PDF}")


# =============================================================================
# OBSERVED VS PREDICTED SCATTER PLOTS
# =============================================================================

print("\n" + "=" * 80)
print("CREATING OBSERVED VS PREDICTED SCATTER PLOTS")
print("=" * 80)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10),
)

for j, site in enumerate(SITES):

    data = prediction_df[
        prediction_df["site"] == site
    ].copy()

    valid = (
        np.isfinite(
            data["observation"].to_numpy(dtype=float)
        )
        & np.isfinite(
            data["prior_prediction"].to_numpy(dtype=float)
        )
        & np.isfinite(
            data["posterior_prediction"].to_numpy(dtype=float)
        )
    )

    data = data.loc[valid]

    observed = data["observation"].to_numpy(dtype=float)
    prior = data["prior_prediction"].to_numpy(dtype=float)
    posterior = data["posterior_prediction"].to_numpy(dtype=float)

    if len(data) == 0:

        for ax, model_name in [
            (axes[0, j], "Prior"),
            (axes[1, j], "Posterior"),
        ]:

            ax.set_title(
                f"{site} — {model_name}",
                fontweight="bold",
            )

            ax.set_xlabel("Observed CH$_4$ (ppm)")
            ax.set_ylabel("Predicted CH$_4$ (ppm)")

            ax.text(
                0.5,
                0.5,
                "No matched observations",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

            ax.grid(True, alpha=0.3)

        continue

    combined = np.concatenate([
        observed,
        prior,
        posterior,
    ])

    lo = float(np.min(combined))
    hi = float(np.max(combined))

    span = hi - lo

    if span > 0:
        margin = 0.05 * span
    else:
        margin = max(abs(lo) * 0.05, 0.01)

    lo -= margin
    hi += margin

    for row, predictions, model_name in [
        (0, prior, "Prior"),
        (1, posterior, "Posterior"),
    ]:

        ax = axes[row, j]

        ax.scatter(
            observed,
            predictions,
            s=18,
            alpha=0.6,
        )

        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--",
            linewidth=1.5,
        )

        metrics = calculate_metrics(
            observed,
            predictions,
        )

        ax.set_title(
            f"{site} — {model_name} (N={metrics['N']})",
            fontweight="bold",
        )

        ax.set_xlabel("Observed CH$_4$ (ppm)")
        ax.set_ylabel("Predicted CH$_4$ (ppm)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.05,
            0.95,
            f"R = {metrics['R']:.3f}\n"
            f"R² = {metrics['R2']:.3f}\n"
            f"RMSE = {metrics['RMSE']:.3f} ppm\n"
            f"MAE = {metrics['MAE']:.3f} ppm\n"
            f"FAC2 = {metrics['FAC2']:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
        )

fig.suptitle(
    "Upper Hunter CH$_4$ — Observed vs Predicted\n"
    "Prior and Posterior Bayesian Inversion, January 2026",
    fontsize=17,
    fontweight="bold",
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

fig.savefig(
    SCATTER_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(f"Saved: {SCATTER_PNG}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL PRIOR VS POSTERIOR SUMMARY")
print("=" * 80)

summary_columns = [
    "site",
    "model",
    "N",
    "R",
    "R2",
    "RMSE",
    "MAE",
    "FAC2",
]

print(
    metrics_df[summary_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

for filename in [
    TIMESERIES_ALL_PNG,
    TIMESERIES_ALL_PDF,
    SCATTER_PNG,
    METRICS_FILE,
    PREDICTIONS_FILE,
]:
    print(filename)

print("\nDONE")
print("=" * 80)
