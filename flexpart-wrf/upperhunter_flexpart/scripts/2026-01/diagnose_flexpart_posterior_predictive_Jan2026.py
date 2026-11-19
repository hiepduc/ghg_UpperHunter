#!/usr/bin/env python3
"""
===============================================================================
FLEXPART-WRF POSTERIOR PREDICTIVE CONCENTRATION DIAGNOSTIC
===============================================================================

Upper Hunter CH4 Bayesian inversion
13 mines x 27 days = 351 daily scaling factors

This script evaluates whether the FLEXPART-WRF posterior is supported by the
Picarro concentration observations.

Forward model:

    CH4_i = BACKGROUND + sum_j H_ij * E0_j * s_j

where:

    H_ij       = FLEXPART sensitivity [ppm / (kg CH4/s)]
    E0_j       = prior mine emission [kg CH4/s]
    s_j        = daily scaling factor
    BACKGROUND = 1.85 ppm

The posterior predictive calculation uses ALL MCMC samples:

    CH4_i^(k) = BACKGROUND + H_i . (E0 * s^(k))

The script calculates:

    PRIOR
        prior concentration prediction
        prior residuals
        prior metrics

    POSTERIOR
        posterior mean prediction
        posterior 2.5 percentile
        posterior 97.5 percentile
        posterior residuals
        posterior metrics
        posterior predictive coverage

    DAILY
        observed daily mean
        prior daily prediction
        posterior daily prediction
        posterior uncertainty

    RECEPTOR
        Singleton
        Muswellbrook
        Merriwa

Outputs:

    posterior_predictive_observations.csv
    posterior_predictive_daily.csv
    posterior_predictive_receptor_metrics.csv
    posterior_predictive_overall_metrics.csv
    posterior_predictive_summary.txt

    01_timeseries_all_receptors.png
    02_timeseries_by_receptor.png
    03_residuals_by_receptor.png
    04_posterior_predictive_interval.png
    05_daily_observed_vs_posterior.png

===============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path.home() / "flexpart-wrf" / "upperhunter_flexpart"

HMATRIX_DIR = (
    BASE
    / "hmatrix"
    / "daily_20260105_20260131"
)

MCMC_DIR = (
    HMATRIX_DIR
    / "bayesian_analytical"
    / "mcmc_blocked"
)

OUTPUT_DIR = (
    HMATRIX_DIR
    / "posterior_predictive_diagnostic_Jan2026"
)

H_FILE = (
    HMATRIX_DIR
    / "H_daily_13mine_1124x351_ppm.npy"
)

OBS_FILE = (
    HMATRIX_DIR
    / "observation_vector.csv"
)

MCMC_FILE = (
    MCMC_DIR
    / "mcmc_samples.npy"
)

PRIOR_EMISSION_FILE = (
    Path.home()
    / "flexpart-wrf"
    / "emission_UH"
    / "RH_upperHunter"
    / "UpperHuntercoalmines.csv"
)

BACKGROUND_PPM = 1.85

G_TO_KG = 1.0e-3

MINES = [
    "OC1",
    "OC2",
    "OC3",
    "OC4",
    "OC5",
    "OC6",
    "OC7",
    "OC8",
    "OC9",
    "OC10",
    "OC11",
    "OC12",
    "OC13",
]

RECEPTORS = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(obs, pred):

    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(pred)

    obs = obs[mask]
    pred = pred[mask]

    n = len(obs)

    if n == 0:
        return {
            "N": 0,
            "R": np.nan,
            "R2": np.nan,
            "Bias": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "FAC2": np.nan,
            "Mean_obs": np.nan,
            "Mean_pred": np.nan,
        }

    residual = pred - obs

    if n >= 2 and np.std(obs) > 0 and np.std(pred) > 0:
        r = np.corrcoef(obs, pred)[0, 1]
    else:
        r = np.nan

    fac_valid = obs > 0

    if np.any(fac_valid):
        ratio = pred[fac_valid] / obs[fac_valid]

        fac2 = np.mean(
            (ratio >= 0.5) &
            (ratio <= 2.0)
        )
    else:
        fac2 = np.nan

    return {
        "N": n,
        "R": r,
        "R2": r ** 2 if np.isfinite(r) else np.nan,
        "Bias": np.mean(residual),
        "RMSE": np.sqrt(np.mean(residual ** 2)),
        "MAE": np.mean(np.abs(residual)),
        "FAC2": fac2,
        "Mean_obs": np.mean(obs),
        "Mean_pred": np.mean(pred),
    }


# =============================================================================
# LOAD PRIOR EMISSIONS
# =============================================================================

def load_prior_emissions():

    print("\n" + "=" * 80)
    print("READING PRIOR MINE EMISSIONS")
    print("=" * 80)

    df = pd.read_csv(PRIOR_EMISSION_FILE)

    df["Facility_code_name"] = (
        df["Facility_code_name"]
        .astype(str)
        .str.strip()
    )

    mine_df = df[
        df["Facility_code_name"].isin(MINES)
    ].copy()

    if len(mine_df) != len(MINES):

        found = set(mine_df["Facility_code_name"])

        missing = [
            m for m in MINES
            if m not in found
        ]

        raise RuntimeError(
            f"Missing prior emissions: {missing}"
        )

    mine_df["mine_order"] = mine_df[
        "Facility_code_name"
    ].map(
        {m: i for i, m in enumerate(MINES)}
    )

    mine_df = mine_df.sort_values("mine_order")

    prior_kg_s = (
        mine_df["gCH4/s"]
        .astype(float)
        .values
        * G_TO_KG
    )

    print("\nPrior emissions:")

    for mine, value in zip(MINES, prior_kg_s):

        print(
            f"{mine:>5s} : "
            f"{value:.8f} kg/s"
        )

    print(
        f"\nTOTAL  : "
        f"{prior_kg_s.sum():.8f} kg/s"
    )

    return prior_kg_s


# =============================================================================
# LOAD H
# =============================================================================

def load_H():

    print("\n" + "=" * 80)
    print("READING H MATRIX")
    print("=" * 80)

    H = np.load(H_FILE)

    print(f"H shape : {H.shape}")
    print(f"H dtype : {H.dtype}")
    print(f"H min   : {np.nanmin(H):.8g}")
    print(f"H max   : {np.nanmax(H):.8g}")
    print(f"H mean  : {np.nanmean(H):.8g}")

    if H.shape != (1124, 351):

        raise RuntimeError(
            f"Expected H shape (1124,351), got {H.shape}"
        )

    return H


# =============================================================================
# LOAD OBSERVATIONS
# =============================================================================

def load_observations():

    print("\n" + "=" * 80)
    print("READING OBSERVATION VECTOR")
    print("=" * 80)

    df = pd.read_csv(OBS_FILE)

    print("\nObservation columns:")
    print(df.columns.tolist())

    print(f"\nNumber of rows: {len(df)}")

    return df


# =============================================================================
# IDENTIFY OBSERVATION COLUMN
# =============================================================================

def find_observation_column(df):

    candidates = [
        "observation",
        "observed",
        "obs",
        "CH4_obs",
        "ch4_obs",
        "CH4",
        "value",
    ]

    for c in candidates:

        if c in df.columns:
            return c

    # Case-insensitive search

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:

        if c.lower() in lower_map:
            return lower_map[c.lower()]

    # Look for a column containing observation-like terminology

    for c in df.columns:

        cl = str(c).lower()

        if (
            "observ" in cl
            or cl == "obs"
        ):
            return c

    raise RuntimeError(
        "Could not identify observation column in "
        f"{OBS_FILE}\n"
        f"Available columns: {df.columns.tolist()}"
    )


# =============================================================================
# IDENTIFY RECEPTOR COLUMN
# =============================================================================

def find_receptor_column(df):

    candidates = [
        "receptor",
        "site",
        "station",
        "location",
    ]

    for c in candidates:

        if c in df.columns:
            return c

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:

        if c.lower() in lower_map:
            return lower_map[c.lower()]

    for c in df.columns:

        cl = str(c).lower()

        if (
            "receptor" in cl
            or "station" in cl
            or cl == "site"
        ):
            return c

    raise RuntimeError(
        "Could not identify receptor column."
    )


# =============================================================================
# IDENTIFY DATE/TIME COLUMN
# =============================================================================

def find_datetime_column(df):

    candidates = [
        "datetime_aest",
        "datetime_utc",
        "datetime",
        "Datetime",
        "date_time",
        "timestamp",
        "time",
    ]

    for c in candidates:

        if c in df.columns:
            return c

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:

        if c.lower() in lower_map:
            return lower_map[c.lower()]

    raise RuntimeError(
        "Could not identify datetime column."
    )


# =============================================================================
# LOAD MCMC
# =============================================================================

def load_mcmc():

    print("\n" + "=" * 80)
    print("READING MCMC POSTERIOR SAMPLES")
    print("=" * 80)

    samples = np.load(MCMC_FILE)

    print(f"MCMC shape : {samples.shape}")
    print(f"dtype      : {samples.dtype}")

    if samples.ndim != 2:
        raise RuntimeError(
            "MCMC samples must be 2-dimensional."
        )

    if samples.shape[1] != 351:
        raise RuntimeError(
            f"Expected 351 parameters, "
            f"got {samples.shape[1]}"
        )

    print(
        f"Samples    : {samples.shape[0]}"
    )

    return samples


# =============================================================================
# BUILD SOURCE EMISSION VECTOR
# =============================================================================

def build_prior_source_vector(prior_kg_s):

    """

    H columns are ordered:

        Jan 5:  OC1 ... OC13
        Jan 6:  OC1 ... OC13
        ...
        Jan 31: OC1 ... OC13

    Therefore repeat the 13 prior emissions 27 times.

    """

    source_vector = np.tile(
        prior_kg_s,
        27
    )

    if len(source_vector) != 351:

        raise RuntimeError(
            "Source vector is not length 351."
        )

    return source_vector


# =============================================================================
# EXTRACT RECEPTOR LABELS
# =============================================================================

def standardise_receptor_names(df, receptor_col):

    out = df[receptor_col].astype(str).str.strip()

    # Normalise capitalization

    mapping = {
        "singleton": "Singleton",
        "muswellbrook": "Muswellbrook",
        "merriwa": "Merriwa",
    }

    out = out.str.lower().map(mapping).fillna(out)

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 80)
    print("FLEXPART-WRF POSTERIOR PREDICTIVE DIAGNOSTIC")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    H = load_H()

    obs_df = load_observations()

    prior_kg_s = load_prior_emissions()

    samples = load_mcmc()

    # -------------------------------------------------------------------------
    # Identify observation columns
    # -------------------------------------------------------------------------

    obs_col = find_observation_column(obs_df)

    receptor_col = find_receptor_column(obs_df)

    datetime_col = find_datetime_column(obs_df)

    print("\nIdentified columns:")

    print(
        f"datetime  : {datetime_col}"
    )

    print(
        f"receptor  : {receptor_col}"
    )

    print(
        f"observation : {obs_col}"
    )

    # -------------------------------------------------------------------------
    # Check dimensions
    # -------------------------------------------------------------------------

    if len(obs_df) != H.shape[0]:

        raise RuntimeError(
            f"Observation rows ({len(obs_df)}) "
            f"do not match H rows ({H.shape[0]})."
        )

    # -------------------------------------------------------------------------
    # Source vector
    # -------------------------------------------------------------------------

    E0 = build_prior_source_vector(
        prior_kg_s
    )

    print(
        f"\nPrior source vector length: "
        f"{len(E0)}"
    )

    # -------------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------------

    obs = (
        pd.to_numeric(
            obs_df[obs_col],
            errors="coerce"
        )
        .values
    )

    receptor = standardise_receptor_names(
        obs_df,
        receptor_col
    )

    datetime = pd.to_datetime(
        obs_df[datetime_col],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # PRIOR PREDICTION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CALCULATING PRIOR PREDICTION")
    print("=" * 80)

    prior_pred = (
        BACKGROUND_PPM
        + H @ E0
    )

    print(
        f"Prior mean prediction : "
        f"{np.nanmean(prior_pred):.6f} ppm"
    )

    # -------------------------------------------------------------------------
    # POSTERIOR PREDICTION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CALCULATING POSTERIOR PREDICTIVE DISTRIBUTION")
    print("=" * 80)

    print(
        "Using all "
        f"{samples.shape[0]} MCMC samples..."
    )

    # Each MCMC sample contains scaling factors.
    #
    # Convert:
    #
    #     scaling
    #
    # to actual source emission:
    #
    #     E0 * scaling
    #
    # Then:
    #
    #     prediction = background + H @ source

    n_samples = samples.shape[0]

    n_obs = H.shape[0]

    post_mean = np.zeros(n_obs)

    post_M2 = np.zeros(n_obs)

    post_lower = np.full(
        n_obs,
        np.inf
    )

    post_upper = np.full(
        n_obs,
        -np.inf
    )

    # For exact percentiles we need to retain predictions.
    #
    # 6000 x 1124 is only ~27 MB in float32,
    # so this is easily manageable.

    predictions = np.empty(
        (n_samples, n_obs),
        dtype=np.float32
    )

    for k in range(n_samples):

        source_k = (
            E0
            * samples[k, :]
        )

        predictions[k, :] = (
            BACKGROUND_PPM
            + H @ source_k
        )

        if (k + 1) % 500 == 0:

            print(
                f"  processed "
                f"{k + 1:5d} / "
                f"{n_samples}"
            )

    # -------------------------------------------------------------------------
    # Posterior statistics
    # -------------------------------------------------------------------------

    print("\nCalculating posterior statistics...")

    post_mean = np.mean(
        predictions,
        axis=0
    )

    post_std = np.std(
        predictions,
        axis=0,
        ddof=1
    )

    post_lower = np.percentile(
        predictions,
        2.5,
        axis=0
    )

    post_upper = np.percentile(
        predictions,
        97.5,
        axis=0
    )

    # -------------------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------------------

    valid_obs = np.isfinite(obs)

    covered = (
        valid_obs
        & (obs >= post_lower)
        & (obs <= post_upper)
    )

    coverage = (
        np.sum(covered)
        / np.sum(valid_obs)
    )

    print(
        f"\nPosterior predictive coverage: "
        f"{coverage:.4f} "
        f"({coverage * 100:.2f}%)"
    )

    # -------------------------------------------------------------------------
    # Observation-level output
    # -------------------------------------------------------------------------

    result = pd.DataFrame({

        "datetime": datetime,

        "receptor": receptor,

        "observed_ppm": obs,

        "prior_pred_ppm": prior_pred,

        "posterior_mean_ppm": post_mean,

        "posterior_std_ppm": post_std,

        "posterior_lower95_ppm": post_lower,

        "posterior_upper95_ppm": post_upper,

    })

    result["prior_residual_ppm"] = (
        result["prior_pred_ppm"]
        - result["observed_ppm"]
    )

    result["posterior_residual_ppm"] = (
        result["posterior_mean_ppm"]
        - result["observed_ppm"]
    )

    result["posterior_covered95"] = covered

    result.to_csv(
        OUTPUT_DIR
        / "posterior_predictive_observations.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    prior_metrics = calculate_metrics(
        obs,
        prior_pred
    )

    posterior_metrics = calculate_metrics(
        obs,
        post_mean
    )

    overall_metrics = pd.DataFrame([
        {
            "Model": "Prior",
            **prior_metrics,
        },
        {
            "Model": "Posterior",
            **posterior_metrics,
            "Coverage95": coverage,
        },
    ])

    overall_metrics.to_csv(
        OUTPUT_DIR
        / "posterior_predictive_overall_metrics.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Receptor metrics
    # -------------------------------------------------------------------------

    receptor_rows = []

    for rec in RECEPTORS:

        mask = (
            receptor.str.lower()
            == rec.lower()
        ).values

        if not np.any(mask):

            print(
                f"WARNING: no observations "
                f"for {rec}"
            )

            continue

        pmask = (
            mask
            & np.isfinite(obs)
        )

        rec_coverage = np.mean(
            (
                obs[pmask]
                >= post_lower[pmask]
            )
            &
            (
                obs[pmask]
                <= post_upper[pmask]
            )
        )

        pm = calculate_metrics(
            obs[mask],
            prior_pred[mask]
        )

        qm = calculate_metrics(
            obs[mask],
            post_mean[mask]
        )

        receptor_rows.append({
            "Receptor": rec,
            "Model": "Prior",
            **pm,
            "Coverage95": np.nan,
        })

        receptor_rows.append({
            "Receptor": rec,
            "Model": "Posterior",
            **qm,
            "Coverage95": rec_coverage,
        })

    receptor_metrics = pd.DataFrame(
        receptor_rows
    )

    receptor_metrics.to_csv(
        OUTPUT_DIR
        / "posterior_predictive_receptor_metrics.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Daily aggregation
    # -------------------------------------------------------------------------

    result["date"] = (
        result["datetime"]
        .dt.date
    )

    daily_rows = []

    for date, group in result.groupby(
        "date",
        sort=True
    ):

        valid = np.isfinite(
            group["observed_ppm"].values
        )

        if not np.any(valid):
            continue

        idx = group.index.values

        valid_idx = idx[valid]

        # Observation mean

        obs_mean = np.mean(
            result.loc[
                valid_idx,
                "observed_ppm"
            ]
        )

        prior_mean = np.mean(
            result.loc[
                valid_idx,
                "prior_pred_ppm"
            ]
        )

        posterior_mean = np.mean(
            result.loc[
                valid_idx,
                "posterior_mean_ppm"
            ]
        )

        posterior_lower = np.mean(
            result.loc[
                valid_idx,
                "posterior_lower95_ppm"
            ]
        )

        posterior_upper = np.mean(
            result.loc[
                valid_idx,
                "posterior_upper95_ppm"
            ]
        )

        daily_rows.append({

            "date": date,

            "N": len(valid_idx),

            "observed_mean_ppm": obs_mean,

            "prior_mean_ppm": prior_mean,

            "posterior_mean_ppm":
                posterior_mean,

            "posterior_lower95_ppm":
                posterior_lower,

            "posterior_upper95_ppm":
                posterior_upper,

            "prior_bias_ppm":
                prior_mean - obs_mean,

            "posterior_bias_ppm":
                posterior_mean - obs_mean,

        })

    daily = pd.DataFrame(
        daily_rows
    )

    daily.to_csv(
        OUTPUT_DIR
        / "posterior_predictive_daily.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    print(
        overall_metrics.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    print("\n" + "=" * 80)
    print("RECEPTOR METRICS")
    print("=" * 80)

    print(
        receptor_metrics.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    print("\n" + "=" * 80)
    print("DAILY POSTERIOR PREDICTIVE SUMMARY")
    print("=" * 80)

    print(
        daily.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # -------------------------------------------------------------------------
    # Find worst posterior residuals
    # -------------------------------------------------------------------------

    result["abs_post_residual"] = np.abs(
        result["posterior_residual_ppm"]
    )

    worst = result.sort_values(
        "abs_post_residual",
        ascending=False
    ).head(20)

    print("\n" + "=" * 80)
    print("20 LARGEST POSTERIOR RESIDUALS")
    print("=" * 80)

    print(
        worst[
            [
                "datetime",
                "receptor",
                "observed_ppm",
                "posterior_mean_ppm",
                "posterior_lower95_ppm",
                "posterior_upper95_ppm",
                "posterior_residual_ppm",
                "posterior_covered95",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # -------------------------------------------------------------------------
    # Jan 17-21 diagnostic
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("JAN 17-21 POSTERIOR PREDICTIVE DIAGNOSTIC")
    print("=" * 80)

    period = result[
        (
            result["datetime"]
            >= pd.Timestamp("2026-01-17")
        )
        &
        (
            result["datetime"]
            < pd.Timestamp("2026-01-22")
        )
    ]

    if len(period) > 0:

        period_metrics = calculate_metrics(
            period["observed_ppm"],
            period["posterior_mean_ppm"]
        )

        period_prior = calculate_metrics(
            period["observed_ppm"],
            period["prior_pred_ppm"]
        )

        print("\nPrior Jan 17-21:")
        for key, value in period_prior.items():
            print(
                f"{key:>15s} : {value:.6f}"
                if isinstance(value, float)
                else f"{key:>15s} : {value}"
            )

        print("\nPosterior Jan 17-21:")
        for key, value in period_metrics.items():
            print(
                f"{key:>15s} : {value:.6f}"
                if isinstance(value, float)
                else f"{key:>15s} : {value}"
            )

        period_coverage = np.mean(
            period["posterior_covered95"]
        )

        print(
            f"\nPosterior 95% coverage "
            f"Jan 17-21: "
            f"{period_coverage:.4f}"
        )

    # =========================================================================
    # PLOTS
    # =========================================================================

    print("\n" + "=" * 80)
    print("CREATING PLOTS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Plot 1: all observations
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        result["datetime"],
        result["observed_ppm"],
        ".",
        markersize=2,
        label="Observed"
    )

    ax.plot(
        result["datetime"],
        result["prior_pred_ppm"],
        "-",
        linewidth=1,
        label="Prior"
    )

    ax.plot(
        result["datetime"],
        result["posterior_mean_ppm"],
        "-",
        linewidth=1,
        label="Posterior mean"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("CH4 (ppm)")
    ax.set_title(
        "FLEXPART-WRF Prior vs Posterior "
        "Predicted CH4"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_timeseries_all_receptors.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 2: receptor time series
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True
    )

    for ax, rec in zip(
        axes,
        RECEPTORS
    ):

        mask = (
            result["receptor"]
            == rec
        )

        r = result.loc[mask]

        ax.plot(
            r["datetime"],
            r["observed_ppm"],
            ".",
            markersize=3,
            label="Observed"
        )

        ax.plot(
            r["datetime"],
            r["prior_pred_ppm"],
            "-",
            linewidth=1,
            label="Prior"
        )

        ax.plot(
            r["datetime"],
            r["posterior_mean_ppm"],
            "-",
            linewidth=1,
            label="Posterior"
        )

        ax.fill_between(
            r["datetime"],
            r["posterior_lower95_ppm"],
            r["posterior_upper95_ppm"],
            alpha=0.20,
            label="Posterior 95%"
        )

        ax.set_ylabel(
            f"{rec}\nCH4 (ppm)"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend(
            loc="best"
        )

    axes[-1].set_xlabel(
        "Date"
    )

    fig.suptitle(
        "FLEXPART-WRF Posterior Predictive "
        "Concentrations by Receptor"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_timeseries_by_receptor.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 3: residuals
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True
    )

    for ax, rec in zip(
        axes,
        RECEPTORS
    ):

        mask = (
            result["receptor"]
            == rec
        )

        r = result.loc[mask]

        ax.axhline(
            0,
            linewidth=1
        )

        ax.plot(
            r["datetime"],
            r["prior_residual_ppm"],
            ".-",
            markersize=2,
            linewidth=0.6,
            label="Prior residual"
        )

        ax.plot(
            r["datetime"],
            r["posterior_residual_ppm"],
            ".-",
            markersize=2,
            linewidth=0.6,
            label="Posterior residual"
        )

        ax.set_ylabel(
            f"{rec}\nResidual (ppm)"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend()

    axes[-1].set_xlabel(
        "Date"
    )

    fig.suptitle(
        "FLEXPART-WRF Prior and Posterior Residuals"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "03_residuals_by_receptor.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 4: posterior predictive interval
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.fill_between(
        result["datetime"],
        result["posterior_lower95_ppm"],
        result["posterior_upper95_ppm"],
        alpha=0.25,
        label="Posterior 95%"
    )

    ax.plot(
        result["datetime"],
        result["posterior_mean_ppm"],
        "-",
        linewidth=1,
        label="Posterior mean"
    )

    ax.plot(
        result["datetime"],
        result["observed_ppm"],
        ".",
        markersize=2,
        label="Observed"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "CH4 (ppm)"
    )

    ax.set_title(
        "FLEXPART-WRF Posterior Predictive "
        "95% Interval"
    )

    ax.legend()
    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "04_posterior_predictive_interval.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 5: daily observed vs posterior
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        daily["date"],
        daily["observed_mean_ppm"],
        "o-",
        label="Observed daily mean"
    )

    ax.plot(
        daily["date"],
        daily["prior_mean_ppm"],
        "o-",
        label="Prior daily mean"
    )

    ax.plot(
        daily["date"],
        daily["posterior_mean_ppm"],
        "o-",
        label="Posterior daily mean"
    )

    ax.fill_between(
        daily["date"],
        daily["posterior_lower95_ppm"],
        daily["posterior_upper95_ppm"],
        alpha=0.20,
        label="Posterior 95%"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "CH4 (ppm)"
    )

    ax.set_title(
        "Daily Mean Observed vs FLEXPART-WRF "
        "Posterior Prediction"
    )

    ax.legend()
    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "05_daily_observed_vs_posterior.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Summary text
    # -------------------------------------------------------------------------

    with open(
        OUTPUT_DIR
        / "posterior_predictive_summary.txt",
        "w"
    ) as f:

        f.write(
            "FLEXPART-WRF POSTERIOR PREDICTIVE DIAGNOSTIC\n"
        )
        f.write(
            "Upper Hunter CH4, Jan 5-31 2026\n"
        )
        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            f"H shape: {H.shape}\n"
        )

        f.write(
            f"MCMC samples: {samples.shape}\n"
        )

        f.write(
            f"Background: {BACKGROUND_PPM} ppm\n"
        )

        f.write(
            f"Prior total emission: "
            f"{prior_kg_s.sum():.8f} kg/s\n\n"
        )

        f.write(
            "OVERALL METRICS\n"
        )
        f.write(
            overall_metrics.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPOSTERIOR PREDICTIVE COVERAGE\n"
        )

        f.write(
            f"{coverage:.6f} "
            f"({coverage * 100:.2f}%)\n"
        )

        f.write(
            "\n\nRECEPTOR METRICS\n"
        )

        f.write(
            receptor_metrics.to_string(
                index=False
            )
        )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print("\nFiles:")

    for p in sorted(
        OUTPUT_DIR.iterdir()
    ):

        print(
            f"  {p.name}"
        )


if __name__ == "__main__":
    main()

