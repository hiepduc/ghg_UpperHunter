#!/usr/bin/env python3
"""
===============================================================================
ANALYTICAL BAYESIAN INVERSION
Upper Hunter FLEXPART-WRF CH4
13 mines x 27 days = 351 daily scaling factors

Model:
    y = G s + epsilon

where:
    y = observed CH4 enhancement above background [ppm]
    G = FLEXPART sensitivity to unit scaling factor [ppm]
    s = daily mine emission scaling factors
    epsilon ~ N(0, R)

Prior:
    s ~ N(s0, B)

Current assumptions:
    s0 = 1 for every mine/day
    prior uncertainty = 50%
    temporal correlation length = 3 days
    observation uncertainty = 0.05 ppm
    background = 1.85 ppm

IMPORTANT:
    The gCH4/s values in UpperHuntercoalmines.csv are already the
    prior daily/average emission rates. DO NOT divide them by 365.

FLEXPART H:
    H is already constructed using surface layer 0 (0-100 m) only.
===============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

HMATRIX_DIR = (
    BASE
    / "hmatrix"
    / "daily_20260105_20260131"
)

H_FILE = HMATRIX_DIR / "H_daily_13mine_1124x351_ppm.npy"
OBS_FILE = HMATRIX_DIR / "observation_vector.csv"
META_FILE = HMATRIX_DIR / "H_metadata.csv"

PRIOR_FILE = (
    Path.home()
    / "flexpart-wrf/emission_UH/RH_upperHunter/UpperHuntercoalmines.csv"
)

OUT_DIR = HMATRIX_DIR / "bayesian_analytical"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INVERSION SETTINGS
# =============================================================================

BACKGROUND_PPM = 1.85

# Prior scaling factor
PRIOR_MEAN = 1.0

# 1-sigma uncertainty of scaling factor
PRIOR_FRACTIONAL_UNCERTAINTY = 0.50

# Temporal correlation length in days
CORRELATION_LENGTH_DAYS = 3.0

# Observation uncertainty
OBS_SIGMA_PPM = 0.05


# Mine ordering MUST match H:
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

N_MINE = len(MINES)
N_DAY = 27
N_PARAM = N_MINE * N_DAY


# =============================================================================
# PRIOR EMISSIONS
# =============================================================================

def read_prior_emissions():

    print("=" * 80)
    print("READING PRIOR MINE EMISSIONS")
    print("=" * 80)

    df = pd.read_csv(PRIOR_FILE)

    print("\nPrior file:")
    print(PRIOR_FILE)

    print("\nColumns:")
    print(df.columns.tolist())

    rates = []

    for mine in MINES:

        row = df[df["Facility_code_name"].astype(str).str.strip() == mine]

        if len(row) != 1:
            raise ValueError(
                f"Expected exactly one row for {mine}, found {len(row)}"
            )

        value = float(
            str(row.iloc[0]["gCH4/s"])
            .strip()
            .replace(",", "")
        )

        rates.append(value)

    rates = np.asarray(rates, dtype=float)

    print("\nPrior emissions:")
    for mine, rate in zip(MINES, rates):
        print(f"  {mine:5s}: {rate:12.4f} g CH4/s")

    print("-" * 50)
    print(f"  Total : {rates.sum():12.4f} g CH4/s")

    if not np.isclose(rates.sum(), 2553.86, rtol=1e-5):
        print("\nWARNING: total differs from expected 2553.86 g/s")

    # Convert g/s -> kg/s
    rates_kg_s = rates / 1000.0

    return rates, rates_kg_s


# =============================================================================
# READ H AND OBSERVATIONS
# =============================================================================

def read_inputs():

    print("\n" + "=" * 80)
    print("READING H MATRIX AND OBSERVATIONS")
    print("=" * 80)

    H = np.load(H_FILE)

    print(f"\nH shape: {H.shape}")
    print(f"H dtype: {H.dtype}")

    if H.shape != (1124, 351):
        raise ValueError(
            f"Unexpected H shape {H.shape}; expected (1124, 351)"
        )

    obs_df = pd.read_csv(OBS_FILE)

    print(f"Observation file rows: {len(obs_df)}")

    # Find observation column
    possible_obs = [
        "observation_ppm",
        "obs_ppm",
        "CH4_obs",
        "observation"
    ]

    obs_col = None

    for c in possible_obs:
        if c in obs_df.columns:
            obs_col = c
            break

    if obs_col is None:
        raise ValueError(
            f"Could not find observation column. "
            f"Available columns: {obs_df.columns.tolist()}"
        )

    obs = pd.to_numeric(obs_df[obs_col], errors="coerce").to_numpy()

    if len(obs) != H.shape[0]:
        raise ValueError(
            f"Observation length {len(obs)} does not match "
            f"H rows {H.shape[0]}"
        )

    if np.any(~np.isfinite(obs)):
        raise ValueError("Observation vector contains NaN or infinite values.")

    # H was constructed against the original CH4 observation.
    # Remove the fixed 1.85 ppm background.
    y = obs - BACKGROUND_PPM

    print(f"\nObservation column: {obs_col}")
    print(f"Observed CH4 min/max: {obs.min():.6f}, {obs.max():.6f} ppm")
    print(f"Background: {BACKGROUND_PPM:.4f} ppm")
    print(f"Enhancement min/max: {y.min():.6f}, {y.max():.6f} ppm")

    return H, obs, y, obs_df


# =============================================================================
# CONSTRUCT G
# =============================================================================

def construct_G(H, prior_rates_kg_s):

    print("\n" + "=" * 80)
    print("CONSTRUCTING SCALING-FACTOR SENSITIVITY MATRIX G")
    print("=" * 80)

    # H columns are day-major:
    #
    # Jan 5:
    #   OC1 OC2 ... OC13
    # Jan 6:
    #   OC1 OC2 ... OC13
    # ...

    emission_vector = np.tile(
        prior_rates_kg_s,
        N_DAY
    )

    if len(emission_vector) != N_PARAM:
        raise ValueError("Emission vector has incorrect length.")

    # H units:
    #     ppm / (kg/s)
    #
    # prior emission:
    #     kg/s
    #
    # Therefore:
    #     G = H * prior emission
    #
    # giving:
    #     ppm per unit scaling factor

    G = H * emission_vector[np.newaxis, :]

    print(f"\nG shape: {G.shape}")
    print(f"G min/max: {G.min():.6e}, {G.max():.6e}")
    print(f"G mean: {G.mean():.6e}")

    return G, emission_vector


# =============================================================================
# CONSTRUCT B
# =============================================================================

def construct_B():

    print("\n" + "=" * 80)
    print("CONSTRUCTING PRIOR COVARIANCE B")
    print("=" * 80)

    sigma = PRIOR_FRACTIONAL_UNCERTAINTY

    B = np.zeros((N_PARAM, N_PARAM), dtype=float)

    # For each mine, construct a 27-day temporal covariance block.
    #
    # B[(mine,day),(mine,day')] =
    #       sigma^2 exp(-|day-day'| / L)

    for m in range(N_MINE):

        start = m
        indices = np.arange(
            start,
            N_PARAM,
            N_MINE
        )

        for i, ii in enumerate(indices):

            for j, jj in enumerate(indices):

                lag = abs(i - j)

                B[ii, jj] = (
                    sigma**2
                    * np.exp(
                        -lag / CORRELATION_LENGTH_DAYS
                    )
                )

    print(f"\nB shape: {B.shape}")
    print(f"Prior standard deviation: {sigma:.3f}")
    print(f"Temporal correlation length: "
          f"{CORRELATION_LENGTH_DAYS:.2f} days")

    print("\nExample temporal correlations:")
    for lag in range(0, 7):

        corr = np.exp(
            -lag / CORRELATION_LENGTH_DAYS
        )

        print(
            f"  {lag} day(s): {corr:.6f}"
        )

    print(
        f"\nB diagonal range: "
        f"{np.min(np.diag(B)):.6f} - "
        f"{np.max(np.diag(B)):.6f}"
    )

    return B


# =============================================================================
# CONSTRUCT R
# =============================================================================

def construct_R(n_obs):

    print("\n" + "=" * 80)
    print("CONSTRUCTING OBSERVATION COVARIANCE R")
    print("=" * 80)

    variance = OBS_SIGMA_PPM**2

    R = np.eye(
        n_obs,
        dtype=float
    ) * variance

    print(f"\nR shape: {R.shape}")
    print(f"Observation sigma: {OBS_SIGMA_PPM:.4f} ppm")
    print(f"Observation variance: {variance:.6f} ppm^2")

    return R


# =============================================================================
# ANALYTICAL BAYESIAN SOLUTION
# =============================================================================

def analytical_bayesian(G, y, s0, B, R):

    print("\n" + "=" * 80)
    print("ANALYTICAL BAYESIAN INVERSION")
    print("=" * 80)

    print("\nUsing:")
    print("  s ~ N(s0, B)")
    print("  y|s ~ N(Gs, R)")

    # Innovation:
    #
    # d = y - G s0
    #
    innovation = y - G @ s0

    print(
        f"\nInnovation min/max: "
        f"{innovation.min():.6f}, "
        f"{innovation.max():.6f} ppm"
    )

    # R + G B G^T
    #
    # This is only 1124 x 1124 and is much more stable than
    # directly forming G^T R^-1 G + B^-1.

    print("\nConstructing innovation covariance S = G B G^T + R ...")

    S = G @ B @ G.T + R

    print(f"S shape: {S.shape}")

    # Cholesky factorization
    print("Performing Cholesky factorization ...")

    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:

        print(
            "WARNING: Cholesky failed. "
            "Adding tiny numerical jitter to R."
        )

        jitter = 1e-10

        S = S + jitter * np.eye(S.shape[0])

        L = np.linalg.cholesky(S)

    # Solve S x = innovation
    #
    # First:
    #   L z = innovation
    #
    # Then:
    #   L.T x = z

    z = np.linalg.solve(
        L,
        innovation
    )

    x = np.linalg.solve(
        L.T,
        z
    )

    # Posterior mean:
    #
    # s_a = s0 + B G^T S^-1 innovation

    print("\nCalculating posterior mean ...")

    s_post = (
        s0
        + B @ G.T @ x
    )

    # Posterior covariance:
    #
    # B_post = B - B G^T S^-1 G B
    #
    # We solve:
    #
    # S X = G B
    #
    # rather than calculating S^-1.

    print("Calculating posterior covariance ...")

    GB = G @ B

    Z = np.linalg.solve(
        L,
        GB
    )

    X = np.linalg.solve(
        L.T,
        Z
    )

    B_post = (
        B
        - B @ G.T @ X
    )

    # Force numerical symmetry
    B_post = (
        B_post
        + B_post.T
    ) / 2.0

    # Small negative numerical eigenvalues can appear at machine precision.
    # We inspect them but do not arbitrarily modify the covariance.

    eigvals = np.linalg.eigvalsh(B_post)

    print("\nPosterior covariance eigenvalues:")
    print(f"  Minimum: {eigvals.min():.6e}")
    print(f"  Maximum: {eigvals.max():.6e}")

    posterior_sd = np.sqrt(
        np.maximum(
            np.diag(B_post),
            0.0
        )
    )

    return s_post, B_post, posterior_sd, S


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def build_parameter_names():

    names = []

    dates = pd.date_range(
        "2026-01-05",
        "2026-01-31",
        freq="D"
    )

    for date in dates:

        datestr = date.strftime("%Y%m%d")

        for mine in MINES:

            names.append(
                f"{mine}_{datestr}"
            )

    return names


def save_results(
    s0,
    s_post,
    posterior_sd,
    B,
    B_post,
    prior_rates_g_s,
    emission_vector_kg_s,
    parameter_names
):

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # NPY files
    # -------------------------------------------------------------------------

    np.save(
        OUT_DIR / "B_prior.npy",
        B
    )

    np.save(
        OUT_DIR / "B_posterior.npy",
        B_post
    )

    np.save(
        OUT_DIR / "prior_scaling.npy",
        s0
    )

    np.save(
        OUT_DIR / "posterior_mean_scaling.npy",
        s_post
    )

    np.save(
        OUT_DIR / "posterior_sd_scaling.npy",
        posterior_sd
    )

    # -------------------------------------------------------------------------
    # Parameter table
    # -------------------------------------------------------------------------

    prior_rates_parameter = np.asarray(
        emission_vector_kg_s
    )

    df = pd.DataFrame({

        "parameter": parameter_names,

        "prior_scaling":
            s0,

        "posterior_mean":
            s_post,

        "posterior_sd":
            posterior_sd,

        "posterior_lower_95":
            s_post - 1.96 * posterior_sd,

        "posterior_upper_95":
            s_post + 1.96 * posterior_sd,

        "prior_emission_kg_s":
            prior_rates_parameter,

        "posterior_emission_kg_s":
            s_post * prior_rates_parameter,

        "posterior_emission_lower_kg_s":
            (s_post - 1.96 * posterior_sd)
            * prior_rates_parameter,

        "posterior_emission_upper_kg_s":
            (s_post + 1.96 * posterior_sd)
            * prior_rates_parameter,

    })

    df.to_csv(
        OUT_DIR / "posterior_scaling_factors.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Mine summary
    # -------------------------------------------------------------------------

    mine_rows = []

    for m, mine in enumerate(MINES):

        indices = np.arange(
            m,
            N_PARAM,
            N_MINE
        )

        prior_rate = prior_rates_g_s[m]

        posterior_scaling = s_post[indices]
        posterior_unc = posterior_sd[indices]

        mine_rows.append({

            "mine":
                mine,

            "prior_emission_g_s":
                prior_rate,

            "prior_mean_scaling":
                1.0,

            "posterior_mean_scaling":
                np.mean(posterior_scaling),

            "posterior_median_scaling":
                np.median(posterior_scaling),

            "posterior_sd_mean":
                np.mean(posterior_unc),

            "posterior_min_scaling":
                np.min(posterior_scaling),

            "posterior_max_scaling":
                np.max(posterior_scaling),

            "posterior_mean_emission_g_s":
                np.mean(
                    posterior_scaling
                    * prior_rate
                ),

        })

    mine_df = pd.DataFrame(
        mine_rows
    )

    mine_df.to_csv(
        OUT_DIR / "posterior_mine_summary.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    negative_count = np.sum(
        s_post < 0
    )

    diagnostic = pd.DataFrame([{

        "n_observations":
            1124,

        "n_parameters":
            351,

        "prior_fractional_uncertainty":
            PRIOR_FRACTIONAL_UNCERTAINTY,

        "temporal_correlation_length_days":
            CORRELATION_LENGTH_DAYS,

        "observation_sigma_ppm":
            OBS_SIGMA_PPM,

        "background_ppm":
            BACKGROUND_PPM,

        "prior_total_g_s":
            np.sum(prior_rates_g_s),

        "posterior_mean_scaling":
            np.mean(s_post),

        "posterior_median_scaling":
            np.median(s_post),

        "posterior_min_scaling":
            np.min(s_post),

        "posterior_max_scaling":
            np.max(s_post),

        "negative_scaling_count":
            negative_count,

        "negative_scaling_fraction":
            negative_count / N_PARAM,

    }])

    diagnostic.to_csv(
        OUT_DIR / "inversion_diagnostics.csv",
        index=False
    )

    print("\nSaved:")
    print(OUT_DIR / "B_prior.npy")
    print(OUT_DIR / "B_posterior.npy")
    print(OUT_DIR / "prior_scaling.npy")
    print(OUT_DIR / "posterior_mean_scaling.npy")
    print(OUT_DIR / "posterior_sd_scaling.npy")
    print(OUT_DIR / "posterior_scaling_factors.csv")
    print(OUT_DIR / "posterior_mine_summary.csv")
    print(OUT_DIR / "inversion_diagnostics.csv")


# =============================================================================
# PLOTS
# =============================================================================

def make_plots(
    s_post,
    posterior_sd,
    obs,
    y,
    G
):

    print("\n" + "=" * 80)
    print("CREATING DIAGNOSTIC PLOTS")
    print("=" * 80)

    parameter_names = build_parameter_names()

    # -------------------------------------------------------------------------
    # 1. Posterior scaling factors
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(18, 7)
    )

    x = np.arange(
        N_PARAM
    )

    ax.errorbar(
        x,
        s_post,
        yerr=1.96 * posterior_sd,
        fmt=".",
        markersize=3,
        elinewidth=0.5,
        capsize=1
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "Mine-day parameter"
    )

    ax.set_ylabel(
        "Emission scaling factor"
    )

    ax.set_title(
        "Analytical Bayesian posterior scaling factors"
    )

    # Day boundaries
    for d in range(
        1,
        N_DAY
    ):

        ax.axvline(
            d * N_MINE - 0.5,
            linewidth=0.4,
            alpha=0.3
        )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "posterior_scaling_factors.png",
        dpi=200
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Posterior uncertainty
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(18, 6)
    )

    ax.bar(
        x,
        posterior_sd,
        width=1.0
    )

    ax.set_xlabel(
        "Mine-day parameter"
    )

    ax.set_ylabel(
        "Posterior standard deviation"
    )

    ax.set_title(
        "Posterior uncertainty for 351 mine-day factors"
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "posterior_uncertainty.png",
        dpi=200
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Observation fit using posterior mean
    # -------------------------------------------------------------------------

    predicted_enhancement = G @ s_post

    predicted_total = (
        BACKGROUND_PPM
        + predicted_enhancement
    )

    residual = (
        obs
        - predicted_total
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        obs,
        predicted_total,
        s=10
    )

    lo = min(
        obs.min(),
        predicted_total.min()
    )

    hi = max(
        obs.max(),
        predicted_total.max()
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--"
    )

    ax.set_xlabel(
        "Observed CH4 (ppm)"
    )

    ax.set_ylabel(
        "Posterior predicted CH4 (ppm)"
    )

    ax.set_title(
        "Observed vs analytical Bayesian posterior"
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "observed_vs_predicted.png",
        dpi=200
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. Residual histogram
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        residual,
        bins=40
    )

    ax.axvline(
        0,
        linestyle="--"
    )

    ax.set_xlabel(
        "Observed - predicted CH4 (ppm)"
    )

    ax.set_ylabel(
        "Number of observations"
    )

    ax.set_title(
        "Posterior residuals"
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "posterior_residual_histogram.png",
        dpi=200
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. Mine-level posterior distributions
    # -------------------------------------------------------------------------

    mine_means = []
    mine_sds = []

    for m in range(N_MINE):

        indices = np.arange(
            m,
            N_PARAM,
            N_MINE
        )

        mine_means.append(
            np.mean(
                s_post[indices]
            )
        )

        mine_sds.append(
            np.mean(
                posterior_sd[indices]
            )
        )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    xx = np.arange(
        N_MINE
    )

    ax.errorbar(
        xx,
        mine_means,
        yerr=1.96 * np.asarray(mine_sds),
        fmt="o",
        capsize=4
    )

    ax.axhline(
        1.0,
        linestyle="--"
    )

    ax.set_xticks(
        xx
    )

    ax.set_xticklabels(
        MINES
    )

    ax.set_ylabel(
        "Mean daily emission scaling factor"
    )

    ax.set_title(
        "Posterior mine scaling factors"
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "posterior_mine_scaling.png",
        dpi=200
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. Posterior prediction residual time series
    # -------------------------------------------------------------------------

    meta = pd.read_csv(
        META_FILE
    )

    if len(meta) == len(residual):

        dt = pd.to_datetime(
            meta["datetime_aest"]
        )

        fig, ax = plt.subplots(
            figsize=(16, 5)
        )

        ax.plot(
            dt,
            residual,
            ".",
            markersize=3
        )

        ax.axhline(
            0,
            linestyle="--"
        )

        ax.set_xlabel(
            "Time (AEST)"
        )

        ax.set_ylabel(
            "Observed - predicted CH4 (ppm)"
        )

        ax.set_title(
            "Posterior residual time series"
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        fig.savefig(
            OUT_DIR / "posterior_residual_timeseries.png",
            dpi=200
        )

        plt.close(fig)

    print("\nPlots saved in:")
    print(OUT_DIR)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")
    print("=" * 80)
    print("UPPER HUNTER ANALYTICAL BAYESIAN INVERSION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Read prior emissions
    # -------------------------------------------------------------------------

    prior_rates_g_s, prior_rates_kg_s = (
        read_prior_emissions()
    )

    # -------------------------------------------------------------------------
    # Read H and observations
    # -------------------------------------------------------------------------

    H, obs, y, obs_df = (
        read_inputs()
    )

    # -------------------------------------------------------------------------
    # Construct G
    # -------------------------------------------------------------------------

    G, emission_vector_kg_s = (
        construct_G(
            H,
            prior_rates_kg_s
        )
    )

    # -------------------------------------------------------------------------
    # Prior state
    # -------------------------------------------------------------------------

    s0 = np.full(
        N_PARAM,
        PRIOR_MEAN,
        dtype=float
    )

    # -------------------------------------------------------------------------
    # B
    # -------------------------------------------------------------------------

    B = construct_B()

    # -------------------------------------------------------------------------
    # R
    # -------------------------------------------------------------------------

    R = construct_R(
        len(obs)
    )

    # -------------------------------------------------------------------------
    # Bayesian inversion
    # -------------------------------------------------------------------------

    (
        s_post,
        B_post,
        posterior_sd,
        S
    ) = analytical_bayesian(
        G,
        y,
        s0,
        B,
        R
    )

    # -------------------------------------------------------------------------
    # Print important results
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("POSTERIOR SUMMARY")
    print("=" * 80)

    print(
        f"\nPrior scaling factor: "
        f"{PRIOR_MEAN:.4f}"
    )

    print(
        f"Posterior scaling mean: "
        f"{np.mean(s_post):.6f}"
    )

    print(
        f"Posterior scaling median: "
        f"{np.median(s_post):.6f}"
    )

    print(
        f"Posterior scaling minimum: "
        f"{np.min(s_post):.6f}"
    )

    print(
        f"Posterior scaling maximum: "
        f"{np.max(s_post):.6f}"
    )

    negative = np.sum(
        s_post < 0
    )

    print(
        f"\nNegative posterior factors: "
        f"{negative} / {N_PARAM}"
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    parameter_names = (
        build_parameter_names()
    )

    save_results(
        s0,
        s_post,
        posterior_sd,
        B,
        B_post,
        prior_rates_g_s,
        emission_vector_kg_s,
        parameter_names
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    make_plots(
        s_post,
        posterior_sd,
        obs,
        y,
        G
    )

    print("\n" + "=" * 80)
    print("ANALYTICAL BAYESIAN INVERSION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

