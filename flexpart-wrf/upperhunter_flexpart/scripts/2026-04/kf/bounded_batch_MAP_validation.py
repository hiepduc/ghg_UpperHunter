#!/usr/bin/env python3

"""
Upper Hunter CH4 inverse modelling
351-parameter bounded batch MAP validation

13 mines x 27 days = 351 daily scaling factors

This script solves the exact box-constrained quadratic MAP problem:

    minimize_x
        0.5 (x - xb)^T B^-1 (x - xb)
      + 0.5 (y - Hx - background)^T R^-1
                         (y - Hx - background)

subject to

        LOWER_BOUND <= x_i <= UPPER_BOUND

using scipy.optimize.lsq_linear on the whitened system.

The resulting bounded batch MAP is the reference solution for
validating the bounded sequential Kalman filter and, later,
a bounded/constrained smoother.

Important:
- H already contains the prior mine-emission contribution.
- Do NOT multiply H by prior mine emissions again.
- Prior mean = 1.0.
- Prior sigma = 0.5.
- Observation sigma = 0.05 ppm.
- Background = 1.85 ppm.
- Temporal prior correlation length = 3 days.
- Bounds = [0, 5].
"""

import os
import sys
import numpy as np
import pandas as pd

from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import lsq_linear


# =============================================================================
# USER SETTINGS
# =============================================================================

START_DATE = pd.Timestamp("2026-01-05")
END_DATE   = pd.Timestamp("2026-01-31")

MINES = [f"OC{i}" for i in range(1, 14)]

N_MINES = len(MINES)

DATES = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

N_DAYS = len(DATES)
N_PARAM = N_MINES * N_DAYS


# -------------------------------------------------------------------------
# Bayesian prior
# -------------------------------------------------------------------------

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5

CORRELATION_LENGTH_DAYS = 3.0

RHO = np.exp(-1.0 / CORRELATION_LENGTH_DAYS)


# -------------------------------------------------------------------------
# Observation model
# -------------------------------------------------------------------------

BACKGROUND_PPM = 1.85
OBS_SIGMA = 0.05


# -------------------------------------------------------------------------
# Physical bounds
# -------------------------------------------------------------------------

LOWER_BOUND = 0.0
UPPER_BOUND = 5.0


# -------------------------------------------------------------------------
# Input files
# -------------------------------------------------------------------------

BASE_DIR = (
    "/home/duch/flexpart-wrf/"
    "upperhunter_flexpart"
)

H_FILE = (
    BASE_DIR +
    "/H_matrix_flexpart/"
    "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/"
    "emission_UH/RH_upperHunter/"
    "tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)


# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR +
    "/H_matrix_flexpart/"
    "Bounded_batch_MAP_full_footprint_13mine_Jan2026"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parameter_name(mine, date):
    """Return parameter name matching H matrix."""
    return f"{mine}_{date.strftime('%Y%m%d')}"


def build_parameter_names():
    """Build 351 parameter names in mine/day ordering."""
    names = []

    for date in DATES:
        for mine in MINES:
            names.append(parameter_name(mine, date))

    return names


def build_prior_covariance():
    """
    Build B for the 351 daily scaling factors.

    Same mine:
        B_ij = sigma^2 * rho^|i-j|

    Different mines:
        covariance = 0

    Parameter ordering:
        day-major:
            OC1_day1 ... OC13_day1
            OC1_day2 ... OC13_day2
            ...
    """

    B = np.zeros((N_PARAM, N_PARAM), dtype=float)

    sigma2 = PRIOR_SIGMA ** 2

    for day_i in range(N_DAYS):

        for mine_i in range(N_MINES):

            i = day_i * N_MINES + mine_i

            for day_j in range(N_DAYS):

                for mine_j in range(N_MINES):

                    j = day_j * N_MINES + mine_j

                    if mine_i == mine_j:

                        lag = abs(day_i - day_j)

                        B[i, j] = (
                            sigma2 *
                            RHO ** lag
                        )

    return B


def load_H_matrix():
    """
    Read H matrix and reorder columns into the exact day-major
    ordering used by this script.
    """

    print("\n" + "=" * 90)
    print("READING FLEXPART H MATRIX")
    print("=" * 90)

    H_df = pd.read_csv(H_FILE)

    print("H shape:", H_df.shape)

    metadata_columns = [
        "receptor",
        "release_utc",
        "obs_time_aest",
        "flexpart_file"
    ]

    parameter_names = build_parameter_names()

    missing = [
        c for c in parameter_names
        if c not in H_df.columns
    ]

    if missing:

        print("\nERROR: Missing H columns:")
        for c in missing:
            print(" ", c)

        raise RuntimeError(
            f"{len(missing)} expected parameter columns "
            "are missing from H matrix."
        )

    H = H_df[parameter_names].to_numpy(dtype=float)

    metadata = H_df[metadata_columns].copy()

    print("Number of parameters:", len(parameter_names))
    print("H matrix shape:", H.shape)

    return H, metadata, parameter_names


def load_observations(H_metadata):
    """
    Match Picarro observations to FLEXPART H rows.

    Uses:
        receptor
        AEST observation time

    The same matching convention used by the KF is retained:
    timestamps are floored to the hour.
    """

    print("\n" + "=" * 90)
    print("READING PICARRO OBSERVATIONS")
    print("=" * 90)

    obs = pd.read_csv(PICARRO_FILE)

    obs["Datetime"] = pd.to_datetime(
        obs["Datetime"],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    obs = obs.dropna(subset=["Datetime"]).copy()

    # AEST
    obs["time_aest"] = (
        obs["Datetime"]
        .dt.floor("h")
    )

    receptor_columns = {
        "singleton": "Singleton cleaned",
        "muswellbrook": "Muswellbrook cleaned",
        "merriwa": "Merriwa cleaned",
    }

    metadata = H_metadata.copy()

    metadata["time_aest"] = pd.to_datetime(
        metadata["obs_time_aest"],
        errors="coerce"
    )

    metadata["time_aest"] = (
        metadata["time_aest"]
        .dt.floor("h")
    )

    matched_rows = []

    for receptor, column in receptor_columns.items():

        sub_obs = obs[
            obs[column].notna()
        ][
            ["time_aest", column]
        ].copy()

        sub_obs = sub_obs.rename(
            columns={column: "observation"}
        )

        sub_obs["receptor"] = receptor

        sub_obs = sub_obs[
            (sub_obs["time_aest"] >= START_DATE) &
            (
                sub_obs["time_aest"]
                < END_DATE + pd.Timedelta(days=1)
            )
        ]

        matched_rows.append(sub_obs)

    obs_long = pd.concat(
        matched_rows,
        ignore_index=True
    )

    # Keep only the receptor/time combinations that occur in H.
    merged = metadata.merge(
        obs_long,
        on=["receptor", "time_aest"],
        how="inner"
    )

    # Preserve H row ordering.
    key_h = pd.MultiIndex.from_arrays(
        [
            metadata["receptor"].astype(str),
            metadata["time_aest"]
        ]
    )

    key_m = pd.MultiIndex.from_arrays(
        [
            merged["receptor"].astype(str),
            merged["time_aest"]
        ]
    )

    lookup = {
        key: idx
        for idx, key in enumerate(key_m)
    }

    h_indices = []
    obs_values = []

    for i, key in enumerate(key_h):

        if key in lookup:

            j = lookup[key]

            h_indices.append(i)
            obs_values.append(
                merged.iloc[j]["observation"]
            )

    h_indices = np.asarray(
        h_indices,
        dtype=int
    )

    y = np.asarray(
        obs_values,
        dtype=float
    )

    print("Observation rows:", len(obs))
    print("Matched observations:", len(y))

    if len(y) == 0:
        raise RuntimeError(
            "No observations matched the H matrix."
        )

    # Sort chronologically, then receptor.
    sort_key = pd.DataFrame({
        "time": metadata.iloc[h_indices]["time_aest"].values,
        "receptor": metadata.iloc[h_indices]["receptor"].values
    })

    order = np.lexsort(
        [
            sort_key["receptor"].astype(str).to_numpy(),
            sort_key["time"].to_numpy()
        ]
    )

    h_indices = h_indices[order]
    y = y[order]

    return h_indices, y


def check_H_temporal_support(metadata):
    """Diagnostic for maximum backward temporal support."""

    print("\n" + "=" * 90)
    print("CHECKING H TEMPORAL SUPPORT")
    print("=" * 90)

    release = pd.to_datetime(
        metadata["release_utc"],
        errors="coerce"
    )

    obs_time = pd.to_datetime(
        metadata["obs_time_aest"],
        errors="coerce"
    )

    # UTC release converted to AEST.
    release_aest = release + pd.Timedelta(hours=10)

    lag_hours = (
        obs_time - release_aest
    ).dt.total_seconds() / 3600.0

    print(
        "Maximum backward lag:",
        int(np.ceil(lag_hours.max() / 24.0)),
        "day(s)"
    )


def kkt_diagnostics(A, b, z, lower, upper):
    """
    KKT diagnostics for bounded least squares.

    Objective:
        0.5 ||A z - b||^2

    Gradient:
        g = A.T @ (A z - b)

    At lower bound:
        g >= 0

    At upper bound:
        g <= 0

    Free variables:
        g ~= 0
    """

    residual = A @ z - b

    gradient = A.T @ residual

    tol_active = 1e-7
    tol_free = 1e-6

    at_lower = z <= lower + tol_active
    at_upper = z >= upper - tol_active

    free = (
        ~at_lower &
        ~at_upper
    )

    lower_violation = np.maximum(
        0.0,
        -gradient[at_lower]
    )

    upper_violation = np.maximum(
        0.0,
        gradient[at_upper]
    )

    free_abs = (
        np.abs(gradient[free])
        if np.any(free)
        else np.array([0.0])
    )

    return {
        "gradient": gradient,
        "at_lower": at_lower,
        "at_upper": at_upper,
        "free": free,
        "max_lower_KKT_violation": (
            np.max(lower_violation)
            if len(lower_violation)
            else 0.0
        ),
        "max_upper_KKT_violation": (
            np.max(upper_violation)
            if len(upper_violation)
            else 0.0
        ),
        "max_free_gradient": np.max(free_abs),
        "gradient_norm": np.linalg.norm(gradient),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n" + "=" * 90)
    print("BOUNDED BATCH MAP — UPPER HUNTER CH4")
    print("=" * 90)

    print("Start date:", START_DATE.date())
    print("End date:", END_DATE.date())
    print("Number of mines:", N_MINES)
    print("Number of days:", N_DAYS)
    print("Number of parameters:", N_PARAM)

    print("Prior mean:", PRIOR_MEAN)
    print("Prior sigma:", PRIOR_SIGMA)
    print("Observation sigma:", OBS_SIGMA)
    print("Background:", BACKGROUND_PPM)
    print("Correlation length:", CORRELATION_LENGTH_DAYS)
    print("rho:", RHO)

    print("Lower bound:", LOWER_BOUND)
    print("Upper bound:", UPPER_BOUND)


    # =========================================================================
    # 1. H
    # =========================================================================

    H_full, metadata, parameter_names = load_H_matrix()

    check_H_temporal_support(metadata)


    # =========================================================================
    # 2. Observations
    # =========================================================================

    h_indices, y = load_observations(
        metadata
    )

    H = H_full[h_indices, :]

    metadata_matched = metadata.iloc[
        h_indices
    ].copy()

    metadata_matched["observation"] = y

    print("\nMatched H shape:", H.shape)

    print(
        "Matched receptors:",
        metadata_matched["receptor"]
        .value_counts()
        .to_dict()
    )


    # =========================================================================
    # 3. Build prior covariance B
    # =========================================================================

    print("\n" + "=" * 90)
    print("BUILDING PRIOR COVARIANCE")
    print("=" * 90)

    B = build_prior_covariance()

    print("B shape:", B.shape)

    print(
        "B[OC1 Jan5, OC1 Jan5] =",
        B[0, 0]
    )

    print(
        "B[OC1 Jan5, OC1 Jan6] =",
        B[0, N_MINES]
    )

    print(
        "B[OC1 Jan5, OC2 Jan5] =",
        B[0, 1]
    )


    # =========================================================================
    # 4. Cholesky factorization
    # =========================================================================

    print("\n" + "=" * 90)
    print("CHOLESKY FACTORIZATION")
    print("=" * 90)

    L_B = cholesky(
        B,
        lower=True,
        check_finite=True
    )

    B_reconstructed = (
        L_B @ L_B.T
    )

    reconstruction_error = np.max(
        np.abs(
            B_reconstructed - B
        )
    )

    print(
        "Cholesky reconstruction error:",
        reconstruction_error
    )


    # =========================================================================
    # 5. Build whitened batch MAP problem
    #
    # Let:
    #
    #       x = xb + z
    #
    # where xb = prior mean vector.
    #
    # Prior term:
    #
    #       || L_B^-1 z ||^2
    #
    # Observation term:
    #
    #       || (y - background - H xb - H z) / sigma ||^2
    #
    # Therefore:
    #
    #       A z ~= b
    #
    # with bounds:
    #
    #       lower - xb <= z <= upper - xb
    # =========================================================================

    print("\n" + "=" * 90)
    print("BUILDING WHITENED BOUNDED LEAST-SQUARES SYSTEM")
    print("=" * 90)

    xb = np.full(
        N_PARAM,
        PRIOR_MEAN,
        dtype=float
    )

    # Prior:
    #
    # L_B w = z
    #
    # w = L_B^-1 z
    #
    # Thus prior design matrix is L_B^-1.
    #
    # solve_triangular(L_B, I) gives exactly that.

    A_prior = solve_triangular(
        L_B,
        np.eye(N_PARAM),
        lower=True
    )

    b_prior = np.zeros(
        N_PARAM,
        dtype=float
    )

    # Observation part.

    A_obs = H / OBS_SIGMA

    b_obs = (
        y
        - BACKGROUND_PPM
        - H @ xb
    ) / OBS_SIGMA

    A = np.vstack([
        A_prior,
        A_obs
    ])

    b = np.concatenate([
        b_prior,
        b_obs
    ])

    print("Prior rows:", A_prior.shape[0])
    print("Observation rows:", A_obs.shape[0])
    print("Total rows:", A.shape[0])
    print("Parameters:", A.shape[1])

    print(
        "Whitened system shape:",
        A.shape
    )


    # =========================================================================
    # 6. Bounds on z
    # =========================================================================

    lower_z = np.full(
        N_PARAM,
        LOWER_BOUND - PRIOR_MEAN,
        dtype=float
    )

    upper_z = np.full(
        N_PARAM,
        UPPER_BOUND - PRIOR_MEAN,
        dtype=float
    )


    # =========================================================================
    # 7. Solve exact bounded batch MAP
    # =========================================================================

    print("\n" + "=" * 90)
    print("SOLVING BOUNDED BATCH MAP")
    print("=" * 90)

    print(
        "Using scipy.optimize.lsq_linear"
    )

    print(
        "Bounds on scaling factors:",
        f"[{LOWER_BOUND}, {UPPER_BOUND}]"
    )

    result = lsq_linear(
        A,
        b,
        bounds=(
            lower_z,
            upper_z
        ),
        method="trf",
        lsmr_tol="auto",
        lsmr_maxiter=5000,
        max_iter=1000,
        verbose=1,
    )

    if not result.success:

        print("\nWARNING:")
        print(
            "Bounded least-squares solver did not report "
            "successful convergence."
        )

        print(
            "Status:",
            result.status
        )

        print(
            "Message:",
            result.message
        )

    else:

        print("\nSolver converged successfully.")

    print(
        "Solver status:",
        result.status
    )

    print(
        "Solver message:",
        result.message
    )

    print(
        "Iterations:",
        result.nit
    )

    print(
        "Optimality:",
        result.optimality
    )

    print(
        "Cost:",
        result.cost
    )


    # =========================================================================
    # 8. Convert z -> x
    # =========================================================================

    z = result.x

    x_map = (
        PRIOR_MEAN + z
    )

    print("\n" + "=" * 90)
    print("BOUNDED BATCH MAP SOLUTION")
    print("=" * 90)

    print(
        "Minimum:",
        x_map.min()
    )

    print(
        "Maximum:",
        x_map.max()
    )

    print(
        "Mean:",
        x_map.mean()
    )

    lower_active = (
        x_map <= LOWER_BOUND + 1e-7
    )

    upper_active = (
        x_map >= UPPER_BOUND - 1e-7
    )

    free = (
        ~lower_active &
        ~upper_active
    )

    print(
        "Lower-bound parameters:",
        int(lower_active.sum())
    )

    print(
        "Upper-bound parameters:",
        int(upper_active.sum())
    )

    print(
        "Free parameters:",
        int(free.sum())
    )


    # =========================================================================
    # 9. KKT diagnostics
    # =========================================================================

    print("\n" + "=" * 90)
    print("KKT OPTIMALITY CHECK")
    print("=" * 90)

    kkt = kkt_diagnostics(
        A,
        b,
        z,
        lower_z,
        upper_z
    )

    print(
        "Gradient norm:",
        kkt["gradient_norm"]
    )

    print(
        "Maximum free-variable |gradient|:",
        kkt["max_free_gradient"]
    )

    print(
        "Maximum lower-bound KKT violation:",
        kkt["max_lower_KKT_violation"]
    )

    print(
        "Maximum upper-bound KKT violation:",
        kkt["max_upper_KKT_violation"]
    )


    # =========================================================================
    # 10. Save 351-parameter solution
    # =========================================================================

    solution_df = pd.DataFrame({

        "parameter": parameter_names,

        "mine": [
            MINES[i % N_MINES]
            for i in range(N_PARAM)
        ],

        "date": [
            DATES[i // N_MINES]
            for i in range(N_PARAM)
        ],

        "bounded_batch_MAP": x_map,

        "active_lower": lower_active,

        "active_upper": upper_active,

        "free": free,

        "gradient": kkt["gradient"],

    })

    solution_file = (
        OUTPUT_DIR +
        "/bounded_batch_MAP_351_parameters.csv"
    )

    solution_df.to_csv(
        solution_file,
        index=False
    )

    print(
        "\nSaved:",
        solution_file
    )


    # =========================================================================
    # 11. Daily 13-mine matrix
    # =========================================================================

    daily_matrix = np.zeros(
        (N_DAYS, N_MINES)
    )

    for d in range(N_DAYS):
        start = d * N_MINES
        end = start + N_MINES

        daily_matrix[d, :] = (
            x_map[start:end]
        )

    daily_df = pd.DataFrame(
        daily_matrix,
        columns=MINES
    )

    daily_df.insert(
        0,
        "date",
        DATES
    )

    daily_file = (
        OUTPUT_DIR +
        "/daily_bounded_batch_MAP_scaling_factors.csv"
    )

    daily_df.to_csv(
        daily_file,
        index=False
    )

    print(
        "Saved:",
        daily_file
    )


    # =========================================================================
    # 12. Active-bound summary
    # =========================================================================

    active_df = solution_df[
        solution_df["active_lower"] |
        solution_df["active_upper"]
    ].copy()

    active_file = (
        OUTPUT_DIR +
        "/bounded_batch_MAP_active_bounds.csv"
    )

    active_df.to_csv(
        active_file,
        index=False
    )

    print(
        "Saved:",
        active_file
    )


    # =========================================================================
    # 13. Per-mine summary
    # =========================================================================

    mine_summary = []

    for mine in MINES:

        sub = solution_df[
            solution_df["mine"] == mine
        ]

        mine_summary.append({

            "mine": mine,

            "mean": sub[
                "bounded_batch_MAP"
            ].mean(),

            "minimum": sub[
                "bounded_batch_MAP"
            ].min(),

            "maximum": sub[
                "bounded_batch_MAP"
            ].max(),

            "lower_activations": int(
                sub["active_lower"].sum()
            ),

            "upper_activations": int(
                sub["active_upper"].sum()
            ),

        })

    mine_summary_df = pd.DataFrame(
        mine_summary
    )

    mine_summary_file = (
        OUTPUT_DIR +
        "/bounded_batch_MAP_mine_summary.csv"
    )

    mine_summary_df.to_csv(
        mine_summary_file,
        index=False
    )

    print(
        "Saved:",
        mine_summary_file
    )


    # =========================================================================
    # 14. Final-day table
    # =========================================================================

    final_day = solution_df[
        solution_df["date"] == END_DATE
    ].copy()

    print("\n" + "=" * 90)
    print("FINAL-DAY BOUNDED BATCH MAP")
    print("=" * 90)

    for _, row in final_day.iterrows():

        marker = ""

        if row["active_lower"]:
            marker = "  <-- LOWER BOUND"

        elif row["active_upper"]:
            marker = "  <-- UPPER BOUND"

        print(
            f"{row['mine']:5s} "
            f"{row['bounded_batch_MAP']:12.8f}"
            f"{marker}"
        )


    # =========================================================================
    # 15. Compare with unbounded batch MAP if available
    # =========================================================================

    unbounded_candidates = [

        BASE_DIR +
        "/H_matrix_flexpart/"
        "Bounded_batch_MAP_full_footprint_13mine_Jan2026/"
        "unbounded_batch_MAP_351_parameters.csv",

        BASE_DIR +
        "/H_matrix_flexpart/"
        "RTS_full_footprint_13mine_Jan2026/"
        "daily_batch_MAP_scaling_factors.csv",

        BASE_DIR +
        "/H_matrix_flexpart/"
        "KF_full_footprint_13mine_Jan2026/"
        "daily_batch_MAP_scaling_factors.csv",

    ]

    unbounded_file = None

    for candidate in unbounded_candidates:

        if os.path.exists(candidate):

            unbounded_file = candidate
            break


    if unbounded_file is not None:

        print("\n" + "=" * 90)
        print("COMPARING WITH EXISTING UNBOUNDED BATCH MAP")
        print("=" * 90)

        try:

            if (
                "351_parameters"
                in os.path.basename(unbounded_file)
            ):

                ub_df = pd.read_csv(
                    unbounded_file
                )

                if (
                    "unbounded_batch_MAP"
                    in ub_df.columns
                ):

                    x_ub = (
                        ub_df[
                            "unbounded_batch_MAP"
                        ]
                        .to_numpy()
                    )

                else:

                    x_ub = (
                        ub_df.iloc[:, -1]
                        .to_numpy()
                    )

            else:

                ub_daily = pd.read_csv(
                    unbounded_file
                )

                ub_values = []

                for date in DATES:

                    row = ub_daily[
                        pd.to_datetime(
                            ub_daily["date"]
                        ) == date
                    ]

                    if len(row) == 0:
                        continue

                    for mine in MINES:
                        ub_values.append(
                            float(row.iloc[0][mine])
                        )

                x_ub = np.asarray(
                    ub_values
                )


            if len(x_ub) == N_PARAM:

                difference = (
                    x_map - x_ub
                )

                print(
                    "Maximum absolute difference:",
                    np.max(np.abs(difference))
                )

                print(
                    "RMSE:",
                    np.sqrt(
                        np.mean(
                            difference ** 2
                        )
                    )
                )

                print(
                    "MAE:",
                    np.mean(
                        np.abs(difference)
                    )
                )

                comparison = pd.DataFrame({

                    "parameter":
                        parameter_names,

                    "bounded_batch_MAP":
                        x_map,

                    "unbounded_batch_MAP":
                        x_ub,

                    "difference":
                        difference,

                })

                comparison_file = (
                    OUTPUT_DIR +
                    "/bounded_vs_unbounded_batch_MAP.csv"
                )

                comparison.to_csv(
                    comparison_file,
                    index=False
                )

                print(
                    "Saved:",
                    comparison_file
                )

        except Exception as exc:

            print(
                "Could not compare with existing "
                "unbounded MAP:"
            )

            print(
                repr(exc)
            )

    else:

        print(
            "\nNo existing unbounded batch MAP file "
            "found for automatic comparison."
        )


    # =========================================================================
    # 16. Save solver diagnostics
    # =========================================================================

    summary_file = (
        OUTPUT_DIR +
        "/bounded_batch_MAP_summary.txt"
    )

    with open(
        summary_file,
        "w"
    ) as f:

        f.write(
            "UPPER HUNTER CH4 BOUNDED BATCH MAP\n"
        )

        f.write(
            "===================================\n\n"
        )

        f.write(
            f"Start date: {START_DATE.date()}\n"
        )

        f.write(
            f"End date: {END_DATE.date()}\n"
        )

        f.write(
            f"Number of mines: {N_MINES}\n"
        )

        f.write(
            f"Number of days: {N_DAYS}\n"
        )

        f.write(
            f"Number of parameters: {N_PARAM}\n\n"
        )

        f.write(
            f"Prior mean: {PRIOR_MEAN}\n"
        )

        f.write(
            f"Prior sigma: {PRIOR_SIGMA}\n"
        )

        f.write(
            f"Observation sigma: {OBS_SIGMA}\n"
        )

        f.write(
            f"Background ppm: {BACKGROUND_PPM}\n"
        )

        f.write(
            f"Correlation length: "
            f"{CORRELATION_LENGTH_DAYS} days\n"
        )

        f.write(
            f"rho: {RHO}\n\n"
        )

        f.write(
            f"Lower bound: {LOWER_BOUND}\n"
        )

        f.write(
            f"Upper bound: {UPPER_BOUND}\n\n"
        )

        f.write(
            f"Matched observations: {len(y)}\n"
        )

        f.write(
            f"Whitened system: {A.shape}\n\n"
        )

        f.write(
            f"Solver status: {result.status}\n"
        )

        f.write(
            f"Solver success: {result.success}\n"
        )

        f.write(
            f"Solver message: {result.message}\n"
        )

        f.write(
            f"Iterations: {result.nit}\n"
        )

        f.write(
            f"Cost: {result.cost}\n"
        )

        f.write(
            f"Optimality: {result.optimality}\n\n"
        )

        f.write(
            f"Minimum scaling factor: "
            f"{x_map.min()}\n"
        )

        f.write(
            f"Maximum scaling factor: "
            f"{x_map.max()}\n"
        )

        f.write(
            f"Mean scaling factor: "
            f"{x_map.mean()}\n"
        )

        f.write(
            f"Lower-bound parameters: "
            f"{lower_active.sum()}\n"
        )

        f.write(
            f"Upper-bound parameters: "
            f"{upper_active.sum()}\n"
        )

        f.write(
            f"Free parameters: "
            f"{free.sum()}\n\n"
        )

        f.write(
            f"KKT gradient norm: "
            f"{kkt['gradient_norm']}\n"
        )

        f.write(
            f"Maximum free gradient: "
            f"{kkt['max_free_gradient']}\n"
        )

        f.write(
            f"Maximum lower KKT violation: "
            f"{kkt['max_lower_KKT_violation']}\n"
        )

        f.write(
            f"Maximum upper KKT violation: "
            f"{kkt['max_upper_KKT_violation']}\n"
        )


    print(
        "\nSaved:",
        summary_file
    )


    # =========================================================================
    # 17. Save NPZ
    # =========================================================================

    npz_file = (
        OUTPUT_DIR +
        "/bounded_batch_MAP_solution.npz"
    )

    np.savez_compressed(
        npz_file,

        states=daily_matrix,

        parameter_vector=x_map,

        parameter_names=np.asarray(
            parameter_names
        ),

        dates=np.asarray(
            DATES.strftime("%Y-%m-%d"),
            dtype="U10"
        ),

        mines=np.asarray(
            MINES
        ),

        lower_active=lower_active,

        upper_active=upper_active,

        gradient=kkt["gradient"],

        solver_cost=result.cost,

        solver_optimality=result.optimality,

    )

    print(
        "Saved:",
        npz_file
    )


    # =========================================================================
    # FINAL
    # =========================================================================

    print("\n" + "=" * 90)
    print("BOUNDED BATCH MAP VALIDATION COMPLETE")
    print("=" * 90)

    print(
        "Reference solution:",
        OUTPUT_DIR
    )

    print(
        "\nThe bounded batch MAP is now ready to compare "
        "against the bounded sequential KF."
    )

    print(
        "\nNext important comparison:"
    )

    print(
        "  bounded KF  <->  bounded batch MAP"
    )

    print(
        "\nDo NOT run a bounded RTS yet."
    )

    print(
        "First establish how closely the bounded sequential "
        "KF reproduces this constrained batch solution."
    )


if __name__ == "__main__":
    main()

