#!/usr/bin/env python3
"""
Upper Hunter CH4
Full-FLEXPART-Footprint Augmented-State RTS Smoother

State:
    X_d = [x_(d-1), x_d]

Observation:
    y_t = H_(t,d-1) x_(d-1) + H_(t,d) x_d + eps

Daily transition:
    x_(d+1) = rho x_d + (1-rho) xb + w_d

Augmented transition:
    X_(d+1) = F_aug X_d + c_aug + eta_d

This script:
    1. Reads the full-footprint augmented KF output.
    2. Runs an augmented-state RTS smoother.
    3. Extracts the current-day 13-mine smoothed state.
    4. Independently computes the full 351-parameter batch MAP.
    5. Compares the RTS solution against the batch MAP.

IMPORTANT:
    This script does NOT modify the production KF or existing 13-state RTS.
"""

import os

import numpy as np
import pandas as pd

from scipy.linalg import solve_triangular


# =============================================================================
# CONFIGURATION
# =============================================================================

H_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "tapm_output/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

KF_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/KF_full_footprint_13mine_Jan2026"
)

KF_NPZ = os.path.join(
    KF_DIR,
    "full_footprint_KF_daily_states.npz"
)

OUTPUT_DIR = KF_DIR

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

BACKGROUND_PPM = 1.85

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

L_CORR = 3.0

OBS_SIGMA = 0.05
OBS_VARIANCE = OBS_SIGMA ** 2

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
N_AUG = 2 * N_MINE

rho = np.exp(-1.0 / L_CORR)

Q = (
    PRIOR_VARIANCE
    * (1.0 - rho ** 2)
    * np.eye(N_MINE)
)


# =============================================================================
# AUGMENTED STATE TRANSITION
# =============================================================================
#
# X_d = [x_(d-1), x_d]
#
# X_(d+1) = [x_d,
#            rho*x_d + (1-rho)*xb + w_d]
#
# Therefore:
#
# F_aug =
#
# [ 0   I ]
# [ 0 rhoI]
#
# c_aug =
#
# [ 0 ]
# [(1-rho)*xb]
#
# Q_aug =
#
# [ 0  0 ]
# [ 0  Q ]
#
# =============================================================================

F_aug = np.zeros(
    (N_AUG, N_AUG),
    dtype=np.float64
)

F_aug[
    :N_MINE,
    N_MINE:
] = np.eye(N_MINE)

F_aug[
    N_MINE:,
    N_MINE:
] = rho * np.eye(N_MINE)


c_aug = np.zeros(
    N_AUG,
    dtype=np.float64
)

c_aug[
    N_MINE:
] = (
    1.0 - rho
) * PRIOR_MEAN


Q_aug = np.zeros(
    (N_AUG, N_AUG),
    dtype=np.float64
)

Q_aug[
    N_MINE:,
    N_MINE:
] = Q


# =============================================================================
# PARAMETER NAMES
# =============================================================================

def parameter_names():

    names = []

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D"
    )

    for day in dates:

        for mine in MINES:

            names.append(
                f"{mine}_{day.strftime('%Y%m%d')}"
            )

    return names


# =============================================================================
# BUILD FULL PRIOR COVARIANCE
# =============================================================================

def build_prior_covariance():

    """
    Full 351 x 351 prior covariance.

    Same mine:
        B[(d,i),(j,i)]
            = sigma^2 exp(-|d-j|/L)

    Different mines:
        covariance = 0
    """

    B = np.zeros(
        (N_PARAM, N_PARAM),
        dtype=np.float64
    )

    for day_i in range(N_DAY):

        for mine_i in range(N_MINE):

            i = (
                day_i * N_MINE
                +
                mine_i
            )

            for day_j in range(N_DAY):

                j = (
                    day_j * N_MINE
                    +
                    mine_i
                )

                B[i, j] = (
                    PRIOR_VARIANCE
                    *
                    np.exp(
                        -abs(day_i - day_j)
                        / L_CORR
                    )
                )

    return B


# =============================================================================
# READ PICARRO OBSERVATIONS
# =============================================================================

def read_observations():

    print()
    print("=" * 90)
    print("READING PICARRO OBSERVATIONS")
    print("=" * 90)

    df = pd.read_csv(
        PICARRO_FILE
    )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    receptor_columns = {
        "Singleton cleaned": "singleton",
        "Muswellbrook cleaned": "muswellbrook",
        "Merriwa cleaned": "merriwa",
    }

    records = []

    for column, receptor in receptor_columns.items():

        tmp = df[
            [
                "Datetime",
                column
            ]
        ].copy()

        tmp = tmp.rename(
            columns={
                "Datetime": "time_aest",
                column: "observation",
            }
        )

        tmp["receptor"] = receptor

        tmp = tmp.dropna(
            subset=[
                "time_aest",
                "observation"
            ]
        )

        tmp["time_aest"] = (
            tmp["time_aest"]
            .dt.floor("h")
        )

        records.append(
            tmp
        )

    obs = pd.concat(
        records,
        ignore_index=True
    )

    obs = (
        obs
        .groupby(
            [
                "receptor",
                "time_aest"
            ],
            as_index=False
        )["observation"]
        .mean()
    )

    obs = obs[
        (obs["time_aest"] >= START_DATE)
        &
        (
            obs["time_aest"]
            <
            END_DATE + pd.Timedelta(days=1)
        )
    ].copy()

    print(
        "Observations:",
        len(obs)
    )

    return obs


# =============================================================================
# READ AND MATCH H
# =============================================================================

def read_and_match_H(obs):

    print()
    print("=" * 90)
    print("READING FLEXPART H")
    print("=" * 90)

    Hdf = pd.read_csv(
        H_FILE
    )

    print(
        "H shape:",
        Hdf.shape
    )

    parameter_cols = parameter_names()

    print()
    print("=" * 90)
    print("MATCHING H AND OBSERVATIONS")
    print("=" * 90)

    Hdf["obs_time_aest"] = pd.to_datetime(
        Hdf["obs_time_aest"]
    ).dt.floor("h")

    obs2 = obs.copy()

    matched = pd.merge(
        Hdf,
        obs2,
        left_on=[
            "receptor",
            "obs_time_aest"
        ],
        right_on=[
            "receptor",
            "time_aest"
        ],
        how="inner"
    )

    matched = (
        matched
        .sort_values(
            "obs_time_aest"
        )
        .reset_index(drop=True)
    )

    print(
        "Matched observations:",
        len(matched)
    )

    H = matched[
        parameter_cols
    ].to_numpy(
        dtype=np.float64
    )

    y = matched[
        "observation"
    ].to_numpy(
        dtype=np.float64
    )

    print(
        "Full H shape:",
        H.shape
    )

    return (
        matched,
        H,
        y,
        parameter_cols
    )


# =============================================================================
# COMPUTE FULL BATCH MAP
# =============================================================================

def compute_batch_map(
    H,
    y,
    B
):

    print()
    print("=" * 90)
    print("COMPUTING FULL 351-PARAMETER BATCH MAP")
    print("=" * 90)

    xb = np.full(
        N_PARAM,
        PRIOR_MEAN,
        dtype=np.float64
    )

    d = (
        y
        -
        BACKGROUND_PPM
    )

    # -------------------------------------------------------------------------
    # Cholesky:
    #
    # B = L L^T
    #
    # Prior whitening:
    #
    # L^-1 (x-xb)
    #
    # -------------------------------------------------------------------------

    LB = np.linalg.cholesky(
        B
    )

    A_prior = solve_triangular(
        LB,
        np.eye(N_PARAM),
        lower=True
    )

    # -------------------------------------------------------------------------
    # Observation whitening
    # -------------------------------------------------------------------------

    A_obs = (
        H
        /
        OBS_SIGMA
    )

    rhs_obs = (
        d
        -
        H @ xb
    ) / OBS_SIGMA

    A_aug = np.vstack(
        [
            A_prior,
            A_obs
        ]
    )

    b_aug = np.concatenate(
        [
            np.zeros(N_PARAM),
            rhs_obs
        ]
    )

    print(
        "Whitened system shape:",
        A_aug.shape
    )

    z_map, residuals, rank, singular_values = (
        np.linalg.lstsq(
            A_aug,
            b_aug,
            rcond=None
        )
    )

    x_map = (
        xb
        +
        z_map
    )

    print(
        "MAP rank:",
        rank
    )

    if len(residuals) > 0:

        print(
            "Least-squares residual norm:",
            residuals
        )

    # -------------------------------------------------------------------------
    # Gradient validation
    # -------------------------------------------------------------------------

    B_inv_x = np.linalg.solve(
        B,
        x_map - xb
    )

    gradient = (
        B_inv_x
        +
        H.T @ (
            H @ x_map
            -
            d
        ) / OBS_VARIANCE
    )

    print(
        "MAP gradient L2:",
        np.linalg.norm(gradient)
    )

    print(
        "MAP gradient max abs:",
        np.max(
            np.abs(gradient)
        )
    )

    print(
        "Batch MAP mean:",
        np.mean(x_map)
    )

    return x_map


# =============================================================================
# AUGMENTED RTS SMOOTHER
# =============================================================================

def run_augmented_rts(
    x_analysis,
    P_analysis
):

    print()
    print("=" * 90)
    print("RUNNING AUGMENTED-STATE RTS SMOOTHER")
    print("=" * 90)

    n_days = len(
        x_analysis
    )

    x_smooth = np.zeros_like(
        x_analysis,
        dtype=np.float64
    )

    P_smooth = np.zeros_like(
        P_analysis,
        dtype=np.float64
    )

    # -------------------------------------------------------------------------
    # Terminal condition
    # -------------------------------------------------------------------------

    x_smooth[-1] = (
        x_analysis[-1]
    )

    P_smooth[-1] = (
        0.5
        *
        (
            P_analysis[-1]
            +
            P_analysis[-1].T
        )
    )

    # -------------------------------------------------------------------------
    # Backward RTS pass
    # -------------------------------------------------------------------------

    for d in range(
        n_days - 2,
        -1,
        -1
    ):

        # ---------------------------------------------------------------------
        # Forecast from day d to d+1
        # ---------------------------------------------------------------------

        x_forecast = (
            F_aug
            @
            x_analysis[d]
            +
            c_aug
        )

        P_forecast = (
            F_aug
            @
            P_analysis[d]
            @
            F_aug.T
            +
            Q_aug
        )

        P_forecast = (
            0.5
            *
            (
                P_forecast
                +
                P_forecast.T
            )
        )

        # ---------------------------------------------------------------------
        # Cross covariance:
        #
        # Cov(X_d, X_(d+1))
        # =
        # P_d F^T
        # ---------------------------------------------------------------------

        cross_cov = (
            P_analysis[d]
            @
            F_aug.T
        )

        # ---------------------------------------------------------------------
        # P_forecast is singular because:
        #
        # first half of X_(d+1) = second half of X_d
        #
        # Therefore use Moore-Penrose inverse.
        # ---------------------------------------------------------------------

        P_forecast_pinv = np.linalg.pinv(
            P_forecast,
            rcond=1e-12
        )

        C = (
            cross_cov
            @
            P_forecast_pinv
        )

        # ---------------------------------------------------------------------
        # Smoothed mean
        # ---------------------------------------------------------------------

        x_smooth[d] = (
            x_analysis[d]
            +
            C
            @
            (
                x_smooth[d + 1]
                -
                x_forecast
            )
        )

        # ---------------------------------------------------------------------
        # Smoothed covariance
        # ---------------------------------------------------------------------

        P_smooth[d] = (
            P_analysis[d]
            +
            C
            @
            (
                P_smooth[d + 1]
                -
                P_forecast
            )
            @
            C.T
        )

        P_smooth[d] = (
            0.5
            *
            (
                P_smooth[d]
                +
                P_smooth[d].T
            )
        )

        # ---------------------------------------------------------------------
        # Protect numerical diagonal
        # ---------------------------------------------------------------------

        diag = np.diag(
            P_smooth[d]
        ).copy()

        diag = np.maximum(
            diag,
            0.0
        )

        for i in range(N_AUG):

            P_smooth[
                d,
                i,
                i
            ] = diag[i]

    return (
        x_smooth,
        P_smooth
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    dates,
    x_analysis,
    P_analysis,
    x_smooth,
    P_smooth,
    x_map
):

    print()
    print("=" * 90)
    print("SAVING RTS RESULTS")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Current-day component
    #
    # X_d = [x_(d-1), x_d]
    #
    # Therefore current day is the second half.
    # -------------------------------------------------------------------------

    current_filtered = (
        x_analysis[
            :,
            N_MINE:
        ]
    )

    current_smoothed = (
        x_smooth[
            :,
            N_MINE:
        ]
    )

    current_cov = (
        P_smooth[
            :,
            N_MINE:,
            N_MINE:
        ]
    )

    map_daily = x_map.reshape(
        N_DAY,
        N_MINE
    )

    # -------------------------------------------------------------------------
    # Comparison dataframe
    # -------------------------------------------------------------------------

    rows = []

    for d, date in enumerate(dates):

        row = {
            "date":
                pd.Timestamp(date).strftime(
                    "%Y-%m-%d"
                )
        }

        for i, mine in enumerate(MINES):

            row[
                f"{mine}_KF"
            ] = current_filtered[
                d,
                i
            ]

        for i, mine in enumerate(MINES):

            row[
                f"{mine}_RTS"
            ] = current_smoothed[
                d,
                i
            ]

        for i, mine in enumerate(MINES):

            row[
                f"{mine}_MAP"
            ] = map_daily[
                d,
                i
            ]

        rows.append(
            row
        )

    comparison = pd.DataFrame(
        rows
    )

    comparison_file = os.path.join(
        OUTPUT_DIR,
        "full_footprint_RTS_vs_MAP.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # RTS daily scaling factors
    # -------------------------------------------------------------------------

    rts_rows = []

    for d, date in enumerate(dates):

        row = {
            "date":
                pd.Timestamp(date).strftime(
                    "%Y-%m-%d"
                )
        }

        for i, mine in enumerate(MINES):

            row[mine] = (
                current_smoothed[
                    d,
                    i
                ]
            )

        rts_rows.append(
            row
        )

    rts_df = pd.DataFrame(
        rts_rows
    )

    rts_file = os.path.join(
        OUTPUT_DIR,
        "RTS_full_footprint_daily_scaling_factors.csv"
    )

    rts_df.to_csv(
        rts_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Covariance diagonal
    # -------------------------------------------------------------------------

    cov_rows = []

    for d, date in enumerate(dates):

        row = {
            "date":
                pd.Timestamp(date).strftime(
                    "%Y-%m-%d"
                )
        }

        for i, mine in enumerate(MINES):

            row[mine] = (
                current_cov[
                    d,
                    i,
                    i
                ]
            )

        cov_rows.append(
            row
        )

    cov_df = pd.DataFrame(
        cov_rows
    )

    cov_file = os.path.join(
        OUTPUT_DIR,
        "RTS_full_footprint_daily_covariance_diagonal.csv"
    )

    cov_df.to_csv(
        cov_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # NPZ
    # -------------------------------------------------------------------------

    npz_file = os.path.join(
        OUTPUT_DIR,
        "full_footprint_RTS_daily_states.npz"
    )

    np.savez_compressed(
        npz_file,
        augmented_states=x_smooth,
        augmented_covariances=P_smooth,
        current_states=current_smoothed,
        current_covariances=current_cov,
        dates=np.asarray(
            [
                str(
                    pd.Timestamp(x).date()
                )
                for x in dates
            ]
        ),
        mines=np.asarray(
            MINES
        )
    )

    print(
        comparison_file
    )

    print(
        rts_file
    )

    print(
        cov_file
    )

    print(
        npz_file
    )

    return (
        current_filtered,
        current_smoothed,
        comparison
    )


# =============================================================================
# COMPARISON DIAGNOSTICS
# =============================================================================

def print_comparison(
    dates,
    current_filtered,
    current_smoothed,
    x_map
):

    map_daily = x_map.reshape(
        N_DAY,
        N_MINE
    )

    # -------------------------------------------------------------------------
    # RTS versus MAP
    # -------------------------------------------------------------------------

    rts_diff = (
        current_smoothed
        -
        map_daily
    )

    rts_max = np.max(
        np.abs(
            rts_diff
        )
    )

    rts_rmse = np.sqrt(
        np.mean(
            rts_diff ** 2
        )
    )

    rts_mae = np.mean(
        np.abs(
            rts_diff
        )
    )

    rts_corr = np.corrcoef(
        current_smoothed.ravel(),
        map_daily.ravel()
    )[0, 1]

    print()
    print("=" * 90)
    print("FULL-FOOTPRINT RTS VS BATCH MAP")
    print("=" * 90)

    print(
        "Maximum absolute difference:",
        rts_max
    )

    print(
        "RMSE:",
        rts_rmse
    )

    print(
        "MAE:",
        rts_mae
    )

    print(
        "Correlation:",
        rts_corr
    )

    # -------------------------------------------------------------------------
    # Forward KF versus MAP
    # -------------------------------------------------------------------------

    filter_diff = (
        current_filtered
        -
        map_daily
    )

    filter_max = np.max(
        np.abs(
            filter_diff
        )
    )

    filter_rmse = np.sqrt(
        np.mean(
            filter_diff ** 2
        )
    )

    filter_mae = np.mean(
        np.abs(
            filter_diff
        )
    )

    filter_corr = np.corrcoef(
        current_filtered.ravel(),
        map_daily.ravel()
    )[0, 1]

    print()
    print(
        "FORWARD FULL-FOOTPRINT KF VS BATCH MAP"
    )

    print(
        "Maximum absolute difference:",
        filter_max
    )

    print(
        "RMSE:",
        filter_rmse
    )

    print(
        "MAE:",
        filter_mae
    )

    print(
        "Correlation:",
        filter_corr
    )

    # -------------------------------------------------------------------------
    # Daily means
    # -------------------------------------------------------------------------

    print()
    print(
        "DAILY MEAN SCALING FACTORS"
    )

    print(
        f"{'Date':12s}"
        f"{'KF':>14s}"
        f"{'RTS':>14s}"
        f"{'MAP':>14s}"
    )

    print(
        "-" * 54
    )

    for d, date in enumerate(dates):

        print(
            f"{pd.Timestamp(date).strftime('%Y-%m-%d'):12s}"
            f"{np.mean(current_filtered[d]):14.6f}"
            f"{np.mean(current_smoothed[d]):14.6f}"
            f"{np.mean(map_daily[d]):14.6f}"
        )


# =============================================================================
# FINAL DAY
# =============================================================================

def print_final_day(
    dates,
    current_smoothed,
    x_map
):

    map_daily = x_map.reshape(
        N_DAY,
        N_MINE
    )

    final = N_DAY - 1

    print()
    print("=" * 90)
    print("FINAL-DAY RTS VS BATCH MAP")
    print("=" * 90)

    print(
        "Date:",
        pd.Timestamp(
            dates[final]
        ).strftime(
            "%Y-%m-%d"
        )
    )

    print()

    max_difference = 0.0

    for i, mine in enumerate(MINES):

        difference = (
            current_smoothed[
                final,
                i
            ]
            -
            map_daily[
                final,
                i
            ]
        )

        max_difference = max(
            max_difference,
            abs(difference)
        )

        print(
            f"{mine:8s}"
            f"{current_smoothed[final, i]:14.8f}"
            f"{map_daily[final, i]:14.8f}"
            f"{difference:16.8f}"
        )

    print()

    print(
        "Final-day maximum difference:",
        max_difference
    )


# =============================================================================
# WRITE SUMMARY
# =============================================================================

def write_summary(
    x_map,
    current_smoothed
):

    map_daily = x_map.reshape(
        N_DAY,
        N_MINE
    )

    rts_diff = (
        current_smoothed
        -
        map_daily
    )

    rts_max = np.max(
        np.abs(
            rts_diff
        )
    )

    rts_rmse = np.sqrt(
        np.mean(
            rts_diff ** 2
        )
    )

    rts_mae = np.mean(
        np.abs(
            rts_diff
        )
    )

    rts_correlation = np.corrcoef(
        current_smoothed.ravel(),
        map_daily.ravel()
    )[0, 1]

    summary_file = os.path.join(
        OUTPUT_DIR,
        "full_footprint_RTS_summary.txt"
    )

    with open(
        summary_file,
        "w"
    ) as f:

        f.write(
            "FULL-FLEXPART-FOOTPRINT "
            "AUGMENTED RTS SMOOTHER\n"
        )

        f.write(
            "=" * 80
            +
            "\n\n"
        )

        f.write(
            f"Observations: "
            f"{len(current_smoothed)} days\n"
        )

        f.write(
            f"Parameters: "
            f"{N_PARAM}\n"
        )

        f.write(
            f"Augmented state dimension: "
            f"{N_AUG}\n"
        )

        f.write(
            f"rho: "
            f"{rho:.15f}\n"
        )

        f.write(
            f"Batch MAP mean: "
            f"{np.mean(x_map):.15f}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "RTS VS BATCH MAP\n"
        )

        f.write(
            f"Maximum absolute difference: "
            f"{rts_max:.15e}\n"
        )

        f.write(
            f"RMSE: "
            f"{rts_rmse:.15e}\n"
        )

        f.write(
            f"MAE: "
            f"{rts_mae:.15e}\n"
        )

        f.write(
            f"Correlation: "
            f"{rts_correlation:.15f}\n"
        )

    print()
    print(
        "Summary file:"
    )

    print(
        summary_file
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 90
    )

    print(
        "FULL-FLEXPART-FOOTPRINT "
        "AUGMENTED RTS SMOOTHER"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "KF NPZ:"
    )

    print(
        KF_NPZ
    )

    print()
    print(
        "H file:"
    )

    print(
        H_FILE
    )

    print()
    print(
        "Picarro file:"
    )

    print(
        PICARRO_FILE
    )

    print()
    print(
        "Inversion period:"
    )

    print(
        START_DATE.strftime("%Y-%m-%d"),
        "to",
        END_DATE.strftime("%Y-%m-%d")
    )

    print()
    print(
        "Number of days:",
        N_DAY
    )

    print(
        "Number of mines:",
        N_MINE
    )

    print(
        "Full parameters:",
        N_PARAM
    )

    print(
        "Augmented state dimension:",
        N_AUG
    )

    print()
    print(
        "Prior:"
    )

    print(
        "mean =",
        PRIOR_MEAN
    )

    print(
        "sigma =",
        PRIOR_SIGMA
    )

    print(
        "correlation length =",
        L_CORR
    )

    print()
    print(
        "rho =",
        rho
    )

    print(
        "Q diagonal =",
        Q[0, 0]
    )

    # =========================================================================
    # READ KF
    # =========================================================================

    print()
    print("=" * 90)
    print("READING FULL-FOOTPRINT KF STATES")
    print("=" * 90)

    if not os.path.exists(
        KF_NPZ
    ):

        raise FileNotFoundError(
            f"KF NPZ not found:\n{KF_NPZ}"
        )

    data = np.load(
        KF_NPZ,
        allow_pickle=True
    )

    print(
        "NPZ keys:",
        data.files
    )

    x_analysis = np.asarray(
        data["states"],
        dtype=np.float64
    )

    P_analysis = np.asarray(
        data["covariances"],
        dtype=np.float64
    )

    dates = pd.to_datetime(
        data["dates"]
    )

    print(
        "State shape:",
        x_analysis.shape
    )

    print(
        "Covariance shape:",
        P_analysis.shape
    )

    print(
        "Dates:",
        dates[0],
        "to",
        dates[-1]
    )

    # =========================================================================
    # VALIDATE KF DIMENSIONS
    # =========================================================================

    if x_analysis.shape != (
        N_DAY,
        N_AUG
    ):

        raise ValueError(
            "Unexpected augmented state shape: "
            +
            str(x_analysis.shape)
        )

    if P_analysis.shape != (
        N_DAY,
        N_AUG,
        N_AUG
    ):

        raise ValueError(
            "Unexpected covariance shape: "
            +
            str(P_analysis.shape)
        )

    if len(dates) != N_DAY:

        raise ValueError(
            "Unexpected number of dates: "
            +
            str(len(dates))
        )

    # =========================================================================
    # OBSERVATIONS
    # =========================================================================

    obs = read_observations()

    (
        matched,
        H,
        y,
        parameter_cols
    ) = read_and_match_H(
        obs
    )

    if H.shape[1] != N_PARAM:

        raise ValueError(
            "Unexpected H parameter dimension: "
            +
            str(H.shape[1])
        )

    # =========================================================================
    # BUILD PRIOR
    # =========================================================================

    print()
    print("=" * 90)
    print("BUILDING FULL PRIOR COVARIANCE")
    print("=" * 90)

    B = build_prior_covariance()

    print(
        "B shape:",
        B.shape
    )

    print(
        "B[OC1 Jan5, OC1 Jan5] =",
        B[0, 0]
    )

    print(
        "B[OC1 Jan5, OC1 Jan6] =",
        B[0, 13]
    )

    print(
        "B[OC1 Jan5, OC2 Jan5] =",
        B[0, 1]
    )

    LB = np.linalg.cholesky(
        B
    )

    B_reconstructed = (
        LB
        @
        LB.T
    )

    print(
        "Cholesky reconstruction error:",
        np.max(
            np.abs(
                B
                -
                B_reconstructed
            )
        )
    )

    # =========================================================================
    # BATCH MAP
    # =========================================================================

    x_map = compute_batch_map(
        H,
        y,
        B
    )

    # =========================================================================
    # RTS
    # =========================================================================

    (
        x_smooth,
        P_smooth
    ) = run_augmented_rts(
        x_analysis,
        P_analysis
    )

    # =========================================================================
    # EXTRACT CURRENT-DAY STATES
    # =========================================================================

    current_filtered = (
        x_analysis[
            :,
            N_MINE:
        ]
    )

    current_smoothed = (
        x_smooth[
            :,
            N_MINE:
        ]
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    (
        current_filtered,
        current_smoothed,
        comparison
    ) = save_results(
        dates,
        x_analysis,
        P_analysis,
        x_smooth,
        P_smooth,
        x_map
    )

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    print_comparison(
        dates,
        current_filtered,
        current_smoothed,
        x_map
    )

    print_final_day(
        dates,
        current_smoothed,
        x_map
    )

    write_summary(
        x_map,
        current_smoothed
    )

    # =========================================================================
    # DONE
    # =========================================================================

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":

    main()

