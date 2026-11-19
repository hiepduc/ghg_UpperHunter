#!/usr/bin/env python3

"""
FLEXPART-WRF Upper Hunter CH4
Sequential Kalman Filter
--------------------------------------------

State:
    13 daily open-cut coal-mine scaling factors

Observations:
    Hourly Picarro CH4 observations

Assimilation:
    Sequential scalar Kalman updates for each available hourly observation.

The state is constant within each day. All hourly observations during
a day sequentially update the same 13-dimensional daily state.

At the end of each day:
    x_daily = final hourly analysis state

Between days:
    x_(d+1)^f = rho*x_d^a + (1-rho)*x_prior

with:
    rho = exp(-1 / correlation_length_days)

Process noise is chosen so that the stationary variance remains
sigma_prior^2.

Outputs:
    hourly_analysis_scaling_factors.csv
    daily_KF_scaling_factors.csv
    hourly_emission_rates_gCH4_s.csv
    hourly_predictions.csv
    hourly_covariance_diagonal.csv
    KF_metrics.csv
    KF_summary.txt
    KF_hourly_states.npz
    KF_daily_states.npz

Author:
    Upper Hunter FLEXPART-WRF inverse modelling workflow
"""

import os
import re
import sys
import numpy as np
import pandas as pd


# ============================================================
# USER CONFIGURATION
# ============================================================

BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart"

H_FILE = (
    BASE +
    "/H_matrix_flexpart/"
    "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

PRIOR_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/UpperHuntercoalmines.csv"
)

OUTPUT_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "KF_13mine_daily_Jan2026"
)

# Analysis period
START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

# Number of mines
N_MINE = 13

# Prior
PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5

# Observation
BACKGROUND_PPM = 1.85
OBS_SIGMA = 0.05
OBS_VARIANCE = OBS_SIGMA ** 2

# Temporal correlation
CORRELATION_LENGTH_DAYS = 3.0

# Reproducibility / numerical
EPS = 1.0e-12


# ============================================================
# MINE NAMES
# ============================================================

MINES = [f"OC{i}" for i in range(1, N_MINE + 1)]

# Expected H parameter names:
# OC1_20260105 ... OC13_20260105
# OC1_20260106 ... etc.


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_picarro_datetime(obs):
    """
    Parse the Picarro Datetime field exactly as used in the
    previous Upper Hunter inversion workflow.
    """

    obs = obs.copy()

    if "Datetime" not in obs.columns:
        raise KeyError(
            "Picarro file does not contain a 'Datetime' column."
        )

    obs["time_aest"] = pd.to_datetime(
        obs["Datetime"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    return obs


def build_observation_table():
    """
    Convert wide Picarro data to long receptor/hour table.

    Returns columns:
        receptor
        time_aest
        obs_ch4_ppm
    """

    print_header("READING PICARRO OBSERVATIONS")

    obs = pd.read_csv(PICARRO_FILE)

    print("Picarro shape:", obs.shape)
    print("Columns:")
    print(list(obs.columns))

    obs = parse_picarro_datetime(obs)

    receptor_columns = {
        "singleton": "Singleton cleaned",
        "muswellbrook": "Muswellbrook cleaned",
        "merriwa": "Merriwa cleaned",
    }

    records = []

    for receptor, column in receptor_columns.items():

        if column not in obs.columns:
            print(
                f"WARNING: {column} not found. "
                f"No observations for {receptor}."
            )
            continue

        tmp = obs[["time_aest", column]].copy()

        tmp["obs_ch4_ppm"] = pd.to_numeric(
            tmp[column],
            errors="coerce"
        )

        tmp["receptor"] = receptor

        tmp = tmp[
            ["receptor", "time_aest", "obs_ch4_ppm"]
        ]

        tmp = tmp.dropna(
            subset=["time_aest", "obs_ch4_ppm"]
        )

        # January analysis period
        tmp = tmp[
            (tmp["time_aest"] >= START_DATE) &
            (tmp["time_aest"] < END_DATE + pd.Timedelta(days=1))
        ]

        # Match hourly FLEXPART observation times
        tmp["time_aest"] = tmp["time_aest"].dt.floor("h")

        records.append(tmp)

    if not records:
        raise RuntimeError(
            "No usable Picarro receptor data found."
        )

    obs_long = pd.concat(
        records,
        ignore_index=True
    )

    # If duplicate observations occur in the same receptor/hour,
    # average them.
    obs_long = (
        obs_long
        .groupby(
            ["receptor", "time_aest"],
            as_index=False
        )["obs_ch4_ppm"]
        .mean()
    )

    obs_long = obs_long.sort_values(
        ["time_aest", "receptor"]
    ).reset_index(drop=True)

    print()
    print("Usable observations:", len(obs_long))

    print()
    print(
        obs_long
        .groupby("receptor")
        .size()
        .to_string()
    )

    print()
    print(
        "Observed CH4 range:",
        f"{obs_long.obs_ch4_ppm.min():.6f}",
        "to",
        f"{obs_long.obs_ch4_ppm.max():.6f}",
        "ppm"
    )

    return obs_long


def load_h_matrix():
    """
    Load H matrix and retain the original row metadata.

    Returns:
        H_all
        metadata
        parameter_names
    """

    print_header("READING FLEXPART H MATRIX")

    H_df = pd.read_csv(H_FILE)

    print("H shape:", H_df.shape)

    metadata_columns = [
        "receptor",
        "release_utc",
        "obs_time_aest",
        "flexpart_file",
    ]

    for col in metadata_columns:
        if col not in H_df.columns:
            raise KeyError(
                f"Required H metadata column missing: {col}"
            )

    parameter_names = [
        c for c in H_df.columns
        if c not in metadata_columns
    ]

    if len(parameter_names) != 13 * 27:
        raise ValueError(
            f"Expected 351 H parameters, found "
            f"{len(parameter_names)}."
        )

    H_all = H_df[parameter_names].to_numpy(
        dtype=np.float64
    )

    metadata = H_df[metadata_columns].copy()

    metadata["obs_time_aest"] = pd.to_datetime(
        metadata["obs_time_aest"],
        errors="coerce"
    )

    metadata["receptor"] = (
        metadata["receptor"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("Number of parameters:", len(parameter_names))
    print("H matrix:", H_all.shape)

    print()
    print("H range:")
    print("  min =", np.nanmin(H_all))
    print("  max =", np.nanmax(H_all))
    print("  mean =", np.nanmean(H_all))

    return H_all, metadata, parameter_names


def align_h_to_observations(H_all, metadata, obs_long):
    """
    Match H rows to Picarro observations.

    Returns:
        H
        y
        aligned_metadata
    """

    print_header("ALIGNING H MATRIX WITH PICARRO")

    H_key = pd.DataFrame({
        "receptor": metadata["receptor"].values,
        "time_aest": metadata["obs_time_aest"]
            .dt.floor("h")
            .values,
    })

    H_key["H_row"] = np.arange(len(H_key))

    obs_key = obs_long[
        ["receptor", "time_aest", "obs_ch4_ppm"]
    ].copy()

    match = H_key.merge(
        obs_key,
        on=["receptor", "time_aest"],
        how="inner"
    )

    if len(match) == 0:
        raise RuntimeError(
            "No H rows matched Picarro observations."
        )

    # Protect against duplicate H rows for the same
    # receptor/hour.
    if match["H_row"].duplicated().any():
        print(
            "WARNING: duplicate H row matches detected."
        )

    H_row_indices = match[
        "H_row"
    ].to_numpy(dtype=int)

    H = H_all[H_row_indices, :]

    y = match[
        "obs_ch4_ppm"
    ].to_numpy(dtype=np.float64)

    aligned_metadata = metadata.iloc[
        H_row_indices
    ].copy().reset_index(drop=True)

    aligned_metadata["time_aest"] = (
        match["time_aest"]
        .values
    )

    aligned_metadata["obs_ch4_ppm"] = y

    print("Original H rows:", len(H_all))
    print("Matched rows:", len(H))
    print("H final shape:", H.shape)

    print()
    print("Matched observations by receptor:")
    print(
        aligned_metadata
        .groupby("receptor")
        .size()
        .to_string()
    )

    print()
    print(
        "Matched observation range:",
        f"{y.min():.6f}",
        "to",
        f"{y.max():.6f}",
        "ppm"
    )

    return H, y, aligned_metadata


def load_prior_emissions():
    """
    Read prior mine emissions in g CH4/s.

    Expected:
        Facility_code_name
        gCH4/s
    """

    print_header("READING PRIOR MINE EMISSIONS")

    prior = pd.read_csv(PRIOR_FILE)

    print("Prior columns:")
    print(list(prior.columns))

    if "Facility_code_name" not in prior.columns:
        raise KeyError(
            "Expected 'Facility_code_name' in prior CSV."
        )

    if "gCH4/s" not in prior.columns:
        raise KeyError(
            "Expected 'gCH4/s' in prior CSV."
        )

    emissions = {}

    for mine in MINES:

        rows = prior[
            prior["Facility_code_name"]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(mine)
        ]

        if len(rows) == 0:
            raise ValueError(
                f"No prior emission found for {mine}"
            )

        value = float(
            rows.iloc[0]["gCH4/s"]
        )

        emissions[mine] = value

    prior_vector = np.array(
        [emissions[m] for m in MINES],
        dtype=np.float64
    )

    print()
    print("Prior emissions:")
    for mine, value in emissions.items():
        print(
            f"  {mine:>4s}: {value:12.6f} g CH4/s"
        )

    print()
    print(
        "Total prior:",
        f"{prior_vector.sum():.6f}",
        "g CH4/s"
    )

    return emissions, prior_vector


def get_day_parameter_columns(parameter_names, day):
    """
    Return indices corresponding to:
        OC1_day ... OC13_day
    """

    date_string = day.strftime("%Y%m%d")

    expected = [
        f"{mine}_{date_string}"
        for mine in MINES
    ]

    index_map = {
        name: i
        for i, name in enumerate(parameter_names)
    }

    missing = [
        name
        for name in expected
        if name not in index_map
    ]

    if missing:
        raise KeyError(
            "Missing H parameter columns:\n"
            + "\n".join(missing)
        )

    return np.array(
        [index_map[name] for name in expected],
        dtype=int
    )


# ============================================================
# KALMAN FILTER
# ============================================================

def run_kalman_filter(
    H,
    y,
    aligned_metadata,
    parameter_names,
    prior_emissions,
):
    """
    Run sequential Kalman filter.

    State dimension:
        13

    Each day:
        forecast once at the beginning

        then:
            observation 1 -> update
            observation 2 -> update
            ...

        final state = daily estimate
    """

    print_header("RUNNING SEQUENTIAL KALMAN FILTER")

    # --------------------------------------------------------
    # Temporal model
    # --------------------------------------------------------

    rho = np.exp(
        -1.0 / CORRELATION_LENGTH_DAYS
    )

    prior_variance = PRIOR_SIGMA ** 2

    # Stationary AR(1) process noise:
    #
    # P = rho^2 P + Q
    #
    # therefore:
    #
    # Q = P * (1-rho^2)
    #
    Q = (
        prior_variance
        * (1.0 - rho ** 2)
        * np.eye(N_MINE)
    )

    print(f"rho = {rho:.10f}")
    print(f"prior variance = {prior_variance:.6f}")
    print(
        "process noise variance =",
        f"{Q[0,0]:.10f}"
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    x = np.full(
        N_MINE,
        PRIOR_MEAN,
        dtype=np.float64
    )

    P = (
        prior_variance
        * np.eye(N_MINE)
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    hourly_state_records = []
    hourly_prediction_records = []
    hourly_cov_records = []

    daily_state_records = []
    daily_cov_records = []

    # Keep arrays for NPZ
    hourly_states = []
    hourly_covariances = []

    daily_states = []
    daily_covariances = []

    current_day = None

    total_updates = 0

    # --------------------------------------------------------
    # Sort observations chronologically
    # --------------------------------------------------------

    order = np.argsort(
        aligned_metadata["time_aest"]
        .values.astype("datetime64[ns]")
    )

    H_sorted = H[order]
    y_sorted = y[order]
    metadata_sorted = aligned_metadata.iloc[
        order
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Process each observation
    # --------------------------------------------------------

    for row_index in range(len(y_sorted)):

        timestamp = pd.Timestamp(
            metadata_sorted.loc[
                row_index,
                "time_aest"
            ]
        )

        day = timestamp.normalize()

        # ----------------------------------------------------
        # New day
        # ----------------------------------------------------

        if current_day is None:

            current_day = day

            print()
            print(
                f"Starting {day.date()}"
            )

        elif day != current_day:

            # ------------------------------------------------
            # Save final analysed state of previous day
            # ------------------------------------------------

            daily_state_records.append({
                "date": current_day,
                "state_type": "analysis",
                "n_observations": int(np.sum(
                    metadata_sorted["time_aest"].dt.normalize() == current_day
                )),
                **{
                    mine: x[i]
                    for i, mine in enumerate(MINES)
                }
            })

            daily_cov_records.append({
                "date": current_day,
                **{
                    f"{mine}_sigma":
                    np.sqrt(
                        max(P[i, i], 0.0)
                    )
                    for i, mine in enumerate(MINES)
                }
            })

            daily_states.append(x.copy())
            daily_covariances.append(P.copy())

            print(
                f"  final {current_day.date()} state "
                f"mean = {x.mean():.6f}"
            )

            # ------------------------------------------------
            # Forecast through EVERY missing calendar day
            # ------------------------------------------------
            #
            # If observations jump:
            #
            # Jan 8 -> Jan 13
            #
            # we must perform:
            #
            # Jan 8 -> Jan 9
            # Jan 9 -> Jan 10
            # Jan 10 -> Jan 11
            # Jan 11 -> Jan 12
            # Jan 12 -> Jan 13
            #
            # rather than applying rho only once.
            # ------------------------------------------------

            next_day = current_day + pd.Timedelta(days=1)

            while next_day <= day:

                x = (
                    rho * x
                    + (1.0 - rho) * PRIOR_MEAN
                )

                P = (
                    rho ** 2 * P
                    + Q
                )

                P = 0.5 * (P + P.T)

                # ------------------------------------------------
                # Save forecast-only days when there are no
                # observations.
                # ------------------------------------------------

                if next_day < day:

                    daily_state_records.append({
                        "date": next_day,
                        "state_type": "forecast",
                        "n_observations": 0,
                        **{
                            mine: x[i]
                            for i, mine in enumerate(MINES)
                        }
                    })

                    daily_cov_records.append({
                        "date": next_day,
                        **{
                            f"{mine}_sigma":
                            np.sqrt(
                                max(P[i, i], 0.0)
                            )
                            for i, mine in enumerate(MINES)
                        }
                    })

                    daily_states.append(
                        x.copy()
                    )

                    daily_covariances.append(
                        P.copy()
                    )

                    print(
                        f"  forecast-only "
                        f"{next_day.date()} "
                        f"mean = {x.mean():.6f}"
                    )

                next_day = (
                    next_day
                    + pd.Timedelta(days=1)
                )

            current_day = day

            print()
            print(
                f"Starting {day.date()} "
                f"(forecasted state mean "
                f"{x.mean():.6f})"
            )


        # ----------------------------------------------------
        # Determine the 13 H columns for this day
        # ----------------------------------------------------

        day_indices = get_day_parameter_columns(
            parameter_names,
            day
        )

        h = H_sorted[
            row_index,
            day_indices
        ].astype(np.float64)

        observation = y_sorted[row_index]

        # ----------------------------------------------------
        # Forecast prediction before this observation
        # ----------------------------------------------------

        model_before = (
            BACKGROUND_PPM
            + np.dot(h, x)
        )

        innovation = (
            observation
            - model_before
        )

        # ----------------------------------------------------
        # Scalar Kalman update
        # ----------------------------------------------------

        # S = H P H' + R
        HP = np.dot(h, P)

        innovation_variance = (
            np.dot(HP, h)
            + OBS_VARIANCE
        )

        if innovation_variance <= EPS:
            raise RuntimeError(
                "Non-positive innovation variance."
            )

        # K = P H' / S
        K = np.dot(
            P,
            h
        ) / innovation_variance

        # State update
        x = (
            x
            + K * innovation
        )

        # ----------------------------------------------------
        # Joseph covariance update
        # ----------------------------------------------------

        I_KH = (
            np.eye(N_MINE)
            - np.outer(K, h)
        )

        P = (
            I_KH @ P @ I_KH.T
            + np.outer(K, K) * OBS_VARIANCE
        )

        # Numerical cleanup
        P = 0.5 * (P + P.T)

        # Protect against tiny negative diagonal values
        diag = np.maximum(
            np.diag(P),
            0.0
        )

        for i in range(N_MINE):
            P[i, i] = diag[i]

        # ----------------------------------------------------
        # Prediction after update
        # ----------------------------------------------------

        model_after = (
            BACKGROUND_PPM
            + np.dot(h, x)
        )

        residual_after = (
            observation
            - model_after
        )

        # ----------------------------------------------------
        # Save hourly state
        # ----------------------------------------------------

        state_record = {
            "time_aest": timestamp,
            "date": day,
            "receptor":
                metadata_sorted.loc[
                    row_index,
                    "receptor"
                ],
            "obs_ch4_ppm": observation,
            "model_before_ppm": model_before,
            "innovation_ppm": innovation,
            "innovation_sigma_ppm":
                np.sqrt(innovation_variance),
            "model_after_ppm": model_after,
            "residual_after_ppm":
                residual_after,
        }

        for i, mine in enumerate(MINES):
            state_record[mine] = x[i]

        hourly_state_records.append(
            state_record
        )

        # ----------------------------------------------------
        # Save covariance diagonal
        # ----------------------------------------------------

        cov_record = {
            "time_aest": timestamp,
            "date": day,
            "receptor":
                metadata_sorted.loc[
                    row_index,
                    "receptor"
                ],
        }

        for i, mine in enumerate(MINES):
            cov_record[
                f"{mine}_variance"
            ] = P[i, i]

            cov_record[
                f"{mine}_sigma"
            ] = np.sqrt(
                max(P[i, i], 0.0)
            )

        hourly_cov_records.append(
            cov_record
        )

        # ----------------------------------------------------
        # Save hourly predictions
        # ----------------------------------------------------

        prediction_record = {
            "time_aest": timestamp,
            "date": day,
            "receptor":
                metadata_sorted.loc[
                    row_index,
                    "receptor"
                ],
            "obs_ch4_ppm": observation,
            "prior_scaling_model_ppm":
                BACKGROUND_PPM
                + np.dot(
                    h,
                    np.ones(N_MINE)
                ),
            "KF_before_update_ppm":
                model_before,
            "innovation_ppm":
                innovation,
            "KF_after_update_ppm":
                model_after,
            "residual_after_ppm":
                residual_after,
        }

        hourly_prediction_records.append(
            prediction_record
        )

        hourly_states.append(x.copy())
        hourly_covariances.append(P.copy())

        total_updates += 1

    # ========================================================
    # Save final day
    # ========================================================

    if current_day is not None:

        daily_state_records.append({
            "date": current_day,
            "state_type": "analysis",
            "n_observations": int(
                np.sum(
                    metadata_sorted["time_aest"].dt.normalize()
                    == current_day
                )
            ),
            **{
                mine: x[i]
                for i, mine in enumerate(MINES)
            }
        })

        daily_cov_records.append({
            "date": current_day,
            **{
                f"{mine}_sigma":
                np.sqrt(
                    max(P[i, i], 0.0)
                )
                for i, mine in enumerate(MINES)
            }
        })

        daily_states.append(x.copy())
        daily_covariances.append(P.copy())

        print()
        print(
            f"final {current_day.date()} state "
            f"mean = {x.mean():.6f}"
        )

    print()
    print("Total hourly assimilation updates:", total_updates)

    # ========================================================
    # Convert to DataFrames
    # ========================================================

    hourly_states_df = pd.DataFrame(
        hourly_state_records
    )

    hourly_cov_df = pd.DataFrame(
        hourly_cov_records
    )

    hourly_predictions_df = pd.DataFrame(
        hourly_prediction_records
    )

    daily_states_df = pd.DataFrame(
        daily_state_records
    )

    daily_cov_df = pd.DataFrame(
        daily_cov_records
    )

    # ========================================================
    # Save hourly emission rates
    # ========================================================

    emission_records = []

    for _, row in hourly_states_df.iterrows():

        timestamp = row["time_aest"]
        day = pd.Timestamp(
            row["date"]
        )

        for i, mine in enumerate(MINES):

            scaling = float(
                row[mine]
            )

            prior_rate = float(
                prior_emissions[i]
            )

            emission_records.append({
                "time_aest": timestamp,
                "date": day,
                "mine": mine,
                "scaling_factor": scaling,
                "prior_emission_gCH4_s":
                    prior_rate,
                "KF_emission_gCH4_s":
                    scaling * prior_rate,
            })

    hourly_emissions_df = pd.DataFrame(
        emission_records
    )

    # ========================================================
    # Metrics
    # ========================================================

    metrics = calculate_metrics(
        hourly_predictions_df
    )

    # ========================================================
    # Save CSV files
    # ========================================================

    hourly_states_file = (
        OUTPUT_DIR +
        "/hourly_analysis_scaling_factors.csv"
    )

    daily_states_file = (
        OUTPUT_DIR +
        "/daily_KF_scaling_factors.csv"
    )

    emissions_file = (
        OUTPUT_DIR +
        "/hourly_emission_rates_gCH4_s.csv"
    )

    predictions_file = (
        OUTPUT_DIR +
        "/hourly_predictions.csv"
    )

    covariance_file = (
        OUTPUT_DIR +
        "/hourly_covariance_diagonal.csv"
    )

    daily_covariance_file = (
        OUTPUT_DIR +
        "/daily_covariance_diagonal.csv"
    )

    metrics_file = (
        OUTPUT_DIR +
        "/KF_metrics.csv"
    )

    hourly_states_df.to_csv(
        hourly_states_file,
        index=False
    )

    daily_states_df.to_csv(
        daily_states_file,
        index=False
    )

    hourly_emissions_df.to_csv(
        emissions_file,
        index=False
    )

    hourly_predictions_df.to_csv(
        predictions_file,
        index=False
    )

    hourly_cov_df.to_csv(
        covariance_file,
        index=False
    )

    daily_cov_df.to_csv(
        daily_covariance_file,
        index=False
    )

    metrics.to_csv(
        metrics_file,
        index=False
    )

    # ========================================================
    # Save NPZ arrays
    # ========================================================

    np.savez_compressed(
        OUTPUT_DIR +
        "/KF_hourly_states.npz",
        states=np.asarray(
            hourly_states
        ),
        covariances=np.asarray(
            hourly_covariances
        ),
        times=
            hourly_states_df[
                "time_aest"
            ].astype(str).to_numpy(),
        mines=np.asarray(MINES),
    )

    np.savez_compressed(
        OUTPUT_DIR +
        "/KF_daily_states.npz",
        states=np.asarray(
            daily_states
        ),
        covariances=np.asarray(
            daily_covariances
        ),
        dates=
            daily_states_df[
                "date"
            ].astype(str).to_numpy(),
        mines=np.asarray(MINES),
    )

    # ========================================================
    # Summary
    # ========================================================

    write_summary(
        metrics=metrics,
        hourly_states_df=hourly_states_df,
        daily_states_df=daily_states_df,
        rho=rho,
        Q=Q,
    )

    return (
        hourly_states_df,
        daily_states_df,
        hourly_emissions_df,
        hourly_predictions_df,
        hourly_cov_df,
        daily_cov_df,
        metrics,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predictions):

    records = []

    def add_metric(name, y, pred):

        y = np.asarray(y, dtype=float)
        pred = np.asarray(pred, dtype=float)

        valid = (
            np.isfinite(y)
            & np.isfinite(pred)
        )

        y = y[valid]
        pred = pred[valid]

        if len(y) == 0:
            return

        residual = pred - y

        rmse = np.sqrt(
            np.mean(residual ** 2)
        )

        mae = np.mean(
            np.abs(residual)
        )

        bias = np.mean(
            residual
        )

        if len(y) > 1:
            r = np.corrcoef(
                y,
                pred
            )[0, 1]
        else:
            r = np.nan

        if np.std(y) > 0:
            r2 = r ** 2
        else:
            r2 = np.nan

        ratio = pred / y

        fac2 = np.mean(
            (ratio >= 0.5)
            & (ratio <= 2.0)
        )

        records.append({
            "case": name,
            "N": len(y),
            "R": r,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Bias": bias,
            "FAC2": fac2,
        })

    # --------------------------------------------------------
    # All observations
    # --------------------------------------------------------

    add_metric(
        "ALL_prior",
        predictions["obs_ch4_ppm"],
        predictions["prior_scaling_model_ppm"],
    )

    add_metric(
        "ALL_KF_before_update",
        predictions["obs_ch4_ppm"],
        predictions["KF_before_update_ppm"],
    )

    add_metric(
        "ALL_KF_after_update",
        predictions["obs_ch4_ppm"],
        predictions["KF_after_update_ppm"],
    )

    # --------------------------------------------------------
    # Receptors
    # --------------------------------------------------------

    for receptor in sorted(
        predictions["receptor"].unique()
    ):

        subset = predictions[
            predictions["receptor"] == receptor
        ]

        add_metric(
            receptor + "_prior",
            subset["obs_ch4_ppm"],
            subset["prior_scaling_model_ppm"],
        )

        add_metric(
            receptor + "_KF_after_update",
            subset["obs_ch4_ppm"],
            subset["KF_after_update_ppm"],
        )

    return pd.DataFrame(records)


# ============================================================
# SUMMARY FILE
# ============================================================

def write_summary(
    metrics,
    hourly_states_df,
    daily_states_df,
    rho,
    Q,
):

    filename = (
        OUTPUT_DIR +
        "/KF_summary.txt"
    )

    with open(filename, "w") as f:

        f.write(
            "UPPER HUNTER FLEXPART-WRF "
            "SEQUENTIAL KALMAN FILTER\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "State dimension: 13 daily mine scaling factors\n"
        )

        f.write(
            "Observation frequency: hourly\n"
        )

        f.write(
            "Analysis period: "
            f"{START_DATE.date()} to "
            f"{END_DATE.date()}\n\n"
        )

        f.write(
            f"Prior mean: {PRIOR_MEAN}\n"
        )

        f.write(
            f"Prior sigma: {PRIOR_SIGMA}\n"
        )

        f.write(
            f"Observation sigma: {OBS_SIGMA} ppm\n"
        )

        f.write(
            f"Background: {BACKGROUND_PPM} ppm\n"
        )

        f.write(
            f"Correlation length: "
            f"{CORRELATION_LENGTH_DAYS} days\n"
        )

        f.write(
            f"rho: {rho:.10f}\n"
        )

        f.write(
            f"Process variance: {Q[0,0]:.10f}\n\n"
        )

        f.write(
            f"Hourly assimilation updates: "
            f"{len(hourly_states_df)}\n"
        )

        f.write(
            f"Daily estimates: "
            f"{len(daily_states_df)}\n\n"
        )

        f.write(
            "METRICS\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            metrics.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "DAILY FINAL STATE\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            daily_states_df.to_string(
                index=False
            )
        )

        f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print(
        "FLEXPART-WRF UPPER HUNTER "
        "SEQUENTIAL KALMAN FILTER"
    )
    print("=" * 90)

    print()
    print("H file:")
    print(H_FILE)

    print()
    print("Picarro file:")
    print(PICARRO_FILE)

    print()
    print("Prior emission file:")
    print(PRIOR_FILE)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    ensure_output_directory()

    # --------------------------------------------------------
    # Load H
    # --------------------------------------------------------

    (
        H_all,
        metadata,
        parameter_names,
    ) = load_h_matrix()

    # --------------------------------------------------------
    # Picarro
    # --------------------------------------------------------

    obs_long = build_observation_table()

    # --------------------------------------------------------
    # Align
    # --------------------------------------------------------

    (
        H,
        y,
        aligned_metadata,
    ) = align_h_to_observations(
        H_all,
        metadata,
        obs_long,
    )

    # --------------------------------------------------------
    # Prior emissions
    # --------------------------------------------------------

    (
        prior_emissions_dict,
        prior_emissions,
    ) = load_prior_emissions()

    # --------------------------------------------------------
    # Run KF
    # --------------------------------------------------------

    results = run_kalman_filter(
        H=H,
        y=y,
        aligned_metadata=aligned_metadata,
        parameter_names=parameter_names,
        prior_emissions=prior_emissions,
    )

    (
        hourly_states_df,
        daily_states_df,
        hourly_emissions_df,
        hourly_predictions_df,
        hourly_cov_df,
        daily_cov_df,
        metrics,
    ) = results

    # --------------------------------------------------------
    # Final screen output
    # --------------------------------------------------------

    print_header("KALMAN FILTER FINISHED")

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("Files created:")

    files = [
        "hourly_analysis_scaling_factors.csv",
        "daily_KF_scaling_factors.csv",
        "hourly_emission_rates_gCH4_s.csv",
        "hourly_predictions.csv",
        "hourly_covariance_diagonal.csv",
        "daily_covariance_diagonal.csv",
        "KF_metrics.csv",
        "KF_summary.txt",
        "KF_hourly_states.npz",
        "KF_daily_states.npz",
    ]

    for filename in files:
        print(
            "  ",
            os.path.join(
                OUTPUT_DIR,
                filename
            )
        )

    print()
    print("Metrics:")
    print(metrics.to_string(index=False))

    print()
    print("Daily scaling factors:")
    print(
        daily_states_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()

