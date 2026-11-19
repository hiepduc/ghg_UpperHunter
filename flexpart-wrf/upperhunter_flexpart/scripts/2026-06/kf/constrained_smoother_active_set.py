#!/usr/bin/env python3

"""
Constrained global smoother for Upper Hunter CH4 inversion.

13 mines x 27 days = 351 daily scaling factors.

Objective:

    J(x) =
        0.5 (x-xb)^T B^-1 (x-xb)
      + 0.5 (y-background-Hx)^T R^-1 (y-background-Hx)

subject to:

    0 <= x_i <= 5

This is solved using a primal active-set quadratic-programming
algorithm.

The result is compared against the independently computed
bounded batch MAP solution.

Important:
    x = xb + z

The whitened system is

    min ||A z - b||^2

with

    -1 <= z_i <= 4

because xb = 1.

The active-set solver operates in z-space.
"""

import os
import numpy as np
import pandas as pd

from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator, cg


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart"

H_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "H_flexpart_0_10m_13mine_Jan2026.csv",
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

BOUNDED_MAP_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "Bounded_batch_MAP_full_footprint_13mine_Jan2026",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "Constrained_smoother_active_set_13mine_Jan2026",
)

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

BACKGROUND_PPM = 1.85
OBS_SIGMA = 0.05

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5

CORR_LENGTH_DAYS = 3.0

LOWER_X = 0.0
UPPER_X = 5.0

LOWER_Z = LOWER_X - PRIOR_MEAN
UPPER_Z = UPPER_X - PRIOR_MEAN

N_MINE = 13


# ============================================================
# UTILITIES
# ============================================================

def rho_from_length(length_days):
    return np.exp(-1.0 / length_days)


def build_prior_covariance(n_mines, n_days, sigma, corr_length):
    rho = rho_from_length(corr_length)

    n = n_mines * n_days
    B = np.zeros((n, n))

    for mine in range(n_mines):
        for d1 in range(n_days):
            i = d1 * n_mines + mine

            for d2 in range(n_days):
                j = d2 * n_mines + mine

                B[i, j] = sigma**2 * rho**abs(d1 - d2)

    return B, rho


def parameter_columns(df):
    """
    Return OC1_YYYYMMDD ... OC13_YYYYMMDD columns in the
    required day-major / mine-minor ordering.
    """

    cols = []

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    for date in dates:
        datestr = date.strftime("%Y%m%d")

        for mine in range(1, N_MINE + 1):
            col = f"OC{mine}_{datestr}"

            if col not in df.columns:
                raise RuntimeError(
                    f"Missing H parameter column: {col}"
                )

            cols.append(col)

    return cols


def load_problem():

    print("=" * 90)
    print("LOADING CONSTRAINED SMOOTHER PROBLEM")
    print("=" * 90)

    # --------------------------------------------------------
    # Load H matrix
    # --------------------------------------------------------

    Hdf = pd.read_csv(H_FILE)

    print("H dataframe:", Hdf.shape)

    pcols = parameter_columns(Hdf)

    H = Hdf[pcols].to_numpy(dtype=float)

    # Preserve original H-row index.
    Hdf["_row_id"] = np.arange(len(Hdf), dtype=int)

    # Make timestamp type unambiguously datetime64[ns].
    Hdf["obs_time_aest"] = pd.to_datetime(
        Hdf["obs_time_aest"],
        errors="coerce",
    )

    Hdf["receptor"] = (
        Hdf["receptor"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    metadata = Hdf[
        [
            "_row_id",
            "receptor",
            "obs_time_aest",
            "release_utc",
            "flexpart_file",
        ]
    ].copy()

    # --------------------------------------------------------
    # Load Picarro observations
    # --------------------------------------------------------

    obs = pd.read_csv(PICARRO_FILE)

    obs["Datetime"] = pd.to_datetime(
        obs["Datetime"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    receptor_columns = {
        "singleton": "Singleton cleaned",
        "muswellbrook": "Muswellbrook cleaned",
        "merriwa": "Merriwa cleaned",
    }

    rows = []

    for receptor, column in receptor_columns.items():

        tmp = obs[
            [
                "Datetime",
                column,
            ]
        ].copy()

        tmp["receptor"] = receptor

        tmp["value"] = pd.to_numeric(
            tmp[column],
            errors="coerce",
        )

        tmp = tmp[
            tmp["Datetime"].between(
                START_DATE,
                END_DATE
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1),
            )
        ]

        tmp = tmp.dropna(
            subset=[
                "Datetime",
                "value",
            ]
        )

        rows.append(
            tmp[
                [
                    "Datetime",
                    "receptor",
                    "value",
                ]
            ]
        )

    obs_long = pd.concat(
        rows,
        ignore_index=True,
    )

    obs_long = obs_long.rename(
        columns={
            "Datetime": "obs_time_aest"
        }
    )

    obs_long["obs_time_aest"] = pd.to_datetime(
        obs_long["obs_time_aest"],
        errors="coerce",
    )

    obs_long["receptor"] = (
        obs_long["receptor"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Match H rows directly to observations
    #
    # Both keys are now:
    #
    #   receptor       : lowercase string
    #   obs_time_aest  : datetime64[ns]
    #
    # --------------------------------------------------------

    merged = metadata.merge(
        obs_long,
        on=[
            "receptor",
            "obs_time_aest",
        ],
        how="inner",
        validate="many_to_one",
    )

    merged = merged.sort_values(
        [
            "obs_time_aest",
            "receptor",
            "_row_id",
        ]
    ).reset_index(drop=True)

    # Original H rows corresponding to matched observations.
    row_indices = merged[
        "_row_id"
    ].to_numpy(
        dtype=int
    )

    H_match = H[
        row_indices,
        :,
    ]

    y = merged[
        "value"
    ].to_numpy(
        dtype=float
    )

    print(
        "Matched observations:",
        len(y),
    )

    print(
        "Matched receptors:",
        merged[
            "receptor"
        ].value_counts().to_dict()
    )

    print(
        "H matched shape:",
        H_match.shape,
    )

    return (
        H_match,
        y,
        merged,
        pcols,
    )


# ============================================================
# BUILD WHITENED OBJECTIVE
# ============================================================

def build_whitened_system(H, y):

    nparam = H.shape[1]

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    nday = len(dates)

    B, rho = build_prior_covariance(
        N_MINE,
        nday,
        PRIOR_SIGMA,
        CORR_LENGTH_DAYS,
    )

    print()
    print("=" * 90)
    print("PRIOR")
    print("=" * 90)

    print("B shape:", B.shape)
    print("rho:", rho)
    print("B diagonal:", B[0, 0])
    print(
        "Adjacent covariance:",
        B[0, N_MINE],
    )

    # Cholesky
    L = np.linalg.cholesky(B)

    recon_error = np.max(
        np.abs(
            L @ L.T - B
        )
    )

    print(
        "Cholesky reconstruction error:",
        recon_error,
    )

    # --------------------------------------------------------
    # Prior whitening
    #
    # || L^-1 z ||^2
    # --------------------------------------------------------

    I = np.eye(nparam)

    A_prior = solve_triangular(
        L,
        I,
        lower=True,
    )

    b_prior = np.zeros(
        nparam
    )

    # --------------------------------------------------------
    # Observation whitening
    #
    # y - background - H(xb + z)
    #
    # = y-background-Hxb-Hz
    # --------------------------------------------------------

    xb = np.full(
        nparam,
        PRIOR_MEAN,
    )

    A_obs = H / OBS_SIGMA

    b_obs = (
        y
        - BACKGROUND_PPM
        - H @ xb
    ) / OBS_SIGMA

    A = np.vstack(
        [
            A_prior,
            A_obs,
        ]
    )

    b = np.concatenate(
        [
            b_prior,
            b_obs,
        ]
    )

    print()
    print("Whitened system:", A.shape)

    return (
        A,
        b,
        B,
        L,
        xb,
    )


# ============================================================
# QUADRATIC MODEL
# ============================================================

class QuadraticProblem:

    def __init__(self, A, b):

        self.A = A
        self.b = b

        self.n = A.shape[1]

        # Q = A^T A
        #
        # We use matrix-vector products rather than explicitly
        # constructing Q for the iterative free-variable solves.

        self.At_b = A.T @ b

    def gradient(self, z):

        # Gradient of ||Az-b||^2
        return 2.0 * (
            self.A.T @ (
                self.A @ z - self.b
            )
        )

    def hess_vec(self, v):

        return 2.0 * (
            self.A.T @ (
                self.A @ v
            )
        )

    def solve_free_direction(
        self,
        z,
        free,
        active,
        g,
    ):

        """
        Solve

            Q_ff p_f = -g_f

        with p_active = 0.

        This gives the equality-constrained Newton
        direction for the current active set.
        """

        if len(free) == 0:
            return np.zeros(self.n)

        free = np.asarray(
            free,
            dtype=int,
        )

        # Explicit Q_ff is only 343x343 here, so it is
        # perfectly manageable and gives a stable solve.

        A_free = self.A[:, free]

        Qff = (
            2.0
            * A_free.T
            @ A_free
        )

        rhs = -g[free]

        try:
            p_free = np.linalg.solve(
                Qff,
                rhs,
            )

        except np.linalg.LinAlgError:

            p_free = np.linalg.lstsq(
                Qff,
                rhs,
                rcond=None,
            )[0]

        p = np.zeros(
            self.n
        )

        p[free] = p_free

        return p


# ============================================================
# ACTIVE-SET SOLVER
# ============================================================

def active_set_qp(
    problem,
    lower,
    upper,
    z0=None,
    max_iter=1000,
    primal_tol=1e-10,
    dual_tol=1e-9,
):

    n = problem.n

    if z0 is None:
        z = np.zeros(n)

    else:
        z = np.asarray(
            z0,
            dtype=float,
        ).copy()

    z = np.clip(
        z,
        lower,
        upper,
    )

    # --------------------------------------------------------
    # Determine initial active set.
    # --------------------------------------------------------

    active_lower = set(
        np.where(
            z <= lower + primal_tol
        )[0].tolist()
    )

    active_upper = set(
        np.where(
            z >= upper - primal_tol
        )[0].tolist()
    )

    # Start from prior, so nothing should be active.
    active_lower.clear()
    active_upper.clear()

    history = []

    for iteration in range(
        1,
        max_iter + 1,
    ):

        active = (
            active_lower
            | active_upper
        )

        free = [
            i
            for i in range(n)
            if i not in active
        ]

        g = problem.gradient(z)

        # ----------------------------------------------------
        # Compute equality-constrained search direction.
        # ----------------------------------------------------

        p = problem.solve_free_direction(
            z,
            free,
            active,
            g,
        )

        pnorm = np.max(
            np.abs(p)
        )

        # ----------------------------------------------------
        # If there is a search direction, move as far as
        # possible while respecting all bounds.
        # ----------------------------------------------------

        if pnorm > primal_tol:

            alpha = 1.0
            blocking_index = None
            blocking_type = None

            for i in free:

                if p[i] > primal_tol:

                    a = (
                        upper[i] - z[i]
                    ) / p[i]

                    if a < alpha:

                        alpha = a
                        blocking_index = i
                        blocking_type = "upper"

                elif p[i] < -primal_tol:

                    a = (
                        lower[i] - z[i]
                    ) / p[i]

                    if a < alpha:

                        alpha = a
                        blocking_index = i
                        blocking_type = "lower"

            alpha = max(
                0.0,
                min(1.0, alpha),
            )

            z = z + alpha * p

            z = np.clip(
                z,
                lower,
                upper,
            )

            if (
                blocking_index is not None
                and alpha < 1.0 - primal_tol
            ):

                if blocking_type == "lower":
                    z[blocking_index] = lower[
                        blocking_index
                    ]

                    active_lower.add(
                        blocking_index
                    )

                else:
                    z[blocking_index] = upper[
                        blocking_index
                    ]

                    active_upper.add(
                        blocking_index
                    )

            history.append(
                {
                    "iteration": iteration,
                    "action": "move",
                    "alpha": alpha,
                    "blocking_index": blocking_index,
                    "blocking_type": blocking_type,
                    "n_lower": len(active_lower),
                    "n_upper": len(active_upper),
                    "pnorm": pnorm,
                }
            )

            continue

        # ----------------------------------------------------
        # No feasible descent direction.
        #
        # Check KKT multipliers of active constraints.
        #
        # For minimization:
        #
        # lower active: g_i >= 0
        # upper active: g_i <= 0
        # ----------------------------------------------------

        release_index = None
        release_type = None
        worst_violation = 0.0

        for i in active_lower:

            violation = -g[i]

            if violation > worst_violation:

                worst_violation = violation
                release_index = i
                release_type = "lower"

        for i in active_upper:

            violation = g[i]

            if violation > worst_violation:

                worst_violation = violation
                release_index = i
                release_type = "upper"

        history.append(
            {
                "iteration": iteration,
                "action": "kkt_check",
                "alpha": 0.0,
                "blocking_index": release_index,
                "blocking_type": release_type,
                "n_lower": len(active_lower),
                "n_upper": len(active_upper),
                "pnorm": pnorm,
                "worst_dual_violation": worst_violation,
            }
        )

        # ----------------------------------------------------
        # All KKT conditions satisfied.
        # ----------------------------------------------------

        if worst_violation <= dual_tol:

            print()
            print(
                "Active-set solver converged."
            )

            print(
                "Iterations:",
                iteration,
            )

            return (
                z,
                active_lower,
                active_upper,
                history,
            )

        # ----------------------------------------------------
        # Release the most violated active constraint.
        # ----------------------------------------------------

        if release_type == "lower":

            active_lower.remove(
                release_index
            )

        else:

            active_upper.remove(
                release_index
            )

    raise RuntimeError(
        "Active-set solver failed to converge "
        f"after {max_iter} iterations."
    )


# ============================================================
# KKT DIAGNOSTICS
# ============================================================

def kkt_diagnostics(
    problem,
    z,
    lower,
    upper,
    primal_tol=1e-8,
):

    x = z + PRIOR_MEAN

    g = problem.gradient(z)

    active_lower = (
        x <= LOWER_X + primal_tol
    )

    active_upper = (
        x >= UPPER_X - primal_tol
    )

    free = ~(
        active_lower
        | active_upper
    )

    free_grad = np.max(
        np.abs(
            g[free]
        )
    ) if np.any(free) else 0.0

    lower_violation = np.max(
        np.maximum(
            -g[active_lower],
            0.0,
        )
    ) if np.any(active_lower) else 0.0

    upper_violation = np.max(
        np.maximum(
            g[active_upper],
            0.0,
        )
    ) if np.any(active_upper) else 0.0

    return {
        "x": x,
        "gradient": g,
        "active_lower": active_lower,
        "active_upper": active_upper,
        "free": free,
        "max_free_gradient": free_grad,
        "max_lower_violation": lower_violation,
        "max_upper_violation": upper_violation,
    }


# ============================================================
# LOAD BATCH MAP REFERENCE
# ============================================================

def load_batch_map_reference():

    candidates = [
        os.path.join(
            BOUNDED_MAP_DIR,
            "bounded_batch_MAP_351_parameters.csv",
        ),
        os.path.join(
            BOUNDED_MAP_DIR,
            "bounded_batch_MAP_solution.npz",
        ),
    ]

    csv_file = candidates[0]

    if os.path.exists(csv_file):

        df = pd.read_csv(
            csv_file
        )

        print(
            "Reference MAP file:",
            csv_file,
        )

        # Try common parameter-value column names.
        for col in [
            "bounded_batch_MAP",
            "scaling_factor",
            "value",
            "x",
        ]:

            if col in df.columns:

                return df[col].to_numpy(
                    dtype=float
                )

        raise RuntimeError(
            "Could not identify MAP value column in "
            + csv_file
        )

    npz_file = candidates[1]

    if os.path.exists(npz_file):

        data = np.load(
            npz_file
        )

        for key in [
            "x",
            "solution",
            "bounded_batch_MAP",
            "states",
        ]:

            if key in data:

                arr = data[key]

                if arr.size == 351:

                    return arr.reshape(
                        -1
                    )

        raise RuntimeError(
            "Could not identify 351-element MAP solution."
        )

    raise FileNotFoundError(
        "No bounded batch MAP reference found in "
        + BOUNDED_MAP_DIR
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    (
        H,
        y,
        metadata,
        pcols,
    ) = load_problem()

    A, b, B, L, xb = (
        build_whitened_system(
            H,
            y,
        )
    )

    n = A.shape[1]

    problem = QuadraticProblem(
        A,
        b,
    )

    lower = np.full(
        n,
        LOWER_Z,
    )

    upper = np.full(
        n,
        UPPER_Z,
    )

    print()
    print("=" * 90)
    print("ACTIVE-SET CONSTRAINED GLOBAL SMOOTHER")
    print("=" * 90)

    print(
        "Parameters:",
        n,
    )

    print(
        "Bounds in x:",
        LOWER_X,
        UPPER_X,
    )

    print(
        "Bounds in z:",
        LOWER_Z,
        UPPER_Z,
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    z, active_lower, active_upper, history = (
        active_set_qp(
            problem,
            lower,
            upper,
            z0=np.zeros(n),
            max_iter=1000,
            primal_tol=1e-10,
            dual_tol=1e-9,
        )
    )

    x = z + PRIOR_MEAN

    # --------------------------------------------------------
    # KKT
    # --------------------------------------------------------

    diag = kkt_diagnostics(
        problem,
        z,
        lower,
        upper,
    )

    print()
    print("=" * 90)
    print("KKT CHECK")
    print("=" * 90)

    print(
        "Max free |gradient|:",
        diag["max_free_gradient"],
    )

    print(
        "Max lower KKT violation:",
        diag["max_lower_violation"],
    )

    print(
        "Max upper KKT violation:",
        diag["max_upper_violation"],
    )

    print(
        "Solution min:",
        np.min(x),
    )

    print(
        "Solution max:",
        np.max(x),
    )

    print(
        "Solution mean:",
        np.mean(x),
    )

    print(
        "Lower-bound parameters:",
        np.sum(diag["active_lower"]),
    )

    print(
        "Upper-bound parameters:",
        np.sum(diag["active_upper"]),
    )

    # --------------------------------------------------------
    # Compare with bounded batch MAP
    # --------------------------------------------------------

    reference = (
        load_batch_map_reference()
    )

    if len(reference) != n:

        raise RuntimeError(
            "Reference MAP has wrong length: "
            f"{len(reference)}"
        )

    diff = x - reference

    rmse = np.sqrt(
        np.mean(
            diff**2
        )
    )

    mae = np.mean(
        np.abs(diff)
    )

    max_abs = np.max(
        np.abs(diff)
    )

    corr = np.corrcoef(
        x,
        reference,
    )[0, 1]

    print()
    print("=" * 90)
    print("COMPARISON WITH BOUNDED BATCH MAP")
    print("=" * 90)

    print(
        "Max absolute difference:",
        max_abs,
    )

    print(
        "RMSE:",
        rmse,
    )

    print(
        "MAE:",
        mae,
    )

    print(
        "Correlation:",
        corr,
    )

    # --------------------------------------------------------
    # Parameter table
    # --------------------------------------------------------

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    records = []

    for idx, col in enumerate(
        pcols
    ):

        day = idx // N_MINE
        mine = idx % N_MINE + 1

        records.append(
            {
                "date": dates[day],
                "mine": f"OC{mine}",
                "index": idx,
                "constrained_smoother": x[idx],
                "bounded_batch_MAP": reference[idx],
                "difference": diff[idx],
                "gradient": diag["gradient"][idx],
                "active_lower": bool(
                    diag["active_lower"][idx]
                ),
                "active_upper": bool(
                    diag["active_upper"][idx]
                ),
            }
        )

    result_df = pd.DataFrame(
        records
    )

    result_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "constrained_smoother_351.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Active constraints
    # --------------------------------------------------------

    active_df = result_df[
        result_df[
            [
                "active_lower",
                "active_upper",
            ]
        ].any(axis=1)
    ].copy()

    active_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "constrained_smoother_active_bounds.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Daily output
    # --------------------------------------------------------

    daily = result_df.pivot(
        index="date",
        columns="mine",
        values="constrained_smoother",
    ).reset_index()

    daily.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "daily_constrained_smoother_scaling_factors.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "active_set_history.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "constrained_smoother_solution.npz",
        ),
        states=x,
        gradients=diag["gradient"],
        active_lower=diag["active_lower"],
        active_upper=diag["active_upper"],
        history_iterations=len(history),
        mines=np.array(
            [
                f"OC{i}"
                for i in range(
                    1,
                    N_MINE + 1,
                )
            ],
            dtype="U10",
        ),
        dates=np.array(
            dates.strftime(
                "%Y-%m-%d"
            ),
            dtype="U10",
        ),
    )

    # --------------------------------------------------------
    # Final active parameters
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("ACTIVE PARAMETERS")
    print("=" * 90)

    print(
        active_df[
            [
                "date",
                "mine",
                "constrained_smoother",
                "active_lower",
                "active_upper",
                "gradient",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)

    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()

