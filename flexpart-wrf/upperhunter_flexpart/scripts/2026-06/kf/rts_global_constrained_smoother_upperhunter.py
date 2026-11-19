#!/usr/bin/env python3

"""
FLEXPART-WRF GLOBAL CONSTRAINED SMOOTHER
========================================

Global Bayesian constrained smoothing / batch MAP solution
for the Upper Hunter 13-mine daily CH4 inversion.

State:
    13 mines x 27 days = 351 parameters

Prior:
    mean = 1.0
    sigma = 0.5
    AR(1) correlation length = 3 days

Observation:
    Picarro CH4
    sigma = 0.05 ppm
    background = 1.85 ppm

Bounds:
    0 <= x <= 5

The objective is

    J(x) =
        0.5 (x-xb)' B^-1 (x-xb)
      + 0.5 (y-background-Hx)' R^-1
        (y-background-Hx)

subject to

    lower <= x <= upper

This script solves the GLOBAL constrained quadratic problem
using a primal active-set method.

It is therefore intended as the constrained counterpart of
the ordinary RTS/batch equivalence.

Validation target:
    Bounded_batch_MAP_full_footprint_13mine_Jan2026

The final solution should reproduce the validated bounded
batch MAP to numerical tolerance.

Known validated active constraints:

    2026-01-15 OC12 = 0
    2026-01-15 OC13 = 0
    2026-01-19 OC9  = 5
    2026-01-20 OC9  = 5
    2026-01-21 OC3  = 0
    2026-01-21 OC9  = 5
    2026-01-21 OC12 = 0
    2026-01-22 OC3  = 0
"""


import os
import warnings
import numpy as np
import pandas as pd

from scipy.linalg import solve
from scipy.optimize import lsq_linear


# ============================================================
# PATHS
# ============================================================

BASE = (
    "/home/duch/flexpart-wrf/"
    "upperhunter_flexpart/H_matrix_flexpart"
)

H_FILE = os.path.join(
    BASE,
    "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/"
    "emission_UH/RH_upperHunter/"
    "tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

BATCH_MAP_DIR = os.path.join(
    BASE,
    "Bounded_batch_MAP_full_footprint_13mine_Jan2026"
)

BATCH_MAP_FILE = os.path.join(
    BATCH_MAP_DIR,
    "bounded_batch_MAP_351_parameters.csv"
)

BATCH_DAILY_FILE = os.path.join(
    BATCH_MAP_DIR,
    "daily_bounded_batch_MAP_scaling_factors.csv"
)

OUTDIR = os.path.join(
    BASE,
    "Global_constrained_smoother_13mine_Jan2026"
)

os.makedirs(
    OUTDIR,
    exist_ok=True
)


# ============================================================
# INVERSION SETTINGS
# ============================================================

N_MINE = 13
N_DAY = 27
N_PARAM = N_MINE * N_DAY

MINES = [
    f"OC{i}"
    for i in range(1, 14)
]

DATES = pd.date_range(
    "2026-01-05",
    "2026-01-31",
    freq="D"
)

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5

BACKGROUND_PPM = 1.85
OBS_SIGMA = 0.05

CORRELATION_LENGTH_DAYS = 3.0

RHO = np.exp(
    -1.0 / CORRELATION_LENGTH_DAYS
)

Q_SCALAR = (
    PRIOR_SIGMA ** 2
    * (1.0 - RHO ** 2)
)

LOWER = 0.0
UPPER = 5.0


# ============================================================
# ACTIVE-SET SETTINGS
# ============================================================

BOUND_TOL = 1.0e-9
KKT_TOL = 1.0e-8
STEP_TOL = 1.0e-12

MAX_ITER = 1000


# ============================================================
# LOAD H MATRIX
# ============================================================

def load_H():

    print("=" * 78)
    print("LOADING FLEXPART H MATRIX")
    print("=" * 78)

    df = pd.read_csv(
        H_FILE
    )

    print(
        "H dataframe:",
        df.shape
    )

    metadata = [
        "receptor",
        "release_utc",
        "obs_time_aest",
        "flexpart_file",
    ]

    parameter_columns = [
        f"OC{m}_{d.strftime('%Y%m%d')}"
        for d in DATES
        for m in range(1, N_MINE + 1)
    ]

    missing = [
        c for c in parameter_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing H parameter columns:\n"
            + "\n".join(missing)
        )

    H = df[
        parameter_columns
    ].to_numpy(
        dtype=float
    )

    df["obs_time_aest"] = pd.to_datetime(
        df["obs_time_aest"],
        errors="coerce"
    )

    df["receptor"] = (
        df["receptor"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print(
        "H matrix:",
        H.shape
    )

    print(
        "H parameters:",
        len(parameter_columns)
    )

    return (
        df,
        H,
        parameter_columns
    )


# ============================================================
# LOAD PICARRO
# ============================================================

def load_observations():

    print("\n" + "=" * 78)
    print("LOADING PICARRO OBSERVATIONS")
    print("=" * 78)

    obs = pd.read_csv(
        PICARRO_FILE
    )

    obs["Datetime"] = pd.to_datetime(
        obs["Datetime"],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    print(
        "Picarro rows:",
        len(obs)
    )

    receptor_columns = {
        "singleton": "Singleton cleaned",
        "muswellbrook": "Muswellbrook cleaned",
        "merriwa": "Merriwa cleaned",
    }

    rows = []

    for receptor, column in receptor_columns.items():

        if column not in obs.columns:

            raise ValueError(
                f"Missing Picarro column: {column}"
            )

        tmp = obs[
            ["Datetime", column]
        ].copy()

        tmp = tmp.rename(
            columns={
                "Datetime": "obs_time_aest",
                column: "observation",
            }
        )

        tmp["receptor"] = receptor

        tmp = tmp[
            [
                "receptor",
                "obs_time_aest",
                "observation",
            ]
        ]

        rows.append(tmp)

    result = pd.concat(
        rows,
        ignore_index=True
    )

    result["obs_time_aest"] = pd.to_datetime(
        result["obs_time_aest"]
    )

    result["observation"] = pd.to_numeric(
        result["observation"],
        errors="coerce"
    )

    result = result.dropna(
        subset=[
            "obs_time_aest",
            "observation"
        ]
    )

    return result


# ============================================================
# MATCH H AND OBSERVATIONS
# ============================================================

def match_observations(
    Hdf,
    H,
    observations,
):

    print("\n" + "=" * 78)
    print("MATCHING H MATRIX TO PICARRO")
    print("=" * 78)

    lookup = observations.copy()

    lookup["receptor"] = (
        lookup["receptor"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    lookup["obs_time_aest"] = pd.to_datetime(
        lookup["obs_time_aest"]
    )

    # --------------------------------------------------------
    # H rows are already observation timestamps.
    # --------------------------------------------------------

    merged = Hdf[
        [
            "receptor",
            "obs_time_aest",
        ]
    ].copy()

    merged["_H_row"] = np.arange(
        len(merged)
    )

    merged = merged.merge(
        lookup,
        on=[
            "receptor",
            "obs_time_aest",
        ],
        how="inner",
    )

    # --------------------------------------------------------
    # Restrict to requested January period
    # --------------------------------------------------------

    start = pd.Timestamp(
        "2026-01-05 00:00:00"
    )

    end = pd.Timestamp(
        "2026-01-31 23:59:59"
    )

    merged = merged[
        (
            merged["obs_time_aest"] >= start
        )
        &
        (
            merged["obs_time_aest"] <= end
        )
    ]

    # --------------------------------------------------------
    # Remove duplicate matched observation rows.
    # --------------------------------------------------------

    merged = (
        merged
        .sort_values(
            [
                "obs_time_aest",
                "receptor",
                "_H_row",
            ]
        )
        .drop_duplicates(
            subset=[
                "receptor",
                "obs_time_aest",
            ],
            keep="first"
        )
    )

    merged = merged.sort_values(
        [
            "obs_time_aest",
            "receptor",
        ]
    ).reset_index(
        drop=True
    )

    row_index = merged[
        "_H_row"
    ].to_numpy(
        dtype=int
    )

    y = merged[
        "observation"
    ].to_numpy(
        dtype=float
    )

    H_match = H[
        row_index,
        :
    ]

    print(
        "Matched observations:",
        len(y)
    )

    print(
        "Matched H:",
        H_match.shape
    )

    print(
        "Matched receptors:",
        merged["receptor"]
        .value_counts()
        .to_dict()
    )

    if len(y) == 0:

        raise RuntimeError(
            "No H/Picarro observations matched."
        )

    return (
        merged,
        H_match,
        y
    )


# ============================================================
# BUILD AR(1) PRIOR COVARIANCE
# ============================================================

def build_prior():

    print("\n" + "=" * 78)
    print("BUILDING AR(1) PRIOR")
    print("=" * 78)

    B = np.zeros(
        (
            N_PARAM,
            N_PARAM
        )
    )

    for mine in range(
        N_MINE
    ):

        for d1 in range(
            N_DAY
        ):

            i = (
                d1 * N_MINE
                + mine
            )

            for d2 in range(
                N_DAY
            ):

                j = (
                    d2 * N_MINE
                    + mine
                )

                lag = abs(
                    d1 - d2
                )

                B[i, j] = (
                    PRIOR_SIGMA ** 2
                    * RHO ** lag
                )

    print(
        "B shape:",
        B.shape
    )

    print(
        "rho:",
        RHO
    )

    print(
        "B diagonal:",
        B[0, 0]
    )

    print(
        "Adjacent covariance:",
        B[0, N_MINE]
    )

    # --------------------------------------------------------
    # Cholesky
    # --------------------------------------------------------

    L = np.linalg.cholesky(
        B
    )

    recon_error = np.max(
        np.abs(
            B
            -
            L @ L.T
        )
    )

    print(
        "Cholesky reconstruction error:",
        recon_error
    )

    return B, L


# ============================================================
# BUILD GLOBAL POSTERIOR
# ============================================================

def build_posterior(
    H,
    y,
    B,
):

    print("\n" + "=" * 78)
    print("BUILDING GLOBAL POSTERIOR")
    print("=" * 78)

    xb = np.full(
        N_PARAM,
        PRIOR_MEAN
    )

    # --------------------------------------------------------
    # Prior precision
    # --------------------------------------------------------

    B_inv = np.linalg.inv(
        B
    )

    # --------------------------------------------------------
    # Observation precision
    # --------------------------------------------------------

    R_inv_scalar = (
        1.0
        / OBS_SIGMA ** 2
    )

    # d = y - background
    d = (
        y
        -
        BACKGROUND_PPM
    )

    # --------------------------------------------------------
    # Objective:
    #
    # 0.5 x' A x + q' x + const
    #
    # A = B^-1 + H'R^-1H
    #
    # q = -B^-1 xb - H'R^-1 d
    # --------------------------------------------------------

    print(
        "Constructing posterior Hessian..."
    )

    A = (
        B_inv
        +
        R_inv_scalar
        * (H.T @ H)
    )

    q = (
        -B_inv @ xb
        -
        R_inv_scalar
        * (H.T @ d)
    )

    A = 0.5 * (
        A + A.T
    )

    print(
        "Posterior Hessian:",
        A.shape
    )

    eig = np.linalg.eigvalsh(
        A
    )

    print(
        "Hessian minimum eigenvalue:",
        np.min(eig)
    )

    print(
        "Hessian maximum eigenvalue:",
        np.max(eig)
    )

    if np.min(eig) <= 0:

        raise RuntimeError(
            "Posterior Hessian is not positive definite."
        )

    return (
        A,
        q,
        xb
    )


# ============================================================
# OBJECTIVE / GRADIENT
# ============================================================

def objective_from_A(
    x,
    A,
    q,
):

    return (
        0.5 * x @ A @ x
        +
        q @ x
    )


def gradient_from_A(
    x,
    A,
    q,
):

    return (
        A @ x
        +
        q
    )


# ============================================================
# UNCONSTRAINED SOLUTION
# ============================================================

def solve_unconstrained(
    A,
    q,
):

    print("\n" + "=" * 78)
    print("UNCONSTRAINED GLOBAL SOLUTION")
    print("=" * 78)

    x = solve(
        A,
        -q,
        assume_a="pos"
    )

    print(
        "Minimum:",
        np.min(x)
    )

    print(
        "Maximum:",
        np.max(x)
    )

    print(
        "Mean:",
        np.mean(x)
    )

    print(
        "Below 0:",
        np.sum(x < 0)
    )

    print(
        "Above 5:",
        np.sum(x > 5)
    )

    return x


# ============================================================
# EQUALITY-CONSTRAINED FREE SOLVE
# ============================================================

def solve_free_system(
    A,
    q,
    active_lower,
    active_upper,
    x,
):

    active = (
        active_lower
        |
        active_upper
    )

    free = ~active

    free_idx = np.where(
        free
    )[0]

    active_idx = np.where(
        active
    )[0]

    if len(free_idx) == 0:

        return x.copy()

    rhs = -q[
        free_idx
    ]

    if len(active_idx) > 0:

        rhs -= (
            A[
                np.ix_(
                    free_idx,
                    active_idx
                )
            ]
            @
            x[active_idx]
        )

    Aff = A[
        np.ix_(
            free_idx,
            free_idx
        )
    ]

    x_candidate = x.copy()

    x_candidate[
        free_idx
    ] = solve(
        Aff,
        rhs,
        assume_a="pos"
    )

    return x_candidate


# ============================================================
# PRIMAL ACTIVE-SET SOLVER
# ============================================================

def solve_box_active_set(
    A,
    q,
):

    print("\n" + "=" * 78)
    print("GLOBAL BOX-CONSTRAINED ACTIVE-SET SOLVER")
    print("=" * 78)

    n = len(q)

    # --------------------------------------------------------
    # Start from feasible prior mean.
    # --------------------------------------------------------

    x = np.full(
        n,
        PRIOR_MEAN
    )

    active_lower = np.zeros(
        n,
        dtype=bool
    )

    active_upper = np.zeros(
        n,
        dtype=bool
    )

    # --------------------------------------------------------
    # Main active-set iterations
    # --------------------------------------------------------

    for iteration in range(
        1,
        MAX_ITER + 1
    ):

        active = (
            active_lower
            |
            active_upper
        )

        free = ~active

        # ----------------------------------------------------
        # Solve equality-constrained problem.
        # ----------------------------------------------------

        candidate = solve_free_system(
            A,
            q,
            active_lower,
            active_upper,
            x
        )

        direction = (
            candidate - x
        )

        # ----------------------------------------------------
        # Determine first bound encountered.
        # ----------------------------------------------------

        alpha = 1.0
        hit_index = None
        hit_bound = None

        free_idx = np.where(
            free
        )[0]

        for i in free_idx:

            di = direction[i]

            if di > 0:

                ai = (
                    UPPER - x[i]
                ) / di

                if (
                    ai >= 0
                    and ai < alpha
                ):

                    alpha = ai
                    hit_index = i
                    hit_bound = "upper"

            elif di < 0:

                ai = (
                    LOWER - x[i]
                ) / di

                if (
                    ai >= 0
                    and ai < alpha
                ):

                    alpha = ai
                    hit_index = i
                    hit_bound = "lower"

        # ----------------------------------------------------
        # Move toward equality-constrained solution.
        # ----------------------------------------------------

        x_new = (
            x
            +
            alpha * direction
        )

        # ----------------------------------------------------
        # Snap bound if one was encountered.
        # ----------------------------------------------------

        if (
            hit_index is not None
            and alpha < 1.0 - STEP_TOL
        ):

            x_new[
                hit_index
            ] = (
                UPPER
                if hit_bound == "upper"
                else LOWER
            )

            if hit_bound == "upper":

                active_upper[
                    hit_index
                ] = True

                active_lower[
                    hit_index
                ] = False

            else:

                active_lower[
                    hit_index
                ] = True

                active_upper[
                    hit_index
                ] = False

            x = x_new

            if (
                iteration <= 20
                or iteration % 10 == 0
            ):

                print(
                    f"iter={iteration:4d} "
                    f"ADD {hit_bound:5s} "
                    f"index={hit_index:3d} "
                    f"day={hit_index // N_MINE + 1:2d} "
                    f"mine={MINES[hit_index % N_MINE]}"
                )

            continue

        # ----------------------------------------------------
        # Candidate is within bounds.
        # ----------------------------------------------------

        x = np.clip(
            x_new,
            LOWER,
            UPPER
        )

        # ----------------------------------------------------
        # KKT gradient.
        # ----------------------------------------------------

        g = gradient_from_A(
            x,
            A,
            q
        )

        free = (
            ~active_lower
            &
            ~active_upper
        )

        # ----------------------------------------------------
        # Check free gradient.
        # ----------------------------------------------------

        if np.any(free):

            max_free_gradient = np.max(
                np.abs(
                    g[free]
                )
            )

        else:

            max_free_gradient = 0.0

        # ----------------------------------------------------
        # Lower-bound KKT:
        #
        # x=0 requires g >= 0
        # ----------------------------------------------------

        lower_violation = np.where(
            active_lower
            &
            (g < -KKT_TOL)
        )[0]

        # ----------------------------------------------------
        # Upper-bound KKT:
        #
        # x=5 requires g <= 0
        # ----------------------------------------------------

        upper_violation = np.where(
            active_upper
            &
            (g > KKT_TOL)
        )[0]

        # ----------------------------------------------------
        # If KKT satisfied -> done.
        # ----------------------------------------------------

        if (
            max_free_gradient <= KKT_TOL
            and len(lower_violation) == 0
            and len(upper_violation) == 0
        ):

            print(
                "\nActive-set solver converged."
            )

            print(
                "Iterations:",
                iteration
            )

            return (
                x,
                active_lower,
                active_upper,
                g,
                iteration
            )

        # ----------------------------------------------------
        # Release the most strongly violated active variable.
        # ----------------------------------------------------

        if len(lower_violation) > 0:

            # Most negative gradient
            idx = lower_violation[
                np.argmin(
                    g[lower_violation]
                )
            ]

            active_lower[idx] = False

            if (
                iteration <= 20
                or iteration % 10 == 0
            ):

                print(
                    f"iter={iteration:4d} "
                    f"REMOVE lower "
                    f"index={idx:3d} "
                    f"day={idx // N_MINE + 1:2d} "
                    f"mine={MINES[idx % N_MINE]} "
                    f"gradient={g[idx]:.6e}"
                )

            continue

        if len(upper_violation) > 0:

            # Most positive gradient
            idx = upper_violation[
                np.argmax(
                    g[upper_violation]
                )
            ]

            active_upper[idx] = False

            if (
                iteration <= 20
                or iteration % 10 == 0
            ):

                print(
                    f"iter={iteration:4d} "
                    f"REMOVE upper "
                    f"index={idx:3d} "
                    f"day={idx // N_MINE + 1:2d} "
                    f"mine={MINES[idx % N_MINE]} "
                    f"gradient={g[idx]:.6e}"
                )

            continue

        # ----------------------------------------------------
        # Numerical fallback.
        # ----------------------------------------------------

        if (
            max_free_gradient
            > KKT_TOL
        ):

            # Re-solve on free variables.
            x = candidate

            continue

    raise RuntimeError(
        "Global active-set solver did not converge "
        f"after {MAX_ITER} iterations."
    )


# ============================================================
# KKT VALIDATION
# ============================================================

def validate_kkt(
    x,
    g,
    active_lower,
    active_upper,
):

    print("\n" + "=" * 78)
    print("FINAL KKT VALIDATION")
    print("=" * 78)

    free = (
        ~active_lower
        &
        ~active_upper
    )

    max_free = (
        np.max(
            np.abs(
                g[free]
            )
        )
        if np.any(free)
        else 0.0
    )

    lower_violation = (
        np.max(
            np.maximum(
                -g[active_lower],
                0.0
            )
        )
        if np.any(active_lower)
        else 0.0
    )

    upper_violation = (
        np.max(
            np.maximum(
                g[active_upper],
                0.0
            )
        )
        if np.any(active_upper)
        else 0.0
    )

    print(
        "Max free |gradient|:",
        max_free
    )

    print(
        "Max lower KKT violation:",
        lower_violation
    )

    print(
        "Max upper KKT violation:",
        upper_violation
    )

    print(
        "Solution minimum:",
        np.min(x)
    )

    print(
        "Solution maximum:",
        np.max(x)
    )

    print(
        "Solution mean:",
        np.mean(x)
    )

    print(
        "Lower-bound parameters:",
        np.sum(active_lower)
    )

    print(
        "Upper-bound parameters:",
        np.sum(active_upper)
    )

    return (
        max_free,
        lower_violation,
        upper_violation
    )


# ============================================================
# PRINT ACTIVE PARAMETERS
# ============================================================

def print_active_parameters(
    x,
    g,
    active_lower,
    active_upper,
):

    print("\n" + "=" * 78)
    print("ACTIVE PARAMETERS")
    print("=" * 78)

    rows = []

    for i in range(
        N_PARAM
    ):

        if (
            active_lower[i]
            or
            active_upper[i]
        ):

            day = i // N_MINE
            mine = i % N_MINE

            value = x[i]

            lower = active_lower[i]
            upper = active_upper[i]

            print(
                f"{DATES[day].strftime('%Y-%m-%d')} "
                f"{MINES[mine]:5s} "
                f"{value:18.12g} "
                f"lower={str(lower):5s} "
                f"upper={str(upper):5s} "
                f"gradient={g[i]: .9f}"
            )

            rows.append(
                {
                    "date":
                        DATES[day],
                    "mine":
                        MINES[mine],
                    "index":
                        i,
                    "value":
                        value,
                    "active_lower":
                        lower,
                    "active_upper":
                        upper,
                    "gradient":
                        g[i],
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        os.path.join(
            OUTDIR,
            "global_smoother_active_constraints.csv"
        ),
        index=False
    )


# ============================================================
# COMPARE TO VALIDATED BATCH MAP
# ============================================================

def compare_to_batch_map(
    x,
):

    print("\n" + "=" * 78)
    print("COMPARISON WITH VALIDATED BOUNDED BATCH MAP")
    print("=" * 78)

    if not os.path.exists(
        BATCH_MAP_FILE
    ):

        print(
            "Batch MAP parameter file not found:"
        )

        print(
            BATCH_MAP_FILE
        )

        return None

    df = pd.read_csv(
        BATCH_MAP_FILE
    )

    # --------------------------------------------------------
    # Expected long format:
    #
    # date
    # mine
    # scaling_factor
    # --------------------------------------------------------

    required = [
        "date",
        "mine",
    ]

    for c in required:

        if c not in df.columns:

            raise ValueError(
                f"Batch MAP file missing "
                f"column {c}"
            )

    value_column = None

    for c in [
        "bounded_batch_MAP",
        "scaling_factor",
        "value",
    ]:

        if c in df.columns:

            value_column = c
            break

    if value_column is None:

        raise ValueError(
            "Could not identify batch MAP "
            "value column."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["mine"] = (
        df["mine"]
        .astype(str)
        .str.strip()
    )

    lookup = {}

    for _, row in df.iterrows():

        key = (
            pd.Timestamp(
                row["date"]
            ),
            row["mine"]
        )

        lookup[key] = float(
            row[value_column]
        )

    batch = np.zeros(
        N_PARAM
    )

    for d in range(
        N_DAY
    ):

        for m in range(
            N_MINE
        ):

            i = (
                d * N_MINE
                + m
            )

            key = (
                DATES[d],
                MINES[m]
            )

            if key not in lookup:

                raise ValueError(
                    f"Missing batch MAP "
                    f"parameter {key}"
                )

            batch[i] = lookup[key]

    diff = (
        x
        -
        batch
    )

    print(
        "Number of parameters:",
        len(x)
    )

    print(
        "Max absolute difference:",
        np.max(
            np.abs(diff)
        )
    )

    print(
        "RMSE:",
        np.sqrt(
            np.mean(
                diff ** 2
            )
        )
    )

    print(
        "MAE:",
        np.mean(
            np.abs(diff)
        )
    )

    print(
        "Correlation:",
        np.corrcoef(
            x,
            batch
        )[0, 1]
    )

    print(
        "|difference| > 0.1:",
        np.sum(
            np.abs(diff) > 0.1
        ),
        "/",
        N_PARAM
    )

    print(
        "|difference| > 1e-6:",
        np.sum(
            np.abs(diff) > 1.0e-6
        ),
        "/",
        N_PARAM
    )

    # --------------------------------------------------------
    # Daily comparison
    # --------------------------------------------------------

    print("\nDAILY COMPARISON")

    print(
        f"{'date':12s}"
        f"{'Smoother':>12s}"
        f"{'MAP':>12s}"
        f"{'difference':>12s}"
        f"{'RMSE':>12s}"
        f"{'MAE':>12s}"
        f"{'max_abs':>12s}"
    )

    daily_rows = []

    for d in range(
        N_DAY
    ):

        a = x[
            d * N_MINE:
            (d + 1) * N_MINE
        ]

        b = batch[
            d * N_MINE:
            (d + 1) * N_MINE
        ]

        dd = a - b

        rmse = np.sqrt(
            np.mean(
                dd ** 2
            )
        )

        mae = np.mean(
            np.abs(dd)
        )

        max_abs = np.max(
            np.abs(dd)
        )

        difference = np.mean(
            dd
        )

        print(
            f"{DATES[d].strftime('%Y-%m-%d'):12s}"
            f"{np.mean(a):12.6f}"
            f"{np.mean(b):12.6f}"
            f"{difference:12.6f}"
            f"{rmse:12.6f}"
            f"{mae:12.6f}"
            f"{max_abs:12.6f}"
        )

        daily_rows.append(
            {
                "date":
                    DATES[d],
                "smoother_mean":
                    np.mean(a),
                "batch_MAP_mean":
                    np.mean(b),
                "difference_mean":
                    difference,
                "RMSE":
                    rmse,
                "MAE":
                    mae,
                "max_abs_difference":
                    max_abs,
            }
        )

    pd.DataFrame(
        daily_rows
    ).to_csv(
        os.path.join(
            OUTDIR,
            "global_smoother_vs_batch_MAP_daily.csv"
        ),
        index=False
    )

    parameter_rows = []

    for i in range(
        N_PARAM
    ):

        d = i // N_MINE
        m = i % N_MINE

        parameter_rows.append(
            {
                "date":
                    DATES[d],
                "mine":
                    MINES[m],
                "smoother":
                    x[i],
                "batch_MAP":
                    batch[i],
                "difference":
                    diff[i],
                "abs_difference":
                    abs(diff[i]),
            }
        )

    pd.DataFrame(
        parameter_rows
    ).to_csv(
        os.path.join(
            OUTDIR,
            "global_smoother_vs_batch_MAP_parameters.csv"
        ),
        index=False
    )

    return batch


# ============================================================
# SAVE SOLUTION
# ============================================================

def save_solution(
    x,
    g,
    active_lower,
    active_upper,
    H_shape,
    n_obs,
    n_iterations,
):

    # --------------------------------------------------------
    # Long format
    # --------------------------------------------------------

    rows = []

    for i in range(
        N_PARAM
    ):

        d = i // N_MINE
        m = i % N_MINE

        rows.append(
            {
                "date":
                    DATES[d],
                "mine":
                    MINES[m],
                "index":
                    i,
                "scaling_factor":
                    x[i],
                "gradient":
                    g[i],
                "active_lower":
                    active_lower[i],
                "active_upper":
                    active_upper[i],
            }
        )

    long_df = pd.DataFrame(
        rows
    )

    long_df.to_csv(
        os.path.join(
            OUTDIR,
            "global_constrained_smoother_parameters.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Daily wide format
    # --------------------------------------------------------

    daily = np.zeros(
        (
            N_DAY,
            N_MINE
        )
    )

    for d in range(
        N_DAY
    ):

        daily[d, :] = x[
            d * N_MINE:
            (d + 1) * N_MINE
        ]

    daily_df = pd.DataFrame(
        daily,
        columns=MINES
    )

    daily_df.insert(
        0,
        "date",
        DATES
    )

    daily_df.to_csv(
        os.path.join(
            OUTDIR,
            "daily_global_constrained_smoother_scaling_factors.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # NPZ
    # --------------------------------------------------------

    np.savez_compressed(
        os.path.join(
            OUTDIR,
            "global_constrained_smoother_solution.npz"
        ),
        solution=x,
        gradient=g,
        active_lower=active_lower,
        active_upper=active_upper,
        dates=np.array(
            DATES.astype(str)
        ),
        mines=np.array(
            MINES
        ),
        rho=RHO,
        prior_mean=PRIOR_MEAN,
        prior_sigma=PRIOR_SIGMA,
        observation_sigma=OBS_SIGMA,
        background_ppm=BACKGROUND_PPM,
        lower_bound=LOWER,
        upper_bound=UPPER,
        n_observations=n_obs,
        h_shape=np.array(
            H_shape
        ),
        n_iterations=n_iterations,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with open(
        os.path.join(
            OUTDIR,
            "global_constrained_smoother_summary.txt"
        ),
        "w"
    ) as f:

        f.write(
            "FLEXPART-WRF GLOBAL "
            "CONSTRAINED SMOOTHER\n"
        )

        f.write(
            "====================================\n"
        )

        f.write(
            f"Parameters = {N_PARAM}\n"
        )

        f.write(
            f"Observations = {n_obs}\n"
        )

        f.write(
            f"rho = {RHO}\n"
        )

        f.write(
            f"prior mean = {PRIOR_MEAN}\n"
        )

        f.write(
            f"prior sigma = {PRIOR_SIGMA}\n"
        )

        f.write(
            f"observation sigma = {OBS_SIGMA}\n"
        )

        f.write(
            f"background = {BACKGROUND_PPM}\n"
        )

        f.write(
            f"bounds = [{LOWER}, {UPPER}]\n"
        )

        f.write(
            f"iterations = {n_iterations}\n"
        )

        f.write(
            f"solution minimum = {np.min(x)}\n"
        )

        f.write(
            f"solution maximum = {np.max(x)}\n"
        )

        f.write(
            f"solution mean = {np.mean(x)}\n"
        )

        f.write(
            f"lower active = "
            f"{np.sum(active_lower)}\n"
        )

        f.write(
            f"upper active = "
            f"{np.sum(active_upper)}\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # H
    # --------------------------------------------------------

    (
        Hdf,
        H,
        parameter_columns
    ) = load_H()

    # --------------------------------------------------------
    # Observations
    # --------------------------------------------------------

    observations = load_observations()

    (
        matched,
        H_match,
        y
    ) = match_observations(
        Hdf,
        H,
        observations
    )

    # --------------------------------------------------------
    # Prior
    # --------------------------------------------------------

    B, L = build_prior()

    # --------------------------------------------------------
    # Posterior
    # --------------------------------------------------------

    (
        A,
        q,
        xb
    ) = build_posterior(
        H_match,
        y,
        B
    )

    # --------------------------------------------------------
    # Unconstrained solution
    # --------------------------------------------------------

    x_unbounded = solve_unconstrained(
        A,
        q
    )

    pd.DataFrame(
        {
            "date":
                np.repeat(
                    DATES,
                    N_MINE
                ),
            "mine":
                MINES * N_DAY,
            "unbounded":
                x_unbounded,
        }
    ).to_csv(
        os.path.join(
            OUTDIR,
            "unbounded_global_solution.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Global constrained solve
    # --------------------------------------------------------

    (
        x,
        active_lower,
        active_upper,
        g,
        n_iterations
    ) = solve_box_active_set(
        A,
        q
    )

    # --------------------------------------------------------
    # KKT
    # --------------------------------------------------------

    validate_kkt(
        x,
        g,
        active_lower,
        active_upper
    )

    # --------------------------------------------------------
    # Active parameters
    # --------------------------------------------------------

    print_active_parameters(
        x,
        g,
        active_lower,
        active_upper
    )

    # --------------------------------------------------------
    # Compare against validated batch MAP
    # --------------------------------------------------------

    compare_to_batch_map(
        x
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_solution(
        x,
        g,
        active_lower,
        active_upper,
        H_match.shape,
        len(y),
        n_iterations
    )

    # --------------------------------------------------------
    # Final daily result
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("FINAL DAILY GLOBAL CONSTRAINED SOLUTION")
    print("=" * 78)

    for d in range(
        N_DAY
    ):

        values = x[
            d * N_MINE:
            (d + 1) * N_MINE
        ]

        print(
            DATES[d].strftime("%Y-%m-%d"),
            f"mean={np.mean(values):.8f}",
            f"min={np.min(values):.8f}",
            f"max={np.max(values):.8f}",
        )

    print("\nOutput:")
    print(OUTDIR)

    print("\nFinished.")


if __name__ == "__main__":
    main()

