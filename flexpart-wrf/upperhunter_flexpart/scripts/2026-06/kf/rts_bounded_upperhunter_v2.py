#!/usr/bin/env python3

"""
FLEXPART-WRF Bayesian Bounded RTS Smoother
===========================================

IMPORTANT DESIGN
----------------
The forward bounded Kalman filter is NOT reimplemented here.

Instead, this script uses the already validated bounded-KF results:

    KF_bounded_full_footprint_13mine_Jan2026/

This ensures that the forward filter used by the RTS smoother is
identical to the validated bounded KF.

The script then performs a backward constrained smoothing recursion.

State:
    x_t = 13 daily mine scaling factors

Transition:
    x_{t+1} = rho*x_t + (1-rho)*prior_mean + w_t

with:
    rho = exp(-1/3)
    Q   = sigma_prior^2 * (1-rho^2) I

Bounds:
    0 <= x_i <= 5

The backward step solves the constrained conditional MAP:

    min_x
       0.5 (x-x_f)^T P_f^{-1} (x-x_f)
       +
       0.5 (x_next_s-rho*x-offset)^T Q^{-1}
             (x_next_s-rho*x-offset)

subject to:
       0 <= x <= 5

This is a constrained Gaussian conditional MAP / active-set
approximation to the RTS smoother.

It is NOT claimed to be an exact truncated-Gaussian RTS covariance.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.linalg import solve


# ============================================================
# CONFIGURATION
# ============================================================

BASE = (
    "/home/duch/flexpart-wrf/"
    "upperhunter_flexpart/H_matrix_flexpart"
)

KF_DIR = os.path.join(
    BASE,
    "KF_bounded_full_footprint_13mine_Jan2026"
)

KF_NPZ = os.path.join(
    KF_DIR,
    "bounded_full_footprint_KF_daily_states.npz"
)

# Some versions of the bounded KF may use another NPZ name.
# The script will search these alternatives if the main file
# does not exist.
KF_NPZ_ALTERNATIVES = [
    os.path.join(KF_DIR, "KF_bounded_results.npz"),
    os.path.join(KF_DIR, "bounded_KF_results.npz"),
    os.path.join(KF_DIR, "KF_results.npz"),
]

KF_DAILY_CSV = os.path.join(
    KF_DIR,
    "daily_bounded_full_footprint_KF_scaling_factors.csv"
)

OUTDIR = os.path.join(
    BASE,
    "RTS_bounded_full_footprint_13mine_Jan2026_v2"
)

os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# BAYESIAN SETTINGS
# ============================================================

N_MINE = 13
N_DAY = 27
N_STATE = 13

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5

CORR_LENGTH_DAYS = 3.0

RHO = np.exp(-1.0 / CORR_LENGTH_DAYS)

Q_SCALAR = (
    PRIOR_SIGMA ** 2
    * (1.0 - RHO ** 2)
)

LOWER = 0.0
UPPER = 5.0

DATES = pd.date_range(
    "2026-01-05",
    "2026-01-31",
    freq="D"
)

MINES = [f"OC{i}" for i in range(1, 14)]


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

KKT_TOL = 1.0e-7
BOUND_TOL = 1.0e-7

MAXITER = 2000


# ============================================================
# UTILITY
# ============================================================

def find_kf_npz():

    if os.path.exists(KF_NPZ):
        return KF_NPZ

    for f in KF_NPZ_ALTERNATIVES:
        if os.path.exists(f):
            return f

    raise FileNotFoundError(
        "\nCould not find validated bounded-KF NPZ.\n"
        f"Expected directory:\n{KF_DIR}\n\n"
        "Files found there:\n"
        + "\n".join(
            os.path.basename(x)
            for x in sorted(os.listdir(KF_DIR))
        )
        + "\n\n"
        "Please check the NPZ filename produced by "
        "run_kf_upperhunter_full_footprint_bounded.py."
    )


# ============================================================
# LOAD VALIDATED FORWARD KF
# ============================================================

def load_validated_kf():

    npz_file = find_kf_npz()

    print("=" * 78)
    print("LOADING VALIDATED BOUNDED KALMAN FILTER")
    print("=" * 78)
    print("NPZ:")
    print(npz_file)

    z = np.load(
        npz_file,
        allow_pickle=True
    )

    print("\nKeys:")
    print(z.files)

    required = [
        "states",
        "covariances",
        "augmented_states",
        "augmented_covariances",
        "dates",
        "mines",
    ]

    missing = [
        k for k in required
        if k not in z.files
    ]

    if missing:
        raise KeyError(
            "Validated KF NPZ is missing required arrays: "
            + str(missing)
        )

    states = np.asarray(
        z["states"],
        dtype=float
    )

    covariances = np.asarray(
        z["covariances"],
        dtype=float
    )

    augmented_states = np.asarray(
        z["augmented_states"],
        dtype=float
    )

    augmented_covariances = np.asarray(
        z["augmented_covariances"],
        dtype=float
    )

    dates = pd.to_datetime(
        z["dates"]
    )

    mines = [
        str(x)
        for x in z["mines"]
    ]

    print("\nShapes:")
    print(
        f"states                 : {states.shape}"
    )
    print(
        f"covariances            : {covariances.shape}"
    )
    print(
        f"augmented_states       : {augmented_states.shape}"
    )
    print(
        f"augmented_covariances  : "
        f"{augmented_covariances.shape}"
    )

    # --------------------------------------------------------
    # HARD VALIDATION
    # --------------------------------------------------------

    if states.shape != (N_DAY, N_STATE):
        raise ValueError(
            f"Unexpected states shape {states.shape}; "
            f"expected {(N_DAY, N_STATE)}"
        )

    if covariances.shape != (
        N_DAY,
        N_STATE,
        N_STATE
    ):
        raise ValueError(
            f"Unexpected covariance shape "
            f"{covariances.shape}"
        )

    if augmented_states.shape != (
        N_DAY,
        2 * N_STATE
    ):
        raise ValueError(
            f"Unexpected augmented state shape "
            f"{augmented_states.shape}"
        )

    if augmented_covariances.shape != (
        N_DAY,
        2 * N_STATE,
        2 * N_STATE
    ):
        raise ValueError(
            f"Unexpected augmented covariance shape "
            f"{augmented_covariances.shape}"
        )

    if len(dates) != N_DAY:
        raise ValueError(
            "Unexpected number of dates."
        )

    if len(mines) != N_STATE:
        raise ValueError(
            "Unexpected number of mines."
        )

    print("\nDates:")
    print(
        dates[0].strftime("%Y-%m-%d"),
        "to",
        dates[-1].strftime("%Y-%m-%d")
    )

    print("\nMines:")
    print(mines)

    print("\nForward KF state range:")
    print("minimum =", np.min(states))
    print("maximum =", np.max(states))

    # --------------------------------------------------------
    # SYMMETRISE COVARIANCES
    # --------------------------------------------------------

    covariances = 0.5 * (
        covariances
        + np.transpose(
            covariances,
            (0, 2, 1)
        )
    )

    augmented_covariances = 0.5 * (
        augmented_covariances
        + np.transpose(
            augmented_covariances,
            (0, 2, 1)
        )
    )

    return (
        states,
        covariances,
        augmented_states,
        augmented_covariances,
        dates,
        mines,
        npz_file,
    )


# ============================================================
# CHECK FORWARD KF BOUNDS
# ============================================================

def validate_forward_states(states):

    print("\n" + "=" * 78)
    print("FORWARD KF VALIDATION")
    print("=" * 78)

    min_state = np.min(states)
    max_state = np.max(states)

    print(
        f"Minimum filtered scaling factor: "
        f"{min_state:.12g}"
    )

    print(
        f"Maximum filtered scaling factor: "
        f"{max_state:.12g}"
    )

    below = np.sum(
        states < LOWER - BOUND_TOL
    )

    above = np.sum(
        states > UPPER + BOUND_TOL
    )

    print(
        "Below lower bound:",
        below
    )

    print(
        "Above upper bound:",
        above
    )

    if below or above:
        raise ValueError(
            "The supplied bounded KF contains "
            "states outside [0,5]."
        )

    print(
        "\nForward bounded KF passed "
        "the [0,5] state check."
    )


# ============================================================
# BUILD TRANSITION
# ============================================================

def transition_offset():

    return (
        (1.0 - RHO)
        * PRIOR_MEAN
    )


# ============================================================
# SAFE INVERSE / PSEUDOINVERSE
# ============================================================

def precision_matrix(P):

    """
    Return a numerically stable precision matrix.

    The validated bounded KF can have very small covariance
    eigenvalues when a state component has been constrained.
    Therefore use an eigenvalue floor rather than np.linalg.inv.
    """

    P = 0.5 * (P + P.T)

    eigval, eigvec = np.linalg.eigh(P)

    scale = max(
        1.0,
        np.max(np.abs(eigval))
    )

    floor = 1.0e-10 * scale

    eigval = np.maximum(
        eigval,
        floor
    )

    return (
        eigvec
        @ np.diag(1.0 / eigval)
        @ eigvec.T
    )


# ============================================================
# CONSTRAINED BACKWARD STEP
# ============================================================

def constrained_backward_step(
    x_filt,
    P_filt,
    x_next_smooth,
):
    """
    Solve

       min_x 0.5 (x-xf)' Pf^-1 (x-xf)
            + 0.5 (xs-rho*x-offset)' Q^-1
                  (xs-rho*x-offset)

    subject to 0 <= x <= 5.

    Returns:
        x_smooth
        covariance approximation
        active_lower
        active_upper
        gradient
    """

    Pf_inv = precision_matrix(
        P_filt
    )

    Qinv = (
        np.eye(N_STATE)
        / Q_SCALAR
    )

    offset = transition_offset()

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(x):

        d1 = x - x_filt

        d2 = (
            x_next_smooth
            - RHO * x
            - offset
        )

        return 0.5 * (
            d1 @ Pf_inv @ d1
            +
            d2 @ Qinv @ d2
        )

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    def gradient(x):

        d1 = x - x_filt

        d2 = (
            x_next_smooth
            - RHO * x
            - offset
        )

        return (
            Pf_inv @ d1
            -
            RHO * (Qinv @ d2)
        )

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    x0 = np.clip(
        x_filt,
        LOWER,
        UPPER
    )

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[
            (LOWER, UPPER)
            for _ in range(N_STATE)
        ],
        options={
            "ftol": 1.0e-14,
            "gtol": 1.0e-10,
            "maxiter": MAXITER,
            "maxls": 50,
        },
    )

    if not result.success:
        warnings.warn(
            "Backward constrained optimization did not "
            f"report success: {result.message}"
        )

    x = np.clip(
        result.x,
        LOWER,
        UPPER
    )

    g = gradient(x)

    active_lower = (
        x <= LOWER + BOUND_TOL
    )

    active_upper = (
        x >= UPPER - BOUND_TOL
    )

    # --------------------------------------------------------
    # KKT check
    # --------------------------------------------------------

    free = (
        ~active_lower
        &
        ~active_upper
    )

    max_free_gradient = (
        np.max(
            np.abs(g[free])
        )
        if np.any(free)
        else 0.0
    )

    lower_violation = np.max(
        np.maximum(
            -g[active_lower],
            0.0
        )
    ) if np.any(active_lower) else 0.0

    upper_violation = np.max(
        np.maximum(
            g[active_upper],
            0.0
        )
    ) if np.any(active_upper) else 0.0

    if (
        max_free_gradient > KKT_TOL
        or lower_violation > KKT_TOL
        or upper_violation > KKT_TOL
    ):
        warnings.warn(
            "Backward KKT tolerance exceeded: "
            f"free={max_free_gradient:.3e}, "
            f"lower={lower_violation:.3e}, "
            f"upper={upper_violation:.3e}"
        )

    # --------------------------------------------------------
    # Conditional covariance approximation
    #
    # H = Pf^-1 + rho^2 Q^-1
    #
    # For active variables we use zero variance.
    # For free variables use inverse free-free Hessian.
    # --------------------------------------------------------

    Hessian = (
        Pf_inv
        +
        (RHO ** 2) * Qinv
    )

    covariance = np.zeros(
        (N_STATE, N_STATE)
    )

    free_idx = np.where(
        free
    )[0]

    if len(free_idx) > 0:

        Hff = np.asarray(
            Hessian[
                np.ix_(
                    free_idx,
                    free_idx
                )
            ]
        )

        Hff = 0.5 * (
            Hff + Hff.T
        )

        try:

            Cff = np.linalg.inv(
                Hff
            )

        except np.linalg.LinAlgError:

            Cff = np.linalg.pinv(
                Hff,
                rcond=1.0e-10
            )

        covariance[
            np.ix_(
                free_idx,
                free_idx
            )
        ] = 0.5 * (
            Cff + Cff.T
        )

    return (
        x,
        covariance,
        active_lower,
        active_upper,
        g,
        result,
    )


# ============================================================
# RUN CONSTRAINED RTS
# ============================================================

def run_rts(
    filtered_states,
    filtered_covariances,
    dates,
):

    print("\n" + "=" * 78)
    print("BACKWARD CONSTRAINED RTS")
    print("=" * 78)

    smoothed_states = np.zeros_like(
        filtered_states
    )

    smoothed_covariances = np.zeros_like(
        filtered_covariances
    )

    active_lower_all = np.zeros(
        (N_DAY, N_STATE),
        dtype=bool
    )

    active_upper_all = np.zeros(
        (N_DAY, N_STATE),
        dtype=bool
    )

    gradients_all = np.zeros_like(
        filtered_states
    )

    # --------------------------------------------------------
    # FINAL DAY
    #
    # There are no future observations after Jan 31, so
    # smoothing equals filtering.
    # --------------------------------------------------------

    smoothed_states[-1] = (
        filtered_states[-1]
    )

    smoothed_covariances[-1] = (
        filtered_covariances[-1]
    )

    # --------------------------------------------------------
    # BACKWARD RECURSION
    # --------------------------------------------------------

    for t in range(
        N_DAY - 2,
        -1,
        -1
    ):

        (
            xs,
            Ps,
            lower,
            upper,
            grad,
            result,
        ) = constrained_backward_step(
            filtered_states[t],
            filtered_covariances[t],
            smoothed_states[t + 1],
        )

        smoothed_states[t] = xs
        smoothed_covariances[t] = Ps

        active_lower_all[t] = lower
        active_upper_all[t] = upper
        gradients_all[t] = grad

        free_mask = (
            ~lower
            & ~upper
        )

        if np.any(free_mask):
            kkt_value = np.max(
                np.abs(grad[free_mask])
            )
        else:
            kkt_value = 0.0

        print(
            dates[t].strftime("%Y-%m-%d"),
            f"lower={np.sum(lower):2d}",
            f"upper={np.sum(upper):2d}",
            f"min={np.min(xs):.6f}",
            f"max={np.max(xs):.6f}",
            f"KKT={kkt_value:.3e}"
        )

    return (
        smoothed_states,
        smoothed_covariances,
        active_lower_all,
        active_upper_all,
        gradients_all,
    )


# ============================================================
# LOAD BATCH MAP FOR VALIDATION
# ============================================================

def load_batch_map():

    batch_dir = os.path.join(
        BASE,
        "Bounded_batch_MAP_full_footprint_13mine_Jan2026"
    )

    csv_file = os.path.join(
        batch_dir,
        "daily_bounded_batch_MAP_scaling_factors.csv"
    )

    if not os.path.exists(csv_file):

        print(
            "\nBatch MAP file not found:"
        )
        print(csv_file)

        return None

    df = pd.read_csv(
        csv_file
    )

    # Expected layout:
    # date, OC1, ..., OC13

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"]
        )

    return df


# ============================================================
# COMPARISON
# ============================================================

def compare_to_batch_map(
    smoothed_states,
    dates,
    mines,
):

    batch = load_batch_map()

    if batch is None:
        return

    map_values = np.zeros(
        (N_DAY, N_STATE)
    )

    # --------------------------------------------------------
    # Flexible date handling
    # --------------------------------------------------------

    if "date" in batch.columns:

        batch = batch.set_index(
            "date"
        )

        for t, d in enumerate(dates):

            if d not in batch.index:

                raise ValueError(
                    f"Batch MAP missing date {d}"
                )

            for j, mine in enumerate(mines):

                if mine not in batch.columns:

                    raise ValueError(
                        f"Batch MAP missing column "
                        f"{mine}"
                    )

                map_values[t, j] = (
                    batch.loc[d, mine]
                )

    else:

        raise ValueError(
            "Could not identify date column "
            "in batch MAP CSV."
        )

    diff = (
        smoothed_states
        -
        map_values
    )

    print("\n" + "=" * 78)
    print("CONSTRAINED RTS VS BOUNDED BATCH MAP")
    print("=" * 78)

    print(
        "Number of parameters:",
        diff.size
    )

    print(
        "Max absolute difference:",
        np.max(np.abs(diff))
    )

    print(
        "RMSE:",
        np.sqrt(
            np.mean(diff ** 2)
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
            smoothed_states.ravel(),
            map_values.ravel()
        )[0, 1]
    )

    print(
        "|difference| > 0.1:",
        np.sum(
            np.abs(diff) > 0.1
        ),
        "/",
        diff.size
    )

    # --------------------------------------------------------
    # Daily comparison
    # --------------------------------------------------------

    print("\nDAILY COMPARISON")
    print(
        f"{'date':12s}"
        f"{'RTS_mean':>12s}"
        f"{'MAP_mean':>12s}"
        f"{'diff_mean':>12s}"
        f"{'RMSE':>12s}"
        f"{'MAE':>12s}"
        f"{'max_abs':>12s}"
    )

    rows = []

    for t, d in enumerate(dates):

        d0 = diff[t]

        row = {
            "date": d,
            "RTS_mean":
                np.mean(
                    smoothed_states[t]
                ),
            "MAP_mean":
                np.mean(
                    map_values[t]
                ),
            "difference_mean":
                np.mean(d0),
            "RMSE":
                np.sqrt(
                    np.mean(
                        d0 ** 2
                    )
                ),
            "MAE":
                np.mean(
                    np.abs(d0)
                ),
            "max_abs_difference":
                np.max(
                    np.abs(d0)
                ),
        }

        rows.append(row)

        print(
            f"{d.strftime('%Y-%m-%d'):12s}"
            f"{row['RTS_mean']:12.6f}"
            f"{row['MAP_mean']:12.6f}"
            f"{row['difference_mean']:12.6f}"
            f"{row['RMSE']:12.6f}"
            f"{row['MAE']:12.6f}"
            f"{row['max_abs_difference']:12.6f}"
        )

    daily_df = pd.DataFrame(
        rows
    )

    daily_df.to_csv(
        os.path.join(
            OUTDIR,
            "daily_RTS_vs_bounded_batch_MAP.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Parameter comparison
    # --------------------------------------------------------

    parameter_rows = []

    for t, d in enumerate(dates):

        for j, mine in enumerate(mines):

            parameter_rows.append(
                {
                    "date": d,
                    "mine": mine,
                    "RTS": smoothed_states[t, j],
                    "batch_MAP": map_values[t, j],
                    "difference":
                        diff[t, j],
                    "abs_difference":
                        abs(diff[t, j]),
                }
            )

    parameter_df = pd.DataFrame(
        parameter_rows
    )

    parameter_df.to_csv(
        os.path.join(
            OUTDIR,
            "RTS_vs_bounded_batch_MAP_parameters.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Active constraints in batch comparison
    # --------------------------------------------------------

    active_rows = []

    for t, d in enumerate(dates):

        for j, mine in enumerate(mines):

            x = smoothed_states[t, j]
            xb = map_values[t, j]

            if (
                xb <= LOWER + BOUND_TOL
                or
                xb >= UPPER - BOUND_TOL
            ):

                active_rows.append(
                    {
                        "date": d,
                        "mine": mine,
                        "RTS": x,
                        "batch_MAP": xb,
                        "difference": x - xb,
                        "batch_MAP_lower":
                            xb <= LOWER + BOUND_TOL,
                        "batch_MAP_upper":
                            xb >= UPPER - BOUND_TOL,
                    }
                )

    pd.DataFrame(
        active_rows
    ).to_csv(
        os.path.join(
            OUTDIR,
            "batch_MAP_active_constraints_vs_RTS.csv"
        ),
        index=False
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    states,
    covariances,
    dates,
    mines,
    active_lower,
    active_upper,
    gradients,
    kf_npz,
):

    print("\n" + "=" * 78)
    print("SAVING CONSTRAINED RTS RESULTS")
    print("=" * 78)

    np.savez_compressed(
        os.path.join(
            OUTDIR,
            "RTS_bounded_results.npz"
        ),
        states=states,
        covariances=covariances,
        dates=np.array(
            dates.astype(str)
        ),
        mines=np.array(
            mines
        ),
        active_lower=active_lower,
        active_upper=active_upper,
        gradients=gradients,
        rho=RHO,
        q_scalar=Q_SCALAR,
        prior_mean=PRIOR_MEAN,
        prior_sigma=PRIOR_SIGMA,
        lower=LOWER,
        upper=UPPER,
        source_KF_npz=kf_npz,
    )

    # --------------------------------------------------------
    # Daily CSV
    # --------------------------------------------------------

    daily = pd.DataFrame(
        states,
        columns=mines
    )

    daily.insert(
        0,
        "date",
        dates
    )

    daily.to_csv(
        os.path.join(
            OUTDIR,
            "daily_bounded_RTS_scaling_factors.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Long parameter CSV
    # --------------------------------------------------------

    rows = []

    for t, d in enumerate(dates):

        for j, mine in enumerate(mines):

            rows.append(
                {
                    "date": d,
                    "mine": mine,
                    "scaling_factor":
                        states[t, j],
                    "active_lower":
                        active_lower[t, j],
                    "active_upper":
                        active_upper[t, j],
                    "gradient":
                        gradients[t, j],
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        os.path.join(
            OUTDIR,
            "bounded_RTS_parameters.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Covariance per day
    # --------------------------------------------------------

    for t, d in enumerate(dates):

        pd.DataFrame(
            covariances[t],
            index=mines,
            columns=mines,
        ).to_csv(
            os.path.join(
                OUTDIR,
                "covariance_"
                + d.strftime("%Y%m%d")
                + ".csv"
            )
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with open(
        os.path.join(
            OUTDIR,
            "RTS_bounded_summary.txt"
        ),
        "w"
    ) as f:

        f.write(
            "FLEXPART-WRF BAYESIAN "
            "BOUNDED RTS SMOOTHER\n"
        )

        f.write(
            "====================================\n"
        )

        f.write(
            f"Prior mean      = {PRIOR_MEAN}\n"
        )

        f.write(
            f"Prior sigma     = {PRIOR_SIGMA}\n"
        )

        f.write(
            f"Correlation     = {RHO}\n"
        )

        f.write(
            f"Q scalar        = {Q_SCALAR}\n"
        )

        f.write(
            f"Lower bound     = {LOWER}\n"
        )

        f.write(
            f"Upper bound     = {UPPER}\n"
        )

        f.write(
            f"Days            = {N_DAY}\n"
        )

        f.write(
            f"Mines           = {N_STATE}\n"
        )

        f.write(
            f"Forward KF NPZ  = {kf_npz}\n"
        )

        f.write(
            "\nRTS minimum = "
            f"{np.min(states):.12g}\n"
        )

        f.write(
            "RTS maximum = "
            f"{np.max(states):.12g}\n"
        )

        f.write(
            "RTS mean = "
            f"{np.mean(states):.12g}\n"
        )

        f.write(
            "\nActive lower constraints = "
            f"{np.sum(active_lower)}\n"
        )

        f.write(
            "Active upper constraints = "
            f"{np.sum(active_upper)}\n"
        )

    print(
        "\nOutput directory:"
    )

    print(OUTDIR)


# ============================================================
# MAIN
# ============================================================

def main():

    (
        filtered_states,
        filtered_covariances,
        augmented_states,
        augmented_covariances,
        dates,
        mines,
        kf_npz,
    ) = load_validated_kf()

    validate_forward_states(
        filtered_states
    )

    # --------------------------------------------------------
    # Confirm that the dates/mines are exactly the expected
    # Upper Hunter January 2026 experiment.
    # --------------------------------------------------------

    expected_dates = pd.date_range(
        "2026-01-05",
        "2026-01-31",
        freq="D"
    )

    if not dates.equals(
        expected_dates
    ):

        raise ValueError(
            "Validated KF dates do not match "
            "2026-01-05 through 2026-01-31."
        )

    expected_mines = [
        f"OC{i}"
        for i in range(1, 14)
    ]

    if mines != expected_mines:

        raise ValueError(
            "Validated KF mine ordering is not "
            "OC1 ... OC13.\n"
            f"Found: {mines}"
        )

    # --------------------------------------------------------
    # Forward covariance diagnostics
    # --------------------------------------------------------

    print("\nForward covariance diagnostics:")

    for t in [0, 1, 14, 15, 20, 26]:

        eig = np.linalg.eigvalsh(
            filtered_covariances[t]
        )

        print(
            dates[t].strftime("%Y-%m-%d"),
            "min eigenvalue =",
            f"{np.min(eig):.6e}",
            "max eigenvalue =",
            f"{np.max(eig):.6e}",
        )

    # --------------------------------------------------------
    # Run backward smoother
    # --------------------------------------------------------

    (
        smoothed_states,
        smoothed_covariances,
        active_lower,
        active_upper,
        gradients,
    ) = run_rts(
        filtered_states,
        filtered_covariances,
        dates,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("CONSTRAINED RTS SUMMARY")
    print("=" * 78)

    print(
        "Smoothed minimum:",
        np.min(smoothed_states)
    )

    print(
        "Smoothed maximum:",
        np.max(smoothed_states)
    )

    print(
        "Smoothed mean:",
        np.mean(smoothed_states)
    )

    print(
        "Final-day smoothed mean:",
        np.mean(smoothed_states[-1])
    )

    print(
        "Total smoothed lower activations:",
        np.sum(active_lower)
    )

    print(
        "Total smoothed upper activations:",
        np.sum(active_upper)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        smoothed_states,
        smoothed_covariances,
        dates,
        mines,
        active_lower,
        active_upper,
        gradients,
        kf_npz,
    )

    # --------------------------------------------------------
    # Compare to batch MAP
    # --------------------------------------------------------

    compare_to_batch_map(
        smoothed_states,
        dates,
        mines,
    )

    # --------------------------------------------------------
    # Final day
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("FINAL-DAY CONSTRAINED RTS")
    print("=" * 78)

    for j, mine in enumerate(mines):

        print(
            f"{mine:5s} "
            f"{smoothed_states[-1,j]:12.8f}"
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()

