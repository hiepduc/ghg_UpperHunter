#!/usr/bin/env python3
"""Projected constrained RTS approximation for Upper Hunter CH4 daily mine factors.

This is a recursive approximate smoother: standard RTS on the saved 26-D
augmented bounded-KF Gaussian summaries, followed by a box-constrained
Mahalanobis projection at every backward step. Hard bounds make the posterior
non-Gaussian, so this is not an exact truncated-Gaussian RTS smoother.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path.home() / "flexpart-wrf/upperhunter_flexpart/H_matrix_flexpart"
INPUT = ROOT / "KF_bounded_full_footprint_13mine_Jan2026/bounded_full_footprint_KF_daily_states.npz"
OUT = ROOT / "RTS_projected_bounded_full_footprint_13mine_Jan2026"
LOWER, UPPER = 0.0, 5.0
OUT.mkdir(parents=True, exist_ok=True)


def sym(a):
    return (a + a.T) * 0.5


def inv_spd(a):
    """Stable inverse for symmetric positive (semi-)definite matrices."""
    a = sym(np.asarray(a, float))
    try:
        return np.linalg.inv(a + np.eye(a.shape[0]) * 1e-10)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a, rcond=1e-12)


def bounded_projection(mean, covariance):
    """Box-constrained Mahalanobis MAP plus local active-set covariance."""
    mean, covariance = np.asarray(mean, float), sym(covariance)
    precision = inv_spd(covariance)

    def fun(x):
        d = x - mean
        return 0.5 * d @ precision @ d

    def jac(x):
        return precision @ (x - mean)

    res = minimize(fun, np.clip(mean, LOWER, UPPER), jac=jac,
                   method="L-BFGS-B",
                   bounds=[(LOWER, UPPER)] * len(mean),
                   options={"ftol": 1e-14, "gtol": 1e-9,
                            "maxiter": 3000, "maxls": 50})
    x = np.clip(res.x, LOWER, UPPER)
    lo, hi = x <= LOWER + 1e-7, x >= UPPER - 1e-7
    free = ~(lo | hi)
    cov = np.zeros_like(covariance)
    if free.any():
        cov[np.ix_(free, free)] = inv_spd(precision[np.ix_(free, free)])
    g = jac(x)
    diag = {
        "success": bool(res.success),
        "iterations": int(getattr(res, "nit", -1)),
        "lower_active": int(lo.sum()),
        "upper_active": int(hi.sum()),
        "max_free_gradient": float(np.max(np.abs(g[free]))) if free.any() else 0.0,
    }
    return x, sym(cov), diag


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing bounded-KF input: {INPUT}")
    z = np.load(INPUT, allow_pickle=True)
    xf = np.asarray(z["augmented_states"], float)
    Pf = np.asarray(z["augmented_covariances"], float)
    dates = pd.to_datetime(z["dates"])
    mines = [str(m) for m in z["mines"]]
    n = len(mines)
    nt, d = xf.shape
    if d != 2*n or Pf.shape != (nt, d, d):
        raise ValueError(f"Unexpected augmented state/covariance shapes: {xf.shape}, {Pf.shape}")

    prior_mean = float(np.asarray(z["prior_mean"]).squeeze())
    prior_sigma = float(np.asarray(z["prior_sigma"]).squeeze())
    corr_len = float(np.asarray(z["correlation_length"]).squeeze())
    rho = np.exp(-1.0 / corr_len)

    # Augmented state z_t=[x_(t-1), x_t], transition to [x_t, x_(t+1)].
    F = np.zeros((d, d))
    F[:n, n:] = np.eye(n)
    F[n:, n:] = rho * np.eye(n)
    offset = np.zeros(d)
    offset[n:] = (1.0-rho) * prior_mean

    B = np.eye(n) * prior_sigma**2
    Q = (1.0-rho**2) * B
    Qaug = np.zeros((d, d))
    Qaug[n:, n:] = Q

    xs, Ps = xf.copy(), Pf.copy()
    diagnostics = []

    for t in range(nt-2, -1, -1):
        P_f = sym(Pf[t])
        P_pred = sym(F @ P_f @ F.T + Qaug)
        x_pred = F @ xf[t] + offset
        C = P_f @ F.T @ inv_spd(P_pred)
        x_raw = xf[t] + C @ (xs[t+1] - x_pred)
        P_raw = sym(P_f + C @ (Ps[t+1] - P_pred) @ C.T)
        # Repair tiny negative eigenvalues from floating-point roundoff.
        eigmin = np.linalg.eigvalsh(P_raw).min()
        if eigmin < 1e-10:
            P_raw += np.eye(d) * (1e-10 - eigmin)
        xs[t], Ps[t], diag = bounded_projection(x_raw, P_raw)
        diagnostics.append({"date": dates[t].strftime("%Y-%m-%d"), **diag})

    # Current-day component is the second half of the augmented state.
    daily = np.clip(xs[:, n:], LOWER, UPPER)
    wide = pd.DataFrame(daily, columns=mines)
    wide.insert(0, "date", dates.strftime("%Y-%m-%d"))
    wide.to_csv(OUT / "daily_projected_bounded_RTS_scaling_factors.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "backward_projection_diagnostics.csv", index=False)

    cov_rows = []
    for t, date in enumerate(dates):
        for i, mi in enumerate(mines):
            for j, mj in enumerate(mines):
                cov_rows.append({"date": date.strftime("%Y-%m-%d"),
                                 "mine_i": mi, "mine_j": mj,
                                 "covariance": Ps[t, n+i, n+j]})
    pd.DataFrame(cov_rows).to_csv(OUT / "daily_projected_bounded_RTS_covariances.csv", index=False)

    summary = pd.DataFrame({
        "mine": mines,
        "mean_scaling_factor": daily.mean(axis=0),
        "median_scaling_factor": np.median(daily, axis=0),
        "min_scaling_factor": daily.min(axis=0),
        "max_scaling_factor": daily.max(axis=0),
        "final_day_scaling_factor": daily[-1],
    })
    summary.to_csv(OUT / "mine_summary_projected_bounded_RTS.csv", index=False)

    report = [
        "Projected constrained RTS summary",
        f"Input: {INPUT}",
        f"Output: {OUT}",
        f"Dates: {dates[0].date()} to {dates[-1].date()} ({nt} days)",
        f"Mines: {n}; augmented state dimension: {d}",
        f"Prior mean={prior_mean}, sigma={prior_sigma}, correlation length={corr_len}",
        f"rho={rho:.15g}; bounds=[{LOWER}, {UPPER}]",
        f"Scaling-factor min={daily.min():.12g}",
        f"Scaling-factor max={daily.max():.12g}",
        f"Scaling-factor mean={daily.mean():.12g}",
        f"Daily parameters at lower bound={int((daily <= LOWER+1e-7).sum())}",
        f"Daily parameters at upper bound={int((daily >= UPPER-1e-7).sum())}",
        "",
        "LIMITATION: This is a recursive projected/active-set RTS approximation.",
        "Hard bounds make the posterior non-Gaussian; this is not an exact",
        "truncated-Gaussian smoother. Validate against the global constrained",
        "batch MAP reference and report any remaining differences.",
    ]
    (OUT / "projected_bounded_RTS_summary.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
