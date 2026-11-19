#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri


# =============================================================================
# SETTINGS
# =============================================================================

BASE = "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"

ANALYTICAL_DIR = os.path.join(BASE, "inversion_results/2026-06")

MEAN_FILE = os.path.join(
    ANALYTICAL_DIR,
    "analytical_posterior_scaling_factors.csv",
)

COV_FILE = os.path.join(
    ANALYTICAL_DIR,
    "analytical_posterior_covariance.npy",
)

MCMC_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "2026-06",
    "MCMC_13mine_daily_Jun2026",
    "FLEXPART_MCMC_posterior_daily_factors_Jun2026.csv",
)

OUTPUT_FILE = os.path.join(
    ANALYTICAL_DIR,
    "truncated_gaussian_analytical_posterior.csv",
)

N_CHAINS = 4

# Start with a shorter test run.
# Once validated, increase these.
BURN_IN = 2000
N_SAMPLES = 5000
THIN = 2

RANDOM_SEED = 24680


# =============================================================================
# FAST TRUNCATED NORMAL
# =============================================================================

def sample_truncnorm_zero(rng, mu, sigma):
    """
    Fast draw from N(mu, sigma^2) conditional on x >= 0.

    Uses inverse-CDF sampling.

    For very negative standardized means, the direct CDF interval can
    suffer numerical precision problems, so use an exponential-tail
    approximation in that regime.
    """

    if sigma <= 0.0:
        return max(mu, 0.0)

    alpha = -mu / sigma

    # Ordinary inverse-CDF region
    if alpha < 8.0:

        lower = ndtr(alpha)

        # U in [Phi(alpha), 1]
        u = rng.random()

        p = lower + u * (1.0 - lower)

        # Avoid exactly 1
        p = min(p, 1.0 - 1e-15)

        z = ndtri(p)

        return mu + sigma * z

    # Extreme lower-tail truncation.
    #
    # If alpha is large, the distribution is approximately
    # alpha + Exp(1/alpha).
    else:

        # Robert-style exponential proposal
        # for the standard normal tail.
        a = alpha

        while True:

            e = rng.exponential()

            z = a + e / a

            accept = np.exp(
                -0.5 * (z - a) ** 2
            )

            if rng.random() <= accept:
                return mu + sigma * z


# =============================================================================
# LOAD POSTERIOR
# =============================================================================

print("=" * 90)
print("FAST INDEPENDENT 390-DIMENSIONAL TRUNCATED GAUSSIAN")
print("=" * 90)

df = pd.read_csv(MEAN_FILE)

parameters = df["parameter"].astype(str).tolist()
mu = df["posterior_scaling"].to_numpy(float)

N_PARAMS = len(parameters)

print("\nParameters:", N_PARAMS)

if N_PARAMS != 390:
    raise RuntimeError("Expected 390 parameters.")


# =============================================================================
# LOAD COVARIANCE
# =============================================================================

Sigma = np.load(COV_FILE)
Sigma = 0.5 * (Sigma + Sigma.T)

print("Covariance:", Sigma.shape)

np.linalg.cholesky(Sigma)

print("Covariance positive definite: YES")


# =============================================================================
# PRECISION
# =============================================================================

print("\nComputing Q = Sigma^-1 ...")

Q = np.linalg.inv(Sigma)
Q = 0.5 * (Q + Q.T)

qdiag = np.diag(Q)

conditional_sd = 1.0 / np.sqrt(qdiag)

print(
    "Conditional SD:",
    conditional_sd.min(),
    "to",
    conditional_sd.max()
)


# =============================================================================
# INITIAL STATE
# =============================================================================

x_initial = np.maximum(mu, 1e-8)


# =============================================================================
# FAST GIBBS CHAIN
# =============================================================================

def run_chain(chain_id, seed):

    rng = np.random.default_rng(seed)

    x = x_initial.copy()

    # qdx = Q @ (x - mu)
    qdx = Q @ (x - mu)

    total_iterations = BURN_IN + N_SAMPLES * THIN

    samples = np.empty(
        (N_SAMPLES, N_PARAMS),
        dtype=np.float64
    )

    save_i = 0

    print(
        f"\nChain {chain_id}: "
        f"{total_iterations:,} iterations"
    )

    for iteration in range(total_iterations):

        for j in range(N_PARAMS):

            qjj = qdiag[j]

            old = x[j]

            # Q_j,-j (x_-j - mu_-j)
            q_without_j = qdx[j] - qjj * (
                old - mu[j]
            )

            cond_mu = mu[j] - q_without_j / qjj

            cond_sd = conditional_sd[j]

            new = sample_truncnorm_zero(
                rng,
                cond_mu,
                cond_sd
            )

            delta = new - old

            if delta != 0.0:

                x[j] = new

                qdx += Q[:, j] * delta

        if (
            iteration >= BURN_IN
            and
            (iteration - BURN_IN) % THIN == 0
        ):

            samples[save_i] = x
            save_i += 1

        if (
            (iteration + 1) % 500 == 0
            or
            iteration == total_iterations - 1
        ):

            print(
                f"  chain {chain_id}: "
                f"{iteration + 1:,}/"
                f"{total_iterations:,}",
                flush=True
            )

    return samples


# =============================================================================
# RUN CHAINS
# =============================================================================

chains = []

for c in range(N_CHAINS):

    chains.append(
        run_chain(
            c + 1,
            RANDOM_SEED + c
        )
    )


samples = np.concatenate(
    chains,
    axis=0
)

print("\nCombined samples:", samples.shape)


# =============================================================================
# SUMMARY
# =============================================================================

tg_mean = samples.mean(axis=0)
tg_median = np.median(samples, axis=0)
tg_sd = samples.std(axis=0, ddof=1)

tg_q025 = np.quantile(samples, 0.025, axis=0)
tg_q975 = np.quantile(samples, 0.975, axis=0)

tg_prob_lt_1 = np.mean(samples < 1.0, axis=0)
tg_prob_gt_1 = np.mean(samples > 1.0, axis=0)


# =============================================================================
# SAVE
# =============================================================================

result = df.copy()

result["truncated_posterior_mean"] = tg_mean
result["truncated_posterior_median"] = tg_median
result["truncated_posterior_sd"] = tg_sd
result["truncated_q025"] = tg_q025
result["truncated_q975"] = tg_q975
result["truncated_prob_lt_1"] = tg_prob_lt_1
result["truncated_prob_gt_1"] = tg_prob_gt_1

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)


# =============================================================================
# COMPARE WITH MCMC
# =============================================================================

if os.path.exists(MCMC_FILE):

    mcmc = pd.read_csv(MCMC_FILE)

    if "posterior_mean" in mcmc.columns:

        mcmc_map = dict(
            zip(
                mcmc["parameter"].astype(str),
                mcmc["posterior_mean"].astype(float)
            )
        )

        mcmc_mean = np.array([
            mcmc_map[p]
            for p in parameters
        ])

        diff = tg_mean - mcmc_mean

        print("\n" + "=" * 90)
        print("TRUNCATED ANALYTICAL vs FLEXPART MCMC")
        print("=" * 90)

        print("Parameters:", len(diff))
        print("Mean difference:", diff.mean())
        print("MAE:", np.mean(np.abs(diff)))
        print("RMSE:", np.sqrt(np.mean(diff ** 2)))
        print(
            "Correlation:",
            np.corrcoef(tg_mean, mcmc_mean)[0, 1]
        )


# =============================================================================
# SELECTED PARAMETERS
# =============================================================================

selected = [
    "OC1_20260605",
    "OC2_20260605",
    "OC3_20260605",
    "OC4_20260605",
    "OC5_20260605",
    "OC6_20260605",
    "OC7_20260605",
    "OC8_20260605",
    "OC9_20260605",
    "OC10_20260605",
    "OC11_20260605",
    "OC12_20260605",
    "OC13_20260605",
    "OC5_20260606",
    "OC5_20260607",
    "OC5_20260608",
]

lookup = {
    p: i
    for i, p in enumerate(parameters)
}

print("\n" + "=" * 90)
print("SELECTED PARAMETERS")
print("=" * 90)

for p in selected:

    i = lookup[p]

    print(
        f"{p:18s} "
        f"unconstrained={mu[i]:10.6f} "
        f"TG_mean={tg_mean[i]:10.6f} "
        f"TG_sd={tg_sd[i]:10.6f} "
        f"q025={tg_q025[i]:10.6f} "
        f"q975={tg_q975[i]:10.6f}"
    )

print("\nDONE.")


