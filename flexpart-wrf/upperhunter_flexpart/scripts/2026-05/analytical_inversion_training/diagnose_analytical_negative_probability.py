import numpy as np
import pandas as pd
from scipy.special import ndtr

POSTERIOR_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/"
    "inversion_results/2026-05/analytical_posterior_scaling_factors.csv"
)

df = pd.read_csv(POSTERIOR_FILE)

print("Columns:")
print(df.columns.tolist())

# Actual analytical output column names
mu = df["posterior_scaling"].to_numpy(dtype=float)
sd = df["posterior_sigma"].to_numpy(dtype=float)

# Marginal probability that unconstrained Gaussian posterior
# produces a negative scaling factor:
#
# P(s < 0) = Phi((0 - mu) / sigma)
#
p_negative = ndtr(-mu / sd)

df["prob_negative"] = p_negative
df["prob_nonnegative"] = 1.0 - p_negative

print()
print("=" * 100)
print("ANALYTICAL POSTERIOR: EFFECT OF s >= 0 CONSTRAINT")
print("=" * 100)

print()
print(f"Parameters: {len(df)}")

print(
    f"Parameters with P(s < 0) > 1% : "
    f"{np.sum(p_negative > 0.01)}"
)

print(
    f"Parameters with P(s < 0) > 5% : "
    f"{np.sum(p_negative > 0.05)}"
)

print(
    f"Parameters with P(s < 0) > 10%: "
    f"{np.sum(p_negative > 0.10)}"
)

print(
    f"Parameters with P(s < 0) > 25%: "
    f"{np.sum(p_negative > 0.25)}"
)

print(
    f"Parameters with P(s < 0) > 50%: "
    f"{np.sum(p_negative > 0.50)}"
)

print()
print(
    f"Sum of marginal P(s < 0): "
    f"{np.sum(p_negative):.6f}"
)

print()
print(
    f"Mean marginal P(s < 0): "
    f"{np.mean(p_negative):.6f}"
)

print()
print("Largest negative probabilities:")
print()

cols = [
    "parameter",
    "mine",
    "date",
    "posterior_scaling",
    "posterior_sigma",
    "prob_negative",
    "prob_nonnegative",
]

print(
    df.sort_values(
        "prob_negative",
        ascending=False
    )[cols]
    .head(30)
    .to_string(index=False)
)

# ============================================================================
# MINE-LEVEL SUMMARY
# ============================================================================

mine_summary = (
    df.groupby("mine")
    .agg(
        parameters=("parameter", "size"),
        mean_scaling=("posterior_scaling", "mean"),
        mean_sigma=("posterior_sigma", "mean"),
        mean_prob_negative=("prob_negative", "mean"),
        max_prob_negative=("prob_negative", "max"),
        n_prob_negative_gt10pct=(
            "prob_negative",
            lambda x: np.sum(x > 0.10)
        ),
    )
)

print()
print("=" * 100)
print("MINE-LEVEL SUMMARY")
print("=" * 100)

print(
    mine_summary.to_string()
)

