import numpy as np
import pandas as pd
import os

# ============================================================================
# FILES
# ============================================================================

H_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/2026-07/H_flexpart_0_10m_13mine_Jul2026.csv"
)

PICARRO_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

ANALYTICAL_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/"
    "inversion_results/2026-07/analytical_posterior_scaling_factors.csv"
)

BACKGROUND = 1.85
SIGMA_OBS = 0.05
SIGMA_PRIOR = 1.0
PRIOR_MEAN = 1.0

# ============================================================================
# PARAMETER NAMES
# ============================================================================

parameter_names = [
    f"OC{mine}_{date.strftime('%Y%m%d')}"
    for date in pd.date_range("2026-07-01", "2026-07-31")
    for mine in range(1, 14)
]

print("=" * 80)
print("INDEPENDENT ANALYTICAL POSTERIOR CHECK")
print("=" * 80)

# ============================================================================
# LOAD H
# ============================================================================

H_df = pd.read_csv(H_FILE)

metadata_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

H = H_df[parameter_names].to_numpy(dtype=np.float64)

print("\nH:")
print("  full shape :", H_df.shape)
print("  used shape :", H.shape)

# ============================================================================
# NORMALISE RECEPTORS
# ============================================================================

def normalise_receptor(x):
    x = str(x).strip().lower()

    if "singleton" in x:
        return "singleton"

    if "muswellbrook" in x:
        return "muswellbrook"

    if "merriwa" in x:
        return "merriwa"

    return x


H_work = H_df.copy()

H_work["receptor_normalised"] = (
    H_work["receptor"]
    .map(normalise_receptor)
)

H_work["datetime_aest"] = pd.to_datetime(
    H_work["obs_time_aest"],
    errors="coerce",
)

# ============================================================================
# LOAD PICARRO
# ============================================================================

obs = pd.read_csv(PICARRO_FILE)

time_candidates = [
    c for c in obs.columns
    if "time" in c.lower()
    or "date" in c.lower()
    or "datetime" in c.lower()
]

print("\nPossible datetime columns:")
print(time_candidates)

# Use the same full datetime column identified previously.
time_col = "Datetime"

print("Using:", time_col)

obs["datetime_aest"] = pd.to_datetime(
    obs[time_col],
    errors="coerce",
)

# ============================================================================
# BUILD LONG OBSERVATION TABLE
# ============================================================================

receptor_columns = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}

obs_parts = []

for receptor, column in receptor_columns.items():

    tmp = obs[
        ["datetime_aest", column]
    ].copy()

    tmp["receptor_normalised"] = receptor
    tmp["obs_ppm"] = pd.to_numeric(
        tmp[column],
        errors="coerce",
    )

    tmp = tmp[
        [
            "receptor_normalised",
            "datetime_aest",
            "obs_ppm",
        ]
    ]

    obs_parts.append(tmp)

obs_long = pd.concat(
    obs_parts,
    ignore_index=True,
)

obs_long = obs_long[
    obs_long["obs_ppm"].notna()
].copy()

# ============================================================================
# EXACT MATCH
# ============================================================================

matched = H_work.merge(
    obs_long,
    on=[
        "receptor_normalised",
        "datetime_aest",
    ],
    how="left",
)

matched = matched[
    matched["obs_ppm"].notna()
].copy()

print("\nMatching:")
print("  H rows       :", len(H_work))
print("  matched rows :", len(matched))

# ============================================================================
# REBUILD H_obs AND y
# ============================================================================

H_obs = matched[
    parameter_names
].to_numpy(dtype=np.float64)

y = matched[
    "obs_ppm"
].to_numpy(dtype=np.float64)

print("\nReconstructed inversion arrays:")
print("  H_obs:", H_obs.shape)
print("  y    :", y.shape)

# ============================================================================
# INDEPENDENT GAUSSIAN POSTERIOR
# ============================================================================

w = 1.0 / SIGMA_OBS**2

# B^-1 = I because prior sigma = 1
Q = np.eye(len(parameter_names))

Q += w * (H_obs.T @ H_obs)

rhs = (
    np.ones(len(parameter_names))
    +
    w * H_obs.T @ (y - BACKGROUND)
)

mu = np.linalg.solve(Q, rhs)

cov = np.linalg.inv(Q)
sigma = np.sqrt(np.diag(cov))

# ============================================================================
# LOAD SAVED ANALYTICAL RESULT
# ============================================================================

ana = pd.read_csv(ANALYTICAL_FILE)

ana = ana.set_index("parameter")

saved_mu = ana.loc[
    parameter_names,
    "posterior_scaling",
].to_numpy(dtype=np.float64)

saved_sigma = ana.loc[
    parameter_names,
    "posterior_sigma",
].to_numpy(dtype=np.float64)

# ============================================================================
# COMPARE
# ============================================================================

diff = mu - saved_mu

print("\n" + "=" * 80)
print("INDEPENDENT vs SAVED ANALYTICAL")
print("=" * 80)

print("\nPosterior mean:")
print("  independent mean :", mu.mean())
print("  saved mean       :", saved_mu.mean())

print("\nMaximum |mean difference|:")
print(" ", np.max(np.abs(diff)))

print("\nRMS mean difference:")
print(" ", np.sqrt(np.mean(diff**2)))

print("\nMaximum |sigma difference|:")
print(" ", np.max(np.abs(sigma - saved_sigma)))

# ============================================================================
# TARGET PARAMETER
# ============================================================================

target = "OC2_20260707"
j = parameter_names.index(target)

print("\n" + "=" * 80)
print("TARGET:", target)
print("=" * 80)

print("\nIndependent:")
print("  mean  =", mu[j])
print("  sigma =", sigma[j])

print("\nSaved analytical:")
print("  mean  =", saved_mu[j])
print("  sigma =", saved_sigma[j])

print("\nDifference:")
print("  mean  =", diff[j])
print("  sigma =", sigma[j] - saved_sigma[j])

# ============================================================================
# FORWARD MODEL CHECK
# ============================================================================

prediction_prior = BACKGROUND + H_obs @ np.ones(len(parameter_names))
prediction_ind = BACKGROUND + H_obs @ mu
prediction_saved = BACKGROUND + H_obs @ saved_mu

print("\n" + "=" * 80)
print("FORWARD MODEL")
print("=" * 80)

print("\nPrior:")
print("  mean prediction =", prediction_prior.mean())
print("  RMS residual     =",
      np.sqrt(np.mean((y - prediction_prior)**2)))

print("\nIndependent posterior:")
print("  mean prediction =", prediction_ind.mean())
print("  RMS residual     =",
      np.sqrt(np.mean((y - prediction_ind)**2)))

print("\nSaved analytical posterior:")
print("  mean prediction =", prediction_saved.mean())
print("  RMS residual     =",
      np.sqrt(np.mean((y - prediction_saved)**2)))

# ============================================================================
# GRADIENT TEST
# ============================================================================

gradient_ind = (
    np.ones(len(parameter_names))
    +
    w * H_obs.T @ (y - BACKGROUND)
    -
    Q @ mu
)

gradient_saved = (
    np.ones(len(parameter_names))
    +
    w * H_obs.T @ (y - BACKGROUND)
    -
    Q @ saved_mu
)

print("\n" + "=" * 80)
print("POSTERIOR GRADIENT TEST")
print("=" * 80)

print("\nAt independent posterior:")
print("  max |gradient| =",
      np.max(np.abs(gradient_ind)))

print("\nAt saved analytical posterior:")
print("  max |gradient| =",
      np.max(np.abs(gradient_saved)))

print("\nTarget gradient at saved posterior:")
print(" ", gradient_saved[j])

