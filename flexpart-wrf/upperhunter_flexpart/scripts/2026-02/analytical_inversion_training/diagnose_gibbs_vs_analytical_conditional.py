import numpy as np
import pandas as pd

# ============================================================
# FILES
# ============================================================

H_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/2026-02/H_flexpart_0_10m_13mine_Feb2026.csv"
)

ANALYTICAL_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/"
    "inversion_results/2026-02/analytical_posterior_scaling_factors.csv"
)

COV_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/"
    "inversion_results/2026-02/analytical_posterior_covariance.npy"
)

PICARRO_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

BACKGROUND_PPM = 1.85
OBS_SIGMA = 0.05
PRIOR_MEAN = 1.0
PRIOR_SIGMA = 1.0

TARGET = "OC2_20260207"

# ============================================================
# READ ANALYTICAL POSTERIOR FIRST
# ============================================================

post = pd.read_csv(ANALYTICAL_FILE)

parameter_names = post["parameter"].astype(str).tolist()

print("=" * 90)
print("GIBBS CONDITIONAL VS ANALYTICAL POSTERIOR CONDITIONAL")
print("=" * 90)

print()
print("Analytical parameters:", len(parameter_names))
print("First parameters:", parameter_names[:5])
print("Last parameters:", parameter_names[-5:])

# ============================================================
# READ H
# ============================================================

Hdf = pd.read_csv(H_FILE)

print()
print("Raw H shape:", Hdf.shape)

# Use the actual 364 parameter names from the analytical inversion.
missing_H_columns = [
    p for p in parameter_names
    if p not in Hdf.columns
]

if missing_H_columns:
    print()
    print("ERROR: parameters missing from H:")
    for p in missing_H_columns:
        print(" ", p)
    raise SystemExit(1)

print("H parameter columns found:", len(parameter_names))

# ============================================================
# NORMALIZE H METADATA
# ============================================================

Hdf["obs_time_aest"] = pd.to_datetime(
    Hdf["obs_time_aest"]
).dt.floor("h")

Hdf["receptor"] = (
    Hdf["receptor"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# ============================================================
# READ PICARRO
# ============================================================

pic = pd.read_csv(PICARRO_FILE)

pic["Datetime"] = pd.to_datetime(
    pic["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
).dt.floor("h")

mapping = {
    "Singleton cleaned": "singleton",
    "Muswellbrook cleaned": "muswellbrook",
    "Merriwa cleaned": "merriwa",
}

obs_frames = []

for col, receptor in mapping.items():

    tmp = pic[["Datetime", col]].copy()

    tmp = tmp.rename(
        columns={
            "Datetime": "obs_time_aest",
            col: "obs"
        }
    )

    tmp["receptor"] = receptor

    tmp = tmp.dropna(subset=["obs"])

    obs_frames.append(tmp)

obs = pd.concat(
    obs_frames,
    ignore_index=True
)

obs = obs[
    (obs["obs_time_aest"] >= "2026-02-01") &
    (obs["obs_time_aest"] < "2026-03-01")
]

# Average duplicate receptor/hour observations
obs = (
    obs.groupby(
        ["receptor", "obs_time_aest"],
        as_index=False
    )["obs"]
    .mean()
)

# ============================================================
# MATCH H TO OBSERVATIONS
# ============================================================

matched = Hdf.merge(
    obs,
    on=["receptor", "obs_time_aest"],
    how="inner"
)

matched = matched.sort_values(
    ["obs_time_aest", "receptor"]
).reset_index(drop=True)

H = matched[parameter_names].to_numpy(float)
y = matched["obs"].to_numpy(float)

print()
print("Usable observations:", len(obs))
print("Matched H rows:", len(matched))
print("H_obs shape:", H.shape)

# ============================================================
# CHECK TARGET
# ============================================================

if TARGET not in parameter_names:
    raise ValueError(
        f"{TARGET} not found in analytical parameter list"
    )

j = parameter_names.index(TARGET)

print()
print("Target:", TARGET)
print("Index:", j)

# ============================================================
# READ ANALYTICAL POSTERIOR
# ============================================================

mu = (
    post.set_index("parameter")
    .loc[parameter_names, "posterior_scaling"]
    .to_numpy(float)
)

Sigma = np.load(COV_FILE)

if Sigma.shape != (len(parameter_names), len(parameter_names)):
    raise ValueError(
        f"Covariance shape {Sigma.shape} does not match "
        f"{len(parameter_names)} parameters"
    )

Q = np.linalg.inv(Sigma)

print()
print("Analytical posterior:")
print("  mean =", mu[j])
print("  sigma =", np.sqrt(Sigma[j, j]))
print("  precision diagonal =", Q[j, j])

# ============================================================
# OBSERVATION WEIGHTS
# ============================================================

invvar = np.full(
    len(y),
    1.0 / OBS_SIGMA**2
)

# ============================================================
# TEST STATE = ANALYTICAL POSTERIOR MEAN
# ============================================================

s = mu.copy()

hj = H[:, j]

idx = np.flatnonzero(hj != 0.0)

old_value = s[j]

# Prediction excluding parameter j
prediction_without_j = (
    BACKGROUND_PPM
    + H @ s
    - hj * old_value
)

residual_without_j = (
    y
    - prediction_without_j
)

# ============================================================
# GIBBS CONDITIONAL
# ============================================================

likelihood_precision = np.sum(
    hj[idx] ** 2 * invvar[idx]
)

conditional_precision_gibbs = (
    likelihood_precision
    + 1.0 / PRIOR_SIGMA**2
)

conditional_sd_gibbs = (
    1.0 / np.sqrt(
        conditional_precision_gibbs
    )
)

likelihood_linear = np.sum(
    hj[idx]
    * residual_without_j[idx]
    * invvar[idx]
)

conditional_linear = (
    likelihood_linear
    + PRIOR_MEAN / PRIOR_SIGMA**2
)

conditional_mu_gibbs = (
    conditional_linear
    / conditional_precision_gibbs
)

# ============================================================
# ANALYTICAL POSTERIOR CONDITIONAL
# ============================================================

other = np.arange(len(mu)) != j

conditional_variance_Q = (
    1.0 / Q[j, j]
)

conditional_sd_Q = np.sqrt(
    conditional_variance_Q
)

conditional_mu_Q = (
    mu[j]
    -
    np.sum(
        Q[j, other]
        *
        (s[other] - mu[other])
    )
    / Q[j, j]
)

# ============================================================
# REPORT
# ============================================================

print()
print("=" * 90)
print("CONDITIONAL COMPARISON")
print("=" * 90)

print()
print("Gibbs conditional:")
print("  nonzero observations =", len(idx))
print("  precision =", conditional_precision_gibbs)
print("  sd        =", conditional_sd_gibbs)
print("  mean      =", conditional_mu_gibbs)

print()
print("Analytical Q conditional:")
print("  precision =", Q[j, j])
print("  sd        =", conditional_sd_Q)
print("  mean      =", conditional_mu_Q)

print()
print("Differences:")
print(
    "  precision difference =",
    conditional_precision_gibbs - Q[j, j]
)

print(
    "  sd difference =",
    conditional_sd_gibbs - conditional_sd_Q
)

print(
    "  mean difference =",
    conditional_mu_gibbs - conditional_mu_Q
)

# ============================================================
# FULL POSTERIOR CHECK
# ============================================================

prediction = BACKGROUND_PPM + H @ s

residual = y - prediction

loglike = -0.5 * np.sum(
    residual**2 * invvar
)

logprior = -0.5 * np.sum(
    ((s - PRIOR_MEAN) / PRIOR_SIGMA)**2
)

print()
print("=" * 90)
print("AT ANALYTICAL POSTERIOR MEAN")
print("=" * 90)

print(
    "RMS residual =",
    np.sqrt(np.mean(residual**2))
)

print(
    "log likelihood =",
    loglike
)

print(
    "log prior =",
    logprior
)

print("=" * 90)

