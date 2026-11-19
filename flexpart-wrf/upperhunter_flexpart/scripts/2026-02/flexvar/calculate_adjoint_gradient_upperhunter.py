#!/usr/bin/env python3

"""
Upper Hunter CH4 inverse modelling
Discrete adjoint diagnostics for constrained 4D-Var

This script calculates:

    g_adj  = H.T @ R^-1 @ (H @ s - d)

where:

    H = FLEXPART-WRF source-receptor sensitivity matrix
    s = constrained 4D-Var posterior scaling factors
    d = observed CH4 enhancement
    R = observation error covariance

It also calculates:

    g_prior = B^-1 @ (s - sb)

and:

    gradient = g_prior + g_adj

Outputs:
    - 351-parameter adjoint diagnostics
    - 13 x 27 mine/day adjoint matrices
    - heatmaps
    - residual diagnostics
    - CSV files
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
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

CONSTRAINED_POSTERIOR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_constrained/"
    "4dvar_posterior_scaling_factors.csv"
)

OUTPUT_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_constrained/"
    "adjoint_diagnostics"
)

PLOT_DIR = OUTPUT_DIR + "/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# INVERSION SETTINGS
# ============================================================

MINES = [
    f"OC{i}"
    for i in range(1, 14)
]

N_MINES = 13
N_DAYS = 27
N_PARAMETERS = 351

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

DATES = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)

BACKGROUND_PPM = 1.85

OBS_SIGMA = 0.05

PRIOR_MEAN = 1.0

PRIOR_SIGMA = 0.5

CORRELATION_LENGTH_DAYS = 3.0


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("UPPER HUNTER CH4")
print("DISCRETE ADJOINT DIAGNOSTICS FOR CONSTRAINED 4D-VAR")
print("=" * 90)

print()
print("H:")
print(H_FILE)

print()
print("Picarro:")
print(PICARRO_FILE)

print()
print("Constrained posterior:")
print(CONSTRAINED_POSTERIOR)

print()
print("Output:")
print(OUTPUT_DIR)


# ============================================================
# READ H
# ============================================================

print()
print("=" * 90)
print("READING FLEXPART-WRF H MATRIX")
print("=" * 90)

H_df = pd.read_csv(H_FILE)

metadata_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

parameter_columns = [
    c
    for c in H_df.columns
    if c not in metadata_columns
]

H_all = H_df[parameter_columns].to_numpy(
    dtype=np.float64
)

print()
print("H shape:", H_all.shape)

print("Number of parameters:", len(parameter_columns))

if H_all.shape[1] != N_PARAMETERS:
    raise RuntimeError(
        f"Expected {N_PARAMETERS} parameters, "
        f"found {H_all.shape[1]}"
    )


# ============================================================
# READ CONSTRAINED POSTERIOR
# ============================================================

print()
print("=" * 90)
print("READING CONSTRAINED 4D-VAR POSTERIOR")
print("=" * 90)

post = pd.read_csv(
    CONSTRAINED_POSTERIOR
)

if "parameter" not in post.columns:
    raise RuntimeError(
        "Posterior file does not contain 'parameter'."
    )

if "posterior_scaling" not in post.columns:
    raise RuntimeError(
        "Posterior file does not contain "
        "'posterior_scaling'."
    )

if len(post) != N_PARAMETERS:
    raise RuntimeError(
        f"Expected {N_PARAMETERS} posterior parameters, "
        f"found {len(post)}"
    )

# Verify parameter ordering
if not np.array_equal(
    post["parameter"].astype(str).to_numpy(),
    np.asarray(parameter_columns).astype(str),
):
    raise RuntimeError(
        "Parameter ordering in posterior does not match H."
    )

s_post = post[
    "posterior_scaling"
].to_numpy(
    dtype=np.float64
)

print()
print("Posterior min :", s_post.min())
print("Posterior max :", s_post.max())
print("Posterior mean:", s_post.mean())
print("Posterior median:", np.median(s_post))

if np.any(s_post < 0) or np.any(s_post > 3):
    raise RuntimeError(
        "Constrained posterior contains values outside [0, 3]."
    )


# ============================================================
# READ PICARRO
# ============================================================

print()
print("=" * 90)
print("READING PICARRO OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(
    PICARRO_FILE
)

obs["time_aest"] = pd.to_datetime(
    obs["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce",
)

if obs["time_aest"].isna().any():
    print(
        "WARNING:",
        obs["time_aest"].isna().sum(),
        "invalid datetime rows"
    )

receptor_columns = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}

obs_records = []

for receptor, column in receptor_columns.items():

    if column not in obs.columns:
        raise RuntimeError(
            f"Missing Picarro column: {column}"
        )

    tmp = obs[
        [
            "time_aest",
            column,
        ]
    ].copy()

    tmp["receptor"] = receptor

    tmp["obs_ch4_ppm"] = pd.to_numeric(
        tmp[column],
        errors="coerce",
    )

    tmp = tmp[
        [
            "time_aest",
            "receptor",
            "obs_ch4_ppm",
        ]
    ]

    obs_records.append(tmp)

obs_long = pd.concat(
    obs_records,
    ignore_index=True,
)

obs_long = obs_long.dropna(
    subset=[
        "time_aest",
        "obs_ch4_ppm",
    ]
)

obs_long["time_aest"] = (
    obs_long["time_aest"].dt.floor("h")
)

obs_long = obs_long[
    (obs_long["time_aest"] >= START_DATE)
    &
    (
        obs_long["time_aest"]
        < END_DATE + pd.Timedelta(days=1)
    )
]

# Average duplicate receptor/hour observations
obs_long = (
    obs_long
    .groupby(
        [
            "receptor",
            "time_aest",
        ],
        as_index=False,
    )["obs_ch4_ppm"]
    .mean()
)

print()
print("Usable Picarro observations:", len(obs_long))

print()
print(
    obs_long["receptor"]
    .value_counts()
    .sort_index()
)


# ============================================================
# ALIGN H WITH PICARRO
# ============================================================

print()
print("=" * 90)
print("ALIGNING H AND PICARRO")
print("=" * 90)

H_df["receptor"] = (
    H_df["receptor"]
    .astype(str)
    .str.strip()
    .str.lower()
)

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
)

H_key = pd.DataFrame(
    {
        "receptor": H_df["receptor"],
        "time_aest": H_df[
            "obs_time_aest"
        ].dt.floor("h"),
    }
)

obs_key = obs_long[
    [
        "receptor",
        "time_aest",
        "obs_ch4_ppm",
    ]
].copy()

H_key_indexed = H_key.copy()

H_key_indexed["H_row"] = np.arange(
    len(H_key_indexed)
)

match = H_key_indexed.merge(
    obs_key,
    on=[
        "receptor",
        "time_aest",
    ],
    how="inner",
)

H_row_indices = match[
    "H_row"
].to_numpy(
    dtype=int
)

H = H_all[
    H_row_indices,
    :,
]

y = match[
    "obs_ch4_ppm"
].to_numpy(
    dtype=np.float64
)

receptors = match[
    "receptor"
].to_numpy()

times = match[
    "time_aest"
].to_numpy()

print()
print("Matched observations:", len(match))

print()
print(
    pd.Series(receptors)
    .value_counts()
    .sort_index()
)

if len(match) == 0:
    raise RuntimeError(
        "No H/Picarro observations matched."
    )


# ============================================================
# OBSERVATION ENHANCEMENT
# ============================================================

d = y - BACKGROUND_PPM


# ============================================================
# FORWARD MODEL AT POSTERIOR
# ============================================================

print()
print("=" * 90)
print("FORWARD MODEL")
print("=" * 90)

y_model_enhancement = H @ s_post

y_model = (
    BACKGROUND_PPM
    + y_model_enhancement
)

residual = (
    y_model_enhancement - d
)

print()
print("Observed CH4:")
print("  min :", y.min())
print("  max :", y.max())
print("  mean:", y.mean())

print()
print("Model CH4:")
print("  min :", y_model.min())
print("  max :", y_model.max())
print("  mean:", y_model.mean())

print()
print("Residual:")
print("  min :", residual.min())
print("  max :", residual.max())
print("  mean:", residual.mean())
print("  RMSE:", np.sqrt(np.mean(residual**2)))


# ============================================================
# OBSERVATION ERROR COVARIANCE
# ============================================================

R_inv_scalar = (
    1.0 / OBS_SIGMA**2
)

weighted_residual = (
    R_inv_scalar * residual
)


# ============================================================
# ADJOINT OBSERVATION CONTRIBUTION
#
# g_adj = H.T R^-1 (Hs-d)
# ============================================================

print()
print("=" * 90)
print("CALCULATING ADJOINT GRADIENT")
print("=" * 90)

g_adjoint = (
    H.T @ weighted_residual
)

print()
print("Adjoint gradient shape:")
print(g_adjoint.shape)

print()
print("Adjoint gradient:")
print("  min :", g_adjoint.min())
print("  max :", g_adjoint.max())
print("  mean:", g_adjoint.mean())
print("  L2  :", np.linalg.norm(g_adjoint))


# ============================================================
# BUILD PRIOR COVARIANCE B
# ============================================================

print()
print("=" * 90)
print("BUILDING PRIOR COVARIANCE")
print("=" * 90)

B = np.zeros(
    (
        N_PARAMETERS,
        N_PARAMETERS,
    ),
    dtype=np.float64,
)

for mine_index in range(N_MINES):

    start = (
        mine_index
    )

    # Parameters are day-major:
    #
    # OC1_day1 ... OC13_day1
    # OC1_day2 ... OC13_day2
    #
    # Therefore the same mine occurs every 13 parameters.

    indices = np.arange(
        mine_index,
        N_PARAMETERS,
        N_MINES,
    )

    for i in range(N_DAYS):

        for j in range(N_DAYS):

            lag = abs(i - j)

            covariance = (
                PRIOR_SIGMA**2
                * np.exp(
                    -lag
                    / CORRELATION_LENGTH_DAYS
                )
            )

            B[
                indices[i],
                indices[j],
            ] = covariance


# ============================================================
# PRIOR GRADIENT
# ============================================================

print()
print("Calculating prior gradient...")

s_b = np.full(
    N_PARAMETERS,
    PRIOR_MEAN,
    dtype=np.float64,
)

# Solve B x = (s - sb)
# rather than explicitly computing B^-1.

prior_gradient = np.linalg.solve(
    B,
    s_post - s_b,
)


# ============================================================
# TOTAL GRADIENT
# ============================================================

gradient = (
    prior_gradient
    + g_adjoint
)

print()
print("=" * 90)
print("GRADIENT SUMMARY")
print("=" * 90)

print()
print("Adjoint observation contribution:")
print(
    "  L2:",
    np.linalg.norm(g_adjoint)
)

print()
print("Prior contribution:")
print(
    "  L2:",
    np.linalg.norm(prior_gradient)
)

print()
print("Total gradient:")
print(
    "  L2:",
    np.linalg.norm(gradient)
)

print(
    "  max abs:",
    np.max(np.abs(gradient))
)


# ============================================================
# KKT / PROJECTED GRADIENT DIAGNOSTIC
# ============================================================

LOWER = 0.0
UPPER = 3.0

projected_gradient = gradient.copy()

at_lower = np.isclose(
    s_post,
    LOWER,
    atol=1e-8,
)

at_upper = np.isclose(
    s_post,
    UPPER,
    atol=1e-8,
)

interior = (
    ~at_lower
    & ~at_upper
)

# For a minimization problem:
#
# lower bound: gradient >= 0
# upper bound: gradient <= 0
# interior: gradient = 0
#
# Projected gradient measures violation of these
# conditions.

projected_gradient[
    at_lower
    & (gradient > 0)
] = 0.0

projected_gradient[
    at_upper
    & (gradient < 0)
] = 0.0

print()
print("=" * 90)
print("PROJECTED GRADIENT")
print("=" * 90)

print()
print("Interior parameters:", interior.sum())
print("At lower bound    :", at_lower.sum())
print("At upper bound    :", at_upper.sum())

print()
print(
    "Projected gradient L2:",
    np.linalg.norm(projected_gradient)
)

print(
    "Projected gradient max abs:",
    np.max(np.abs(projected_gradient))
)


# ============================================================
# PARAMETER DIAGNOSTICS DATAFRAME
# ============================================================

diag = pd.DataFrame(
    {
        "parameter": parameter_columns,
        "posterior_scaling": s_post,
        "prior_scaling": s_b,
        "adjoint_observation_gradient": g_adjoint,
        "prior_gradient": prior_gradient,
        "total_gradient": gradient,
        "projected_gradient": projected_gradient,
        "at_lower_bound": at_lower,
        "at_upper_bound": at_upper,
    }
)

diag_file = (
    OUTPUT_DIR +
    "/adjoint_gradient_351_parameters.csv"
)

diag.to_csv(
    diag_file,
    index=False,
)

print()
print("Saved:")
print(diag_file)


# ============================================================
# ADD OBSERVATION DIAGNOSTICS
# ============================================================

obs_diag = match[
    [
        "receptor",
        "time_aest",
    ]
].copy()

obs_diag[
    "observed_ch4_ppm"
] = y

obs_diag[
    "observed_enhancement_ppm"
] = d

obs_diag[
    "model_ch4_ppm"
] = y_model

obs_diag[
    "model_enhancement_ppm"
] = y_model_enhancement

obs_diag[
    "residual_ppm"
] = residual

obs_diag[
    "weighted_residual"
] = weighted_residual

obs_file = (
    OUTPUT_DIR +
    "/adjoint_observation_diagnostics.csv"
)

obs_diag.to_csv(
    obs_file,
    index=False,
)

print()
print("Saved:")
print(obs_file)


# ============================================================
# CONVERT 351 PARAMETERS TO 13 × 27
# ============================================================

adjoint_matrix = np.zeros(
    (
        N_MINES,
        N_DAYS,
    )
)

prior_matrix = np.zeros(
    (
        N_MINES,
        N_DAYS,
    )
)

gradient_matrix = np.zeros(
    (
        N_MINES,
        N_DAYS,
    )
)

scaling_matrix = np.zeros(
    (
        N_MINES,
        N_DAYS,
    )
)

for day_index in range(N_DAYS):

    for mine_index in range(N_MINES):

        index = (
            day_index * N_MINES
            + mine_index
        )

        adjoint_matrix[
            mine_index,
            day_index,
        ] = g_adjoint[index]

        prior_matrix[
            mine_index,
            day_index,
        ] = prior_gradient[index]

        gradient_matrix[
            mine_index,
            day_index,
        ] = gradient[index]

        scaling_matrix[
            mine_index,
            day_index,
        ] = s_post[index]


# ============================================================
# SAVE MATRICES
# ============================================================

adjoint_df = pd.DataFrame(
    adjoint_matrix,
    index=MINES,
    columns=[
        d.strftime("%Y-%m-%d")
        for d in DATES
    ],
)

prior_df = pd.DataFrame(
    prior_matrix,
    index=MINES,
    columns=[
        d.strftime("%Y-%m-%d")
        for d in DATES
    ],
)

gradient_df = pd.DataFrame(
    gradient_matrix,
    index=MINES,
    columns=[
        d.strftime("%Y-%m-%d")
        for d in DATES
    ],
)

scaling_df = pd.DataFrame(
    scaling_matrix,
    index=MINES,
    columns=[
        d.strftime("%Y-%m-%d")
        for d in DATES
    ],
)

adjoint_csv = (
    OUTPUT_DIR +
    "/adjoint_observation_gradient_13x27.csv"
)

prior_csv = (
    OUTPUT_DIR +
    "/prior_gradient_13x27.csv"
)

gradient_csv = (
    OUTPUT_DIR +
    "/total_gradient_13x27.csv"
)

scaling_csv = (
    OUTPUT_DIR +
    "/posterior_scaling_13x27.csv"
)

adjoint_df.to_csv(adjoint_csv)
prior_df.to_csv(prior_csv)
gradient_df.to_csv(gradient_csv)
scaling_df.to_csv(scaling_csv)

print()
print("Saved:")
print(adjoint_csv)
print(prior_csv)
print(gradient_csv)
print(scaling_csv)


# ============================================================
# HEATMAP FUNCTION
# ============================================================

def save_heatmap(
    matrix,
    title,
    filename,
    cmap="RdBu_r",
    symmetric=True,
):

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    if symmetric:

        vmax = np.nanmax(
            np.abs(matrix)
        )

        if vmax == 0:
            vmax = 1.0

        vmin = -vmax

    else:

        vmin = np.nanmin(matrix)
        vmax = np.nanmax(matrix)

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(
        np.arange(N_DAYS)
    )

    ax.set_xticklabels(
        [
            d.strftime("%d")
            for d in DATES
        ],
        rotation=0,
    )

    ax.set_yticks(
        np.arange(N_MINES)
    )

    ax.set_yticklabels(
        MINES
    )

    ax.set_xlabel(
        "January 2026 day"
    )

    ax.set_ylabel(
        "Mine"
    )

    ax.set_title(
        title
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Gradient"
    )

    ax.grid(
        False
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print(
        "Saved:",
        filename
    )


# ============================================================
# ADJOINT HEATMAP
# ============================================================

save_heatmap(
    adjoint_matrix,
    "Upper Hunter CH4 — Adjoint observation gradient\n"
    r"$H^T R^{-1}(Hs-d)$",
    PLOT_DIR +
    "/adjoint_observation_gradient_13x27.png",
)


# ============================================================
# TOTAL GRADIENT HEATMAP
# ============================================================

save_heatmap(
    gradient_matrix,
    "Upper Hunter CH4 — Total 4D-Var gradient\n"
    r"$B^{-1}(s-s_b)+H^T R^{-1}(Hs-d)$",
    PLOT_DIR +
    "/total_gradient_13x27.png",
)


# ============================================================
# POSTERIOR SCALING HEATMAP
# ============================================================

save_heatmap(
    scaling_matrix,
    "Upper Hunter CH4 — Constrained 4D-Var posterior scaling",
    PLOT_DIR +
    "/posterior_scaling_13x27.png",
    cmap="viridis",
    symmetric=False,
)


# ============================================================
# ABSOLUTE ADJOINT HEATMAP
# ============================================================

save_heatmap(
    np.abs(adjoint_matrix),
    "Upper Hunter CH4 — Absolute adjoint sensitivity",
    PLOT_DIR +
    "/absolute_adjoint_sensitivity_13x27.png",
    cmap="viridis",
    symmetric=False,
)


# ============================================================
# DAILY TOTAL ADJOINT MAGNITUDE
# ============================================================

daily_adjoint_norm = np.linalg.norm(
    adjoint_matrix,
    axis=0,
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    DATES,
    daily_adjoint_norm,
    "-o",
    linewidth=1.8,
    markersize=4,
)

ax.set_title(
    "Upper Hunter CH4 — Daily magnitude of adjoint observation gradient"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    r"$\|H^T R^{-1}(Hs-d)\|$"
)

ax.grid(
    True,
    alpha=0.3,
)

fig.autofmt_xdate()

plt.tight_layout()

daily_adjoint_file = (
    PLOT_DIR +
    "/daily_adjoint_gradient_magnitude.png"
)

plt.savefig(
    daily_adjoint_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()
plt.close()

print()
print("Saved:")
print(daily_adjoint_file)


# ============================================================
# MINE TOTAL ADJOINT MAGNITUDE
# ============================================================

mine_adjoint_norm = np.linalg.norm(
    adjoint_matrix,
    axis=1,
)

fig, ax = plt.subplots(
    figsize=(11, 5)
)

ax.bar(
    MINES,
    mine_adjoint_norm,
)

ax.set_title(
    "Upper Hunter CH4 — Total adjoint sensitivity by mine"
)

ax.set_xlabel(
    "Mine"
)

ax.set_ylabel(
    r"$\|g_{\mathrm{adj}}\|$"
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

mine_adjoint_file = (
    PLOT_DIR +
    "/mine_total_adjoint_sensitivity.png"
)

plt.savefig(
    mine_adjoint_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()
plt.close()

print()
print("Saved:")
print(mine_adjoint_file)


# ============================================================
# TOP MINE-DAY ADJOINT CONTRIBUTIONS
# ============================================================

top = diag.copy()

top[
    "absolute_adjoint"
] = np.abs(
    top["adjoint_observation_gradient"]
)

top = top.sort_values(
    "absolute_adjoint",
    ascending=False,
)

top_file = (
    OUTPUT_DIR +
    "/top_50_mine_day_adjoint_sensitivities.csv"
)

top.head(50).to_csv(
    top_file,
    index=False,
)

print()
print("Saved:")
print(top_file)

print()
print("=" * 90)
print("TOP 20 MINE-DAY ADJOINT SENSITIVITIES")
print("=" * 90)

print(
    top[
        [
            "parameter",
            "posterior_scaling",
            "adjoint_observation_gradient",
            "total_gradient",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 90)
print("ADJOINT DIAGNOSTICS COMPLETE")
print("=" * 90)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Plots:")
print(PLOT_DIR)

print()
print("Main adjoint heatmap:")
print(
    PLOT_DIR +
    "/adjoint_observation_gradient_13x27.png"
)

print()
print("Total gradient heatmap:")
print(
    PLOT_DIR +
    "/total_gradient_13x27.png"
)

print()
print("Posterior scaling heatmap:")
print(
    PLOT_DIR +
    "/posterior_scaling_13x27.png"
)

print()
print("=" * 90)

