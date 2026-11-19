#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart"

MCMC_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "MCMC_13mine_daily_Jan2026"
)

VAR_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "bayesian_4dvar"
)

OUT_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "MCMC_vs_4DVar"
)

os.makedirs(OUT_DIR, exist_ok=True)


MCMC_SAMPLES = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_posterior_samples_Jan2026.npz"
)

MCMC_DAILY = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_daily_factor_matrix_Jan2026.csv"
)

VAR_FACTORS = os.path.join(
    VAR_DIR,
    "4dvar_posterior_scaling_factors.csv"
)

VAR_SD = os.path.join(
    VAR_DIR,
    "posterior_sd_scaling.npy"
)


# ============================================================
# SETTINGS
# ============================================================

N_MINE = 13
N_DAY = 27

DATES = pd.date_range(
    "2026-01-05",
    "2026-01-31",
    freq="D"
)

MINE_NAMES = [
    f"OC{i}" for i in range(1, N_MINE + 1)
]

PARAMETER_NAMES = [
    f"{mine}_{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINE_NAMES
]


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_mcmc_samples():

    print()
    print("=" * 90)
    print("READING MCMC POSTERIOR SAMPLES")
    print("=" * 90)

    z = np.load(MCMC_SAMPLES, allow_pickle=True)

    print("NPZ keys:")
    print(z.files)

    if "samples" not in z.files:
        raise RuntimeError(
            "MCMC NPZ does not contain 'samples'."
        )

    samples = np.asarray(z["samples"])

    print("MCMC array shape:", samples.shape)

    # Current MCMC file:
    #
    #   samples = (20000, 351)
    #
    # The four chains have already been combined.
    #
    # Also support the older possible format:
    #
    #   samples = (4, 5000, 351)

    if samples.ndim == 2:

        if samples.shape[1] != 351:
            raise RuntimeError(
                f"Expected 351 parameters, got {samples.shape[1]}."
            )

        combined = samples

    elif samples.ndim == 3:

        if samples.shape[-1] != 351:
            raise RuntimeError(
                f"Expected 351 parameters, got {samples.shape[-1]}."
            )

        combined = samples.reshape(
            -1,
            samples.shape[-1]
        )

    else:

        raise RuntimeError(
            f"Expected 2-D or 3-D MCMC array, "
            f"got {samples.ndim}-D."
        )

    print("Combined samples:", combined.shape)

    # --------------------------------------------------------
    # Read parameter names from NPZ if available
    # --------------------------------------------------------

    if "parameters" in z.files:

        parameters = np.asarray(
            z["parameters"]
        ).astype(str)

        print(
            "MCMC parameter names:",
            parameters.shape
        )

        if len(parameters) != 351:
            raise RuntimeError(
                f"MCMC parameter array contains "
                f"{len(parameters)} names; expected 351."
            )

    else:

        print(
            "WARNING: MCMC NPZ does not contain "
            "'parameters'. Reconstructing names."
        )

        parameters = np.array([
            f"{mine}_{date.strftime('%Y%m%d')}"
            for date in DATES
            for mine in MINE_NAMES
        ])

    mcmc_mean = combined.mean(axis=0)
    mcmc_median = np.median(
        combined,
        axis=0
    )
    mcmc_sd = combined.std(
        axis=0,
        ddof=1
    )

    return (
        combined,
        parameters,
        mcmc_mean,
        mcmc_median,
        mcmc_sd
    )


def load_4dvar():

    print()
    print("=" * 90)
    print("READING 4D-VAR POSTERIOR")
    print("=" * 90)

    df = pd.read_csv(VAR_FACTORS)

    print("4D-Var dataframe shape:", df.shape)
    print("Columns:")
    print(list(df.columns))

    # Try common column names first.
    parameter_col = find_column(
        df,
        [
            "parameter",
            "Parameter",
            "name",
            "Name",
            "factor",
            "Factor",
        ]
    )

    mean_col = find_column(
        df,
        [
            "posterior_mean",
            "Posterior_mean",
            "mean",
            "Mean",
            "scaling_factor",
            "posterior_scaling",
        ]
    )

    sd_col = find_column(
        df,
        [
            "posterior_sd",
            "Posterior_sd",
            "sd",
            "SD",
            "std",
        ]
    )

    if parameter_col is None:
        # Look for a column containing OC1_20260105 style names.
        for c in df.columns:
            vals = df[c].astype(str)
            if vals.str.match(r"^OC\d+_\d{8}$").sum() > 10:
                parameter_col = c
                break

    if mean_col is None:
        # Identify a numeric column with 351 rows.
        numeric = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        for c in numeric:
            if len(df[c]) == 351:
                mean_col = c
                break

    if parameter_col is None or mean_col is None:
        raise RuntimeError(
            "Could not identify parameter/mean columns in 4D-Var file."
        )

    print("Parameter column:", parameter_col)
    print("Mean column     :", mean_col)

    var_mean = np.full(351, np.nan)

    name_to_value = dict(
        zip(
            df[parameter_col].astype(str),
            pd.to_numeric(df[mean_col], errors="coerce")
        )
    )

    missing = []

    for i, name in enumerate(PARAMETER_NAMES):
        if name not in name_to_value:
            missing.append(name)
        else:
            var_mean[i] = name_to_value[name]

    if missing:
        raise RuntimeError(
            "Missing 4D-Var parameters:\n" +
            "\n".join(missing[:20])
        )

    # SD is stored separately by the 4D-Var script.
    if os.path.exists(VAR_SD):
        var_sd = np.asarray(np.load(VAR_SD)).reshape(-1)

        if len(var_sd) != 351:
            raise RuntimeError(
                f"4D-Var SD has {len(var_sd)} values, expected 351."
            )
    else:
        var_sd = np.full(351, np.nan)

    return df, var_mean, var_sd


# ============================================================
# LOAD
# ============================================================

combined_samples, mcmc_parameters, mcmc_mean, mcmc_median, mcmc_sd = \
    load_mcmc_samples()

var_df, var_mean, var_sd = load_4dvar()


# ============================================================
# PARAMETER-BY-PARAMETER COMPARISON
# ============================================================

print()
print("=" * 90)
print("MCMC vs 4D-VAR PARAMETER COMPARISON")
print("=" * 90)

comparison = pd.DataFrame({
    "parameter": PARAMETER_NAMES,
    "MCMC_mean": mcmc_mean,
    "MCMC_median": mcmc_median,
    "MCMC_SD": mcmc_sd,
    "4DVar_mean": var_mean,
    "4DVar_SD": var_sd,
})

comparison["difference_MCMC_minus_4DVar"] = (
    comparison["MCMC_mean"]
    - comparison["4DVar_mean"]
)

comparison["absolute_difference"] = (
    comparison["difference_MCMC_minus_4DVar"].abs()
)

comparison["relative_difference_percent"] = (
    100.0
    * comparison["difference_MCMC_minus_4DVar"]
    / comparison["4DVar_mean"].replace(0, np.nan)
)

comparison["within_2SD_combined"] = (
    comparison["absolute_difference"]
    <= 2.0 * np.sqrt(
        comparison["MCMC_SD"] ** 2
        + comparison["4DVar_SD"] ** 2
    )
)

comparison_file = os.path.join(
    OUT_DIR,
    "MCMC_vs_4DVar_scaling_factor_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

print()
print("Comparison saved:")
print(comparison_file)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

valid = (
    np.isfinite(comparison["MCMC_mean"])
    &
    np.isfinite(comparison["4DVar_mean"])
)

x = comparison.loc[valid, "4DVar_mean"].to_numpy()
y = comparison.loc[valid, "MCMC_mean"].to_numpy()

corr = np.corrcoef(x, y)[0, 1]

rmse = np.sqrt(
    np.mean((y - x) ** 2)
)

mae = np.mean(
    np.abs(y - x)
)

bias = np.mean(y - x)

print()
print("Global comparison:")
print(f"Parameters compared : {len(x)}")
print(f"Correlation         : {corr:.6f}")
print(f"RMSE difference     : {rmse:.6f}")
print(f"MAE difference      : {mae:.6f}")
print(f"Mean difference     : {bias:.6f}")
print(
    f"Max abs difference  : "
    f"{np.max(np.abs(y-x)):.6f}"
)

print()
print("MCMC:")
print(f"  mean   = {np.mean(y):.6f}")
print(f"  median = {np.median(y):.6f}")
print(f"  min    = {np.min(y):.6f}")
print(f"  max    = {np.max(y):.6f}")

print()
print("4D-Var:")
print(f"  mean   = {np.mean(x):.6f}")
print(f"  median = {np.median(x):.6f}")
print(f"  min    = {np.min(x):.6f}")
print(f"  max    = {np.max(x):.6f}")


# ============================================================
# LARGEST DIFFERENCES
# ============================================================

largest = (
    comparison
    .sort_values(
        "absolute_difference",
        ascending=False
    )
    .head(20)
)

print()
print("=" * 90)
print("20 LARGEST MCMC vs 4D-VAR DIFFERENCES")
print("=" * 90)

print(
    largest[
        [
            "parameter",
            "MCMC_mean",
            "4DVar_mean",
            "difference_MCMC_minus_4DVar",
            "absolute_difference",
        ]
    ].to_string(index=False)
)


# ============================================================
# PLOT 1: 351 PARAMETER COMPARISON
# ============================================================

plt.figure(figsize=(10, 8))

plt.scatter(
    comparison["4DVar_mean"],
    comparison["MCMC_mean"],
    s=20,
    alpha=0.7
)

lims = [
    min(
        comparison["4DVar_mean"].min(),
        comparison["MCMC_mean"].min()
    ),
    max(
        comparison["4DVar_mean"].max(),
        comparison["MCMC_mean"].max()
    )
]

plt.plot(
    lims,
    lims,
    linestyle="--",
    linewidth=1.5,
    label="1:1"
)

plt.xlabel("4D-Var posterior mean scaling factor")
plt.ylabel("MCMC posterior mean scaling factor")

plt.title(
    "MCMC vs 4D-Var posterior scaling factors\n"
    "13 mines × 27 days = 351 parameters"
)

plt.grid(True, alpha=0.3)
plt.legend()

plt.text(
    0.03,
    0.97,
    f"r = {corr:.4f}\n"
    f"RMSE difference = {rmse:.4f}\n"
    f"MAE difference = {mae:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plot1 = os.path.join(
    OUT_DIR,
    "MCMC_vs_4DVar_scatter.png"
)

plt.savefig(
    plot1,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# PLOT 2: DIFFERENCE FOR ALL 351 PARAMETERS
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    np.arange(1, 352),
    comparison["difference_MCMC_minus_4DVar"],
    linewidth=1
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Parameter index")
plt.ylabel("MCMC − 4D-Var scaling factor")

plt.title(
    "Difference between MCMC and 4D-Var posterior means"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plot2 = os.path.join(
    OUT_DIR,
    "MCMC_minus_4DVar_all_351_parameters.png"
)

plt.savefig(
    plot2,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# RESHAPE INTO DAILY MATRICES
# ============================================================

mcmc_daily = mcmc_mean.reshape(
    N_DAY,
    N_MINE
)

var_daily = var_mean.reshape(
    N_DAY,
    N_MINE
)

mcmc_sd_daily = mcmc_sd.reshape(
    N_DAY,
    N_MINE
)

var_sd_daily = var_sd.reshape(
    N_DAY,
    N_MINE
)


# ============================================================
# DAILY TABLE
# ============================================================

daily_rows = []

for d, date in enumerate(DATES):

    for m, mine in enumerate(MINE_NAMES):

        daily_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "mine": mine,
            "MCMC_mean": mcmc_daily[d, m],
            "MCMC_median": mcmc_median.reshape(
                N_DAY, N_MINE
            )[d, m],
            "MCMC_SD": mcmc_sd_daily[d, m],
            "4DVar_mean": var_daily[d, m],
            "4DVar_SD": var_sd_daily[d, m],
            "difference": (
                mcmc_daily[d, m]
                - var_daily[d, m]
            ),
        })

daily = pd.DataFrame(daily_rows)

daily_file = os.path.join(
    OUT_DIR,
    "MCMC_vs_4DVar_daily_scaling_factors.csv"
)

daily.to_csv(
    daily_file,
    index=False
)

print()
print("Daily comparison saved:")
print(daily_file)


# ============================================================
# PLOT 3–15: ONE TIME SERIES PER MINE
# ============================================================

print()
print("=" * 90)
print("PLOTTING DAILY SCALING FACTOR TIME SERIES")
print("=" * 90)

for mine in MINE_NAMES:

    sub = daily[
        daily["mine"] == mine
    ].copy()

    sub["date"] = pd.to_datetime(sub["date"])

    plt.figure(figsize=(11, 5.5))

    # 4D-Var uncertainty
    if np.isfinite(sub["4DVar_SD"]).any():

        plt.fill_between(
            sub["date"],
            sub["4DVar_mean"] - 2 * sub["4DVar_SD"],
            sub["4DVar_mean"] + 2 * sub["4DVar_SD"],
            alpha=0.15,
            label="4D-Var ±2 SD"
        )

    # MCMC uncertainty
    plt.fill_between(
        sub["date"],
        sub["MCMC_mean"] - 2 * sub["MCMC_SD"],
        sub["MCMC_mean"] + 2 * sub["MCMC_SD"],
        alpha=0.15,
        label="MCMC ±2 SD"
    )

    plt.plot(
        sub["date"],
        sub["MCMC_mean"],
        marker="o",
        linewidth=1.8,
        label="MCMC posterior mean"
    )

    plt.plot(
        sub["date"],
        sub["4DVar_mean"],
        marker="s",
        linewidth=1.8,
        label="4D-Var posterior mean"
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        label="Prior factor = 1"
    )

    plt.xlabel("Date")
    plt.ylabel("Emission scaling factor")

    plt.title(
        f"{mine}: daily CH₄ emission scaling factor\n"
        "January 2026"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    outfile = os.path.join(
        OUT_DIR,
        f"{mine}_MCMC_vs_4DVar_timeseries.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(outfile)


# ============================================================
# PLOT 16: ALL MINES — MCMC
# ============================================================

plt.figure(figsize=(14, 8))

for m, mine in enumerate(MINE_NAMES):

    plt.plot(
        DATES,
        mcmc_daily[:, m],
        marker="o",
        linewidth=1.3,
        label=mine
    )

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Date")
plt.ylabel("MCMC posterior mean scaling factor")

plt.title(
    "Daily CH₄ emission scaling factors — MCMC\n"
    "13 open-cut mines, January 2026"
)

plt.grid(True, alpha=0.3)
plt.legend(
    ncol=2,
    fontsize=9
)

plt.xticks(rotation=45)

plt.tight_layout()

plot_all_mcmc = os.path.join(
    OUT_DIR,
    "ALL_mines_MCMC_daily_scaling_factors.png"
)

plt.savefig(
    plot_all_mcmc,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# PLOT 17: ALL MINES — 4D-VAR
# ============================================================

plt.figure(figsize=(14, 8))

for m, mine in enumerate(MINE_NAMES):

    plt.plot(
        DATES,
        var_daily[:, m],
        marker="s",
        linewidth=1.3,
        label=mine
    )

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Date")
plt.ylabel("4D-Var posterior mean scaling factor")

plt.title(
    "Daily CH₄ emission scaling factors — 4D-Var\n"
    "13 open-cut mines, January 2026"
)

plt.grid(True, alpha=0.3)
plt.legend(
    ncol=2,
    fontsize=9
)

plt.xticks(rotation=45)

plt.tight_layout()

plot_all_var = os.path.join(
    OUT_DIR,
    "ALL_mines_4DVar_daily_scaling_factors.png"
)

plt.savefig(
    plot_all_var,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# PLOT 18: DAILY MEAN ACROSS 13 MINES
# ============================================================

daily_mean = pd.DataFrame({
    "date": DATES,
    "MCMC": mcmc_daily.mean(axis=1),
    "4DVar": var_daily.mean(axis=1),
})

plt.figure(figsize=(11, 5.5))

plt.plot(
    daily_mean["date"],
    daily_mean["MCMC"],
    marker="o",
    linewidth=2,
    label="MCMC"
)

plt.plot(
    daily_mean["date"],
    daily_mean["4DVar"],
    marker="s",
    linewidth=2,
    label="4D-Var"
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Prior = 1"
)

plt.xlabel("Date")
plt.ylabel("Mean scaling factor across 13 mines")

plt.title(
    "Daily mean CH₄ emission scaling factor\n"
    "13 mines — January 2026"
)

plt.grid(True, alpha=0.3)
plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plot_daily_mean = os.path.join(
    OUT_DIR,
    "daily_mean_MCMC_vs_4DVar.png"
)

plt.savefig(
    plot_daily_mean,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "Number of parameters",
        "MCMC posterior samples",
        "Correlation",
        "RMSE of MCMC - 4DVar",
        "MAE of MCMC - 4DVar",
        "Mean difference",
        "Maximum absolute difference",
        "MCMC mean scaling",
        "4DVar mean scaling",
        "MCMC minimum",
        "4DVar minimum",
        "MCMC maximum",
        "4DVar maximum",
    ],
    "value": [
        len(x),
        len(combined_samples),
        corr,
        rmse,
        mae,
        bias,
        np.max(np.abs(y-x)),
        np.mean(y),
        np.mean(x),
        np.min(y),
        np.min(x),
        np.max(y),
        np.max(x),
    ]
})

summary_file = os.path.join(
    OUT_DIR,
    "MCMC_vs_4DVar_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


print()
print("=" * 90)
print("COMPARISON COMPLETE")
print("=" * 90)

print()
print("Output directory:")
print(OUT_DIR)

print()
print("Main files:")
print(comparison_file)
print(daily_file)
print(summary_file)
print(plot1)
print(plot2)
print(plot_all_mcmc)
print(plot_all_var)
print(plot_daily_mean)

