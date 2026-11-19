#!/usr/bin/env python3

"""
===============================================================================
UPPER HUNTER CH4
ANALYTICAL BAYESIAN vs MCMC POSTERIOR SCALING FACTORS
===============================================================================

13 open-cut mines × 27 days

Comparison:
    Direct Gibbs truncated  Gaussian posterior mean
vs
    Gibbs/MCMC posterior mean

MCMC file:
    FLEXPART_MCMC_posterior_daily_factors_Jan2026.csv

Analytical file:
    direct_Gibbs_truncated_posterior_scaling_factors.csv

Outputs:
    1. 13-panel time-series comparison
    2. Individual mine plots
    3. Comparison CSV
    4. Summary statistics
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

ANALYTICAL_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/"
    "upperhunter_flexpart/inversion_results/2026-01/MCMC_direct_Gibbs_13mine_daily_Jan2026/"
    "direct_Gibbs_truncated_posterior_scaling_factors.csv"
#    "truncated_gaussian_analytical_posterior.csv"
#    "analytical_posterior_scaling_factors.csv"
#     "analytical_posterior_scaling_factors_constrained.csv"
)

MCMC_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/"
    "upperhunter_flexpart/H_matrix_flexpart/2026-01/"
    "MCMC_13mine_daily_Jan2026/"
    "FLEXPART_MCMC_posterior_daily_factors_Jan2026.csv"
)

OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/"
    "upperhunter_flexpart/inversion_results/2026-01/"
    "directGibbs_vs_mcmc"
)


# =============================================================================
# MINE / DATE DEFINITIONS
# =============================================================================

MINE_NAMES = [
    "OC1",
    "OC2",
    "OC3",
    "OC4",
    "OC5",
    "OC6",
    "OC7",
    "OC8",
    "OC9",
    "OC10",
    "OC11",
    "OC12",
    "OC13",
]


DATES = pd.date_range(
    "2026-01-05",
    "2026-01-31",
    freq="D"
)


# =============================================================================
# LOAD ANALYTICAL RESULTS
# =============================================================================

def load_analytical():

    print("\n" + "=" * 80)
    print("LOADING ANALYTICAL POSTERIOR")
    print("=" * 80)

    print(
        "\nFile:"
    )

    print(
        ANALYTICAL_FILE
    )

    if not os.path.exists(
        ANALYTICAL_FILE
    ):

        raise FileNotFoundError(
            ANALYTICAL_FILE
        )


    df = pd.read_csv(
        ANALYTICAL_FILE
    )


    print(
        f"\nShape: {df.shape}"
    )


    print(
        "\nColumns:"
    )

    print(
        list(df.columns)
    )


    required = [
        "parameter",
        "prior_scaling",
        "posterior_scaling",
        "posterior_sigma",
        "mine",
        "date",
    ]


    missing = [
        x
        for x in required
        if x not in df.columns
    ]


    if missing:

        raise RuntimeError(
            "Analytical file is missing:\n"
            + "\n".join(missing)
        )


    df["date"] = pd.to_datetime(
        df["date"]
    )


    df["mine"] = df["mine"].astype(
        str
    )


    return df


# =============================================================================
# LOAD MCMC RESULTS
# =============================================================================

def load_mcmc():

    print("\n" + "=" * 80)
    print("LOADING MCMC POSTERIOR")
    print("=" * 80)

    print(
        "\nFile:"
    )

    print(
        MCMC_FILE
    )


    if not os.path.exists(
        MCMC_FILE
    ):

        raise FileNotFoundError(
            MCMC_FILE
        )


    df = pd.read_csv(
        MCMC_FILE
    )


    print(
        f"\nShape: {df.shape}"
    )


    print(
        "\nColumns:"
    )

    print(
        list(df.columns)
    )


    required = [
        "parameter",
        "mine",
        "date",
        "posterior_mean",
        "posterior_median",
        "posterior_sd",
        "q025",
        "q975",
    ]


    missing = [
        x
        for x in required
        if x not in df.columns
    ]


    if missing:

        raise RuntimeError(
            "MCMC file is missing:\n"
            + "\n".join(missing)
        )


    df["date"] = pd.to_datetime(
        df["date"]
    )


    df["mine"] = df["mine"].astype(
        str
    )


    return df


# =============================================================================
# CHECK PARAMETER COVERAGE
# =============================================================================

def check_data(
    analytical,
    mcmc
):

    print("\n" + "=" * 80)
    print("CHECKING PARAMETER COVERAGE")
    print("=" * 80)


    expected = pd.MultiIndex.from_product(
        [
            DATES,
            MINE_NAMES,
        ],
        names=[
            "date",
            "mine",
        ]
    )


    analytical_index = pd.MultiIndex.from_frame(
        analytical[
            [
                "date",
                "mine",
            ]
        ]
    )


    mcmc_index = pd.MultiIndex.from_frame(
        mcmc[
            [
                "date",
                "mine",
            ]
        ]
    )


    missing_analytical = (
        expected
        .difference(
            analytical_index
        )
    )


    missing_mcmc = (
        expected
        .difference(
            mcmc_index
        )
    )


    print(
        f"\nExpected parameters: "
        f"{len(expected)}"
    )


    print(
        f"Analytical rows:     "
        f"{len(analytical)}"
    )


    print(
        f"MCMC rows:           "
        f"{len(mcmc)}"
    )


    print(
        f"Missing analytical:  "
        f"{len(missing_analytical)}"
    )


    print(
        f"Missing MCMC:        "
        f"{len(missing_mcmc)}"
    )


    if len(missing_analytical) > 0:

        print(
            "\nMissing analytical parameters:"
        )

        print(
            missing_analytical
        )


    if len(missing_mcmc) > 0:

        print(
            "\nMissing MCMC parameters:"
        )

        print(
            missing_mcmc
        )


# =============================================================================
# MERGE ANALYTICAL AND MCMC
# =============================================================================

def merge_results(
    analytical,
    mcmc
):

    print("\n" + "=" * 80)
    print("MERGING ANALYTICAL AND MCMC RESULTS")
    print("=" * 80)


    analytical_use = analytical[
        [
            "parameter",
            "mine",
            "date",
            "prior_scaling",
            "truncated_posterior_mean",
            "truncated_posterior_sd",
        ]
    ].copy()


    analytical_use = (
        analytical_use
        .rename(
            columns={
                "truncated_posterior_mean":
                    "analytical_truncated_mean",

                "truncated_posterior_sd":
                    "analytical_truncated_sigma",
            }
        )
    )


    mcmc_use = mcmc[
        [
            "parameter",
            "mine",
            "date",
            "posterior_mean",
            "posterior_median",
            "posterior_sd",
            "q025",
            "q975",
        ]
    ].copy()


    mcmc_use = (
        mcmc_use
        .rename(
            columns={
                "posterior_mean":
                    "mcmc_mean",

                "posterior_median":
                    "mcmc_median",

                "posterior_sd":
                    "mcmc_sd",

                "q025":
                    "mcmc_q025",

                "q975":
                    "mcmc_q975",
            }
        )
    )


    merged = analytical_use.merge(
        mcmc_use,
        on=[
            "parameter",
            "mine",
            "date",
        ],
        how="inner",
    )


    print(
        f"\nMatched parameters: "
        f"{len(merged)}"
    )


    if len(merged) == 0:

        raise RuntimeError(
            "No analytical/MCMC parameters matched."
        )


    # =========================================================================
    # Differences
    # =========================================================================

    merged["difference"] = (
        merged["analytical_truncated_mean"]
        -
        merged["mcmc_mean"]
    )


    merged["absolute_difference"] = (
        np.abs(
            merged["difference"]
        )
    )


    merged["relative_difference_percent"] = (
        100.0
        *
        merged["difference"]
        /
        np.where(
            np.abs(
                merged["mcmc_mean"]
            ) > 1e-12,

            np.abs(
                merged["mcmc_mean"]
            ),

            np.nan
        )
    )


    merged = merged.sort_values(
        [
            "date",
            "mine",
        ]
    )


    return merged


# =============================================================================
# PRINT COMPARISON STATISTICS
# =============================================================================

def print_statistics(
    merged
):

    print("\n" + "=" * 80)
    print("ANALYTICAL vs MCMC STATISTICS")
    print("=" * 80)


    print(
        "\nOverall:"
    )


    difference = merged[
        "difference"
    ].to_numpy()


    abs_difference = merged[
        "absolute_difference"
    ].to_numpy()


    rmse = np.sqrt(
        np.mean(
            difference ** 2
        )
    )


    mae = np.mean(
        abs_difference
    )


    correlation = np.corrcoef(
        merged["analytical_truncated_mean"],
        merged["mcmc_mean"]
    )[0, 1]


    print(
        f"  Mean difference: "
        f"{np.mean(difference):.8f}"
    )


    print(
        f"  MAE:             "
        f"{mae:.8f}"
    )


    print(
        f"  RMSE:            "
        f"{rmse:.8f}"
    )


    print(
        f"  Correlation:     "
        f"{correlation:.8f}"
    )


    # =========================================================================
    # Mine-by-mine
    # =========================================================================

    print(
        "\nMine-by-mine:"
    )


    rows = []


    for mine in MINE_NAMES:

        tmp = merged[
            merged["mine"] == mine
        ]


        if len(tmp) == 0:

            continue


        d = (
            tmp["analytical_truncated_mean"]
            -
            tmp["mcmc_mean"]
        )


        r = np.corrcoef(
            tmp["analytical_truncated_mean"],
            tmp["mcmc_mean"]
        )[0, 1]


        rows.append(
            {
                "mine":
                    mine,

                "n":
                    len(tmp),

                "analytical_truncated_mean":
                    tmp[
                        "analytical_truncated_mean"
                    ].mean(),

                "mcmc_mean":
                    tmp[
                        "mcmc_mean"
                    ].mean(),

                "mean_difference":
                    d.mean(),

                "MAE":
                    np.abs(d).mean(),

                "RMSE":
                    np.sqrt(
                        np.mean(
                            d ** 2
                        )
                    ),

                "correlation":
                    r,
            }
        )


    stats = pd.DataFrame(
        rows
    )


    print(
        stats.to_string(
            index=False,

            float_format=
            lambda x: f"{x:.6f}"
        )
    )


    return stats


# =============================================================================
# PLOT 13-PANEL TIME SERIES
# =============================================================================

def plot_all_mines(
    merged
):

    print("\n" + "=" * 80)
    print("PLOTTING 13-MINE TIME SERIES")
    print("=" * 80)


    fig, axes = plt.subplots(
        7,
        2,
        figsize=(16, 24),
        sharex=True,
    )


    axes = axes.flatten()


    for i, mine in enumerate(
        MINE_NAMES
    ):

        ax = axes[i]


        tmp = merged[
            merged["mine"] == mine
        ].sort_values(
            "date"
        )


        # ---------------------------------------------------------------------
        # Prior
        # ---------------------------------------------------------------------

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.0,
            label="Prior = 1"
        )


        # ---------------------------------------------------------------------
        # MCMC credible interval
        # ---------------------------------------------------------------------

        ax.fill_between(
            tmp["date"],
            tmp["mcmc_q025"],
            tmp["mcmc_q975"],
            alpha=0.20,
            label="MCMC 95% CI"
        )


        # ---------------------------------------------------------------------
        # MCMC posterior mean
        # ---------------------------------------------------------------------

        ax.plot(
            tmp["date"],
            tmp["mcmc_mean"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="MCMC mean"
        )


        # ---------------------------------------------------------------------
        # Analytical posterior mean
        # ---------------------------------------------------------------------

        ax.plot(
            tmp["date"],
            tmp["analytical_truncated_mean"],
            marker="s",
            markersize=3,
            linewidth=1.5,
            label="Analytical truncated mean"
        )


        ax.set_title(
            mine
        )


        ax.set_ylabel(
            "Emission scaling factor"
        )


        ax.grid(
            True,
            alpha=0.3
        )


        ax.set_xticks(
            DATES[::4]
        )


        ax.tick_params(
            axis="x",
            rotation=45
        )


    # =========================================================================
    # Hide unused 14th panel
    # =========================================================================

    axes[-1].axis(
        "off"
    )


    # =========================================================================
    # Common legend
    # =========================================================================

    handles, labels = axes[0].get_legend_handles_labels()


    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995)
    )


    fig.suptitle(
        "Upper Hunter CH4 Daily Mine Emission Scaling Factors\n"
        "Analytical Bayesian vs Gibbs/MCMC",
        fontsize=16,
        y=1.01
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.975
        ]
    )


    output = os.path.join(
        OUTPUT_DIR,
        "analytical_vs_mcmc_13_mines_timeseries.png"
    )


    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()
    plt.close(
        fig
    )


    print(
        f"\nSaved:\n{output}"
    )


# =============================================================================
# INDIVIDUAL MINE PLOTS
# =============================================================================

def plot_individual_mines(
    merged
):

    individual_dir = os.path.join(
        OUTPUT_DIR,
        "individual_mines"
    )


    os.makedirs(
        individual_dir,
        exist_ok=True
    )


    for mine in MINE_NAMES:

        tmp = merged[
            merged["mine"] == mine
        ].sort_values(
            "date"
        )


        if len(tmp) == 0:

            continue


        fig, ax = plt.subplots(
            figsize=(11, 6)
        )


        # Prior

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.0,
            label="Prior = 1"
        )


        # MCMC uncertainty

        ax.fill_between(
            tmp["date"],
            tmp["mcmc_q025"],
            tmp["mcmc_q975"],
            alpha=0.20,
            label="MCMC 95% CI"
        )


        # MCMC mean

        ax.plot(
            tmp["date"],
            tmp["mcmc_mean"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label="MCMC mean"
        )


        # Analytical truncated mean

        ax.plot(
            tmp["date"],
            tmp["analytical_truncated_mean"],
            marker="s",
            markersize=5,
            linewidth=1.8,
            label="Analytical truncated mean"
        )


        ax.set_title(
            f"{mine} — Daily CH4 emission scaling"
        )


        ax.set_xlabel(
            "Date"
        )


        ax.set_ylabel(
            "Emission scaling factor"
        )


        ax.grid(
            True,
            alpha=0.3
        )


        ax.legend()


        fig.autofmt_xdate()


        fig.tight_layout()


        output = os.path.join(
            individual_dir,
            f"{mine}_analytical_vs_mcmc.png"
        )


        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight"
        )


        plt.show()
        plt.close(
            fig
        )


    print(
        f"\nIndividual mine plots saved in:"
    )

    print(
        individual_dir
    )


# =============================================================================
# PLOT DIRECT ANALYTICAL vs MCMC SCATTER
# =============================================================================

def plot_scatter(
    merged
):

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )


    x = merged[
        "mcmc_mean"
    ].to_numpy()


    y = merged[
        "analytical_truncated_mean"
    ].to_numpy()


    ax.scatter(
        x,
        y,
        s=25,
        alpha=0.65
    )


    # -------------------------------------------------------------------------
    # 1:1 line
    # -------------------------------------------------------------------------

    minimum = min(
        np.min(x),
        np.min(y)
    )


    maximum = max(
        np.max(x),
        np.max(y)
    )


    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.2
    )


    correlation = np.corrcoef(
        x,
        y
    )[0, 1]


    rmse = np.sqrt(
        np.mean(
            (y - x) ** 2
        )
    )


    ax.set_xlabel(
        "MCMC posterior mean"
    )


    ax.set_ylabel(
        "Analytical posterior mean"
    )


    ax.set_title(
        "Analytical vs MCMC posterior scaling factors"
    )


    ax.text(
        0.05,
        0.95,
        f"Correlation = {correlation:.4f}\n"
        f"RMSE = {rmse:.6f}",
        transform=ax.transAxes,
        verticalalignment="top"
    )


    ax.grid(
        True,
        alpha=0.3
    )


    fig.tight_layout()


    output = os.path.join(
        OUTPUT_DIR,
        "analytical_vs_mcmc_scatter.png"
    )


    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()
    plt.close(
        fig
    )


    print(
        f"\nSaved:\n{output}"
    )


# =============================================================================
# SAVE COMPARISON CSV
# =============================================================================

def save_comparison(
    merged,
    stats
):

    comparison_file = os.path.join(
        OUTPUT_DIR,
        "analytical_vs_mcmc_comparison.csv"
    )


    merged.to_csv(
        comparison_file,
        index=False
    )


    stats_file = os.path.join(
        OUTPUT_DIR,
        "analytical_vs_mcmc_statistics.csv"
    )


    stats.to_csv(
        stats_file,
        index=False
    )


    print(
        f"\nSaved comparison:"
    )

    print(
        comparison_file
    )


    print(
        "\nSaved statistics:"
    )

    print(
        stats_file
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")


    print("=" * 80)

    print(
        "UPPER HUNTER CH4"
    )

    print(
        "ANALYTICAL BAYESIAN vs MCMC"
    )

    print("=" * 80)


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # =========================================================================
    # Load
    # =========================================================================

    analytical = load_analytical()

    mcmc = load_mcmc()


    # =========================================================================
    # Check
    # =========================================================================

    check_data(
        analytical,
        mcmc
    )


    # =========================================================================
    # Merge
    # =========================================================================

    merged = merge_results(
        analytical,
        mcmc
    )


    # =========================================================================
    # Statistics
    # =========================================================================

    stats = print_statistics(
        merged
    )


    # =========================================================================
    # Plots
    # =========================================================================

    plot_all_mines(
        merged
    )


    plot_individual_mines(
        merged
    )


    plot_scatter(
        merged
    )


    # =========================================================================
    # Save
    # =========================================================================

    save_comparison(
        merged,
        stats
    )


    # =========================================================================
    # Final diagnostics
    # =========================================================================

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )


    print(
        "\nMain time-series plot:"
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "analytical_vs_mcmc_13_mines_timeseries.png"
        )
    )


    print(
        "\nScatter plot:"
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "analytical_vs_mcmc_scatter.png"
        )
    )


    print(
        "\nComparison CSV:"
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "analytical_vs_mcmc_comparison.csv"
        )
    )


if __name__ == "__main__":

    main()

