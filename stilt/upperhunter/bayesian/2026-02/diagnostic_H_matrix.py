#!/usr/bin/env python3

"""
===============================================================================
STILT H-MATRIX DIAGNOSTIC
Upper Hunter CH4 - 13 Open-Cut Mines
===============================================================================

Input
-----
    STILT_H_matrix_mines_only_Feb2026.csv

Expected columns
----------------
    site
    datetime_utc
    datetime_aest
    OC1 ... OC13
    OC_TOTAL

Purpose
-------
Diagnose whether the 13 mines are independently identifiable from the
current STILT source-receptor sensitivities.

Diagnostics
-----------
1. H matrix dimensions
2. Number of nonzero observations for each mine
3. Basic H statistics
4. 13 x 13 Pearson correlation matrix
5. Strongest correlated mine pairs
6. SVD singular values
7. Singular-value ratios
8. Condition number
9. Numerical rank
10. Effective rank
11. Normalized source fingerprints
12. Correlation heatmap
13. Singular-value plot

Important
---------
The H matrix currently represents:

    C_i = C_background + sum_j H_ij * s_j

where:

    s_j = emission scaling factor for mine j

This diagnostic therefore tests the identifiability of 13 CONSTANT
mine scaling factors.

It does NOT yet test the 13 x 365 daily-factor problem.

===============================================================================
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_H_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-02/"
    "H_matrix/STILT_H_matrix_mines_only_Feb2026.csv"
)

OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-02/"
    "H_diagnostic_Feb2026"
)

MINES = [
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

# Correlation threshold for reporting strong pairs
STRONG_CORRELATION_THRESHOLD = 0.90

# Numerical-rank tolerance.
#
# Singular values smaller than:
#
#     max(singular_values) * RANK_TOLERANCE
#
# are treated as numerically insignificant.
RANK_TOLERANCE = 1.0e-10


# =============================================================================
# FUNCTIONS
# =============================================================================

def print_header(title):
    """Print a section header."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def load_H_matrix(filename):
    """Load H matrix CSV."""

    print_header("LOADING H MATRIX")

    print("Input file:")
    print(filename)
    print()

    df = pd.read_csv(filename)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()

    missing = [
        mine for mine in MINES
        if mine not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing H columns: "
            + ", ".join(missing)
        )

    H = df[MINES].to_numpy(
        dtype=np.float64
    )

    print("H matrix shape:")
    print(
        H.shape[0],
        "observations x",
        H.shape[1],
        "mines"
    )

    return df, H


def calculate_basic_statistics(df, H):
    """Calculate basic H statistics."""

    print_header("BASIC H MATRIX STATISTICS")

    rows = []

    for j, mine in enumerate(MINES):

        values = H[:, j]

        nonzero = np.count_nonzero(
            np.abs(values) > 0.0
        )

        positive = np.count_nonzero(
            values > 0.0
        )

        rows.append({
            "mine": mine,
            "n_observations": len(values),
            "n_nonzero": nonzero,
            "fraction_nonzero": (
                nonzero / len(values)
            ),
            "n_positive": positive,
            "min_ppm": np.min(values),
            "max_ppm": np.max(values),
            "mean_ppm": np.mean(values),
            "median_ppm": np.median(values),
            "std_ppm": np.std(values),
            "sum_ppm": np.sum(values),
        })

    stats = pd.DataFrame(rows)

    print(
        stats.to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}"
        )
    )

    return stats


def calculate_correlation(H):
    """Calculate Pearson correlation matrix."""

    print_header("MINE-MINE CORRELATION MATRIX")

    correlation = np.corrcoef(
        H,
        rowvar=False
    )

    corr_df = pd.DataFrame(
        correlation,
        index=MINES,
        columns=MINES
    )

    print(
        corr_df.to_string(
            float_format=lambda x: f"{x: .4f}"
        )
    )

    return corr_df


def find_strong_correlations(correlation):
    """
    Find strongest off-diagonal correlations.
    """

    pairs = []

    for i in range(len(MINES)):

        for j in range(i + 1, len(MINES)):

            r = correlation.iloc[i, j]

            pairs.append({
                "mine_1": MINES[i],
                "mine_2": MINES[j],
                "correlation": r,
                "absolute_correlation": abs(r),
            })

    pairs_df = pd.DataFrame(
        pairs
    ).sort_values(
        "absolute_correlation",
        ascending=False
    ).reset_index(
        drop=True
    )

    return pairs_df


def calculate_svd(H):
    """Calculate singular-value decomposition."""

    print_header("SINGULAR VALUE DECOMPOSITION")

    # Remove columns that are exactly zero.
    #
    # This should normally not happen for your current H matrix,
    # but the check makes the diagnostic robust.

    column_norms = np.linalg.norm(
        H,
        axis=0
    )

    active_columns = column_norms > 0.0

    H_active = H[
        :,
        active_columns
    ]

    active_mines = [
        MINES[i]
        for i in range(len(MINES))
        if active_columns[i]
    ]

    print(
        "Active mines:",
        len(active_mines)
    )

    if len(active_mines) < len(MINES):

        inactive = [
            MINES[i]
            for i in range(len(MINES))
            if not active_columns[i]
        ]

        print(
            "Inactive mines:",
            ", ".join(inactive)
        )

    # SVD
    U, singular_values, Vt = np.linalg.svd(
        H_active,
        full_matrices=False
    )

    print()
    print("Singular values:")
    print()

    for i, value in enumerate(
        singular_values,
        start=1
    ):

        relative = (
            value
            / singular_values[0]
        )

        print(
            f"  {i:2d}  "
            f"{value:.8e}   "
            f"relative = {relative:.8e}"
        )

    # Numerical rank
    tolerance = (
        singular_values[0]
        * RANK_TOLERANCE
    )

    numerical_rank = np.count_nonzero(
        singular_values > tolerance
    )

    # Condition number
    if singular_values[-1] > 0.0:

        condition_number = (
            singular_values[0]
            / singular_values[-1]
        )

    else:

        condition_number = np.inf

    # Effective rank based on entropy
    normalized = (
        singular_values
        / np.sum(singular_values)
    )

    entropy = -np.sum(
        normalized
        * np.log(normalized)
    )

    effective_rank = np.exp(
        entropy
    )

    print()
    print(
        "Numerical rank:",
        numerical_rank
    )

    print(
        "Effective rank:",
        f"{effective_rank:.4f}"
    )

    print(
        "Condition number:",
        f"{condition_number:.6e}"
    )

    print(
        "Rank tolerance:",
        f"{RANK_TOLERANCE:.1e}"
    )

    return {
        "U": U,
        "singular_values": singular_values,
        "Vt": Vt,
        "active_mines": active_mines,
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "tolerance": tolerance,
    }


def calculate_normalized_fingerprints(H):
    """
    Normalize each mine H column to unit Euclidean norm.

    This is useful for comparing the SHAPE of source-receptor
    sensitivities independently of total magnitude.
    """

    print_header("NORMALIZED SOURCE FINGERPRINTS")

    norms = np.linalg.norm(
        H,
        axis=0
    )

    normalized = np.zeros_like(H)

    for j in range(H.shape[1]):

        if norms[j] > 0.0:

            normalized[:, j] = (
                H[:, j]
                / norms[j]
            )

    normalized_df = pd.DataFrame(
        normalized,
        columns=MINES
    )

    return normalized_df


def calculate_pairwise_distance(H):
    """
    Calculate cosine similarity between H columns.

    Because H is non-negative, this provides another measure of how
    similar the source-receptor fingerprints are.
    """

    print_header("SOURCE FINGERPRINT COSINE SIMILARITY")

    norms = np.linalg.norm(
        H,
        axis=0
    )

    similarity = np.zeros(
        (len(MINES), len(MINES))
    )

    for i in range(len(MINES)):

        for j in range(len(MINES)):

            if (
                norms[i] > 0.0
                and norms[j] > 0.0
            ):

                similarity[i, j] = (
                    np.dot(
                        H[:, i],
                        H[:, j]
                    )
                    / (
                        norms[i]
                        * norms[j]
                    )
                )

    similarity_df = pd.DataFrame(
        similarity,
        index=MINES,
        columns=MINES
    )

    print(
        similarity_df.to_string(
            float_format=lambda x: f"{x: .4f}"
        )
    )

    return similarity_df


def make_correlation_plot(correlation, filename):
    """Create correlation heatmap."""

    print()
    print(
        "Creating correlation heatmap..."
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        correlation.values,
        vmin=-1.0,
        vmax=1.0,
        aspect="equal"
    )

    ax.set_xticks(
        np.arange(len(MINES))
    )

    ax.set_yticks(
        np.arange(len(MINES))
    )

    ax.set_xticklabels(
        MINES,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        MINES
    )

    ax.set_xlabel(
        "Mine"
    )

    ax.set_ylabel(
        "Mine"
    )

    ax.set_title(
        "STILT H-Matrix Mine Correlation"
    )

    # Add numerical correlation values
    for i in range(len(MINES)):

        for j in range(len(MINES)):

            value = correlation.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Pearson correlation"
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=200
    )

    plt.show()
    plt.close(fig)


def make_singular_value_plot(
    singular_values,
    filename
):
    """Create singular-value plot."""

    print(
        "Creating singular-value plot..."
    )

    indices = np.arange(
        1,
        len(singular_values) + 1
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.semilogy(
        indices,
        singular_values,
        marker="o"
    )

    ax.set_xlabel(
        "Singular value index"
    )

    ax.set_ylabel(
        "Singular value"
    )

    ax.set_title(
        "STILT H-Matrix Singular Values"
    )

    ax.grid(
        True,
        which="both",
        axis="both"
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=200
    )

    plt.show()
    plt.close(fig)


def make_h_distribution_plot(
    H,
    filename
):
    """
    Plot distribution of H values for each mine.

    Uses log10(H) for positive values because the H values span
    several orders of magnitude.
    """

    print(
        "Creating H-value distribution plot..."
    )

    data = []

    labels = []

    for j, mine in enumerate(MINES):

        values = H[:, j]

        positive = values[
            values > 0.0
        ]

        if len(positive) > 0:

            data.append(
                np.log10(positive)
            )

            labels.append(
                mine
            )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.boxplot(
        data,
        labels=labels,
        showfliers=False
    )

    ax.set_xlabel(
        "Mine"
    )

    ax.set_ylabel(
        "log10(H [ppm])"
    )

    ax.set_title(
        "Distribution of Positive STILT H Values"
    )

    ax.grid(
        True,
        axis="y"
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=200
    )

    plt.show()
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Load H matrix
    # -------------------------------------------------------------------------

    df, H = load_H_matrix(
        INPUT_H_FILE
    )

    # -------------------------------------------------------------------------
    # Basic statistics
    # -------------------------------------------------------------------------

    statistics_df = calculate_basic_statistics(
        df,
        H
    )

    statistics_file = os.path.join(
        OUTPUT_DIR,
        "H_basic_statistics.csv"
    )

    statistics_df.to_csv(
        statistics_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Correlation
    # -------------------------------------------------------------------------

    correlation_df = calculate_correlation(
        H
    )

    correlation_file = os.path.join(
        OUTPUT_DIR,
        "H_mine_correlation.csv"
    )

    correlation_df.to_csv(
        correlation_file
    )

    # -------------------------------------------------------------------------
    # Strong correlations
    # -------------------------------------------------------------------------

    print_header(
        "STRONGEST MINE-MINE CORRELATIONS"
    )

    pairs_df = find_strong_correlations(
        correlation_df
    )

    print(
        pairs_df.head(30).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    strong_pairs = pairs_df[
        pairs_df[
            "absolute_correlation"
        ]
        >= STRONG_CORRELATION_THRESHOLD
    ]

    print()

    print(
        f"Pairs with |r| >= "
        f"{STRONG_CORRELATION_THRESHOLD:.2f}: "
        f"{len(strong_pairs)}"
    )

    print()

    if len(strong_pairs) > 0:

        print(
            strong_pairs.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    else:

        print(
            "No strongly correlated pairs found."
        )

    pairs_file = os.path.join(
        OUTPUT_DIR,
        "H_strongest_mine_pairs.csv"
    )

    pairs_df.to_csv(
        pairs_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # SVD
    # -------------------------------------------------------------------------

    svd_results = calculate_svd(
        H
    )

    singular_values = (
        svd_results[
            "singular_values"
        ]
    )

    singular_df = pd.DataFrame({
        "index": np.arange(
            1,
            len(singular_values) + 1
        ),
        "singular_value":
            singular_values,
        "relative_to_largest":
            singular_values
            / singular_values[0],
    })

    singular_file = os.path.join(
        OUTPUT_DIR,
        "H_singular_values.csv"
    )

    singular_df.to_csv(
        singular_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Normalized source fingerprints
    # -------------------------------------------------------------------------

    normalized_df = (
        calculate_normalized_fingerprints(
            H
        )
    )

    normalized_file = os.path.join(
        OUTPUT_DIR,
        "H_normalized_source_fingerprints.csv"
    )

    # Add observation metadata
    normalized_output = pd.concat(
        [
            df[
                [
                    "site",
                    "datetime_utc",
                    "datetime_aest",
                ]
            ].reset_index(drop=True),
            normalized_df,
        ],
        axis=1
    )

    normalized_output.to_csv(
        normalized_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Cosine similarity
    # -------------------------------------------------------------------------

    cosine_df = calculate_pairwise_distance(
        H
    )

    cosine_file = os.path.join(
        OUTPUT_DIR,
        "H_mine_cosine_similarity.csv"
    )

    cosine_df.to_csv(
        cosine_file
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    correlation_plot = os.path.join(
        OUTPUT_DIR,
        "H_mine_correlation.png"
    )

    make_correlation_plot(
        correlation_df,
        correlation_plot
    )

    singular_plot = os.path.join(
        OUTPUT_DIR,
        "H_singular_values.png"
    )

    make_singular_value_plot(
        singular_values,
        singular_plot
    )

    distribution_plot = os.path.join(
        OUTPUT_DIR,
        "H_value_distributions.png"
    )

    make_h_distribution_plot(
        H,
        distribution_plot
    )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    report_file = os.path.join(
        OUTPUT_DIR,
        "H_diagnostic_summary.txt"
    )

    with open(
        report_file,
        "w"
    ) as f:

        f.write(
            "STILT H-MATRIX DIAGNOSTIC\n"
        )

        f.write(
            "Upper Hunter CH4 - 13 mines\n"
        )

        f.write(
            "====================================\n\n"
        )

        f.write(
            f"Observations: {H.shape[0]}\n"
        )

        f.write(
            f"Mine parameters: {H.shape[1]}\n"
        )

        f.write(
            f"Numerical rank: "
            f"{svd_results['numerical_rank']}\n"
        )

        f.write(
            f"Effective rank: "
            f"{svd_results['effective_rank']:.6f}\n"
        )

        f.write(
            f"Condition number: "
            f"{svd_results['condition_number']:.6e}\n"
        )

        f.write(
            f"Rank tolerance: "
            f"{RANK_TOLERANCE:.6e}\n"
        )

        f.write(
            f"\nStrong correlation threshold: "
            f"|r| >= "
            f"{STRONG_CORRELATION_THRESHOLD:.2f}\n"
        )

        f.write(
            f"Number of strong pairs: "
            f"{len(strong_pairs)}\n"
        )

        f.write(
            "\nSingular values:\n"
        )

        for i, value in enumerate(
            singular_values,
            start=1
        ):

            f.write(
                f"{i:2d} "
                f"{value:.10e}\n"
            )

        f.write(
            "\nStrongest correlation pairs:\n"
        )

        for _, row in pairs_df.head(20).iterrows():

            f.write(
                f"{row['mine_1']:5s} "
                f"{row['mine_2']:5s} "
                f"r={row['correlation']:.6f}\n"
            )

    # -------------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------------

    print_header(
        "FINAL H-MATRIX DIAGNOSTIC SUMMARY"
    )

    print(
        "Number of observations:",
        H.shape[0]
    )

    print(
        "Number of mines:",
        H.shape[1]
    )

    print(
        "Numerical rank:",
        svd_results["numerical_rank"]
    )

    print(
        "Effective rank:",
        f"{svd_results['effective_rank']:.4f}"
    )

    print(
        "Condition number:",
        f"{svd_results['condition_number']:.6e}"
    )

    print(
        "Strongly correlated pairs:",
        len(strong_pairs)
    )

    print()
    print("Output directory:")
    print(
        OUTPUT_DIR
    )

    print()
    print("Output files:")
    print(
        "  H_basic_statistics.csv"
    )
    print(
        "  H_mine_correlation.csv"
    )
    print(
        "  H_strongest_mine_pairs.csv"
    )
    print(
        "  H_singular_values.csv"
    )
    print(
        "  H_normalized_source_fingerprints.csv"
    )
    print(
        "  H_mine_cosine_similarity.csv"
    )
    print(
        "  H_mine_correlation.png"
    )
    print(
        "  H_singular_values.png"
    )
    print(
        "  H_value_distributions.png"
    )
    print(
        "  H_diagnostic_summary.txt"
    )

    print()
    print(
        "Diagnostic complete."
    )


if __name__ == "__main__":
    main()

