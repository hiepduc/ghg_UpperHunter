#!/usr/bin/env python3

"""
Upper Hunter CH4 inversion
KF vs RTS smoother vs batch MAP comparison

13 mines x 27 days = 351 daily scaling factors.

Inputs
------
1. Sequential KF:
   full_footprint_KF_daily_states.npz

2. RTS smoother:
   daily_RTS_smoothed_scaling_factors.csv
   or RTS_full_footprint_daily_states.npz

3. Batch MAP:
   daily_batch_MAP_scaling_factors.csv

Outputs
-------
- all_methods_daily_scaling_factors.csv
- daily_method_differences.csv
- method_comparison_diagnostics.csv
- RTS_vs_KF_vs_MAP_diagnostics.txt
- all_mines_KF_RTS_MAP.png
- RTS_minus_KF_heatmap.png
- all_351_KF_vs_RTS_MAP_scatter.png
- OC1_KF_RTS_MAP.png ... OC13_KF_RTS_MAP.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# PATHS
# ============================================================================

BASE = Path.home() / "flexpart-wrf" / "upperhunter_flexpart"

OUTDIR = (
    BASE
    / "H_matrix_flexpart"
    / "KF_full_footprint_13mine_Jan2026"
)

KF_NPZ = OUTDIR / "full_footprint_KF_daily_states.npz"

RTS_CSV = OUTDIR / "daily_RTS_smoothed_scaling_factors.csv"
RTS_NPZ = OUTDIR / "RTS_full_footprint_daily_states.npz"

MAP_CSV = OUTDIR / "daily_batch_MAP_scaling_factors.csv"


# ============================================================================
# EXPECTED MODEL DIMENSIONS
# ============================================================================

N_DAYS = 27
N_MINES = 13

MINES = [f"OC{i}" for i in range(1, 14)]


# ============================================================================
# PLOTTING SETTINGS
# ============================================================================

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


# ============================================================================
# BASIC STATISTICS
# ============================================================================

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.mean((a - b) ** 2))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.mean(np.abs(a - b))


def max_abs(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.max(np.abs(a - b))


def correlation(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


# ============================================================================
# LOAD KF
# ============================================================================

def load_kf():

    print("\nLoading KF NPZ:")
    print(KF_NPZ)

    z = np.load(KF_NPZ, allow_pickle=True)

    print("KF keys:")
    print(list(z.keys()))

    states = np.asarray(z["states"], dtype=float)
    dates = pd.to_datetime(z["dates"])
    mines = [str(x) for x in z["mines"]]

    print("KF states:", states.shape)
    print("KF dates:", dates.min(), "to", dates.max())
    print("KF mines:", mines)

    if states.shape != (N_DAYS, N_MINES):
        raise ValueError(
            f"Unexpected KF state shape: {states.shape}; "
            f"expected {(N_DAYS, N_MINES)}"
        )

    df = pd.DataFrame(
        states,
        index=dates,
        columns=mines,
    )

    df.index.name = "date"

    return df.sort_index()


# ============================================================================
# IDENTIFY DATE COLUMN
# ============================================================================

def find_date_column(df):

    preferred = [
        "date",
        "Date",
        "day",
        "Day",
        "datetime",
        "Datetime",
        "time",
        "Time",
    ]

    for col in preferred:
        if col in df.columns:
            return col

    # Try to identify automatically
    for col in df.columns:

        parsed = pd.to_datetime(
            df[col],
            errors="coerce",
        )

        if parsed.notna().sum() >= max(5, len(df) // 2):
            return col

    raise ValueError(
        "Could not identify date column.\n"
        f"Columns: {list(df.columns)}"
    )


# ============================================================================
# FIND MINE COLUMNS
# ============================================================================

def find_mine_columns(df):

    result = {}

    for mine in MINES:

        # Exact column
        if mine in df.columns:
            result[mine] = mine
            continue

        # Try stripped names
        matches = []

        for col in df.columns:

            name = str(col).strip()

            if name == mine:
                matches.append(col)

            elif name.startswith(mine + "_"):
                matches.append(col)

            elif name.startswith(mine + " "):
                matches.append(col)

        if len(matches) == 1:
            result[mine] = matches[0]

    return result


# ============================================================================
# LOAD A DAILY CSV
# ============================================================================

def load_daily_csv(path, label):

    print(f"\nLoading {label}:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    print("Columns:")
    print(list(df.columns))

    date_col = find_date_column(df)

    print("Date column:", date_col)

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="raise",
    )

    mine_columns = find_mine_columns(df)

    print("Mine columns:")
    print(mine_columns)

    if len(mine_columns) != N_MINES:

        raise ValueError(
            f"{label}: found {len(mine_columns)} mine columns, "
            f"expected {N_MINES}.\n"
            f"Detected: {mine_columns}\n"
            f"All columns: {list(df.columns)}"
        )

    result = pd.DataFrame(index=df[date_col])

    for mine in MINES:

        result[mine] = pd.to_numeric(
            df[mine_columns[mine]],
            errors="coerce",
        ).values

    result.index.name = "date"

    return result.sort_index()


# ============================================================================
# LOAD RTS
# ============================================================================

def load_rts():

    if RTS_CSV.exists():

        print("\nUsing RTS CSV:")
        return load_daily_csv(
            RTS_CSV,
            "RTS smoother",
        )

    print("\nRTS CSV not found.")
    print("Trying RTS NPZ.")

    if not RTS_NPZ.exists():
        raise FileNotFoundError(
            "Neither RTS CSV nor RTS NPZ exists."
        )

    z = np.load(
        RTS_NPZ,
        allow_pickle=True,
    )

    print("RTS NPZ keys:")
    print(list(z.keys()))

    candidate_keys = [
        "smoothed_states",
        "states",
        "rts_states",
        "scaling_factors",
    ]

    state_key = None

    for key in candidate_keys:

        if key in z:

            arr = np.asarray(
                z[key],
                dtype=float,
            )

            if arr.shape == (N_DAYS, N_MINES):

                state_key = key
                break

    if state_key is None:

        raise ValueError(
            "Could not find a (27,13) RTS state array."
        )

    dates = pd.to_datetime(z["dates"])

    if "mines" in z:
        mines = [str(x) for x in z["mines"]]
    else:
        mines = MINES

    result = pd.DataFrame(
        np.asarray(z[state_key], dtype=float),
        index=dates,
        columns=mines,
    )

    result.index.name = "date"

    return result.sort_index()


# ============================================================================
# ALIGN ALL THREE METHODS
# ============================================================================

def align_methods(kf, rts, map_df):

    print("\n" + "=" * 90)
    print("ALIGNING KF / RTS / MAP")
    print("=" * 90)

    common_dates = (
        kf.index
        .intersection(rts.index)
        .intersection(map_df.index)
    )

    if len(common_dates) != N_DAYS:

        raise ValueError(
            f"Expected {N_DAYS} common dates, "
            f"found {len(common_dates)}."
        )

    for df_name, df in [
        ("KF", kf),
        ("RTS", rts),
        ("MAP", map_df),
    ]:

        missing = [
            mine
            for mine in MINES
            if mine not in df.columns
        ]

        if missing:

            raise ValueError(
                f"{df_name} missing mines: {missing}"
            )

    kf = kf.loc[common_dates, MINES].sort_index()
    rts = rts.loc[common_dates, MINES].sort_index()
    map_df = map_df.loc[common_dates, MINES].sort_index()

    print("Dates:")
    print(
        kf.index[0].date(),
        "to",
        kf.index[-1].date(),
    )

    print("Days:", len(kf))
    print("Mines:", len(MINES))
    print("Total parameters:", len(kf) * len(MINES))

    return kf, rts, map_df


# ============================================================================
# SAVE LONG-FORM COMPARISON
# ============================================================================

def save_combined_table(kf, rts, map_df):

    rows = []

    for date in kf.index:

        for mine in MINES:

            k = kf.loc[date, mine]
            r = rts.loc[date, mine]
            m = map_df.loc[date, mine]

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "mine": mine,
                    "KF": k,
                    "RTS": r,
                    "MAP": m,
                    "RTS_minus_KF": r - k,
                    "MAP_minus_KF": m - k,
                    "RTS_minus_MAP": r - m,
                }
            )

    result = pd.DataFrame(rows)

    path = OUTDIR / "all_methods_daily_scaling_factors.csv"

    result.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================================
# DAILY DIFFERENCES
# ============================================================================

def save_difference_table(kf, rts, map_df):

    rows = []

    for date in kf.index:

        for mine in MINES:

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "mine": mine,
                    "RTS_minus_KF":
                        rts.loc[date, mine]
                        - kf.loc[date, mine],

                    "MAP_minus_KF":
                        map_df.loc[date, mine]
                        - kf.loc[date, mine],

                    "RTS_minus_MAP":
                        rts.loc[date, mine]
                        - map_df.loc[date, mine],
                }
            )

    result = pd.DataFrame(rows)

    path = OUTDIR / "daily_method_differences.csv"

    result.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================================
# DIAGNOSTICS PER MINE
# ============================================================================

def make_diagnostics(kf, rts, map_df):

    rows = []

    for mine in MINES:

        rows.append(
            {
                "mine": mine,

                "KF_mean":
                    kf[mine].mean(),

                "RTS_mean":
                    rts[mine].mean(),

                "MAP_mean":
                    map_df[mine].mean(),

                "KF_std":
                    kf[mine].std(),

                "RTS_std":
                    rts[mine].std(),

                "MAP_std":
                    map_df[mine].std(),

                "KF_RTS_RMSE":
                    rmse(kf[mine], rts[mine]),

                "KF_RTS_MAE":
                    mae(kf[mine], rts[mine]),

                "KF_RTS_MaxAbs":
                    max_abs(kf[mine], rts[mine]),

                "KF_RTS_Correlation":
                    correlation(kf[mine], rts[mine]),

                "KF_MAP_RMSE":
                    rmse(kf[mine], map_df[mine]),

                "KF_MAP_MAE":
                    mae(kf[mine], map_df[mine]),

                "KF_MAP_MaxAbs":
                    max_abs(kf[mine], map_df[mine]),

                "KF_MAP_Correlation":
                    correlation(kf[mine], map_df[mine]),

                "RTS_MAP_RMSE":
                    rmse(rts[mine], map_df[mine]),

                "RTS_MAP_MAE":
                    mae(rts[mine], map_df[mine]),

                "RTS_MAP_MaxAbs":
                    max_abs(rts[mine], map_df[mine]),

                "RTS_MAP_Correlation":
                    correlation(rts[mine], map_df[mine]),
            }
        )

    result = pd.DataFrame(rows)

    path = OUTDIR / "method_comparison_diagnostics.csv"

    result.to_csv(
        path,
        index=False,
    )

    return result, path


# ============================================================================
# GLOBAL DIAGNOSTICS
# ============================================================================

def save_global_diagnostics(kf, rts, map_df):

    comparisons = [
        ("KF_vs_RTS", kf, rts),
        ("KF_vs_MAP", kf, map_df),
        ("RTS_vs_MAP", rts, map_df),
    ]

    lines = []

    lines.append(
        "UPPER HUNTER CH4 — KF vs RTS vs BATCH MAP"
    )

    lines.append("=" * 80)

    lines.append(
        f"Period: {kf.index.min().date()} "
        f"to {kf.index.max().date()}"
    )

    lines.append(
        f"Parameters: {N_MINES} mines x {N_DAYS} days = "
        f"{N_MINES * N_DAYS}"
    )

    lines.append("")

    for name, a, b in comparisons:

        lines.append(name)
        lines.append("-" * 80)

        lines.append(
            f"RMSE       = "
            f"{rmse(a.values, b.values):.16e}"
        )

        lines.append(
            f"MAE        = "
            f"{mae(a.values, b.values):.16e}"
        )

        lines.append(
            f"Max abs    = "
            f"{max_abs(a.values, b.values):.16e}"
        )

        lines.append(
            f"Correlation= "
            f"{correlation(a.values, b.values):.16f}"
        )

        lines.append("")

    path = OUTDIR / "RTS_vs_KF_vs_MAP_diagnostics.txt"

    path.write_text(
        "\n".join(lines)
    )

    return path


# ============================================================================
# PLOT 1 — ALL 13 MINES
# ============================================================================

def plot_all_mines(kf, rts, map_df):

    fig, axes = plt.subplots(
        N_MINES,
        1,
        figsize=(15, 32),
        sharex=True,
    )

    for i, mine in enumerate(MINES):

        ax = axes[i]

        ax.plot(
            kf.index,
            kf[mine],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="KF",
        )

        ax.plot(
            rts.index,
            rts[mine],
            marker="s",
            markersize=3,
            linewidth=1.2,
            label="RTS smoother",
        )

        ax.plot(
            map_df.index,
            map_df[mine],
            marker="^",
            markersize=3,
            linewidth=1.2,
            label="Batch MAP",
        )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=0.8,
            label="Prior = 1" if i == 0 else None,
        )

        ax.set_ylabel(mine)

        ax.set_title(
            mine,
            loc="left",
            fontsize=10,
        )

        if i == 0:
            ax.legend(
                ncol=4,
                loc="upper right",
            )

    axes[-1].set_xlabel(
        "Date"
    )

    fig.suptitle(
        "Upper Hunter CH4 daily scaling factors\n"
        "Kalman Filter vs RTS smoother vs Batch MAP",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.985]
    )

    path = OUTDIR / "all_mines_KF_RTS_MAP.png"

    fig.savefig(
        path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    return path


# ============================================================================
# PLOT 2 — RTS MINUS KF HEATMAP
# ============================================================================

def plot_rts_minus_kf_heatmap(kf, rts):

    diff = (
        rts - kf
    ).T

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    image = ax.imshow(
        diff.values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_yticks(
        np.arange(N_MINES)
    )

    ax.set_yticklabels(
        MINES
    )

    ax.set_xticks(
        np.arange(N_DAYS)
    )

    ax.set_xticklabels(
        [
            d.strftime("%d %b")
            for d in diff.columns
        ],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Mine"
    )

    ax.set_title(
        "RTS smoother − Kalman filter"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "RTS − KF scaling factor"
    )

    fig.tight_layout()

    path = OUTDIR / "RTS_minus_KF_heatmap.png"

    fig.savefig(
        path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    return path


# ============================================================================
# PLOT 3 — ALL 351 PARAMETERS
# ============================================================================

def plot_351_scatter(kf, rts, map_df):

    x = kf.values.ravel()
    y_rts = rts.values.ravel()
    y_map = map_df.values.ravel()

    lo = min(
        x.min(),
        y_rts.min(),
        y_map.min(),
    )

    hi = max(
        x.max(),
        y_rts.max(),
        y_map.max(),
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        x,
        y_rts,
        s=20,
        alpha=0.7,
        label="RTS vs KF",
    )

    ax.scatter(
        x,
        y_map,
        s=20,
        alpha=0.7,
        marker="x",
        label="MAP vs KF",
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "KF daily scaling factor"
    )

    ax.set_ylabel(
        "RTS / MAP daily scaling factor"
    )

    ax.set_title(
        "All 351 daily parameters"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        OUTDIR
        / "all_351_KF_vs_RTS_MAP_scatter.png"
    )

    fig.savefig(
        path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    return path


# ============================================================================
# PLOT 4 — ONE PLOT PER MINE
# ============================================================================

def plot_each_mine(kf, rts, map_df):

    paths = []

    for mine in MINES:

        fig, ax = plt.subplots(
            figsize=(12, 5.5)
        )

        ax.plot(
            kf.index,
            kf[mine],
            marker="o",
            label="KF",
        )

        ax.plot(
            rts.index,
            rts[mine],
            marker="s",
            label="RTS smoother",
        )

        ax.plot(
            map_df.index,
            map_df[mine],
            marker="^",
            label="Batch MAP",
        )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1,
            label="Prior = 1",
        )

        ax.set_title(
            f"{mine}: daily posterior scaling factor"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Scaling factor"
        )

        ax.legend()

        fig.tight_layout()

        path = (
            OUTDIR
            / f"{mine}_KF_RTS_MAP.png"
        )

        fig.savefig(
            path,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

        paths.append(path)

    return paths


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 90)
    print(
        "UPPER HUNTER CH4 — "
        "KF vs RTS vs BATCH MAP"
    )
    print("=" * 90)

    print("\nOutput directory:")
    print(OUTDIR)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    kf = load_kf()

    rts = load_rts()

    map_df = load_daily_csv(
        MAP_CSV,
        "Batch MAP",
    )

    # ------------------------------------------------------------------
    # Align
    # ------------------------------------------------------------------

    kf, rts, map_df = align_methods(
        kf,
        rts,
        map_df,
    )

    # ------------------------------------------------------------------
    # Global diagnostics
    # ------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("GLOBAL DIAGNOSTICS")
    print("=" * 90)

    comparisons = [
        ("KF vs RTS", kf, rts),
        ("KF vs MAP", kf, map_df),
        ("RTS vs MAP", rts, map_df),
    ]

    for name, a, b in comparisons:

        print(
            f"{name:12s} "
            f"RMSE={rmse(a.values,b.values):.6e}  "
            f"MAE={mae(a.values,b.values):.6e}  "
            f"MAX={max_abs(a.values,b.values):.6e}  "
            f"CORR={correlation(a.values,b.values):.12f}"
        )

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------

    print("\nSaving comparison tables...")

    p1 = save_combined_table(
        kf,
        rts,
        map_df,
    )

    p2 = save_difference_table(
        kf,
        rts,
        map_df,
    )

    diagnostics_df, p3 = make_diagnostics(
        kf,
        rts,
        map_df,
    )

    p4 = save_global_diagnostics(
        kf,
        rts,
        map_df,
    )

    print(p1)
    print(p2)
    print(p3)
    print(p4)

    # ------------------------------------------------------------------
    # Print per-mine summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("PER-MINE RTS vs KF")
    print("=" * 90)

    print(
        diagnostics_df[
            [
                "mine",
                "KF_RTS_RMSE",
                "KF_RTS_MAE",
                "KF_RTS_MaxAbs",
                "KF_RTS_Correlation",
                "RTS_MAP_RMSE",
                "RTS_MAP_MaxAbs",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    print("\nGenerating plots...")

    plot_paths = []

    plot_paths.append(
        plot_all_mines(
            kf,
            rts,
            map_df,
        )
    )

    plot_paths.append(
        plot_rts_minus_kf_heatmap(
            kf,
            rts,
        )
    )

    plot_paths.append(
        plot_351_scatter(
            kf,
            rts,
            map_df,
        )
    )

    plot_paths.extend(
        plot_each_mine(
            kf,
            rts,
            map_df,
        )
    )

    print("\nGenerated plots:")

    for path in plot_paths:
        print(path)

    print("\n" + "=" * 90)
    print("COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()

