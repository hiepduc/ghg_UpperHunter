#!/usr/bin/env python3
"""
Compare FLEXPART-WRF H sensitivities between:

    3-layer:
        0-10 m
        10-100 m
        100-20000 m

    2-layer:
        0-100 m
        100-20000 m

Same:
    - receptor
    - release
    - WRF meteorology
    - horizontal grid
    - mine weights
    - 500 particles
    - 24-h backward simulation

The script calculates:

    H_hourly_mine
        shape = (24, 13)

    H_layer_hourly
        shape = (24, nlayer)

    H_total_hourly
        shape = (24,)

and compares the two configurations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path.home() / "flexpart-wrf" / "upperhunter_flexpart"

RUN_3L = BASE / "run" / "singleton_10m"
RUN_2L = BASE / "run" / "singleton_2layer_clean"

WEIGHT_FILE = (
    BASE
    / "hmatrix"
    / "test_20260105"
    / "mine_area_fraction_20260105.npy"
)

OUTPUT_DIR = BASE / "hmatrix" / "daily_test" / "vertical_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONC_FILE_NAME = "flxout_d01_20260110_010000.nc"
HEADER_FILE_NAME = "header_d01.nc"

PPMV_FACTOR = 1.0e6 * 28.97 / 16.043

MINES = [f"OC{i}" for i in range(1, 14)]

RELEASE_UTC = "2026-01-10 02:00:00"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def read_times(nc):
    """
    Read FLEXPART Times and convert to pandas UTC timestamps.
    """
    times = nc.variables["Times"][:]

    result = []

    for row in times:
        if hasattr(row, "tobytes"):
            text = row.tobytes().decode("ascii")
        else:
            text = "".join(
                x.decode("ascii") if isinstance(x, bytes) else str(x)
                for x in row
            )

        text = text.replace("\x00", "").strip()

        result.append(
            pd.to_datetime(
                text,
                format="%Y%m%d_%H%M%S",
                utc=True,
            )
        )

    return pd.DatetimeIndex(result)


def read_geometry(run_dir):
    """
    Read FLEXPART grid geometry and vertical layer thicknesses.
    """

    header = run_dir / HEADER_FILE_NAME

    with Dataset(header, "r") as nc:

        gridarea = np.asarray(
            nc.variables["GRIDAREA"][:],
            dtype=np.float64,
        )

        ztop = np.asarray(
            nc.variables["ZTOP"][:],
            dtype=np.float64,
        )

        xlong = np.asarray(
            nc.variables["XLONG"][:],
            dtype=np.float64,
        )

        xlat = np.asarray(
            nc.variables["XLAT"][:],
            dtype=np.float64,
        )

    thickness = np.empty(len(ztop), dtype=np.float64)

    thickness[0] = ztop[0]

    for k in range(1, len(ztop)):
        thickness[k] = ztop[k] - ztop[k - 1]

    volume = (
        gridarea[None, :, :]
        * thickness[:, None, None]
    )

    return (
        gridarea,
        ztop,
        thickness,
        volume,
        xlong,
        xlat,
    )


def read_concentration(run_dir):
    """
    Read FLEXPART concentration and Times.

    Returns:
        conc:
            (time, layer, y, x)

        times:
            UTC DatetimeIndex
    """

    path = run_dir / CONC_FILE_NAME

    with Dataset(path, "r") as nc:

        var = nc.variables["CONC"]

        print(f"CONC dimensions: {var.dimensions}")
        print(f"CONC shape     : {var.shape}")
        print(f"CONC units     : {getattr(var, 'units', 'unknown')}")

        conc = np.asarray(
            var[:, 0, 0, :, :, :],
            dtype=np.float64,
        )

        times = read_times(nc)

    return conc, times


def calculate_H(conc, volume, weights):
    """
    Calculate hourly mine H.

    conc:
        (time, layer, y, x)

    volume:
        (layer, y, x)

    weights:
        (mine, y, x)

    Returns:
        H_mine:
            (time, mine)

        H_layer:
            (time, layer)

        H_total:
            (time,)
    """

    # Concentration / layer volume
    #
    # conc:   time, layer, y, x
    # volume:       layer, y, x
    #
    # result:
    #         time, layer, y, x
    kernel = conc / volume[None, :, :, :]

    # Mine-weighted sensitivity.
    #
    # kernel:
    #     time, layer, y, x
    #
    # weights:
    #     mine, y, x
    #
    # result:
    #     time, mine, layer
    H_time_mine_layer = np.einsum(
        "tkyx,myx->tmk",
        kernel,
        weights,
        optimize=True,
    )

    # Sum vertical layers.
    H_mine_mass = np.sum(
        H_time_mine_layer,
        axis=2,
    )

    # Convert kg/s sensitivity to ppmv/(kg/s).
    H_mine = H_mine_mass * PPMV_FACTOR

    # Vertical-layer contribution summed over all mines.
    H_layer_mass = np.sum(
        H_time_mine_layer,
        axis=1,
    )

    H_layer = H_layer_mass * PPMV_FACTOR

    # Total receptor sensitivity.
    H_total = np.sum(
        H_mine,
        axis=1,
    )

    return (
        H_mine,
        H_layer,
        H_total,
    )


def process_run(run_dir, weights, label):
    """
    Process one FLEXPART run.
    """

    print()
    print("=" * 80)
    print(f"PROCESSING {label}")
    print("=" * 80)
    print(f"Run directory : {run_dir}")

    conc, times = read_concentration(run_dir)

    (
        gridarea,
        ztop,
        thickness,
        volume,
        xlong,
        xlat,
    ) = read_geometry(run_dir)

    print()
    print("VERTICAL GEOMETRY")
    print("-" * 80)
    print(f"ZTOP           : {ztop}")
    print(f"Layer thickness: {thickness}")
    print(f"Volume shape   : {volume.shape}")

    print()
    print("GRID")
    print("-" * 80)
    print(
        f"GRIDAREA min/max/mean : "
        f"{gridarea.min():.3f} / "
        f"{gridarea.max():.3f} / "
        f"{gridarea.mean():.3f}"
    )

    print()
    print("CONCENTRATION")
    print("-" * 80)
    print(f"Shape          : {conc.shape}")
    print(f"Min            : {conc.min():.8e}")
    print(f"Max            : {conc.max():.8e}")

    H_mine, H_layer, H_total = calculate_H(
        conc,
        volume,
        weights,
    )

    return {
        "label": label,
        "run_dir": run_dir,
        "conc": conc,
        "times": times,
        "ztop": ztop,
        "thickness": thickness,
        "volume": volume,
        "H_mine": H_mine,
        "H_layer": H_layer,
        "H_total": H_total,
    }


# =============================================================================
# MAIN
# =============================================================================

print()
print("=" * 80)
print("FLEXPART-WRF 2-LAYER vs 3-LAYER H COMPARISON")
print("=" * 80)
print()

print(f"PPMV factor : {PPMV_FACTOR:.15f}")
print(f"Weight file : {WEIGHT_FILE}")
print(f"3-layer run : {RUN_3L}")
print(f"2-layer run : {RUN_2L}")
print()

# -----------------------------------------------------------------------------
# Read mine weights
# -----------------------------------------------------------------------------

weights = np.load(WEIGHT_FILE)

print("MINE WEIGHTS")
print("-" * 80)
print(f"Shape : {weights.shape}")
print(f"Dtype : {weights.dtype}")

if weights.shape[0] != 13:
    raise ValueError(
        f"Expected 13 mines, found {weights.shape[0]}"
    )

# -----------------------------------------------------------------------------
# Process both runs
# -----------------------------------------------------------------------------

result_3l = process_run(
    RUN_3L,
    weights,
    "3-LAYER",
)

result_2l = process_run(
    RUN_2L,
    weights,
    "2-LAYER",
)

# =============================================================================
# TIME CHECK
# =============================================================================

print()
print("=" * 80)
print("TIME COMPARISON")
print("=" * 80)

t3 = result_3l["times"]
t2 = result_2l["times"]

print(f"3-layer times: {len(t3)}")
print(f"2-layer times: {len(t2)}")

if not np.array_equal(t3.values, t2.values):
    raise ValueError(
        "The 2-layer and 3-layer runs do not have identical FLEXPART Times."
    )

print("Times are identical.")

print()
print("FLEXPART OUTPUT TIME")
print("-" * 80)

for i, t in enumerate(t3):
    print(
        f"{i:2d}  "
        f"{t.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

# =============================================================================
# HOURLY TOTAL H COMPARISON
# =============================================================================

H3 = result_3l["H_total"]
H2 = result_2l["H_total"]

ratio_total = np.divide(
    H2,
    H3,
    out=np.full_like(H2, np.nan),
    where=H3 != 0,
)

print()
print("=" * 80)
print("HOURLY TOTAL H")
print("=" * 80)

hourly_rows = []

for i, t in enumerate(t3):

    hourly_rows.append(
        {
            "index": i,
            "datetime_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
            "datetime_aest": (
                t + pd.Timedelta(hours=10)
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "H_3layer": H3[i],
            "H_2layer": H2[i],
            "H_2L_over_3L": ratio_total[i],
        }
    )

    print(
        f"{i:2d} "
        f"{t.strftime('%Y-%m-%d %H:%M')} "
        f"{H3[i]:14.8e} "
        f"{H2[i]:14.8e} "
        f"{ratio_total[i]:10.4f}"
    )

hourly_df = pd.DataFrame(hourly_rows)

# =============================================================================
# TOTAL 24-H COMPARISON
# =============================================================================

sum3 = np.sum(H3)
sum2 = np.sum(H2)

ratio_sum = sum2 / sum3 if sum3 != 0 else np.nan

print()
print("=" * 80)
print("24-HOUR TOTAL H")
print("=" * 80)

print(f"3-layer total H : {sum3:.12e}")
print(f"2-layer total H : {sum2:.12e}")
print(f"2L / 3L         : {ratio_sum:.8f}")
print(f"2L / 3L (%)     : {100.0 * ratio_sum:.4f}%")

# =============================================================================
# MINE-BY-MINE COMPARISON
# =============================================================================

H3_mine = result_3l["H_mine"]
H2_mine = result_2l["H_mine"]

mine_sum_3 = np.sum(H3_mine, axis=0)
mine_sum_2 = np.sum(H2_mine, axis=0)

mine_ratio = np.divide(
    mine_sum_2,
    mine_sum_3,
    out=np.full_like(mine_sum_2, np.nan),
    where=mine_sum_3 != 0,
)

print()
print("=" * 80)
print("MINE-BY-MINE 24-HOUR H")
print("=" * 80)

mine_rows = []

for j, mine in enumerate(MINES):

    mine_rows.append(
        {
            "mine": mine,
            "H_3layer": mine_sum_3[j],
            "H_2layer": mine_sum_2[j],
            "H_2L_over_3L": mine_ratio[j],
            "H_2L_over_3L_percent": 100.0 * mine_ratio[j],
        }
    )

    print(
        f"{mine:5s} "
        f"{mine_sum_3[j]:14.8e} "
        f"{mine_sum_2[j]:14.8e} "
        f"{mine_ratio[j]:10.4f} "
        f"{100.0 * mine_ratio[j]:8.2f}%"
    )

mine_df = pd.DataFrame(mine_rows)

# =============================================================================
# VERTICAL LAYER CONTRIBUTION
# =============================================================================

print()
print("=" * 80)
print("VERTICAL LAYER CONTRIBUTION")
print("=" * 80)

layer_rows = []

for label, result in [
    ("3-layer", result_3l),
    ("2-layer", result_2l),
]:

    H_layer = result["H_layer"]

    print()
    print(label)
    print("-" * 80)

    total_layer = np.sum(H_layer)

    for k in range(len(result["ztop"])):

        layer_sum = np.sum(H_layer[:, k])

        if k == 0:
            zbottom = 0.0
        else:
            zbottom = result["ztop"][k - 1]

        ztop = result["ztop"][k]

        fraction = (
            layer_sum / total_layer
            if total_layer != 0
            else np.nan
        )

        print(
            f"Layer {k+1}: "
            f"{zbottom:.0f}-{ztop:.0f} m  "
            f"H = {layer_sum:.12e}  "
            f"fraction = {fraction:.6f} "
            f"({100.0 * fraction:.3f}%)"
        )

        layer_rows.append(
            {
                "configuration": label,
                "layer": k + 1,
                "zbottom_m": zbottom,
                "ztop_m": ztop,
                "H_total": layer_sum,
                "fraction": fraction,
                "fraction_percent": 100.0 * fraction,
            }
        )

layer_df = pd.DataFrame(layer_rows)

# =============================================================================
# HOURLY MINE DATAFRAME
# =============================================================================

hourly_mine_rows = []

for i, t in enumerate(t3):

    row = {
        "index": i,
        "datetime_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
        "datetime_aest": (
            t + pd.Timedelta(hours=10)
        ).strftime("%Y-%m-%d %H:%M:%S"),
    }

    for j, mine in enumerate(MINES):

        row[f"{mine}_3layer"] = H3_mine[i, j]
        row[f"{mine}_2layer"] = H2_mine[i, j]

        if H3_mine[i, j] != 0:
            row[f"{mine}_2L_over_3L"] = (
                H2_mine[i, j]
                / H3_mine[i, j]
            )
        else:
            row[f"{mine}_2L_over_3L"] = np.nan

    hourly_mine_rows.append(row)

hourly_mine_df = pd.DataFrame(hourly_mine_rows)

# =============================================================================
# SAVE RESULTS
# =============================================================================

hourly_file = OUTPUT_DIR / "hourly_H_2L_vs_3L.csv"
mine_file = OUTPUT_DIR / "mine_H_2L_vs_3L.csv"
layer_file = OUTPUT_DIR / "vertical_layer_H_2L_vs_3L.csv"

hourly_df.to_csv(
    hourly_file,
    index=False,
)

mine_df.to_csv(
    mine_file,
    index=False,
)

layer_df.to_csv(
    layer_file,
    index=False,
)

hourly_mine_file = OUTPUT_DIR / "hourly_mine_H_2L_vs_3L.csv"

hourly_mine_df.to_csv(
    hourly_mine_file,
    index=False,
)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()
print(
    f"3-layer total H : {sum3:.12e} ppmv/(kg/s)"
)
print(
    f"2-layer total H : {sum2:.12e} ppmv/(kg/s)"
)
print(
    f"2-layer / 3-layer: {100.0 * ratio_sum:.3f}%"
)

print()
print("Maximum hourly H:")
print(
    f"3-layer: {np.max(H3):.12e}"
)
print(
    f"2-layer: {np.max(H2):.12e}"
)

print()
print("Maximum mine H:")
print(
    f"3-layer: {np.max(H3_mine):.12e}"
)
print(
    f"2-layer: {np.max(H2_mine):.12e}"
)

print()
print("OUTPUT FILES")
print("-" * 80)
print(hourly_file)
print(mine_file)
print(layer_file)
print(hourly_mine_file)

print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

