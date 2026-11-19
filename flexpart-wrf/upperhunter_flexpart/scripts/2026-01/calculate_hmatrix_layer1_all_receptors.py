#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import xarray as xr

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path("/mnt/scratch_lustre/nzem_meim_scratch") / "flexpart-wrf/upperhunter_flexpart"

WEIGHTFILE = (
    BASE / "H_matrix_flexpart/test_20260105/mine_area_fraction_20260105.npy"
)

OUTDIR = BASE / "H_matrix_flexpart/test_20260105"
OUTDIR.mkdir(parents=True, exist_ok=True)

LAYER = 1
LAYER_NAME = "100-20000m"

RECEPTORS = {
    "singleton":
        BASE / "run/2026-01/production/singleton/20260105_0000"
        / "flxout_d01_20260104_230000.nc",

    "muswellbrook":
        BASE / "run/2026-01/production/muswellbrook/20260105_0000"
        / "flxout_d01_20260104_230000.nc",

    "merriwa":
        BASE / "run/2026-01/production/merriwa/20260105_0000"
        / "flxout_d01_20260104_230000.nc",
}

# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("FLEXPART-WRF H-MATRIX — ALL RECEPTORS")
print("=" * 80)

print("\nVertical layer:")
print(f"  bottom_top index : {LAYER}")
print(f"  physical layer   : {LAYER_NAME}")
print("  units             : s m3 kg-1")

# =============================================================================
# READ MINE AREA FRACTIONS
# =============================================================================

print("\nReading mine area fractions:")
print(WEIGHTFILE)

area_fraction = np.load(WEIGHTFILE)

print("  shape:", area_fraction.shape)

if area_fraction.shape[0] != 13:
    raise RuntimeError(
        f"Expected 13 mines, found {area_fraction.shape[0]}"
    )

# =============================================================================
# NORMALIZE MINE SOURCE WEIGHTS
# =============================================================================

weights = np.zeros_like(area_fraction, dtype=np.float64)

print("\nNormalizing mine source weights:")
print("-" * 80)

for m in range(13):

    total = area_fraction[m].sum()

    if total <= 0:
        raise RuntimeError(
            f"Mine OC{m+1} has zero grid-overlap area."
        )

    weights[m] = area_fraction[m] / total

    print(
        f"OC{m+1:<3} "
        f"area-fraction sum = {total:12.8f}   "
        f"weight sum = {weights[m].sum():12.8f}"
    )

# =============================================================================
# FUNCTION TO CALCULATE ONE RECEPTOR
# =============================================================================

def calculate_hmatrix(receptor, concfile):

    print("\n" + "=" * 80)
    print(f"PROCESSING: {receptor.upper()}")
    print("=" * 80)

    print("\nCONC file:")
    print(concfile)

    if not concfile.exists():
        raise FileNotFoundError(
            f"CONC file not found:\n{concfile}"
        )

    ds = xr.open_dataset(concfile)

    try:

        conc = ds["CONC"].values

        print("\nOriginal CONC shape:")
        print(" ", conc.shape)

        print("\nCONC units:")
        print(" ", ds["CONC"].attrs.get("units", "unknown"))

        if conc.ndim != 6:
            raise RuntimeError(
                f"Unexpected CONC dimensions for {receptor}: "
                f"{conc.shape}"
            )

        # Expected:
        #
        # time
        # ageclass
        # releases
        # bottom_top
        # south_north
        # west_east
        #

        nt = conc.shape[0]
        nz = conc.shape[3]
        ny = conc.shape[4]
        nx = conc.shape[5]

        if LAYER >= nz:
            raise RuntimeError(
                f"Layer {LAYER} unavailable. "
                f"CONC has {nz} vertical layers."
            )

        # ---------------------------------------------------------------------
        # Select layer 1 = 100–20,000 m
        #
        # Sum ageclass and release dimensions.
        # ---------------------------------------------------------------------

        foot = conc[:, :, :, LAYER, :, :].sum(axis=(1, 2))

        print("\nSelected footprint:")
        print("  shape :", foot.shape)
        print("  layer :", LAYER_NAME)
        print("  min   :", foot.min())
        print("  max   :", foot.max())
        print("  mean  :", foot.mean())

        # ---------------------------------------------------------------------
        # Check grid dimensions
        # ---------------------------------------------------------------------

        if area_fraction.shape[1:] != (ny, nx):

            raise RuntimeError(
                f"Grid mismatch for {receptor}:\n"
                f"  mine fractions : {area_fraction.shape[1:]}\n"
                f"  FLEXPART CONC : {(ny, nx)}"
            )

        # ---------------------------------------------------------------------
        # Calculate H
        #
        # H[t,m] = sum(
        #              footprint[t,i,j]
        #              *
        #              source_weight[m,i,j]
        #           )
        #
        # ---------------------------------------------------------------------

        print("\nCalculating H matrix...")

        H = np.zeros((nt, 13), dtype=np.float64)

        for t in range(nt):

            for m in range(13):

                H[t, m] = np.sum(
                    foot[t] * weights[m]
                )

        # ---------------------------------------------------------------------
        # Output files
        # ---------------------------------------------------------------------

        npy_file = (
            OUTDIR
            / f"H_layer1_{receptor}_20260105.npy"
        )

        csv_file = (
            OUTDIR
            / f"H_layer1_{receptor}_20260105.csv"
        )

        np.save(npy_file, H)

        header = ",".join(
            [f"OC{i}" for i in range(1, 14)]
        )

        np.savetxt(
            csv_file,
            H,
            delimiter=",",
            header=header,
            comments=""
        )

        # ---------------------------------------------------------------------
        # Print first 10 rows
        # ---------------------------------------------------------------------

        print("\n" + "-" * 80)
        print(f"H MATRIX — {receptor.upper()}")
        print("-" * 80)

        print(
            f"{'t':>4} "
            + " ".join(
                [f"OC{i:>8}" for i in range(1, 14)]
            )
        )

        for t in range(min(10, nt)):

            print(
                f"{t+1:4d} "
                + " ".join(
                    [f"{H[t,m]:10.5g}" for m in range(13)]
                )
            )

        # ---------------------------------------------------------------------
        # 24-hour integrated H
        # ---------------------------------------------------------------------

        print("\n24-h source-integrated H:")

        print(
            f"{'Mine':<8}"
            f"{'sum(H)':>18}"
            f"{'mean(H)':>18}"
            f"{'max(H)':>18}"
        )

        print("-" * 62)

        for m in range(13):

            print(
                f"OC{m+1:<5}"
                f"{H[:,m].sum():18.8g}"
                f"{H[:,m].mean():18.8g}"
                f"{H[:,m].max():18.8g}"
            )

        print("-" * 62)

        print(
            f"{'Total':<8}"
            f"{H.sum():18.8g}"
            f"{H.mean():18.8g}"
            f"{H.max():18.8g}"
        )

        # ---------------------------------------------------------------------
        # Save basic metadata
        # ---------------------------------------------------------------------

        print("\nSaved:")
        print(" ", npy_file)
        print(" ", csv_file)

        return H

    finally:

        ds.close()


# =============================================================================
# PROCESS ALL THREE RECEPTORS
# =============================================================================

H_all = {}

for receptor, concfile in RECEPTORS.items():

    H_all[receptor] = calculate_hmatrix(
        receptor,
        concfile
    )

# =============================================================================
# COMBINED SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("COMBINED 24-HOUR H SUMMARY")
print("=" * 80)

print(
    f"{'Mine':<8}"
    f"{'Singleton':>18}"
    f"{'Muswellbrook':>18}"
    f"{'Merriwa':>18}"
)

print("-" * 62)

for m in range(13):

    print(
        f"OC{m+1:<5}"
        f"{H_all['singleton'][:,m].sum():18.8g}"
        f"{H_all['muswellbrook'][:,m].sum():18.8g}"
        f"{H_all['merriwa'][:,m].sum():18.8g}"
    )

print("-" * 62)

print(
    f"{'Total':<8}"
    f"{H_all['singleton'].sum():18.8g}"
    f"{H_all['muswellbrook'].sum():18.8g}"
    f"{H_all['merriwa'].sum():18.8g}"
)

# =============================================================================
# FINISHED
# =============================================================================

print("\n" + "=" * 80)
print("ALL THREE H MATRICES COMPLETED")
print("=" * 80)

print("\nOutput directory:")
print(OUTDIR)

print("\nFiles:")

for receptor in RECEPTORS:

    print(
        f"  H_layer1_{receptor}_20260105.npy"
    )

    print(
        f"  H_layer1_{receptor}_20260105.csv"
    )

print("\nDONE")
