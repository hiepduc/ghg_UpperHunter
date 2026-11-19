#!/usr/bin/env python3

from netCDF4 import Dataset
from pathlib import Path
import numpy as np

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

CONC_FILE = (
    BASE / "run/production/singleton/20260105_0000/"
    "flxout_d01_20260104_230000.nc"
)

HEADER_FILE = (
    BASE / "run/production/singleton/20260105_0000/"
    "header_d01.nc"
)

AREA_FILE = (
    BASE / "hmatrix/test_20260105/"
    "mine_area_fraction_20260105.npy"
)

MW_AIR = 28.97
MW_CH4 = 16.043
PPMV_FACTOR = MW_AIR / MW_CH4 * 1.0e6

# ------------------------------------------------------------
# Read FLEXPART CONC
# ------------------------------------------------------------

with Dataset(CONC_FILE) as nc:
    conc = np.asarray(nc.variables["CONC"][:, 0, 0, 0, :, :])
    times = nc.variables["Times"][:]

print("CONC shape:", conc.shape)

# ------------------------------------------------------------
# Grid cell area and 0-100 m volume
# ------------------------------------------------------------

with Dataset(HEADER_FILE) as nc:
    gridarea = np.asarray(nc.variables["GRIDAREA"][:, :])

volume = gridarea * 100.0

# ------------------------------------------------------------
# Mine overlap areas
# ------------------------------------------------------------

area_fraction = np.load(AREA_FILE)

print("Mine area array:", area_fraction.shape)

# ------------------------------------------------------------
# Calculate H
#
# H[t,m] =
#   sum_ij CONC[t,ij] * weight_m[ij] / volume[ij]
#
# This gives mass mixing ratio per kg/s emission.
# Then convert kg/kg -> ppmv.
# ------------------------------------------------------------

H_mass = np.zeros((conc.shape[0], 13), dtype=np.float64)

for m in range(13):

    area = area_fraction[m]

    total_area = np.sum(area)

    if total_area <= 0:
        raise RuntimeError(f"Mine OC{m+1} has zero area.")

    weights = area / total_area

    H_mass[:, m] = np.sum(
        conc * (weights[None, :, :] / volume[None, :, :]),
        axis=(1, 2)
    )

H_ppm = H_mass * PPMV_FACTOR

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print()
print("=" * 95)
print("13-MINE FLEXPART-WRF H MATRIX TEST")
print("=" * 95)

print()
print("H units:")
print("  mass basis :", "mass mixing ratio per kg/s")
print("  final      :", "ppmv per kg/s")

print()
print(f"MW air       = {MW_AIR}")
print(f"MW CH4       = {MW_CH4}")
print(f"ppmv factor  = {PPMV_FACTOR:.8f}")

print()
print("Hourly/source-time sensitivities")
print("-" * 95)

header = "Time                    " + " ".join(
    [f"OC{i:02d}".rjust(14) for i in range(1, 14)]
)

print(header)

for t in range(conc.shape[0]):

    ts = bytes(times[t]).decode().replace("*", "")

    vals = " ".join(
        f"{H_ppm[t,m]:14.6e}" for m in range(13)
    )

    print(f"{ts:22s} {vals}")

# ------------------------------------------------------------
# Integrated diagnostic
# ------------------------------------------------------------

H_sum = np.sum(H_ppm, axis=0)

print()
print("=" * 95)
print("24-HOUR INTEGRATED SOURCE-TIME SENSITIVITY")
print("=" * 95)

for m in range(13):
    print(
        f"OC{m+1:02d} : "
        f"{H_sum[m]:.10e} ppmv/(kg/s)"
    )

print()
print("Non-zero mines:")
for m in range(13):
    if H_sum[m] > 0:
        print(f"OC{m+1:02d}")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTDIR = BASE / "hmatrix/test_20260105"
OUTDIR.mkdir(parents=True, exist_ok=True)

np.save(
    OUTDIR / "H_singleton_20260105_13mine_ppm.npy",
    H_ppm
)

np.save(
    OUTDIR / "H_singleton_20260105_13mine_mass.npy",
    H_mass
)

np.savetxt(
    OUTDIR / "H_singleton_20260105_13mine_ppm.csv",
    H_ppm,
    delimiter=",",
    header=",".join([f"OC{i}" for i in range(1,14)]),
    comments=""
)

print()
print("Saved:")
print(OUTDIR / "H_singleton_20260105_13mine_ppm.npy")
print(OUTDIR / "H_singleton_20260105_13mine_13mine_mass.npy")
print(OUTDIR / "H_singleton_20260105_13mine_ppm.csv")
