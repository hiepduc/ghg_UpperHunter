#!/usr/bin/env python3

from pathlib import Path
from netCDF4 import Dataset
import numpy as np
import csv

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

RECEPTORS = ["singleton", "muswellbrook", "merriwa"]

TEST_DATE = "20260105_0000"

AREA_FILE = (
    BASE / "hmatrix/test_20260105/"
    "mine_area_fraction_20260105.npy"
)

OUTDIR = BASE / "hmatrix/hourly"
OUTDIR.mkdir(parents=True, exist_ok=True)

MW_AIR = 28.97
MW_CH4 = 16.043
PPMV_FACTOR = MW_AIR / MW_CH4 * 1.0e6

area_fraction = np.load(AREA_FILE)

if area_fraction.shape[0] != 13:
    raise RuntimeError(
        f"Expected 13 mines, got {area_fraction.shape}"
    )


def clean_time(x):
    arr = np.asarray(x)

    if arr.dtype.kind == "S":
        return b"".join(arr.tolist()).decode().replace("*", "")

    if arr.dtype.kind == "U":
        return "".join(arr.tolist()).replace("*", "")

    if isinstance(x, bytes):
        return x.decode().replace("*", "")

    return str(x).replace("*", "")


def process_run(receptor, run_id):

    run_dir = BASE / "run/production" / receptor / run_id

    files = sorted(run_dir.glob("flxout_d01_*.nc"))

    if len(files) != 1:
        raise RuntimeError(
            f"{receptor}/{run_id}: expected 1 flxout file, "
            f"found {len(files)}"
        )

    conc_file = files[0]

    header_file = run_dir / "header_d01.nc"

    if not header_file.exists():
        raise FileNotFoundError(header_file)

    with Dataset(conc_file) as nc:

        conc = np.asarray(
            nc.variables["CONC"][:, 0, 0, 0, :, :],
            dtype=np.float64
        )

        times = nc.variables["Times"][:]

    with Dataset(header_file) as nc:

        gridarea = np.asarray(
            nc.variables["GRIDAREA"][:, :],
            dtype=np.float64
        )

    volume = gridarea * 100.0

    H = np.zeros((conc.shape[0], 13), dtype=np.float64)

    for m in range(13):

        area = area_fraction[m]
        total_area = np.sum(area)

        if total_area <= 0:
            raise RuntimeError(
                f"OC{m+1}: zero polygon area"
            )

        weights = area / total_area

        H[:, m] = np.sum(
            conc *
            (weights[None, :, :] / volume[None, :, :]),
            axis=(1, 2)
        )

    H *= PPMV_FACTOR

    rows = []

    for t in range(len(times)):

        source_time = clean_time(times[t])

        row = {
            "receptor": receptor,
            "receptor_run": run_id,
            "source_time": source_time,
        }

        for m in range(13):
            row[f"OC{m+1}"] = H[t, m]

        rows.append(row)

    return rows


all_rows = []

print("=" * 80)
print("BUILD HOURLY FLEXPART-WRF H")
print("=" * 80)

for receptor in RECEPTORS:

    print()
    print("Processing:", receptor)

    rows = process_run(receptor, TEST_DATE)

    all_rows.extend(rows)

    print("  rows:", len(rows))


outfile = OUTDIR / f"H_hourly_test_{TEST_DATE}.csv"

fields = [
    "receptor",
    "receptor_run",
    "source_time",
] + [f"OC{i}" for i in range(1, 14)]


with open(outfile, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(all_rows)


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print("Total rows:", len(all_rows))
print("Expected  :", 3 * 24)
print("Output    :", outfile)

print()
print("First rows:")

for row in all_rows[:5]:
    print(
        row["receptor"],
        row["source_time"],
        row["OC1"],
        row["OC2"],
        row["OC3"]
    )
