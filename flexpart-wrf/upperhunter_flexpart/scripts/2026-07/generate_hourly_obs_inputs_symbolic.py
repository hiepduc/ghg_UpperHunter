#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate FLEXPART-WRF backward runs for every VALID hourly Picarro
observation from 1-30 July 2026 AEST.

Picarro timestamps:
    AEST = UTC + 10 hours

FLEXPART release time:
    release_UTC = observation_AEST - 10 hours

One FLEXPART run is generated for each valid receptor observation.

Pathname changes:
    1. FLEXPART output directory uses the short uh3_jul symlink.
    2. WRFOUT directory remains the original full path.
    3. AVAILABLE file uses the short uh3_jul symlink.

The original FLEXPART settings and AVAILABLE generation logic
are retained.

Current data:
    Singleton       568
    Muswellbrook    598
    Merriwa         573
    -------------------
    Total          1739
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "flexpart-wrf/upperhunter_flexpart"
)

# Full WRFOUT directory: KEEP THIS UNCHANGED
WRFOUT = BASE / "WRFOUT" / "2026-07"

# Actual June three-layer production directory
HOURLY_OBS = (
    BASE / "run" / "2026-07" / "hourly_obs_3layer"
)

# Short symbolic link to the three-layer production directory
SHORT_RUN_ROOT = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "flexpart-wrf/uh3_jul"
)

PICO_CSV = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "data/piccaro_obs/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

START = datetime(2026, 7, 1, 0, 0, 0)
END = datetime(2026, 7, 31, 23, 0, 0)

BACKWARD_HOURS = 24
PARTICLES = 500

RECEPTORS = {
    "singleton": {
        "lat": -32.57123,
        "lon": 151.16132,
        "column": "Singleton cleaned",
    },

    "muswellbrook": {
        "lat": -32.26202,
        "lon": 150.89297,
        "column": "Muswellbrook cleaned",
    },

    "merriwa": {
        "lat": -32.14097,
        "lon": 150.35767,
        "column": "Merriwa cleaned",
    },
}


# ============================================================================
# CHECK SYMBOLIC LINK
# ============================================================================

if not SHORT_RUN_ROOT.exists():
    raise FileNotFoundError(
        f"Short symbolic link does not exist:\n{SHORT_RUN_ROOT}"
    )

if SHORT_RUN_ROOT.resolve() != HOURLY_OBS.resolve():
    raise RuntimeError(
        "The short symbolic link does not point to the "
        "June three-layer production directory.\n"
        f"Short link resolves to: {SHORT_RUN_ROOT.resolve()}\n"
        f"Expected:                {HOURLY_OBS.resolve()}"
    )


# ============================================================================
# TIME FORMATTING
# ============================================================================

def flex_time(dt):
    """FLEXPART date/time format."""
    return dt.strftime("%Y%m%d %H%M%S")


def run_name(dt):
    """Run directory name."""
    return dt.strftime("%Y%m%d_%H%M%S")


# ============================================================================
# FIND DAILY WRFOUT FILE
# ============================================================================

def get_wrfout_file(day):
    """
    Find the daily WRF meteorology file for the requested UTC day.

    WRFOUT files are daily files at 00:00 UTC.

    IMPORTANT:
    Path.is_file() follows valid symbolic links, so this also
    works with the WRFOUT symbolic links in WRFOUT/2026-07.
    """

    day = day.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    filename = (
        f"wrfout_met_d03_{day:%Y-%m-%d}_00:00:00"
    )

    path = WRFOUT / filename

    if path.is_file():
        return path

    return None


# ============================================================================
# CREATE AVAILABLE
# ============================================================================

def make_available(run_dir, release_utc):
    """
    Create a FLEXPART-WRF legacy-format AVAILABLE file.

    The file contains one record for every hourly FLEXPART
    meteorological time required by the backward simulation.

    Daily WRFOUT files are referenced by filename. The same
    daily filename is therefore used for all hourly records
    belonging to that UTC day.

    Total records = 25, including both endpoints.
    """

    start_utc = (
        release_utc - timedelta(hours=BACKWARD_HOURS)
    )

    lines = [
        "FLEXPART-WRF Upper Hunter hourly observation",
        "WRF-Chem d03 hourly meteorological fields",
        "DATE TIME FILE SPEC",
    ]

    current = start_utc

    while current <= release_utc:

        day = current.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        wrf_file = get_wrfout_file(day)

        if wrf_file is None:
            raise FileNotFoundError(
                "No WRFOUT file found for "
                f"{day:%Y-%m-%d}\n"
                f"Expected: {WRFOUT / f'wrfout_met_d03_{day:%Y-%m-%d}_00:00:00'}"
            )

        # AVAILABLE requires filename only, not full path.
        filename = wrf_file.name

        lines.append(
            f"{current:%Y%m%d} "
            f"{current:%H%M%S} "
            f"{filename} 1"
        )

        current += timedelta(hours=1)

    available = run_dir / "AVAILABLE"

    available.write_text(
        "\n".join(lines) + "\n"
    )

    return available


# ============================================================================
# CREATE FLEXWRF.INPUT
# ============================================================================

def make_input(
    run_dir,
    release_utc,
    receptor_name,
    lat,
    lon,
):

    start_utc = (
        release_utc - timedelta(hours=BACKWARD_HOURS)
    )

    # Get the relative case path, e.g.
    # singleton/20260131_200000
    relative_case = run_dir.relative_to(HOURLY_OBS)

    # Short output directory through uh3_jul
    short_run_dir = SHORT_RUN_ROOT / relative_case

    # Short AVAILABLE path through uh3_jul
    short_available = short_run_dir / "AVAILABLE"

    input_text = f"""===================== PATHNAMES =============================
{short_run_dir}/
{WRFOUT}/
{short_available}
=====
===================== COMMAND ===============================
-1
{flex_time(start_utc)}
{flex_time(release_utc)}
3600
3600
180
999999999
180
10.
10
5
1
0
0
3600.
0
0
1
1
0
1
2
0
0
1
1
1
0
0
0
0
1
2
24
100
===================== AGECLASSES ============================
1
86400
===================== OUTGRID ==============================
0
0
165
147
0
1000.
1000.
3
10.
100.
20000.
===================== RECEPTORS =============================
0
===================== SPECIES ===============================
1
XXXX|NAME    |decaytime |wetscava  |wetsb|drydif|dryhenry|drya|partrho  |parmean|partsig|dryvelo|weight |
    AIRTRACER     -999.9   -9.9E-09         -9.9                 -9.9E09                   -9.99   29.00
===================== RELEASES ==============================
1
0
1
1
{flex_time(release_utc)}
{flex_time(release_utc)}
{lon}
{lat}
{lon}
{lat}
1
5.0
5.0
{PARTICLES}
1.0
{receptor_name}
"""

    (run_dir / "flexwrf.input").write_text(input_text)


# ============================================================================
# READ PICARRO
# ============================================================================

print("=" * 80)
print("FLEXPART-WRF HOURLY OBSERVATION INPUT GENERATION")
print("=" * 80)

print(f"Picarro CSV : {PICO_CSV}")
print(f"Output      : {HOURLY_OBS}")
print(f"Short path  : {SHORT_RUN_ROOT}")
print(f"WRFOUT      : {WRFOUT}")
print()

df = pd.read_csv(PICO_CSV)

# IMPORTANT:
# Use the existing Datetime column.
# Do NOT reconstruct Date + Time because the CSV contains 24:00 records.

df["Datetime"] = pd.to_datetime(
    df["Datetime"],
    errors="coerce",
)

df = df[
    (df["Datetime"] >= START)
    & (df["Datetime"] <= END)
].copy()

df = df.sort_values("Datetime")

print(
    f"Observation period: "
    f"{START:%Y-%m-%d %H:%M} to "
    f"{END:%Y-%m-%d %H:%M} AEST"
)

print(f"Actual timestamps : {len(df)}")
print(f"Unique timestamps : {df['Datetime'].nunique()}")
print()


# ============================================================================
# GENERATE RUNS
# ============================================================================

HOURLY_OBS.mkdir(
    parents=True,
    exist_ok=True,
)

counts = {}
total = 0

for receptor_name, info in RECEPTORS.items():

    column = info["column"]

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    valid = df[values.notna()].copy()

    valid = valid.sort_values("Datetime")

    receptor_dir = HOURLY_OBS / receptor_name

    receptor_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    print("-" * 80)
    print(receptor_name.upper())
    print(f"Column             : {column}")
    print(f"Valid observations : {len(valid)}")

    for _, row in valid.iterrows():

        obs_aest = row["Datetime"].to_pydatetime()

        # AEST = UTC + 10 hours
        release_utc = (
            obs_aest - timedelta(hours=10)
        )

        run_dir = (
            receptor_dir / run_name(release_utc)
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        make_available(
            run_dir,
            release_utc,
        )

        make_input(
            run_dir,
            release_utc,
            receptor_name,
            info["lat"],
            info["lon"],
        )

        count += 1

    counts[receptor_name] = count
    total += count

    print(f"Generated runs    : {count}")


# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("GENERATION SUMMARY")
print("=" * 80)

for receptor_name in RECEPTORS:

    print(
        f"{receptor_name:15s}: "
        f"{counts[receptor_name]:5d}"
    )

print("-" * 80)

print(
    f"{'TOTAL':15s}: "
    f"{total:5d}"
)

print()

if total == 1739:

    print(
        "SUCCESS: generated all 1739 valid "
        "observation-receptor cases."
    )

else:

    print(
        f"WARNING: expected 1739 cases, "
        f"but generated {total}."
    )

print()
print("Output directory:")
print(HOURLY_OBS)

print()
print("Pathnames in flexwrf.input:")
print("  Line 2: short uh3_jul case output path")
print("  Line 3: original full WRFOUT path")
print("  Line 4: short uh3_jul AVAILABLE path")
print()

