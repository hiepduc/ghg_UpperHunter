#!/usr/bin/env python3

"""
Generate FLEXPART-WRF backward runs for every VALID hourly Picarro
observation from 1-31 March 2026 AEST.

Picarro timestamps:
    AEST = UTC + 10 hours

FLEXPART release time:
    release_UTC = observation_AEST - 10 hours

One FLEXPART run is generated for each valid receptor observation.

Current data:
    Singleton       624
    Muswellbrook    624
    Merriwa         598
    -------------------
    Total          1846

Existing run/production/ is not touched.
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE = Path("/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart")

WRFOUT = BASE / "WRFOUT" / "2026-03"
HOURLY_OBS = BASE / "run" / "2026-03" / "hourly_obs"

PICO_CSV = Path("/mnt/scratch_lustre/nzem_meim_scratch/data/piccaro_obs/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv")

START = datetime(2026, 3, 1, 0, 0, 0)
END   = datetime(2026, 3, 31, 23, 0, 0)

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
# TIME FORMATTING
# ============================================================================

def flex_time(dt):
    """FLEXPART date/time format."""
    return dt.strftime("%Y%m%d %H%M%S")


def run_name(dt):
    """Run directory name."""
    return dt.strftime("%Y%m%d_%H%M%S")


# ============================================================================
# FIND WRFOUT FILE
# ============================================================================

def get_wrfout_file(day):
    """Find the daily WRF meteorology file for the requested UTC day."""

    day = day.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    filename = f"wrfout_met_d03_{day:%Y-%m-%d}_00:00:00"
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

    The file contains one record for every hourly WRF meteorological
    time required by the backward simulation, including both endpoints.

    Example for:
        start   = 2026-03-03 14:00 UTC
        release = 2026-03-04 14:00 UTC

    records:
        20260303 140000 ... 1
        ...
        20260304 140000 ... 1

    Total = 25 hourly records.
    """

    start_utc = release_utc - timedelta(hours=BACKWARD_HOURS)

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
                f"No WRFOUT file found for {day:%Y-%m-%d}"
            )

        # AVAILABLE requires the filename only, not the full path.
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

    start_utc = release_utc - timedelta(hours=BACKWARD_HOURS)

    input_text = f"""===================== PATHNAMES =============================
{run_dir}/
{WRFOUT}/
{run_dir}/AVAILABLE
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
0
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
2
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
        release_utc = obs_aest - timedelta(hours=10)

        run_dir = receptor_dir / run_name(release_utc)

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

if total == 1846:
    print(
        "SUCCESS: generated all 1846 valid "
        "observation-receptor cases."
    )
else:
    print(
        f"WARNING: expected 1846 cases, "
        f"but generated {total}."
    )

print()
print("Existing run/production/ was not modified.")
print()
