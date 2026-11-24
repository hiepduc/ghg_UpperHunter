#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timedelta

#BASE = Path.home() / "flexpart-wrf" / "upperhunter_flexpart"
BASE = Path ("/mnt/scratch_lustre/nzem_meim_scratch/"  "flexpart-wrf/"  "upperhunter_flexpart")
SOURCE_ROOT = BASE / "run" / "2026-07"/ "hourly_obs"
TARGET_ROOT = BASE / "run" / "2026-07"/ "hourly_obs_3layer"
WRFOUT = BASE / "WRFOUT" / "2026-07"

# ----------------------------------------------------------------------
# Receptors
# ----------------------------------------------------------------------
RECEPTORS = {
    "singleton": {
        "name": "Singleton",
        "lat": -32.57123,
        "lon": 151.16132,
    },
    "muswellbrook": {
        "name": "Muswellbrook",
        "lat": -32.26202,
        "lon": 150.89297,
    },
    "merriwa": {
        "name": "Merriwa",
        "lat": -32.14097,
        "lon": 150.35767,
    },
}

# ----------------------------------------------------------------------
# FLEXPART-WRF configuration taken from successful singleton_10m run
# ----------------------------------------------------------------------
BACKWARD_HOURS = 24
OUTPUT_INTERVAL = 3600
OUTGRID_NX = 165
OUTGRID_NY = 147
DX = 1000.0
DY = 1000.0

VERTICAL_LEVELS = [
    10.0,
    100.0,
    20000.0,
]

RECEPTOR_HEIGHT = 5.0
PARTICLES = 500

# WRF-Chem d03 hourly meteorology
WRF_PREFIX = "wrfout_met_d03_"


# ----------------------------------------------------------------------
# Format helpers
# ----------------------------------------------------------------------
def flex_date(dt):
    return dt.strftime("%Y%m%d")


def flex_time(dt):
    return dt.strftime("%H%M%S")


def wrf_filename(dt):
    """
    WRFOUT is stored as one file per day at 00 UTC containing hourly fields.
    """
    midnight = dt.replace(hour=0, minute=0, second=0)
    return f"{WRF_PREFIX}{midnight:%Y-%m-%d}_00:00:00"


def read_manifest_times(source_dir):
    """
    Read timestamp directories from existing hourly_obs structure.
    We deliberately use these directories as the exact production manifest,
    preserving gaps in the existing observation/run set.
    """
    times = []

    for p in sorted(source_dir.iterdir()):
        if not p.is_dir():
            continue

        try:
            dt = datetime.strptime(p.name, "%Y%m%d_%H%M%S")
            times.append(dt)
        except ValueError:
            continue

    return times


# ----------------------------------------------------------------------
# Generate AVAILABLE
# ----------------------------------------------------------------------
def write_available(run_dir, start_dt, end_dt, receptor_name):
    available = run_dir / "AVAILABLE"

    lines = [
        f"FLEXPART-WRF Upper Hunter {receptor_name} production 3-layer",
        "WRF-Chem d03 hourly meteorological fields",
        "DATE TIME FILE SPEC",
    ]

    dt = start_dt

    while dt <= end_dt:
        filename = wrf_filename(dt)

        lines.append(
            f"{flex_date(dt)} {flex_time(dt)} {filename} 1"
        )

        dt += timedelta(hours=1)

    available.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Generate flexwrf.input
# ----------------------------------------------------------------------
def write_flexwrf_input(run_dir, obs_dt, receptor):
    start_dt = obs_dt - timedelta(hours=BACKWARD_HOURS)

    run_path = str(run_dir) + "/"
    wrf_path = str(WRFOUT) + "/"
    available_path = str(run_dir / "AVAILABLE")

    lines = f"""===================== PATHNAMES =============================
{run_path}
{wrf_path}
{available_path}
=====
===================== COMMAND ===============================
-1
{flex_date(start_dt)} {flex_time(start_dt)}
{flex_date(obs_dt)} {flex_time(obs_dt)}
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
{OUTGRID_NX}
{OUTGRID_NY}
0
{DX}
{DY}
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
{flex_date(obs_dt)} {flex_time(obs_dt)}
{flex_date(obs_dt)} {flex_time(obs_dt)}
{receptor["lon"]}
{receptor["lat"]}
{receptor["lon"]}
{receptor["lat"]}
1
{RECEPTOR_HEIGHT}
{RECEPTOR_HEIGHT}
{PARTICLES}
1.0
{receptor["name"]}
"""

    (run_dir / "flexwrf.input").write_text(lines)


# ----------------------------------------------------------------------
# Validate WRF files
# ----------------------------------------------------------------------
def check_wrf_files(start_dt, end_dt):
    missing = []

    dt = start_dt.replace(hour=0, minute=0, second=0)

    while dt <= end_dt:
        f = WRFOUT / wrf_filename(dt)

        if not f.exists():
            missing.append(f)

        dt += timedelta(days=1)

    return missing


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():

    print("=" * 80)
    print("CREATING FLEXPART-WRF 3-LAYER PRODUCTION RUN DIRECTORIES")
    print("=" * 80)
    print(f"Source : {SOURCE_ROOT}")
    print(f"Target : {TARGET_ROOT}")
    print(f"WRFOUT : {WRFOUT}")
    print()

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    total = 0
    total_missing_wrf = 0

    manifest = []

    for receptor_key, receptor in RECEPTORS.items():

        source_dir = SOURCE_ROOT / receptor_key
        target_dir = TARGET_ROOT / receptor_key

        if not source_dir.exists():
            print(f"WARNING: source directory missing: {source_dir}")
            continue

        times = read_manifest_times(source_dir)

        print(f"\n{receptor['name']}")
        print("-" * 80)
        print(f"Source cases found: {len(times)}")

        target_dir.mkdir(parents=True, exist_ok=True)

        for obs_dt in times:

            run_dir = target_dir / obs_dt.strftime("%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)

            start_dt = obs_dt - timedelta(hours=BACKWARD_HOURS)

            # Check meteorological files needed by this run
            missing = check_wrf_files(start_dt, obs_dt)

            if missing:
                total_missing_wrf += 1
                status = "MISSING_WRF"
            else:
                status = "READY"

            write_available(
                run_dir,
                start_dt,
                obs_dt,
                receptor["name"],
            )

            write_flexwrf_input(
                run_dir,
                obs_dt,
                receptor,
            )

            manifest.append(
                (
                    receptor_key,
                    obs_dt.strftime("%Y%m%d_%H%M%S"),
                    status,
                    str(run_dir),
                )
            )

            total += 1

        print(f"Created: {len(times)}")

    # ------------------------------------------------------------------
    # Write global manifest
    # ------------------------------------------------------------------
    manifest_file = TARGET_ROOT / "production_manifest.txt"

    with manifest_file.open("w") as f:
        f.write(
            "receptor,timestamp,status,run_directory\n"
        )

        for receptor, timestamp, status, run_dir in manifest:
            f.write(
                f"{receptor},{timestamp},{status},{run_dir}\n"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Total cases created : {total}")
    print(f"Cases with missing WRF files : {total_missing_wrf}")
    print(f"Manifest : {manifest_file}")
    print()

    # Summary
    for receptor_key in RECEPTORS:
        n = sum(1 for x in manifest if x[0] == receptor_key)
        print(f"{receptor_key:15s}: {n}")

    print()
    print("No FLEXPART-WRF simulations were executed.")
    print("Only input directories/files were generated.")


if __name__ == "__main__":
    main()
