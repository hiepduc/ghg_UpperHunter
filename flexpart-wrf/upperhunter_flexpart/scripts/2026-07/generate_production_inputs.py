#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path("/mnt/scratch_lustre/nzem_meim_scratch") / "flexpart-wrf/upperhunter_flexpart"

WRFOUT = BASE / "WRFOUT" / "2026-03"
PRODUCTION = BASE / "run" / "2026-03" / "production"

START_DAY = datetime(2026, 3, 1)
END_DAY   = datetime(2026, 3, 31)

BACKWARD_HOURS = 24
PARTICLES = 500

RECEPTORS = {
    "singleton": {
        "lat": -32.57123,
        "lon": 151.16132,
    },
    "muswellbrook": {
        "lat": -32.26202,
        "lon": 150.89297,
    },
    "merriwa": {
        "lat": -32.14097,
        "lon": 150.35767,
    },
}

# ============================================================
# FUNCTIONS
# ============================================================

def wrf_filename(day):
    return f"wrfout_met_d03_{day:%Y-%m-%d}_00:00:00"


def make_available(release_time, run_dir):

    first_hour = release_time - timedelta(hours=BACKWARD_HOURS)

    # FLEXPART-WRF needs the meteorological file containing
    # every hourly time from start through release time.
    #
    # WRF files are daily files at 00 UTC. The readwind routine
    # reads the requested hourly Times from those daily files.

    lines = [
        "FLEXPART-WRF Upper Hunter production",
        "WRF-Chem d03 hourly meteorological fields",
        "DATE TIME FILE SPEC",
    ]

    t = first_hour

    while t <= release_time:

        # The WRF daily file corresponding to this timestamp.
        wrf_day = t.replace(hour=0, minute=0, second=0)

        filename = wrf_filename(wrf_day)

        # Confirm that the required WRF file exists.
        wrf_path = WRFOUT / filename

        if not wrf_path.exists():
            raise FileNotFoundError(
                f"Missing WRF file:\n{wrf_path}\n"
                f"Required for release {release_time:%Y-%m-%d %H:%M}"
            )

        lines.append(
            f"{t:%Y%m%d} {t:%H%M%S} {filename} 1"
        )

        t += timedelta(hours=1)

    (run_dir / "AVAILABLE").write_text("\n".join(lines) + "\n")


def make_input(receptor_name, release_time, run_dir):

    rec = RECEPTORS[receptor_name]

    start_time = release_time - timedelta(hours=BACKWARD_HOURS)

    lat = rec["lat"]
    lon = rec["lon"]

    input_text = f"""===================== PATHNAMES =============================
{run_dir}/
{WRFOUT}/
{run_dir}/AVAILABLE
=====
===================== COMMAND ===============================
-1
{start_time:%Y%m%d} {start_time:%H%M%S}
{release_time:%Y%m%d} {release_time:%H%M%S}
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
{release_time:%Y%m%d %H%M%S}
{release_time:%Y%m%d %H%M%S}
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


# ============================================================
# MAIN
# ============================================================

def main():

    PRODUCTION.mkdir(parents=True, exist_ok=True)

    total = 0
    missing = []

    day = START_DAY

    while day <= END_DAY:

        for hour in range(24):

            release_time = day.replace(hour=hour)

            for receptor_name in RECEPTORS:

                run_dir = (
                    PRODUCTION
                    / receptor_name
                    / release_time.strftime("%Y%m%d_%H%M")
                )

                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    make_available(release_time, run_dir)
                    make_input(receptor_name, release_time, run_dir)

                    total += 1

                except FileNotFoundError as e:
                    print("\nERROR:", e)
                    missing.append(str(e))

        day += timedelta(days=1)

    print()
    print("=" * 70)
    print("FLEXPART-WRF PRODUCTION INPUT GENERATION")
    print("=" * 70)
    print(f"Date range       : {START_DAY:%Y-%m-%d} to {END_DAY:%Y-%m-%d}")
    print(f"Days             : {(END_DAY - START_DAY).days + 1}")
    print("Hours per day    : 24")
    print(f"Receptors        : {len(RECEPTORS)}")
    print(f"Expected runs    : {31 * 24 * 3}")
    print(f"Generated runs   : {total}")
    print(f"Missing WRF refs : {len(missing)}")

    if missing:
        print("\nPRODUCTION INPUT GENERATION FAILED.")
        raise SystemExit(1)

    expected = 31 * 24 * 3

    if total != expected:
        print("\nERROR: generated run count does not match expected count.")
        raise SystemExit(1)

    print()
    print("SUCCESS: all production inputs generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()

