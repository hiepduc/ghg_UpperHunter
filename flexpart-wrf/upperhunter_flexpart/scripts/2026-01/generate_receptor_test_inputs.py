#!/usr/bin/env python3

import os
from datetime import datetime, timedelta


BASE = os.path.expanduser(
    "~/flexpart-wrf/upperhunter_flexpart"
)

RUN_BASE = os.path.join(
    BASE,
    "run",
    "production"
)

WRFOUT = os.path.join(
    BASE,
    "WRFOUT"
)


RECEPTORS = {

    "singleton": {
        "name": "Singleton",
        "lat": -32.57123,
        "lon": 151.16132,
        "zagl": 5.0,
    },

    "muswellbrook": {
        "name": "Muswellbrook",
        "lat": -32.26202,
        "lon": 150.89297,
        "zagl": 5.0,
    },

    "merriwa": {
        "name": "Merriwa",
        "lat": -32.14097,
        "lon": 150.35767,
        "zagl": 5.0,
    },
}


RELEASE_TIME = datetime(2026, 1, 5, 0, 0)
BACKWARD_HOURS = 24
PARTICLES = 500

START_TIME = RELEASE_TIME - timedelta(hours=BACKWARD_HOURS)


def flex_date(dt):
    return dt.strftime("%Y%m%d")


def flex_time(dt):
    return dt.strftime("%H%M%S")


def wrf_filename(dt):

    return (
        f"wrfoutUH_d03_"
        f"{dt.strftime('%Y-%m-%d')}_00:00:00"
    )


def build_available():

    lines = [
        "FLEXPART-WRF Upper Hunter production",
        "WRF-Chem d03 hourly meteorological fields",
        "DATE TIME FILE SPEC",
    ]

    current = START_TIME

    while current <= RELEASE_TIME:

        lines.append(
            f"{flex_date(current)} "
            f"{flex_time(current)} "
            f"{wrf_filename(current)} 1"
        )

        current += timedelta(hours=1)

    return "\n".join(lines) + "\n"


def build_command():

    return f"""-1
{flex_date(START_TIME)} {flex_time(START_TIME)}
{flex_date(RELEASE_TIME)} {flex_time(RELEASE_TIME)}
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
"""


def build_input(run_dir, rec):

    available_file = os.path.join(
        run_dir,
        "AVAILABLE"
    )

    return f"""===================== PATHNAMES =============================
{run_dir}/
{WRFOUT}/
{available_file}
=====
===================== COMMAND ===============================
{build_command()}===================== AGECLASSES ============================
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
{flex_date(RELEASE_TIME)} {flex_time(RELEASE_TIME)}
{flex_date(RELEASE_TIME)} {flex_time(RELEASE_TIME)}
{rec['lon']}
{rec['lat']}
{rec['lon']}
{rec['lat']}
1
{rec['zagl']}
{rec['zagl']}
{PARTICLES}
1.0
{rec['name']}
"""


def main():

    run_name = RELEASE_TIME.strftime("%Y%m%d_%H%M")

    for receptor, rec in RECEPTORS.items():

        run_dir = os.path.join(
            RUN_BASE,
            receptor,
            run_name
        )

        os.makedirs(
            run_dir,
            exist_ok=True
        )

        available_file = os.path.join(
            run_dir,
            "AVAILABLE"
        )

        with open(available_file, "w") as f:
            f.write(build_available())

        input_file = os.path.join(
            run_dir,
            "flexwrf.input"
        )

        with open(input_file, "w") as f:
            f.write(build_input(run_dir, rec))

        print("=" * 70)
        print(receptor.upper())
        print("=" * 70)

        print("Name      :", rec["name"])
        print("Latitude  :", rec["lat"])
        print("Longitude :", rec["lon"])
        print("Height    :", rec["zagl"])
        print("Run       :", run_dir)
        print("Input     :", input_file)
        print()


if __name__ == "__main__":
    main()
