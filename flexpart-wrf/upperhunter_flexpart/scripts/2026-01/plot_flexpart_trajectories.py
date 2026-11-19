#!/usr/bin/env python3

from pathlib import Path
import struct
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pyproj import CRS, Transformer


# ============================================================
# PATHS
# ============================================================

RUN_DIR = Path.home() / "flexpart-wrf/upperhunter_flexpart/run/singleton"
HEADER = RUN_DIR / "header_d01.nc"

OUTPUT = RUN_DIR / "FLEXPART_WW_Singleton_trajectories.png"


# ============================================================
# RECEPTOR
# ============================================================

RECEPTOR_LON = 151.16132
RECEPTOR_LAT = -32.57123


# ============================================================
# READ WRF/FLEXPART HEADER
# ============================================================

with Dataset(HEADER) as nc:

    XLONG = nc.variables["XLONG"][:]
    XLAT = nc.variables["XLAT"][:]

    cen_lat = float(nc.getncattr("CEN_LAT"))
    cen_lon = float(nc.getncattr("CEN_LON"))
    stand_lon = float(nc.getncattr("STAND_LON"))
    truelat1 = float(nc.getncattr("TRUELAT1"))
    truelat2 = float(nc.getncattr("TRUELAT2"))
    earth_radius = float(nc.getncattr("EARTH_RADIUS_M"))

print("WRF grid:")
print("  nx =", XLONG.shape[1])
print("  ny =", XLONG.shape[0])
print("  lon range =", float(XLONG.min()), "to", float(XLONG.max()))
print("  lat range =", float(XLAT.min()), "to", float(XLAT.max()))

# ============================================================
# FLEXPART/WFR GRID PARAMETERS
#
# The particle positions are stored in the same grid-meter
# coordinate system used by FLEXPART-WRF.
#
# We determine the grid origin from the FLEXPART header grid.
# ============================================================

# FLEXPART output grid:
DX = 1000.0
DY = 1000.0

# The first WRF grid-cell centre corresponds to approximately
# x = 0, y = 0 in the FLEXPART output-grid coordinate system.
#
# However, the particle x/y values in partposit are measured
# from the FLEXPART meteorological-grid origin.
#
# We therefore use the actual WRF XLONG/XLAT grid and identify
# its projected coordinates.

crs_wrf = CRS.from_proj4(
    f"+proj=lcc "
    f"+lat_1={truelat1} "
    f"+lat_2={truelat2} "
    f"+lat_0={cen_lat} "
    f"+lon_0={stand_lon} "
    f"+a={earth_radius} "
    f"+b={earth_radius} "
    f"+units=m "
    f"+no_defs"
)

transformer_fwd = Transformer.from_crs(
    CRS.from_epsg(4326),
    crs_wrf,
    always_xy=True
)

transformer_inv = Transformer.from_crs(
    crs_wrf,
    CRS.from_epsg(4326),
    always_xy=True
)

# Project the WRF grid origin/reference point.
#
# FLEXPART's particle x/y values can be interpreted relative
# to its meteorological grid origin. We determine that origin
# from the WRF lower-left grid point.

x_wrf, y_wrf = transformer_fwd.transform(
    XLONG[0, 0],
    XLAT[0, 0]
)

print("\nProjected WRF reference:")
print("  x =", x_wrf)
print("  y =", y_wrf)


# ============================================================
# READ ONE PARTICLE FILE
# ============================================================

def read_partposit(filename):

    records = []

    with open(filename, "rb") as f:

        # ----------------------------------------------------
        # Header record
        # ----------------------------------------------------

        raw = f.read(4)
        if len(raw) != 4:
            return records

        nbytes = struct.unpack("<i", raw)[0]

        data = f.read(nbytes)

        f.read(4)

        if nbytes != 12:
            raise RuntimeError(
                f"Unexpected header size {nbytes} in {filename}"
            )

        itime, npart, outgrid = struct.unpack("<3i", data)

        # ----------------------------------------------------
        # Particle records
        # ----------------------------------------------------

        while True:

            raw = f.read(4)

            if not raw:
                break

            nbytes = struct.unpack("<i", raw)[0]

            data = f.read(nbytes)

            end = f.read(4)

            if len(data) != nbytes:
                raise RuntimeError(
                    f"Incomplete record in {filename}"
                )

            # Sentinel record
            if nbytes == 52:

                values = struct.unpack("<ifffiffffffff", data)

                pid = values[0]

                if pid == -99999:
                    break

                # ------------------------------------------------
                # Exact binary structure from partoutput.f90:
                #
                # integer npoint
                # real    xlon
                # real    ylat
                # real    ztra1
                # integer itramem
                # real topo
                # real pvi
                # real qvi
                # real rhoi
                # real hmixi
                # real tri
                # real tti
                # real xmass1
                # ------------------------------------------------

                records.append({
                    "itime": itime,
                    "particle": pid,
                    "x": values[1],
                    "y": values[2],
                    "z": values[3],
                    "age": values[4],
                })

            else:
                raise RuntimeError(
                    f"Unexpected particle record size {nbytes} "
                    f"in {filename}"
                )

    return records


# ============================================================
# READ ALL PARTPOSIT FILES
# ============================================================

files = sorted(
    glob.glob(str(RUN_DIR / "partposit_*"))
)

print("\nParticle files:", len(files))

all_records = []

for filename in files:

    name = Path(filename).name

    # Ignore any possible partposit_end file
    if name == "partposit_end":
        continue

    recs = read_partposit(filename)

    # Extract actual output timestamp from filename
    m = re.search(r"partposit_(\d{14})$", name)

    if m:
        timestamp = m.group(1)
    else:
        timestamp = name

    for r in recs:
        r["timestamp"] = timestamp

    all_records.extend(recs)

print("Total particle records:", len(all_records))

if not all_records:
    raise RuntimeError("No particle records found.")


# ============================================================
# CONVERT PARTICLE GRID X/Y TO LAT/LON
# ============================================================

# FLEXPART particle coordinates are grid metres.
#
# The WRF grid and FLEXPART output grid have their own origin.
# We use the WRF projected grid as the geographic reference.

xs = np.array([r["x"] for r in all_records])
ys = np.array([r["y"] for r in all_records])

print("\nParticle coordinate range:")
print("  x:", xs.min(), "to", xs.max())
print("  y:", ys.min(), "to", ys.max())

# First attempt: FLEXPART x/y are referenced to the lower-left
# WRF/FLEXPART grid origin represented by x_wrf/y_wrf.

xproj = x_wrf + xs
yproj = y_wrf + ys

lon, lat = transformer_inv.transform(xproj, yproj)

for r, lo, la in zip(all_records, lon, lat):
    r["lon"] = lo
    r["lat"] = la


print("\nConverted geographic range:")
print("  lon:", lon.min(), "to", lon.max())
print("  lat:", lat.min(), "to", lat.max())


# ============================================================
# PLOT TRAJECTORIES
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))

# Group by particle ID
particle_ids = sorted(
    set(r["particle"] for r in all_records)
)

print("\nParticles:", len(particle_ids))

for pid in particle_ids:

    p = [
        r for r in all_records
        if r["particle"] == pid
    ]

    # Sort chronologically backward/forward using timestamp
    p.sort(key=lambda r: r["timestamp"])

    plon = [r["lon"] for r in p]
    plat = [r["lat"] for r in p]

    ax.plot(
        plon,
        plat,
        linewidth=0.5,
        alpha=0.25
    )


# Receptor
ax.plot(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    markersize=14,
    markeredgecolor="black",
    label="Singleton receptor"
)


ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title(
    "FLEXPART-WRF 24-hour Backward Particle Trajectories\n"
    "Singleton — 2026-01-10 02:00 UTC"
)

ax.grid(True, alpha=0.3)
ax.legend()

ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved:")
print(OUTPUT)

plt.show()
