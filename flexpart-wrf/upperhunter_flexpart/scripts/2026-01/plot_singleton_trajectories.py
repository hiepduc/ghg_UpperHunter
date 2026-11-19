#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
import shapefile

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE = Path.home() / "flexpart-wrf"

TRAJ_FILE = (
    BASE / "upperhunter_flexpart/run/singleton/trajectories.txt"
)

MINE_DIR = (
    BASE / "emission_UH/RH_upperHunter"
)

OUTFILE = (
    BASE / "upperhunter_flexpart/run/singleton/"
    "FLEXPART_WRF_Singleton_trajectories.png"
)

# ---------------------------------------------------------------------
# Read trajectory file
#
# Data records begin with:
#     release_number  age_seconds  lon  lat ...
#
# We use the first lon/lat pair, which is the primary trajectory
# position reported by FLEXPART-WRF.
# ---------------------------------------------------------------------
trajectory = []

with open(TRAJ_FILE) as f:
    for line in f:
        parts = line.split()

        if len(parts) < 4:
            continue

        try:
            release = int(parts[0])
            age = int(parts[1])
            lon = float(parts[2])
            lat = float(parts[3])
        except ValueError:
            continue

        trajectory.append((age, lon, lat))

if not trajectory:
    raise RuntimeError("No trajectory records found.")

# Sort from receptor backwards in time
trajectory.sort(key=lambda x: x[0], reverse=True)

ages = [x[0] / 3600.0 for x in trajectory]
lons = [x[1] for x in trajectory]
lats = [x[2] for x in trajectory]

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 9))

# ---------------------------------------------------------------------
# Mine polygons
# ---------------------------------------------------------------------
for i in range(1, 14):

    shp = MINE_DIR / f"OC{i}.shp"

    if not shp.exists():
        print(f"WARNING: {shp} not found")
        continue

    sf = shapefile.Reader(str(shp))

    for shape in sf.shapes():

        pts = shape.points

        if len(pts) < 2:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        ax.plot(
            xs,
            ys,
            linewidth=1.0,
            alpha=0.8
        )

        # Label approximately at polygon centre
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        ax.text(
            cx,
            cy,
            f"OC{i}",
            fontsize=8,
            ha="center",
            va="center"
        )

# ---------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------
ax.plot(
    lons,
    lats,
    "-o",
    linewidth=2.5,
    markersize=5,
    label="FLEXPART-WRF trajectory"
)

# Label trajectory times
for age, lon, lat in zip(ages, lons, lats):

    ax.text(
        lon,
        lat,
        f"{age:.0f} h",
        fontsize=8,
        ha="left",
        va="bottom"
    )

# ---------------------------------------------------------------------
# Receptor
# ---------------------------------------------------------------------
receptor_lon = 151.16132
receptor_lat = -32.57123

ax.scatter(
    receptor_lon,
    receptor_lat,
    marker="*",
    s=220,
    zorder=10,
    label="Singleton receptor"
)

ax.text(
    receptor_lon,
    receptor_lat,
    "  Singleton",
    fontsize=10,
    va="bottom"
)

# ---------------------------------------------------------------------
# Map limits based on trajectory + receptor + mines
# ---------------------------------------------------------------------
all_lon = lons + [receptor_lon]
all_lat = lats + [receptor_lat]

ax.set_xlim(
    min(all_lon) - 0.10,
    max(all_lon) + 0.10
)

ax.set_ylim(
    min(all_lat) - 0.10,
    max(all_lat) + 0.10
)

# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title(
    "FLEXPART-WRF backward trajectory — Singleton\n"
    "10 January 2026 02:00 UTC"
)

ax.grid(True, alpha=0.3)
ax.legend()

ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

plt.savefig(
    OUTFILE,
    dpi=300,
    bbox_inches="tight"
)

print()
print("Trajectory records:", len(trajectory))
print("Trajectory ages:", ages)
print("Output:", OUTFILE)

plt.show()
