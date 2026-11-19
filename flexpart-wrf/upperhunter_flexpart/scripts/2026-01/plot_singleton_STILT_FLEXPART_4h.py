#!/usr/bin/env python3
"""
===============================================================================
STILT vs FLEXPART — Singleton 4-hour footprint comparison
===============================================================================

Receptor:
    Singleton
    lat = -32.57123
    lon = 151.16132
    z   = 5 m

Matching UTC hours:
    2026-01-09 22:00
    2026-01-09 23:00
    2026-01-10 00:00
    2026-01-10 01:00

STILT:
    surface influence footprint
    178 x 133 grid
    0.01 degree resolution

FLEXPART:
    CONC
    165 x 147 grid
    1 km grid
    layer 0 = 0-100 m AGL

The FLEXPART footprint is interpolated onto the STILT grid.

The two footprints are normalized independently over the common
valid domain because their native units are different.

Output:
    PNG figure
    NPZ file containing plotted fields

If DISPLAY is available, the figure is also shown interactively.
===============================================================================
"""

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.interpolate import griddata


# =============================================================================
# USER SETTINGS
# =============================================================================

FLEX_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "run/singleton/flxout_d01_20260110_010000.nc"
)

FLEX_HEADER = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "run/singleton/header_d01.nc"
)

STILT_FILE = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Singleton/by-id/"
    "202601100200_151.16132_-32.57123_5/"
    "202601100200_151.16132_-32.57123_5_foot.nc"
)

MINE_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter"
)

OUT_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "hmatrix/daily_20260105_20260131/"
    "footprint_plots"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(
    OUT_DIR,
    "Singleton_STILT_vs_FLEXPART_4h.png"
)

OUT_PDF = os.path.join(
    OUT_DIR,
    "Singleton_STILT_vs_FLEXPART_4h.pdf"
)

OUT_NPZ = os.path.join(
    OUT_DIR,
    "Singleton_STILT_vs_FLEXPART_4h.npz"
)

# Singleton receptor
RECEPTOR_LON = 151.16132
RECEPTOR_LAT = -32.57123

# FLEXPART indices corresponding to STILT's four hours
FLEX_TIME_INDEX = [3, 2, 1, 0]

MINE_NAMES = [
    f"OC{i}" for i in range(1, 14)
]


# =============================================================================
# READ FLEXPART
# =============================================================================

print("=" * 80)
print("READING FLEXPART")
print("=" * 80)

fds = xr.open_dataset(FLEX_FILE)
hds = xr.open_dataset(FLEX_HEADER)

flex_hourly = fds["CONC"].isel(
    Time=FLEX_TIME_INDEX,
    ageclass=0,
    releases=0,
    bottom_top=0
)

flex_4h = flex_hourly.sum(dim="Time").values

flex_lon = hds["XLONG"].values
flex_lat = hds["XLAT"].values

print("FLEXPART shape:", flex_4h.shape)
print("FLEXPART min:", np.nanmin(flex_4h))
print("FLEXPART max:", np.nanmax(flex_4h))
print("FLEXPART sum:", np.nansum(flex_4h))


# =============================================================================
# READ STILT
# =============================================================================

print("\n" + "=" * 80)
print("READING STILT")
print("=" * 80)

sds = xr.open_dataset(STILT_FILE)

stilt_hourly = sds["foot"]

stilt_4h = stilt_hourly.sum(dim="time").values

stilt_lon = sds["lon"].values
stilt_lat = sds["lat"].values

SLON, SLAT = np.meshgrid(
    stilt_lon,
    stilt_lat
)

print("STILT shape:", stilt_4h.shape)
print("STILT min:", np.nanmin(stilt_4h))
print("STILT max:", np.nanmax(stilt_4h))
print("STILT sum:", np.nansum(stilt_4h))


# =============================================================================
# INTERPOLATE FLEXPART TO STILT GRID
# =============================================================================

print("\n" + "=" * 80)
print("INTERPOLATING FLEXPART TO STILT GRID")
print("=" * 80)

points = np.column_stack(
    (
        flex_lon.ravel(),
        flex_lat.ravel()
    )
)

values = flex_4h.ravel()

valid_flex = (
    np.isfinite(points[:, 0])
    & np.isfinite(points[:, 1])
    & np.isfinite(values)
)

flex_on_stilt = griddata(
    points[valid_flex],
    values[valid_flex],
    (SLON, SLAT),
    method="linear"
)

print("Interpolated FLEXPART shape:", flex_on_stilt.shape)


# =============================================================================
# COMMON VALID DOMAIN
# =============================================================================

valid = (
    np.isfinite(stilt_4h)
    & np.isfinite(flex_on_stilt)
    & (stilt_4h > 0)
    & (flex_on_stilt > 0)
)

print("\nCommon valid cells:", np.sum(valid))
print("Total STILT cells:", stilt_4h.size)
print(
    "Common fraction:",
    np.sum(valid) / stilt_4h.size
)


# =============================================================================
# NORMALIZE
# =============================================================================
#
# Because STILT and FLEXPART have different physical units, we compare
# normalized spatial distributions.
#
# Each model is normalized so the integral over the common domain is 1.
#
# =============================================================================

stilt_norm = np.full_like(
    stilt_4h,
    np.nan,
    dtype=float
)

flex_norm = np.full_like(
    flex_on_stilt,
    np.nan,
    dtype=float
)

stilt_sum_common = np.nansum(
    stilt_4h[valid]
)

flex_sum_common = np.nansum(
    flex_on_stilt[valid]
)

stilt_norm[valid] = (
    stilt_4h[valid] /
    stilt_sum_common
)

flex_norm[valid] = (
    flex_on_stilt[valid] /
    flex_sum_common
)

# Difference in normalized spatial sensitivity
difference = np.full_like(
    stilt_norm,
    np.nan
)

difference[valid] = (
    flex_norm[valid] -
    stilt_norm[valid]
)


# =============================================================================
# SPATIAL CORRELATION
# =============================================================================

stilt_vec = stilt_norm[valid]
flex_vec = flex_norm[valid]

correlation = np.corrcoef(
    stilt_vec,
    flex_vec
)[0, 1]

overlap = np.sum(
    np.minimum(
        stilt_vec,
        flex_vec
    )
)

rmse = np.sqrt(
    np.mean(
        (flex_vec - stilt_vec) ** 2
    )
)

print("\n" + "=" * 80)
print("FOOTPRINT STATISTICS")
print("=" * 80)

print(f"Spatial correlation       = {correlation:.6f}")
print(f"Overlap coefficient       = {overlap:.6f}")
print(f"Normalized RMS difference = {rmse:.6e}")


# =============================================================================
# CENTROIDS
# =============================================================================

stilt_lon_centroid = np.sum(
    SLON[valid] * stilt_vec
)

stilt_lat_centroid = np.sum(
    SLAT[valid] * stilt_vec
)

flex_lon_centroid = np.sum(
    SLON[valid] * flex_vec
)

flex_lat_centroid = np.sum(
    SLAT[valid] * flex_vec
)

print("\nCentroids:")
print(
    f"  STILT    = "
    f"{stilt_lat_centroid:.6f}, "
    f"{stilt_lon_centroid:.6f}"
)

print(
    f"  FLEXPART = "
    f"{flex_lat_centroid:.6f}, "
    f"{flex_lon_centroid:.6f}"
)


# =============================================================================
# MINE POLYGONS
# =============================================================================

print("\n" + "=" * 80)
print("READING MINE POLYGONS")
print("=" * 80)

mine_gdfs = {}

for mine in MINE_NAMES:

    shp = os.path.join(
        MINE_DIR,
        mine + ".shp"
    )

    if not os.path.exists(shp):
        print("WARNING: missing", shp)
        continue

    gdf = gpd.read_file(shp)

    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")

    mine_gdfs[mine] = gdf

    print(
        f"  {mine}: {len(gdf)} polygon feature(s)"
    )


# =============================================================================
# PLOT SETTINGS
# =============================================================================

# Small values are displayed on a log scale.
# The lower limit is relative to each model's maximum normalized value.
LOG_VMIN = 1e-5
LOG_VMAX = 1.0

# Difference is normalized probability density difference.
diff_max = np.nanmax(
    np.abs(difference[valid])
)

# Avoid zero-width normalization
if diff_max <= 0:
    diff_max = 1.0


# =============================================================================
# CREATE FIGURE
# =============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 13),
    constrained_layout=True
)

ax1, ax2 = axes[0]
ax3, ax4 = axes[1]


# =============================================================================
# PANEL 1 — STILT
# =============================================================================

stilt_plot = np.ma.masked_invalid(
    stilt_norm
)

im1 = ax1.pcolormesh(
    SLON,
    SLAT,
    stilt_plot / np.nanmax(stilt_plot),
    shading="auto",
    cmap="turbo",
    norm=LogNorm(
        vmin=LOG_VMIN,
        vmax=LOG_VMAX
    )
)

ax1.set_title(
    "STILT footprint — normalized",
    fontsize=14,
    fontweight="bold"
)

ax1.set_xlabel("Longitude (°)")
ax1.set_ylabel("Latitude (°)")

cbar1 = fig.colorbar(
    im1,
    ax=ax1,
    pad=0.02
)

cbar1.set_label(
    "Normalized footprint (relative to maximum)"
)

# Mine polygons
for mine, gdf in mine_gdfs.items():

    gdf.boundary.plot(
        ax=ax1,
        linewidth=1.2,
        edgecolor="black"
    )

    # representative point for label
    for geom in gdf.geometry:

        if geom is None or geom.is_empty:
            continue

        p = geom.representative_point()

        ax1.text(
            p.x,
            p.y,
            mine,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1
            )
        )

ax1.plot(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    markersize=15,
    markeredgecolor="black",
    markerfacecolor="red",
    label="Singleton receptor",
    zorder=10
)

ax1.legend(
    loc="lower right",
    fontsize=9
)


# =============================================================================
# PANEL 2 — FLEXPART
# =============================================================================

flex_plot = np.ma.masked_invalid(
    flex_norm
)

im2 = ax2.pcolormesh(
    SLON,
    SLAT,
    flex_plot / np.nanmax(flex_plot),
    shading="auto",
    cmap="turbo",
    norm=LogNorm(
        vmin=LOG_VMIN,
        vmax=LOG_VMAX
    )
)

ax2.set_title(
    "FLEXPART footprint — normalized",
    fontsize=14,
    fontweight="bold"
)

ax2.set_xlabel("Longitude (°)")
ax2.set_ylabel("Latitude (°)")

cbar2 = fig.colorbar(
    im2,
    ax=ax2,
    pad=0.02
)

cbar2.set_label(
    "Normalized footprint (relative to maximum)"
)

for mine, gdf in mine_gdfs.items():

    gdf.boundary.plot(
        ax=ax2,
        linewidth=1.2,
        edgecolor="black"
    )

    for geom in gdf.geometry:

        if geom is None or geom.is_empty:
            continue

        p = geom.representative_point()

        ax2.text(
            p.x,
            p.y,
            mine,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1
            )
        )

ax2.plot(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    markersize=15,
    markeredgecolor="black",
    markerfacecolor="red",
    label="Singleton receptor",
    zorder=10
)

ax2.legend(
    loc="lower right",
    fontsize=9
)


# =============================================================================
# PANEL 3 — DIFFERENCE
# =============================================================================

diff_plot = np.ma.masked_invalid(
    difference
)

im3 = ax3.pcolormesh(
    SLON,
    SLAT,
    diff_plot,
    shading="auto",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(
        vmin=-diff_max,
        vcenter=0.0,
        vmax=diff_max
    )
)

ax3.set_title(
    "Difference: FLEXPART − STILT",
    fontsize=14,
    fontweight="bold"
)

ax3.set_xlabel("Longitude (°)")
ax3.set_ylabel("Latitude (°)")

cbar3 = fig.colorbar(
    im3,
    ax=ax3,
    pad=0.02
)

cbar3.set_label(
    "Difference in normalized sensitivity"
)

for mine, gdf in mine_gdfs.items():

    gdf.boundary.plot(
        ax=ax3,
        linewidth=1.2,
        edgecolor="black"
    )

    for geom in gdf.geometry:

        if geom is None or geom.is_empty:
            continue

        p = geom.representative_point()

        ax3.text(
            p.x,
            p.y,
            mine,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1
            )
        )

ax3.plot(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    markersize=15,
    markeredgecolor="black",
    markerfacecolor="black",
    zorder=10
)


# =============================================================================
# PANEL 4 — OVERLAY
# =============================================================================
#
# STILT shown as filled contours.
# FLEXPART shown as contour lines.
#
# This makes the spatial displacement easy to see.
# =============================================================================

stilt_overlay = (
    stilt_norm /
    np.nanmax(stilt_norm)
)

flex_overlay = (
    flex_norm /
    np.nanmax(flex_norm)
)

levels = np.logspace(
    -4,
    0,
    8
)

# STILT filled contours
ax4.contourf(
    SLON,
    SLAT,
    np.ma.masked_invalid(stilt_overlay),
    levels=levels,
    cmap="Blues",
    alpha=0.55
)

# FLEXPART contours
cs = ax4.contour(
    SLON,
    SLAT,
    np.ma.masked_invalid(flex_overlay),
    levels=levels,
    cmap="Reds",
    linewidths=1.5
)

ax4.clabel(
    cs,
    inline=True,
    fontsize=7,
    fmt="%.0e"
)

ax4.set_title(
    "Spatial comparison",
    fontsize=14,
    fontweight="bold"
)

ax4.set_xlabel("Longitude (°)")
ax4.set_ylabel("Latitude (°)")

for mine, gdf in mine_gdfs.items():

    gdf.boundary.plot(
        ax=ax4,
        linewidth=1.2,
        edgecolor="black"
    )

    for geom in gdf.geometry:

        if geom is None or geom.is_empty:
            continue

        p = geom.representative_point()

        ax4.text(
            p.x,
            p.y,
            mine,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1
            )
        )

ax4.plot(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    markersize=16,
    markeredgecolor="black",
    markerfacecolor="red",
    zorder=10
)

# Centroids
ax4.plot(
    stilt_lon_centroid,
    stilt_lat_centroid,
    marker="o",
    markersize=8,
    markerfacecolor="none",
    markeredgecolor="blue",
    markeredgewidth=2,
    label="STILT centroid"
)

ax4.plot(
    flex_lon_centroid,
    flex_lat_centroid,
    marker="s",
    markersize=8,
    markerfacecolor="none",
    markeredgecolor="red",
    markeredgewidth=2,
    label="FLEXPART centroid"
)

ax4.legend(
    loc="lower right",
    fontsize=9
)


# =============================================================================
# COMMON AXIS LIMITS
# =============================================================================

xmin = max(
    np.nanmin(stilt_lon),
    np.nanmin(flex_lon)
)

xmax = min(
    np.nanmax(stilt_lon),
    np.nanmax(flex_lon)
)

ymin = max(
    np.nanmin(stilt_lat),
    np.nanmin(flex_lat)
)

ymax = min(
    np.nanmax(stilt_lat),
    np.nanmax(flex_lat)
)

# Use the common geographic area for all panels
for ax in axes.ravel():

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.set_ylim(
        ymin,
        ymax
    )

    ax.grid(
        True,
        alpha=0.25
    )


# =============================================================================
# MAIN TITLE
# =============================================================================

fig.suptitle(
    "STILT vs FLEXPART — Singleton receptor\n"
    "4-hour footprint comparison, 2026-01-09 22:00 to "
    "2026-01-10 01:00 UTC, FLEXPART layer 0 (0–100 m)",
    fontsize=17,
    fontweight="bold"
)

# =============================================================================
# STATISTICS BOX
# =============================================================================

# Approximate centroid separation in km.
# 1 degree latitude  ~ 111 km
# 1 degree longitude ~ 92 km at Upper Hunter latitude

centroid_distance_km = np.sqrt(
    (
        (flex_lon_centroid - stilt_lon_centroid) * 92.0
    ) ** 2
    +
    (
        (flex_lat_centroid - stilt_lat_centroid) * 111.0
    ) ** 2
)

fig.text(
    0.5,
    0.005,
    (
        f"Spatial correlation = {correlation:.3f}    |    "
        f"Overlap = {overlap:.3f}    |    "
        f"Normalized RMS difference = {rmse:.3e}    |    "
        f"Centroid separation ≈ {centroid_distance_km:.1f} km"
    ),
    ha="center",
    fontsize=11
)


# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 80)
print("SAVING FIGURE")
print("=" * 80)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

np.savez(
    OUT_NPZ,
    stilt_lon=stilt_lon,
    stilt_lat=stilt_lat,
    stilt_4h=stilt_4h,
    flex_on_stilt=flex_on_stilt,
    stilt_normalized=stilt_norm,
    flexpart_normalized=flex_norm,
    difference=difference,
    valid=valid,
    correlation=correlation,
    overlap=overlap,
    rmse=rmse,
    stilt_lon_centroid=stilt_lon_centroid,
    stilt_lat_centroid=stilt_lat_centroid,
    flex_lon_centroid=flex_lon_centroid,
    flex_lat_centroid=flex_lat_centroid
)

print("PNG:")
print(" ", OUT_PNG)

print("PDF:")
print(" ", OUT_PDF)

print("NPZ:")
print(" ", OUT_NPZ)

print("\nDone.")


# =============================================================================
# DISPLAY INTERACTIVELY IF X11 IS AVAILABLE
# =============================================================================

if os.environ.get("DISPLAY"):
    print("\nDISPLAY detected:", os.environ["DISPLAY"])
    print("Showing interactive figure...")
    plt.show()
else:
    print("\nNo DISPLAY detected.")
    print("Figure was saved to PNG/PDF.")


# Close datasets
fds.close()
hds.close()
sds.close()

