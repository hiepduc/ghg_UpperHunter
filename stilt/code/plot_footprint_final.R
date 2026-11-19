#!/usr/bin/env Rscript

library(ncdf4)
library(terra)

# ============================================================
# STILT footprint plotting
# ============================================================

foot_file <- "/mnt/scratch_lustre/duch/stilt/upperhunter/test_UH_ARL_Jan10/out_oldgood/by-id/202601100200_151.16132_-32.57123_5_foot.nc"
foot_file <- "/mnt/scratch_lustre/duch/stilt/upperhunter/test_UH_ARL_Jan10/out_oldgood/by-id/202601100200_151.16132_-32.57123_5/202601100200_151.16132_-32.57123_5_foot.nc"

outdir <- "/mnt/scratch_lustre/duch/stilt/upperhunter/test_UH_ARL_Jan10/plots"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# 1. Read NetCDF
# ------------------------------------------------------------

nc <- nc_open(foot_file)

lon  <- ncvar_get(nc, "lon")
lat  <- ncvar_get(nc, "lat")
time <- ncvar_get(nc, "time")
foot <- ncvar_get(nc, "foot")

nc_close(nc)

cat("Footprint dimensions:\n")
print(dim(foot))

cat("\nLongitude range:\n")
print(range(lon))

cat("\nLatitude range:\n")
print(range(lat))

cat("\nNumber of time layers:", length(time), "\n")

# Convert Unix time to UTC
time_utc <- as.POSIXct(
    time,
    origin = "1970-01-01",
    tz = "UTC"
)

cat("\nFootprint times:\n")
print(time_utc)


# ============================================================
# 2. Load mine polygons
# ============================================================

mine_dir <- "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"

mines <- lapply(
    1:13,
    function(i) {
        vect(file.path(mine_dir, paste0("OC", i, ".shp")))
    }
)

names(mines) <- paste0("OC", 1:13)

# Make sure all mines are WGS84
mines <- lapply(
    mines,
    function(x) {
        project(x, "EPSG:4326")
    }
)


# ============================================================
# 3. Receptor
# ============================================================

receptor_lon <- 151.16132
receptor_lat <- -32.57123


# ============================================================
# 4. Correct footprint -> terra raster conversion
# ============================================================

make_foot_raster <- function(k) {

    # --------------------------------------------------------
    # ncvar_get() returns:
    #
    # foot[longitude, latitude, time]
    #
    # Therefore:
    #   dimension 1 = longitude
    #   dimension 2 = latitude
    # --------------------------------------------------------

    z <- foot[, , k]

    # longitude x latitude
    # ->
    # latitude x longitude
    z <- t(z)

    # STILT latitude increases south -> north.
    # terra raster rows run north -> south.
    z <- z[nrow(z):1, ]

    # --------------------------------------------------------
    # Calculate cell spacing from actual coordinate centres
    # --------------------------------------------------------

    dx <- median(diff(lon))
    dy <- median(diff(lat))

    # lon/lat are CELL CENTRES, therefore raster extent
    # must extend half a grid cell beyond the first/last centre.
    xmin <- min(lon) - dx / 2
    xmax <- max(lon) + dx / 2

    ymin <- min(lat) - dy / 2
    ymax <- max(lat) + dy / 2

    r <- rast(
        nrows = length(lat),
        ncols = length(lon),
        xmin = xmin,
        xmax = xmax,
        ymin = ymin,
        ymax = ymax,
        crs = "EPSG:4326"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # R matrix is stored column-major.
    # terra values are assigned row-wise.
    #
    # Therefore use:
    #
    #     as.vector(t(z))
    # --------------------------------------------------------

    values(r) <- as.vector(t(z))

    return(r)
}


# ============================================================
# 5. Create the four footprint rasters
# ============================================================

foot_rasters <- lapply(
    seq_len(dim(foot)[3]),
    make_foot_raster
)

names(foot_rasters) <- format(
    time_utc,
    "%Y%m%d_%H%MUTC"
)


# ============================================================
# 6. Check raster geometry
# ============================================================

cat("\nRaster geometry:\n")
print(foot_rasters[[1]])

cat("\nRaster extent:\n")
print(ext(foot_rasters[[1]]))

cat("\nRaster resolution:\n")
print(res(foot_rasters[[1]]))


# ============================================================
# 7. Plot function
# ============================================================

plot_footprint <- function(r, title_text) {

    r_plot <- ifel(
        r > 0,
        log10(r),
        NA
    )

    plot(
        r_plot,
        main = title_text,
        xlab = "Longitude",
        ylab = "Latitude",
        col = hcl.colors(100, "YlOrRd", rev = TRUE)
    )

    # Mine polygons
    for (i in seq_along(mines)) {
        lines(
            mines[[i]],
            lwd = 2
        )
    }

    # Mine labels
    for (i in seq_along(mines)) {

        xy <- crds(
            centroids(mines[[i]])
        )

        text(
            x = xy[1, 1],
            y = xy[1, 2],
            labels = names(mines)[i],
            cex = 0.75,
            font = 2
        )
    }

    # Singleton
    points(
        receptor_lon,
        receptor_lat,
        pch = 19,
        cex = 1.3
    )

    text(
        receptor_lon,
        receptor_lat,
        labels = "Singleton",
        pos = 4,
        cex = 0.8,
        font = 2
    )
}


# ============================================================
# 8. Open X11 display
# ============================================================

if (capabilities("X11")) {
    X11(width = 10, height = 8)
}


# ============================================================
# 9. Plot the four individual footprints
# ============================================================

# ============================================================
# Plot footprints one at a time
# Press ENTER in terminal to continue
# ============================================================

# ============================================================
# Open X11 display
# ============================================================

if (capabilities("X11")) {
    X11(width = 10, height = 8)
}


# ============================================================
# Plot footprints ONE AT A TIME
# Click on the plot window to continue
# ============================================================

# ============================================================
# Open X11
# ============================================================

X11(width = 10, height = 8)


# ============================================================
# Display footprints one at a time
# ============================================================

for (k in seq_along(foot_rasters)) {

    timestamp <- format(
        time_utc[k],
        "%Y-%m-%d %H:%M UTC"
    )

    outfile <- file.path(
        outdir,
        paste0(
            "footprint_",
            format(time_utc[k], "%Y%m%d_%H%M"),
            ".png"
        )
    )

    title_text <- paste0(
        "STILT footprint\n",
        timestamp,
        "\nlog10(foot)"
    )

    # --------------------------------------------------------
    # Draw directly on X11
    # --------------------------------------------------------

    plot_footprint(
        foot_rasters[[k]],
        title_text
    )

    # --------------------------------------------------------
    # Wait FIRST
    # --------------------------------------------------------

    cat("\n============================================\n")
    cat("Footprint ", k, " of ", length(foot_rasters), "\n", sep = "")
    cat("Time: ", timestamp, "\n", sep = "")
    cat("============================================\n")
    cat("CLICK ON THE PLOT WINDOW TO CONTINUE...\n")

    locator(1)

    # --------------------------------------------------------
    # After user clicks, save this footprint as PNG
    # --------------------------------------------------------

    save_footprint_png(
        foot_rasters[[k]],
        title_text,
        outfile
    )

    cat("Saved: ", outfile, "\n", sep = "")
}


# ============================================================
# 10. Combined footprint
# ============================================================

combined <- sum(
    rast(foot_rasters),
    na.rm = TRUE
)

combined_file <- file.path(
    outdir,
    "footprint_combined_4hours.png"
)

plot_footprint(
    combined,
    paste0(
        "STILT combined footprint\n",
        format(time_utc[1], "%Y-%m-%d %H:%M"),
        " to ",
        format(time_utc[length(time_utc)], "%Y-%m-%d %H:%M"),
        " UTC\n",
        "sum of ", length(foot_rasters), " hourly footprints\n",
        "log10(sum foot)"
    ),
    combined_file
)

cat("\nSaved combined footprint:\n")
cat(combined_file, "\n")


# ============================================================
# 11. Also save combined raster as GeoTIFF
# ============================================================

combined_tif <- file.path(
    outdir,
    "footprint_combined_4hours.tif"
)

writeRaster(
    combined,
    combined_tif,
    overwrite = TRUE
)

cat("\nSaved GeoTIFF:\n")
cat(combined_tif, "\n")


# ============================================================
# 12. Summary
# ============================================================

cat("\n============================================================\n")
cat("STILT footprint plotting completed\n")
cat("============================================================\n")
cat("Number of layers :", length(foot_rasters), "\n")
cat("Grid resolution  :", res(combined), "\n")
cat("Longitude range  :", xmin(combined), "to", xmax(combined), "\n")
cat("Latitude range   :", ymin(combined), "to", ymax(combined), "\n")
cat("PNG output       :", outdir, "\n")
cat("============================================================\n")

