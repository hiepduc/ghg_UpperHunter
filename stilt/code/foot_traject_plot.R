library(ncdf4)
library(terra)

mine_dir <- "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"

mines <- vector("list", 13)

for (i in 1:13) {

    shp <- file.path(
        mine_dir,
        paste0("OC", i, ".shp")
    )

    mines[[i]] <- vect(shp)

    mines[[i]] <- project(
        mines[[i]],
        "EPSG:4326"
    )

    cat("Loaded OC", i, "\n", sep="")
}

plot(
    r1,
    main = paste(
        "STILT footprint + Upper Hunter mines\n",
        format(dates[1], "%Y-%m-%d %H:%M UTC")
    ),
    xlab = "Longitude",
    ylab = "Latitude"
)

for (i in 1:13) {

    lines(
        mines[[i]],
        lwd = 2
    )

    xy <- crds(centroids(mines[[i]]))

    text(
        xy[,1],
        xy[,2],
        labels = paste0("OC", i),
        cex = 0.8,
        pos = 3
    )
}

points(
    151.16132,
    -32.57123,
    pch = 8,
    cex = 2.5,
    lwd = 3
)

text(
    151.16132,
    -32.57123,
    "Singleton",
    pos = 4,
    font = 2
)

plot_foot <- function(k) {

    r <- rast(
        nrows = length(lat),
        ncols = length(lon),
        xmin = min(lon),
        xmax = max(lon),
        ymin = min(lat),
        ymax = max(lat),
        crs = "EPSG:4326"
    )

    v <- t(foot[,,k])
    v <- v[nrow(v):1, ]

    values(r) <- as.vector(v)

    plot(
        r,
        main = paste(
            "STILT footprint + mines\n",
            format(dates[k], "%Y-%m-%d %H:%M UTC")
        ),
        xlab = "Longitude",
        ylab = "Latitude"
    )

    for (i in 1:13) {

        lines(
            mines[[i]],
            lwd = 2
        )

        xy <- crds(centroids(mines[[i]]))

        text(
            xy[,1],
            xy[,2],
            paste0("OC", i),
            cex = 0.7,
            pos = 3
        )
    }

    points(
        151.16132,
        -32.57123,
        pch = 8,
        cex = 2.5,
        lwd = 3
    )
}
plot_foot(1)
plot_foot(2)
plot_foot(3)
plot_foot(4)
plot_foot(2)
range(lon)
range(lat)

head(lon)
tail(lon)

head(lat)
tail(lat)

cat("Singleton:\n")
cat("lon =", 151.16132, "\n")
cat("lat =", -32.57123, "\n")

for (k in 1:4) {
    m <- which(foot[k,,] == max(foot[k,,], na.rm=TRUE), arr.ind=TRUE)

    cat("\nLayer", k, "\n")
    cat("max =", max(foot[k,,], na.rm=TRUE), "\n")
    cat("lon =", lon[m[1,2]], "\n")
    cat("lat =", lat[m[1,1]], "\n")
}

###########################
# Plot trjectories
###########################
traj <- readRDS(
  "/home/duch/software/hysplit_data2arl/test_upperhunter/stilt_singleton_20260110/out_clean/by-id/202601100200_151.16132_-32.57123_5/202601100200_151.16132_-32.57123_5_traj.rds"
)

names(traj)

particle <- traj$particle

range(particle$long, na.rm=TRUE)
range(particle$lati, na.rm=TRUE)
range(particle$time, na.rm=TRUE)

plot(
    particle$long,
    particle$lati,
    pch = ".",
    asp = 1,
    xlab = "Longitude",
    ylab = "Latitude",
    main = "STILT backward particle positions"
)

points(
    151.16132,
    -32.57123,
    pch = 8,
    cex = 2,
    lwd = 2
)
for (i in 1:13) {
    lines(mines[[i]], lwd = 1.5)
}
unique(sort(floor(particle$time / 60)))
table(floor(particle$time / 60))

max(particle$time)
min(particle$time)

particle$time_hours <- particle$time / 60

range(particle$time_hours)

particle$layer <- floor(particle$time / 60)

aggregate(
    foot ~ layer,
    data = particle,
    FUN = function(x) sum(x > 0, na.rm = TRUE)
)

particle$layer <- floor(particle$time / 60)

aggregate(
    foot ~ layer,
    data = particle,
    FUN = function(x) {
        c(
            n = length(x),
            positive = sum(x > 0, na.rm = TRUE),
            sum = sum(x, na.rm = TRUE)
        )
    }
)
sort(unique(particle$layer))

points(151.16132, -32.57123, pch = 8, cex = 2, lwd = 2)

text(
    151.16132,
    -32.57123,
    labels = "Singleton",
    pos = 1,
    font = 2
)

