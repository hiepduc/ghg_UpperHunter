(0) First use spack
source /mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/share/spack/setup-env.sh

spack load netcdf-c
spack load netcdf-fortran

spack load intel-oneapi-compilers/phxvcdf
spack load intel-oneapi-compilers-classic
spack load cmake bison flex curl
spack load hdf5
spack load openmpi

(1) Enter the conda stilt_env2 envrionemnt

source /home/duch/.bashrc.conda
conda activate stilt_env2

export R_LIBS_USER=$HOME/R/x86_64-pc-linux-gnu-library/4.2

NCF="/mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/opt/spack/linux-x86_64_v3/netcdf-fortran-4.6.2-6xxxvdhat3tdlcydf56dey23pmxod4sx"
NCC="/mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/opt/spack/linux-x86_64_v3/netcdf-c-4.9.3-o3xhoky7mwcvtan6cswmuxlhafuz7ial"

export LD_LIBRARY_PATH="$NCF/lib:$NCC/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

(2) Check that netcdf4 is installed

Rscript -e '
cat("R.home =", R.home(), "\n")
cat("R.version =", R.version.string, "\n")
cat("R_LIBS_USER =", Sys.getenv("R_LIBS_USER"), "\n")
cat("Libraries:\n")
print(.libPaths())
cat("ncdf4:\n")
print(find.package("ncdf4", quiet=TRUE))
'

(3) Check all other packages installed and available

Rscript -e '
.libPaths(c("/home/duch/miniconda3/envs/stilt_env2/lib/R/library", .libPaths()))

pkgs <- c("ncdf4","dplyr","raster","terra","sp","lubridate","rslurm")

for (p in pkgs) {
    cat("\n---", p, "---\n")
    ok <- requireNamespace(p, quietly=TRUE)
    cat("available:", ok, "\n")
    if (ok)
        cat("version:", as.character(packageVersion(p)), "\n")
}

cat("\nDONE\n")
'
(4) Check they are loaded ok

"$CONDA_PREFIX/bin/Rscript" -e '
cat("R:", R.version.string, "\n")
cat("R.home:", R.home(), "\n")
cat("CONDA_PREFIX:", Sys.getenv("CONDA_PREFIX"), "\n")
cat("\n.libPaths():\n")
print(.libPaths())

library(ncdf4)
cat("\nncdf4:", as.character(packageVersion("ncdf4")), "\n")
cat("ncdf4 loaded OK\n")

library(dplyr)
cat("dplyr:", as.character(packageVersion("dplyr")), "\n")
cat("dplyr loaded OK\n")

cat("\nALL OK\n")
'

(5) Workflow steps

(a) Convert WRF to HYSPLIT ARL

Use  arw2arl executable to convert the WRF NetCDF files into ARL meteorological files.

For Janaruary 2026 wrf to arl 

/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/arl_mine_2026/2026-01
/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/arl_mine_2026/2026-02
....

It contains the key components we need to inspect:
- arl/ — converted ARL meteorological files.
- arljanuary2026.bsh — your January conversion script.
- config/ and scripts/ — configuration and supporting scripts.
- footprints/ — footprint output.
- ARLDATA.CFG, WRFRAIN.BIN and hybrid_wrfvcoords.txt — supporting files for your setup.

Run the script arljanuary2026.bsh (arlfebruary2026.bsh ....)

or run the script in /mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter

convert_wrf_uh_to_arl.sh

(b) Then  Run STILT / HYSPLIT

Generate backward footprints for each receptor and release time, retaining your existing 5 m receptor height and 24-hour backward duration if those remain your intended settings.

run the script run_stilt_feb2026.sh 
(in /mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian)

(c) Process footprints

Extract footprints and combine them with your 13 Upper Hunter coal-mine emissions to construct the observation sensitivity matrix for the inversion.

(6) The work flow

                    Bayesian inversion
                           │
             ┌─────────────┴─────────────┐
             │                           │
         WRF-GHG                     STILT-WRF
         Eulerian                    Lagrangian
             │                           │
      ┌──────┴──────┐             ┌──────┴──────┐
      │             │             │             │
   Grouped       13 mines      Grouped       13 mines
   sources      resolved       sources       resolved
      │             │             │             │
   mine+ag+pw     15 sources   mine+ag+pw     15 sources


Picarro observation
       │
       ▼
receptor location/time
       │
       ▼
STILT backward particles
       │
       ▼
STILT footprint
       │
       ├──────────────┐
       ▼              ▼
 OC1 emission       OC2 emission
       │              │
       ▼              ▼
 footprint × E      footprint × E
       │              │
       └──────┬───────┘
              ▼
       source contributions
              │
              ▼
     + agriculture
     + power
     + background
              │
              ▼
       predicted CH4
              │
              ▼
     compare with Picarro
