#!/bin/bash
#SBATCH --job-name=UH_FLEXPART3
#SBATCH --partition=general
#SBATCH --array=1-1000%48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/run/2026-06/slurm_logs/flexpart3_%A_%a.out
#SBATCH --error=/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/run/2026-06/slurm_logs/flexpart3_%A_%a.err

set -u

# ================================================================
# FLEXPART-WRF Upper Hunter
# 3-layer hourly observation footprints
# ================================================================

BASE="/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"
EXE="/home/duch/flexpart-wrf/flexpart-wrf/flexwrf33_omp"

MANIFEST="${BASE}/run/2026-06/hourly_obs_3layer/production_manifest.txt"
LOGDIR="${BASE}/run/2026-06/slurm_logs"

mkdir -p "${LOGDIR}"

# ================================================================
# Intel runtime library
# ================================================================

export INTEL_LIB="/mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/opt/spack/linux-x86_64_v3/intel-oneapi-compilers-2023.2.4-phxvcdfl6i64vlmkipsngnsfpqnrxosy/compiler/latest/linux/compiler/lib/intel64_lin"

export LD_LIBRARY_PATH="${INTEL_LIB}:${LD_LIBRARY_PATH:-}"

# ================================================================
# One CPU
# ================================================================

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ================================================================
# Check manifest
# ================================================================

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found:"
    echo "${MANIFEST}"
    exit 1
fi

# ================================================================
# Get case from manifest
#
# Manifest:
# receptor,timestamp,status,run_directory
# ================================================================

MANIFEST_OFFSET="${MANIFEST_OFFSET:-0}"
MANIFEST_LINE=$((SLURM_ARRAY_TASK_ID + MANIFEST_OFFSET + 1))

LINE=$(sed -n "${MANIFEST_LINE}p" "${MANIFEST}")

if [ -z "${LINE}" ]; then
    echo "ERROR: No manifest entry for array index:"
    echo "${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Extract fields
IFS=',' read -r RECEPTOR TIMESTAMP STATUS CASE <<< "${LINE}"

if [ -z "${CASE}" ]; then
    echo "ERROR: Could not extract case directory"
    echo "LINE: ${LINE}"
    exit 1
fi

# ================================================================
# Header
# ================================================================

echo "=============================================================="
echo "FLEXPART-WRF Upper Hunter - 3 LAYER"
echo "=============================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Array ID     : ${SLURM_ARRAY_JOB_ID}"
echo "Task ID      : ${SLURM_ARRAY_TASK_ID}"
echo "Host         : $(hostname)"
echo "Start        : $(date)"
echo "Receptor     : ${RECEPTOR}"
echo "Timestamp    : ${TIMESTAMP}"
echo "Manifest    : ${MANIFEST}"
echo "Status       : ${STATUS}"
echo "Case         : ${CASE}"
echo "Executable   : ${EXE}"
echo "=============================================================="

# ================================================================
# Check case directory
# ================================================================

if [ ! -d "${CASE}" ]; then
    echo "ERROR: Case directory does not exist:"
    echo "${CASE}"
    exit 1
fi

cd "${CASE}" || exit 1

echo "Working directory:"
pwd
echo

# ================================================================
# Skip already completed cases
# ================================================================

shopt -s nullglob
OUTPUT_FILES=(flxout_d01_*.nc)
shopt -u nullglob

if [ ${#OUTPUT_FILES[@]} -gt 0 ]; then

    echo "OUTPUT ALREADY EXISTS -- SKIPPING"

    for f in "${OUTPUT_FILES[@]}"; do
        ls -lh "${f}"
    done

    echo
    echo "Finished: $(date)"
    echo "=============================================================="

    exit 0
fi

# ================================================================
# Check input files
# ================================================================

if [ ! -f "flexwrf.input" ]; then
    echo "ERROR: flexwrf.input not found"
    exit 1
fi

if [ ! -f "AVAILABLE" ]; then
    echo "ERROR: AVAILABLE not found"
    exit 1
fi

# ================================================================
# Run FLEXPART-WRF
# ================================================================

echo
echo "STARTING FLEXPART-WRF"
echo "Directory: $(pwd)"
echo

"${EXE}" ./flexwrf.input

STATUS=$?

echo
echo "FLEXPART exit status: ${STATUS}"

# ================================================================
# Check FLEXPART status
# ================================================================

if [ ${STATUS} -ne 0 ]; then
    echo
    echo "ERROR: FLEXPART-WRF failed"
    echo "Case: ${CASE}"
    echo "Finished: $(date)"
    exit ${STATUS}
fi

# ================================================================
# Verify NetCDF output
# ================================================================

shopt -s nullglob
OUTPUT_FILES=(flxout_d01_*.nc)
shopt -u nullglob

if [ ${#OUTPUT_FILES[@]} -eq 0 ]; then
    echo
    echo "ERROR: FLEXPART finished but no"
    echo "flxout_d01_*.nc was produced"
    exit 1
fi

# ================================================================
# Success
# ================================================================

echo
echo "FLEXPART-WRF SUCCESS"
echo "Output files:"

for f in "${OUTPUT_FILES[@]}"; do
    ls -lh "${f}"
done

echo
echo "Finished: $(date)"
echo "=============================================================="

