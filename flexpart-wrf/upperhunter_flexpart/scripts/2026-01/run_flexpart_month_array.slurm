#!/bin/bash
#SBATCH --job-name=UH_FLEXPART
#SBATCH --partition=general
#SBATCH --array=1-1000%48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/run/2026-01/slurm_logs/flexpart_%A_%a.out
#SBATCH --error=/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/run/2026-01/slurm_logs/flexpart_%A_%a.err

set -u

# ================================================================
# FLEXPART-WRF Upper Hunter monthly production
# ================================================================

BASE="/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"
EXE="/home/duch/flexpart-wrf/flexpart-wrf/flexwrf33_omp"

# Intel runtime library
export INTEL_LIB="/mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/opt/spack/linux-x86_64_v3/intel-oneapi-compilers-2023.2.4-phxvcdfl6i64vlmkipsngnsfpqnrxosy/compiler/latest/linux/compiler/lib/intel64_lin"

export LD_LIBRARY_PATH="${INTEL_LIB}:${LD_LIBRARY_PATH:-}"

# One CPU
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ================================================================
# Directories
# ================================================================

LOGDIR="${BASE}/run/2026-01/slurm_logs"
MANIFEST="${BASE}/run/2026-01/flexpart_month_manifest.txt"

mkdir -p "${LOGDIR}"

# ================================================================
# Check manifest
# ================================================================

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found:"
    echo "${MANIFEST}"
    exit 1
fi

# ================================================================
# Get case corresponding to SLURM_ARRAY_TASK_ID
# ================================================================

CASE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")

if [ -z "${CASE}" ]; then
    echo "ERROR: No case found for array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Convert relative manifest paths to absolute paths.
case "${CASE}" in
    /*) ;;
    *)  CASE="${BASE}/${CASE}" ;;
esac

if [ -z "${CASE}" ]; then
    echo "ERROR: No case found for array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "=============================================================="
echo "FLEXPART-WRF Upper Hunter"
echo "=============================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Array ID     : ${SLURM_ARRAY_JOB_ID}"
echo "Task ID      : ${SLURM_ARRAY_TASK_ID}"
echo "Host         : $(hostname)"
echo "Start        : $(date)"
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

# ================================================================
# Skip already completed FLEXPART cases
# ================================================================

shopt -s nullglob
OUTPUT_FILES=(flxout_d01_*.nc)
shopt -u nullglob

if [ ${#OUTPUT_FILES[@]} -gt 0 ]; then

    echo "OUTPUT ALREADY EXISTS -- SKIPPING"

    for f in "${OUTPUT_FILES[@]}"; do
        echo "  ${f}"
    done

    echo "Finished: $(date)"
    exit 0
fi

# ================================================================
# Check input
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
# Verify output
# ================================================================

if [ ${STATUS} -ne 0 ]; then
    echo "ERROR: FLEXPART-WRF failed"
    exit ${STATUS}
fi

shopt -s nullglob
OUTPUT_FILES=(flxout_d01_*.nc)
shopt -u nullglob

if [ ${#OUTPUT_FILES[@]} -eq 0 ]; then
    echo "ERROR: FLEXPART finished but no flxout_d01_*.nc was produced"
    exit 1
fi

echo
echo "FLEXPART-WRF SUCCESS"
echo "Output files:"

for f in "${OUTPUT_FILES[@]}"; do
    ls -lh "${f}"
done

echo
echo "Finished: $(date)"
echo "=============================================================="

