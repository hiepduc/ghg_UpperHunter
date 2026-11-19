#!/bin/bash
#SBATCH --job-name=fpwrf_test
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/duch/flexpart-wrf/upperhunter_flexpart/run/production/slurm_test_%A_%a.out
#SBATCH --error=/home/duch/flexpart-wrf/upperhunter_flexpart/run/production/slurm_test_%A_%a.err
#SBATCH --array=0-2

set -e

BASE=/home/duch/flexpart-wrf/upperhunter_flexpart
FLEXPART=/home/duch/flexpart-wrf/flexpart-wrf/flexwrf33_omp

export INTEL_LIB=/mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/opt/spack/linux-x86_64_v3/intel-oneapi-compilers-2023.2.4-phxvcdfl6i64vlmkipsngnsfpqnrxosy/compiler/latest/linux/compiler/lib/intel64_lin
export LD_LIBRARY_PATH=$INTEL_LIB:$LD_LIBRARY_PATH

case $SLURM_ARRAY_TASK_ID in
    0)
        RECEPTOR=singleton
        ;;
    1)
        RECEPTOR=muswellbrook
        ;;
    2)
        RECEPTOR=merriwa
        ;;
    *)
        echo "ERROR: invalid array task ID"
        exit 1
        ;;
esac

RUN_DIR=$BASE/run/production/$RECEPTOR/20260105_0000

echo "============================================================"
echo "FLEXPART-WRF SLURM TEST"
echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Array task   : $SLURM_ARRAY_TASK_ID"
echo "Receptor     : $RECEPTOR"
echo "Run directory: $RUN_DIR"
echo "Host         : $(hostname)"
echo "Start time   : $(date)"
echo "============================================================"

cd "$RUN_DIR"

"$FLEXPART" "$RUN_DIR/flexwrf.input"

echo "============================================================"
echo "Completed: $RECEPTOR"
echo "End time: $(date)"
echo "============================================================"

