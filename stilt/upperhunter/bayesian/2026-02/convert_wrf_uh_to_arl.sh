#!/bin/bash

set -euo pipefail

# ============================================================
# Upper Hunter WRF -> ARL conversion for STILT
#
# WRF source:
#   /mnt/scratch_lustre/nzem_meim_scratch/WRF-GHG/runnetzero/wrf/met_nochem/wrfout_met_d03_*
#
# Conversion period:
#   2026-02-01 through 2026-02-28
#
# Output:
#   UpperHunter_d03_YYYYMMDD_mine.arl
# ============================================================

WRF_DIR="/mnt/scratch_lustre/nzem_meim_scratch/WRF-GHG/runnetzero/wrf"

PROJECT="/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/arl_mine_2026/2026-02"

CFG="${PROJECT}/config/WRFDATA.CFG.stilt"
ARL_DIR="${PROJECT}/arl"
LOG_DIR="${PROJECT}/logs"

ARW2ARL="${HOME}/software/hysplit_data2arl/arw2arl/arw2arl"

mkdir -p "${ARL_DIR}"
mkdir -p "${LOG_DIR}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

if [ ! -x "${ARW2ARL}" ]; then
    echo "ERROR: arw2arl not found:"
    echo "${ARW2ARL}"
    exit 1
fi

if [ ! -f "${CFG}" ]; then
    echo "ERROR: WRFDATA.CFG.stilt not found:"
    echo "${CFG}"
    exit 1
fi

echo "============================================================"
echo "Upper Hunter WRF -> ARL"
echo "============================================================"
echo "WRF directory : ${WRF_DIR}"
echo "ARL directory : ${ARL_DIR}"
echo "CFG           : ${CFG}"
echo "arw2arl       : ${ARW2ARL}"
echo "============================================================"

# ------------------------------------------------------------
# Convert February 1 through February 28
# ------------------------------------------------------------

for day in $(seq -w 1 28); do

    DATE="202601${day}"

    WRF_FILE="${WRF_DIR}/wrfoutUH_d03_2026-02-${day}_00:00:00"
    ARL_FILE="${ARL_DIR}/UpperHunter_d03_${DATE}_mine.arl"
    LOG_FILE="${LOG_DIR}/arw2arl_${DATE}.log"

    echo
    echo "------------------------------------------------------------"
    echo "Processing ${DATE}"
    echo "------------------------------------------------------------"
    echo "Input : ${WRF_FILE}"
    echo "Output: ${ARL_FILE}"

    if [ ! -f "${WRF_FILE}" ]; then
        echo "ERROR: WRF file does not exist:"
        echo "${WRF_FILE}"
        exit 1
    fi

    if [ -f "${ARL_FILE}" ] && [ -s "${ARL_FILE}" ]; then
        echo "ARL already exists -- skipping."
        continue
    fi

    rm -f "${ARL_FILE}"

    "${ARW2ARL}" \
        -i"${WRF_FILE}" \
        -o"${ARL_FILE}" \
        -v"${CFG}" \
        > "${LOG_FILE}" 2>&1

    if [ ! -f "${ARL_FILE}" ] || [ ! -s "${ARL_FILE}" ]; then
        echo
        echo "ERROR: ARL output was not created:"
        echo "${ARL_FILE}"
        echo
        echo "Last 40 lines of log:"
        tail -40 "${LOG_FILE}"
        exit 1
    fi

    echo "SUCCESS:"
    ls -lh "${ARL_FILE}"

done

echo
echo "============================================================"
echo "ALL WRF -> ARL CONVERSIONS COMPLETED"
echo "============================================================"

echo
echo "ARL files:"
ls -lh "${ARL_DIR}"/UpperHunter_d03_202602*_mine.arl
