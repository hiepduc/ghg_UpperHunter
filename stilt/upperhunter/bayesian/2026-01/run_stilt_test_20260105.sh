#!/bin/bash

set -u

# ============================================================
# Upper Hunter STILT production test
#
# Date: 2026-01-05 00:00 UTC
#
# Receptors:
#   Singleton
#   Muswellbrook
#   Merriwa
#
# Based exactly on the successful Singleton test:
# 2026-01-10 02:00 UTC
# ============================================================

PROJECT="/mnt/scratch_lustre/duch/stilt/upperhunter/arl_mine_jan2026"
STILT_WD="/home/duch/software/stilt"

ARL_DIR="${PROJECT}/arl"
FOOT_DIR="${PROJECT}/footprints"
LOG_DIR="${PROJECT}/logs"

TIME="2026-01-05T00:00:00Z"

# WRF d03 domain
XMN=150.22491
XMX=152.01050
YMN=-33.217213
YMX=-31.885994

# STILT settings
NUMPAR=500
N_HOURS=-24

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Upper Hunter STILT test"
echo "============================================================"
echo "Time      : ${TIME}"
echo "Particles : ${NUMPAR}"
echo "n_hours   : ${N_HOURS}"
echo "ARL dir   : ${ARL_DIR}"
echo "============================================================"
echo

conda activate stilt_env2

unset R_HOME
unset R_LIBS_USER
unset R_LIBS

# ------------------------------------------------------------
# Check ARL files
# ------------------------------------------------------------

for DATE in 20260104 20260105
do
    FILE="${ARL_DIR}/UpperHunter_d03_${DATE}_mine.arl"

    if [ ! -f "${FILE}" ]; then
        echo "ERROR: Missing ARL file:"
        echo "       ${FILE}"
        exit 1
    fi

    echo "Found ARL:"
    echo "  ${FILE}"
done

echo

# ------------------------------------------------------------
# Function to run one receptor
# ------------------------------------------------------------

run_receptor()
{
    NAME="$1"
    LAT="$2"
    LON="$3"
    ZAGL="$4"

    OUT="${FOOT_DIR}/${NAME}"
    LOG="${LOG_DIR}/stilt_${NAME}_20260105_0000UTC.log"

    mkdir -p "${OUT}"

    echo
    echo "------------------------------------------------------------"
    echo "Running STILT"
    echo "Receptor : ${NAME}"
    echo "Latitude : ${LAT}"
    echo "Longitude: ${LON}"
    echo "ZAGL     : ${ZAGL} m"
    echo "Output   : ${OUT}"
    echo "Log      : ${LOG}"
    echo "------------------------------------------------------------"

    cd "${STILT_WD}" || exit 1

    Rscript r/stilt_cli.r \
        stilt_wd="${STILT_WD}" \
        output_wd="${OUT}" \
        r_run_time="${TIME}" \
        r_lati="${LAT}" \
        r_long="${LON}" \
        r_zagl="${ZAGL}" \
        met_path="${ARL_DIR}" \
        met_file_format=%Y%m%d \
        met_file_tres="24 hours" \
        n_hours=${N_HOURS} \
        numpar=${NUMPAR} \
        xmn=${XMN} \
        xmx=${XMX} \
        xres=0.01 \
        ymn=${YMN} \
        ymx=${YMX} \
        yres=0.01 \
        projection="+proj=longlat" \
        run_trajec=T \
        run_foot=T \
        rm_dat=T \
        timeout=3600 \
        > "${LOG}" 2>&1

    STATUS=$?

    if [ ${STATUS} -ne 0 ]; then
        echo
        echo "ERROR: STILT failed for ${NAME}"
        echo "See:"
        echo "  ${LOG}"
        return ${STATUS}
    fi

    echo
    echo "STILT completed successfully for ${NAME}"
}

# ------------------------------------------------------------
# Run the three receptors
# ------------------------------------------------------------

run_receptor \
    "Singleton" \
    -32.57123 \
    151.16132 \
    5

run_receptor \
    "Muswellbrook" \
    -32.26202 \
    150.89297 \
    5

run_receptor \
    "Merriwa" \
    -32.14097 \
    150.35767 \
    5

echo
echo "============================================================"
echo "January 5 STILT test completed"
echo "============================================================"
echo

