#!/bin/bash

set -u

STILT_WD="/home/duch/software/stilt"

PROJECT="/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/arl_mine_2026/2026-02"

MET_PATH="${PROJECT}/arl"
OUTPUT_BASE="${PROJECT}/footprints"

XMN=150.22491
XMX=152.01050
YMN=-33.217213
YMX=-31.885994

NUMPAR=500
N_HOURS=-24

mkdir -p "${OUTPUT_BASE}/Singleton"
mkdir -p "${OUTPUT_BASE}/Muswellbrook"
mkdir -p "${OUTPUT_BASE}/Merriwa"

run_stilt()
{
    NAME="$1"
    LAT="$2"
    LON="$3"
    TIME="$4"

    OUT="${OUTPUT_BASE}/${NAME}"

    echo ""
    echo "============================================================"
    echo "STILT simulation"
    echo "Receptor : ${NAME}"
    echo "Time     : ${TIME}"
    echo "Lat      : ${LAT}"
    echo "Lon      : ${LON}"
    echo "============================================================"

    Rscript "${STILT_WD}/r/stilt_cli.r" \
        stilt_wd="${STILT_WD}" \
        output_wd="${OUT}" \
        r_run_time="${TIME}" \
        r_lati="${LAT}" \
        r_long="${LON}" \
        r_zagl=5 \
        met_path="${MET_PATH}" \
        met_file_format=%Y%m%d \
        met_file_tres="24 hours" \
        n_hours="${N_HOURS}" \
        numpar="${NUMPAR}" \
        xmn="${XMN}" \
        xmx="${XMX}" \
        xres=0.01 \
        ymn="${YMN}" \
        ymx="${YMX}" \
        yres=0.01 \
        projection="+proj=longlat" \
        run_trajec=T \
        run_foot=T \
        rm_dat=T \
        timeout=3600

    STATUS=$?

    if [ ${STATUS} -ne 0 ]; then
        echo "ERROR: ${NAME} ${TIME}"
        echo "Exit status = ${STATUS}"
        return ${STATUS}
    fi

    echo "Completed: ${NAME} ${TIME}"
    return 0
}


# ============================================================
# February 1 through February 28, 2026
# Every hour, every receptor
# ============================================================

for DAY in $(seq -w 01 28); do

    for HOUR in $(seq -w 00 23); do

        TIME="2026-02-${DAY}T${HOUR}:00:00Z"

        run_stilt \
            "Singleton" \
            "-32.57123" \
            "151.16132" \
            "${TIME}" || exit 1

        run_stilt \
            "Muswellbrook" \
            "-32.26202" \
            "150.89297" \
            "${TIME}" || exit 1

        run_stilt \
            "Merriwa" \
            "-32.14097" \
            "150.35767" \
            "${TIME}" || exit 1

    done

done

echo ""
echo "============================================================"
echo "ALL STILT SIMULATIONS COMPLETED"
echo "============================================================"

