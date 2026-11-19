conda activate stilt_env2

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd /home/duch/software/stilt

Rscript r/stilt_cli.r \
    stilt_wd=/home/duch/software/stilt \
    output_wd=/mnt/scratch_lustre/duch/stilt/upperhunter/arl_mine_jan2026/footprints/Singleton \
    r_run_time=2026-01-10T02:00:00Z \
    r_lati=-32.57123 \
    r_long=151.16132 \
    r_zagl=5 \
    met_path=/mnt/scratch_lustre/duch/stilt/upperhunter/arl_mine_jan2026/arl \
    met_file_format=%Y%m%d \
    met_file_tres="24 hours" \
    n_hours=-24 \
    numpar=500 \
    xmn=150.22491 \
    xmx=152.01050 \
    xres=0.01 \
    ymn=-33.217213 \
    ymx=-31.885994 \
    yres=0.01 \
    projection="+proj=longlat" \
    run_trajec=T \
    run_foot=T \
    rm_dat=T \
    timeout=3600
