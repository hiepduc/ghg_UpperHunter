import numpy as np
import pandas as pd

H_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/"
    "/H_matrix_flexpart/2026-02/H_flexpart_0_10m_13mine_Feb2026.csv"
)

OBS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

START_DATE = pd.Timestamp("2026-02-01 00:00")
END_DATE = pd.Timestamp("2026-02-28 00:00")

RECEPTOR_COLUMNS = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}


def norm(x):
    x = str(x).lower().strip()

    if "singleton" in x:
        return "singleton"
    if "muswellbrook" in x:
        return "muswellbrook"
    if "merriwa" in x:
        return "merriwa"

    return x


print("=" * 90)
print("DIRECT COMPARISON OF H/y PREPROCESSING")
print("=" * 90)


# ============================================================================
# LOAD H
# ============================================================================

H_df = pd.read_csv(H_FILE)

meta_cols = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

mine_columns = [
    c for c in H_df.columns
    if c not in meta_cols
]

H_values = (
    H_df[mine_columns]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float)
)

H_time = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
).dt.floor("h")

H_receptor = H_df["receptor"].map(norm)

H_key = pd.DataFrame({
    "receptor": H_receptor,
    "time_aest": H_time,
})

H_key["H_row"] = np.arange(len(H_key))

print()
print(f"H shape: {H_values.shape}")
print(f"H keys : {len(H_key)}")


# ============================================================================
# LOAD PICARRO — MCMC STYLE
# ============================================================================

obs = pd.read_csv(OBS_FILE)

obs["time_aest"] = pd.to_datetime(
    obs["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce",
)

obs_long_list = []

for receptor, column in RECEPTOR_COLUMNS.items():

    tmp = pd.DataFrame({
        "receptor": receptor,
        "time_aest": obs["time_aest"],
        "obs_ch4_ppm": pd.to_numeric(
            obs[column],
            errors="coerce",
        ),
    })

    tmp = tmp.dropna(
        subset=["time_aest", "obs_ch4_ppm"]
    )

    tmp = tmp[
        (tmp["time_aest"] >= START_DATE)
        &
        (
            tmp["time_aest"]
            < END_DATE + pd.Timedelta(days=1)
        )
    ]

    tmp["time_aest"] = tmp["time_aest"].dt.floor("h")

    obs_long_list.append(tmp)

obs_long = pd.concat(
    obs_long_list,
    ignore_index=True,
)

# Average duplicates exactly as MCMC does
obs_long = (
    obs_long
    .groupby(
        ["receptor", "time_aest"],
        as_index=False,
    )["obs_ch4_ppm"]
    .mean()
)

print()
print(f"Usable Picarro observations: {len(obs_long)}")


# ============================================================================
# MCMC-STYLE MATCH
# ============================================================================

match_mcmc = H_key.merge(
    obs_long,
    on=["receptor", "time_aest"],
    how="inner",
)

H_mcmc = H_values[
    match_mcmc["H_row"].to_numpy(dtype=int),
    :
]

y_mcmc = match_mcmc[
    "obs_ch4_ppm"
].to_numpy(dtype=float)

keys_mcmc = match_mcmc[
    ["receptor", "time_aest"]
].reset_index(drop=True)


# ============================================================================
# ANALYTICAL-STYLE MATCH
# ============================================================================

H_work = H_df.copy()

H_work["datetime_aest"] = pd.to_datetime(
    H_work["obs_time_aest"],
    errors="coerce",
).dt.floor("h")

H_work["receptor_normalised"] = (
    H_work["receptor"].map(norm)
)

obs_work = obs_long.copy()

obs_work["receptor_normalised"] = (
    obs_work["receptor"].map(norm)
)

obs_work["datetime_aest"] = (
    pd.to_datetime(
        obs_work["time_aest"],
        errors="coerce",
    )
    .dt.floor("h")
)

matched = H_work.merge(
    obs_work[
        [
            "receptor_normalised",
            "datetime_aest",
            "obs_ch4_ppm",
        ]
    ],
    on=[
        "receptor_normalised",
        "datetime_aest",
    ],
    how="left",
)

matched = matched[
    matched["obs_ch4_ppm"].notna()
].copy()

H_analytical = matched[
    mine_columns
].to_numpy(dtype=float)

y_analytical = matched[
    "obs_ch4_ppm"
].to_numpy(dtype=float)

keys_analytical = (
    matched[
        ["receptor_normalised", "datetime_aest"]
    ]
    .rename(
        columns={
            "receptor_normalised": "receptor",
            "datetime_aest": "time_aest",
        }
    )
    .reset_index(drop=True)
)


# ============================================================================
# REPORT
# ============================================================================

print()
print("=" * 90)
print("RESULTS")
print("=" * 90)

print()
print(f"MCMC-style rows       : {len(y_mcmc)}")
print(f"Analytical-style rows : {len(y_analytical)}")

print()
print(
    "Same receptor/time sequence:",
    keys_mcmc.equals(keys_analytical)
)

if not keys_mcmc.equals(keys_analytical):

    print()
    print("FIRST DIFFERENCES IN ROW ORDER:")

    n = min(len(keys_mcmc), len(keys_analytical))

    shown = 0

    for i in range(n):

        a = tuple(keys_mcmc.iloc[i])
        b = tuple(keys_analytical.iloc[i])

        if a != b:

            print()
            print("row", i)
            print("MCMC       :", a)
            print("Analytical :", b)

            shown += 1

            if shown >= 10:
                break


print()
print(
    "Same y values:",
    np.allclose(
        y_mcmc,
        y_analytical,
        rtol=0,
        atol=1e-12,
    )
)

print()
print(
    "Same H matrix:",
    np.allclose(
        H_mcmc,
        H_analytical,
        rtol=0,
        atol=1e-12,
    )
)

print()
print(
    "Maximum |H difference|:",
    np.max(
        np.abs(
            H_mcmc - H_analytical
        )
    )
)

print()
print(
    "Maximum |y difference|:",
    np.max(
        np.abs(
            y_mcmc - y_analytical
        )
    )
)

print()
print("=" * 90)
print("MCMC H/y first 10 rows")
print("=" * 90)

print(
    match_mcmc[
        ["receptor", "time_aest", "obs_ch4_ppm"]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("=" * 90)
print("ANALYTICAL H/y first 10 rows")
print("=" * 90)

print(
    matched[
        [
            "receptor_normalised",
            "datetime_aest",
            "obs_ch4_ppm",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

