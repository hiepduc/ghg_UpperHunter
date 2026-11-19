#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

root = Path.home() / "flexpart-wrf/upperhunter_flexpart/H_matrix_flexpart"

rts_file = (
    root / "RTS_projected_bounded_full_footprint_13mine_Jan2026"
    / "daily_projected_bounded_RTS_scaling_factors.csv"
)
map_file = (
    root / "Global_constrained_smoother_13mine_Jan2026"
    / "daily_global_constrained_smoother_scaling_factors.csv"
)

mines = [f"OC{i}" for i in range(1, 14)]

rts = pd.read_csv(rts_file, parse_dates=["date"])
ref = pd.read_csv(map_file, parse_dates=["date"])

df = rts.merge(ref, on="date", suffixes=("_RTS", "_MAP"))

a = df[[f"{m}_RTS" for m in mines]].to_numpy()
b = df[[f"{m}_MAP" for m in mines]].to_numpy()
diff = a - b
absdiff = np.abs(diff)

print("=" * 76)
print("PROJECTED CONSTRAINED RTS vs GLOBAL CONSTRAINED BATCH MAP")
print("=" * 76)
print(f"Days: {len(df)}")
print(f"Mines: {len(mines)}")
print(f"Parameters compared: {diff.size}")
print(f"RTS range: [{a.min():.8f}, {a.max():.8f}]")
print(f"MAP range: [{b.min():.8f}, {b.max():.8f}]")
print(f"Maximum absolute difference: {absdiff.max():.10f}")
print(f"RMSE: {np.sqrt(np.mean(diff**2)):.10f}")
print(f"MAE: {np.mean(absdiff):.10f}")
print(f"Mean signed difference (RTS - MAP): {diff.mean():.10f}")
print(f"Correlation: {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.10f}")
for tol in [0.001, 0.01, 0.1, 0.5, 1.0]:
    print(f"Parameters with |difference| > {tol}: "
          f"{np.sum(absdiff > tol)}/{diff.size}")

# Per-day comparison
daily_rows = []
for i, row in df.iterrows():
    da = np.array([row[f"{m}_RTS"] for m in mines])
    db = np.array([row[f"{m}_MAP"] for m in mines])
    dd = da - db
    daily_rows.append({
        "date": row["date"].strftime("%Y-%m-%d"),
        "RMSE": np.sqrt(np.mean(dd**2)),
        "MAE": np.mean(np.abs(dd)),
        "max_abs_difference": np.max(np.abs(dd)),
        "mean_RTS": da.mean(),
        "mean_MAP": db.mean(),
    })

daily = pd.DataFrame(daily_rows)
daily.to_csv("projected_RTS_vs_global_MAP_daily_comparison.csv", index=False)

print("\nDays with largest maximum absolute differences:")
print(
    daily.sort_values("max_abs_difference", ascending=False)
         .head(10).to_string(index=False, float_format=lambda x: f"{x:.8f}")
)

# Individual mine-day differences
records = []
for i, row in df.iterrows():
    for m in mines:
        x_rts = row[f"{m}_RTS"]
        x_map = row[f"{m}_MAP"]
        records.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "mine": m,
            "RTS": x_rts,
            "global_MAP": x_map,
            "difference": x_rts - x_map,
            "abs_difference": abs(x_rts - x_map),
        })

params = pd.DataFrame(records)
params.to_csv("projected_RTS_vs_global_MAP_parameter_comparison.csv", index=False)

print("\n20 mine-day parameters with largest differences:")
print(
    params.sort_values("abs_difference", ascending=False)
          .head(20).to_string(index=False, float_format=lambda x: f"{x:.8f}")
)

print("\nFinal day (2026-01-31):")
last = params[params["date"] == "2026-01-31"]
print(last.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

print("\nSaved comparison CSVs in the current directory:")
print("  projected_RTS_vs_global_MAP_daily_comparison.csv")
print("  projected_RTS_vs_global_MAP_parameter_comparison.csv")
