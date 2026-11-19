import xarray as xr, matplotlib.pyplot as plt, numpy as np, os

f = "/home/duch/software/hysplit_data2arl/test_upperhunter/stilt_singleton_20260110/out_clean/by-id/202601100200_151.16132_-32.57123_5/202601100200_151.16132_-32.57123_5_foot.nc"

ds = xr.open_dataset(f)
foot = ds["foot"].values
lon = ds["lon"].values
lat = ds["lat"].values
times = ds["time"].values

# Plot the four footprint layers separately.
outdir = "/mnt/data/stilt_footprint_plots"
os.makedirs(outdir, exist_ok=True)

for k in range(foot.shape[0]):
    fig, ax = plt.subplots(figsize=(9, 7))
    v = foot[k]
    im = ax.pcolormesh(lon, lat, v, shading="auto")
    ax.scatter(151.16132, -32.57123, marker="*", s=120, label="Singleton receptor")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"STILT footprint: {str(times[k])[:16].replace('T',' ')} UTC")
    ax.legend()
    fig.colorbar(im, ax=ax, label="Footprint (ppm (µmol m⁻² s⁻¹)⁻¹)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/footprint_{k+1:02d}.png", dpi=180)
    plt.close(fig)

# Combined diagnostic sum of the four layers.
combined = np.sum(foot, axis=0)
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.pcolormesh(lon, lat, combined, shading="auto")
ax.scatter(151.16132, -32.57123, marker="*", s=120, label="Singleton receptor")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("STILT footprint: sum of 4 output time slices")
ax.legend()
fig.colorbar(im, ax=ax, label="Summed footprint")
fig.tight_layout()
fig.savefig(f"{outdir}/footprint_combined_4slices.png", dpi=180)

plt.show()
plt.close(fig)

print(outdir)
print("Created 5 PNG plots.")


