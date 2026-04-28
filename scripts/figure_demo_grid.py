from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from depopulation.radial_f import load_radial_f

# Choose a reference city
# Colima is a good option given the grid is clearly visible
cve = "06.1.01"
year = 2020
dmax = 8

plt.rcParams["font.size"] = 14
fs = 16

# Load the grid
mesh = gpd.read_parquet("outputs/mesh.geoparquet").loc[cve, 2020]
# Get center point
center = mesh.centroid_hist.iloc[0:1]

# Load rem values
rem = (
    pd.read_csv("outputs/remoteness_brackets.csv")
    .drop(columns="CVE_NAME")
    .set_index("CVE_MET")
    .loc[cve]
)

# Load radial density functions
radial_f = load_radial_f([cve], Path("outputs/radial_f/"), core=True)[cve]
fig, axes = plt.subplot_mosaic(
    """
    MMMMMM
    MMMMMM
    MMMMMM
    MMMMMM
    AAAAAA
    BBBBBB
    """,
    figsize=(10, 16),
    layout="constrained",
)


ax = axes["M"]
mesh.plot(
    column="POB_URB", cmap="Blues", alpha=0.5, ax=ax, aspect="equal", edgecolor="black"
)

# Plot rings
for r in radial_f["r_disk"][1:]:
    center.buffer(r).plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.7)

# Plot rem brakets
for _, r in rem.items():
    center.buffer(r * 1000).plot(
        ax=ax, facecolor="none", edgecolor="black", alpha=1, lw=2, ls="--"
    )

# Choose a ring to demo sigma
idx_sigma = 30
r_sigma_low = radial_f["r_disk"][idx_sigma]
r_sigma_high = radial_f["r_disk"][idx_sigma + 1]
disk_sigma_low = center.buffer(r_sigma_low)
disk_sigma_high = center.buffer(r_sigma_high)
ring = disk_sigma_high.difference(disk_sigma_low)
ring.plot(color="brown", ax=ax, alpha=0.5)


# Choose a disk to demo sigma bar
idx_sigma_bar = 10
r_sigmabar = radial_f["r_disk"][idx_sigma_bar]
disk_sigmabar = center.buffer(r_sigmabar)
disk_sigmabar.plot(color="green", ax=ax, alpha=0.5)

# Setup a square region
xc, yc = center.iloc[0].coords.xy
xc = xc[0]
yc = yc[0]
dmax = 7000
xmax = dmax - 700
ymin = yc - dmax + 1700
ax.set_xlim(xc - dmax + 900, xc + xmax)
ax.set_ylim(ymin, yc + dmax)
xticks_m = radial_f["r_disk"][:70:10]
P = radial_f["cumpop_2020"][-1]
xticks_r = np.round(xticks_m / np.sqrt(P), 2)
ax.set_xticks(
    xc + xticks_m,
    [
        f"{int(s / 1000)}\n({r})" for s, r in zip(xticks_m, xticks_r)
    ],  # (xticks_m / 1000).astype(int)
)
ax.set_yticks([])
ax.text(xc - 2300, ymin - 350, r"s")
ax.text(xc - 2300, ymin - 700, r"$r=1000/\sqrt{P}$")

#### A ###################################
ax = axes["A"]
ax.plot(
    radial_f["r_ring"] / 1000,  # change to km
    radial_f["sigma_2020"] * 1e6,  # change to people per km2
    color="brown",
)
# ax.fill_between(
#    [r_sigma_low/1000, r_sigma_high/1000],
#    [(radial_f["sigma_2020"]*1e6)[idx_sigma], (radial_f["sigma_2020"]*1e6)[idx_sigma]],
#    color="brown", alpha=0.5
# )
midp = (r_sigma_low / 1000 + r_sigma_high / 1000) / 2
ax.plot(
    [midp, midp],
    [0, (radial_f["sigma_2020"] * 1e6)[idx_sigma]],
    color="brown",
)
ax.scatter([midp], [(radial_f["sigma_2020"] * 1e6)[idx_sigma]], color="brown", zorder=3)
# for _, r in rem.items():
#    ax.plot(
#        [r, r],
#        [0, np.interp(r*1000, radial_f["r_ring"], 1e6*radial_f["sigma_2020"])],
#        color="black"
#    )
ax.set_ylabel(r"$\sigma$ / $\bar\sigma$ (people/km$^2$)")
ax.set_xlabel("s (km)")
ax.set_xlim(0, xmax / 1000)
ax.set_ylim(0, None)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6])

ax.text(
    3.2,
    500,
    r"$\sigma(s) = \frac{\text{population}(\circledcirc)}{\text{area}(\circledcirc)}$",
    fontsize=fs,
)


#### B ##################################
ax = axes["B"]
ax.plot(
    radial_f["r_ring"] / 1000,  # change to km
    1000 * radial_f["rho_2020"],  # change to keep norm
    color="black",
)
aidx = 4
bidx = 28
ax.fill_between(
    (radial_f["r_ring"] / 1000)[aidx:bidx],
    1000 * radial_f["rho_2020"][aidx:bidx],
    color="grey",
    alpha=0.5,
)
midp = (r_sigma_low / 1000 + r_sigma_high / 1000) / 2
ax.plot(
    [midp, midp],
    [0, (radial_f["rho_2020"] * 1e3)[idx_sigma]],
    color="brown",
)
ax.scatter([midp], [(radial_f["rho_2020"] * 1e3)[idx_sigma]], color="brown", zorder=3)
# for _, r in rem.items():
#    ax.plot(
#        [r, r],
#        [0, np.interp(r*1000, radial_f["r_ring"], 1e3*radial_f["rho_2020"])],
#        color="black"
#    )
ax.set_ylabel(r"$\rho$")
ax.set_xlabel("s (km)")
ax.set_xlim(0, xmax / 1000)
ax.set_ylim(0, None)
ax.set_xticks(
    [
        0,
        radial_f["r_ring"][aidx] / 1000,
        1,
        2,
        radial_f["r_ring"][bidx - 1] / 1000,
        3,
        4,
        5,
        6,
    ],
    [0, "a", 1, 2, "b", 3, 4, 5, 6],
)

ax.text(
    3.1,
    0.06,
    r"$\rho(s) ds = \frac{2\pi s \sigma(s)}{P} ds \approx \frac{population(\circledcirc)}{P}$",
    fontsize=fs,
)

ax.text(3.6, 0.02, r"$= Prob(s < \quad < s + ds)$", fontsize=fs)

ax.text(0.6, 0.025, r"$P\int_a^b \rho(s) ds =$", fontsize=fs)
ax.text(1.7, 0.016, "population \n in [a, b]", fontsize=fs)

#### AA ##########################################################
ax = axes["A"]
s = radial_f["r_disk"][1:] / 1000
barsigma = radial_f["cumpop_2020"][1:] / (np.pi * s**2)

ax.plot(
    s,
    barsigma,
    color="green",
)
# ax.fill_between(
#    [0, r_sigmabar/1000],
#    [barsigma[idx_sigma], barsigma[idx_sigma]],
#    color="green", alpha=0.5
# )
ax.plot(
    [r_sigmabar / 1000, r_sigmabar / 1000],
    [0, barsigma[idx_sigma_bar - 1]],
    color="green",
)
ax.scatter([r_sigmabar / 1000], [barsigma[idx_sigma_bar - 1]], color="green", zorder=3)
# for _, r in rem.items():
#    ax.plot(
#        [r, r],
#        [0, np.interp(r*1000, radial_f["r_ring"], 1e3*radial_f["rho_2020"])],
#        color="black"
#    )
# ax.set_ylabel(r"$\bar\sigma$")
# ax.set_xlabel("s")
ax.set_xlim(0, xmax / 1000)
ax.set_ylim(0, None)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6])

ax.text(
    1.1,
    2000,
    r"$\bar\sigma(s) = \frac{\text{population}(\bigcirc)}{\text{area}(\bigcirc)}$",
    fontsize=fs,
)


# plt.savefig("figures/demo_grid.svg")
plt.savefig("figures/demo_grid.pdf")
plt.close()
