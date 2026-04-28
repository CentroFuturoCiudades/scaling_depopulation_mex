from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from depopulation.radial_f import gen_pop_ar, load_radial_f

plt.rcParams["font.size"] = 14
# plt.rcParams['mathtext.fontset'] = 'cm'
fs = 14

fig, axes = plt.subplot_mosaic(
    """
    .SS..sss
    .SS..sss
    LLLL.sss
    LLLL.lll
    LLLL.lll
    LLLL.lll
    EEEE.HHH
    """,
    figsize=(10, 9),
    layout="constrained",
)

large_cve = "14.1.01"
small_cve = "06.1.01"
year = 2020

large_mesh = gpd.read_parquet("outputs/mesh.geoparquet").loc[large_cve, 2020]
small_mesh = gpd.read_parquet("outputs/mesh.geoparquet").loc[small_cve, 2020]

large_center = large_mesh.centroid_hist.iloc[0:1]
small_center = small_mesh.centroid_hist.iloc[0:1]
lxc, lyc = large_center.iloc[0].coords.xy
lxc = lxc[0]
lyc = lyc[0]
sxc, syc = small_center.iloc[0].coords.xy
sxc = sxc[0]
syc = syc[0]

# Load rem values
large_rem = (
    pd.read_csv("outputs/remoteness_brackets.csv")
    .drop(columns="CVE_NAME")
    .set_index("CVE_MET")
    .loc[large_cve]
)
small_rem = (
    pd.read_csv("outputs/remoteness_brackets.csv")
    .drop(columns="CVE_NAME")
    .set_index("CVE_MET")
    .loc[small_cve]
)

# Load radial density functions
large_radial_f = load_radial_f([large_cve], Path("outputs/radial_f/"), core=True)[
    large_cve
]
small_radial_f = load_radial_f([small_cve], Path("outputs/radial_f/"), core=True)[
    small_cve
]

pop_ar = gen_pop_ar([large_cve, small_cve], Path("outputs/radial_f/"))
P_large = pop_ar[0, 3]
P_small = pop_ar[1, 3]


def plot_mesh(ax, mesh, R, rem, center, xticks, P, rem_color, scale=False):
    xc, yc = center.iloc[0].coords.xy
    xc = xc[0]
    yc = yc[0]

    mesh.plot(column="POB_URB", cmap="Blues", alpha=0.5, ax=ax, aspect="equal")
    if scale:
        ax.set_xticks(xc + np.sqrt(P) * xticks, xticks)
    else:
        ax.set_xticks(xc + xticks * 1000, xticks)
    ax.set_yticks([])
    ax.set_xlim(xc - R, xc + R)
    ax.set_ylim(yc - R, yc + R)
    # Plot rem brakets
    for _, r in rem.items():
        center.buffer(r * 1000).plot(
            ax=ax, facecolor="none", edgecolor=rem_color, alpha=1, lw=2, ls="--"
        )

    cx.add_basemap(
        ax,
        crs=mesh.crs,
        source=cx.providers.CartoDB.PositronNoLabels,
        attribution_size=6,
        # zoom_adjust=0,
    )


def plot_sigma(ax, radial_f, P, R, xticks, color, scale=False):
    R_idx = np.searchsorted(radial_f["r_ring"], R)
    x = radial_f["r_ring"][:R_idx] / 1000
    y = radial_f["sigma_2020"][:R_idx] * 1e6

    if scale:
        x = 1000 * x / np.sqrt(P)
        R = 1000 * R / np.sqrt(P)
        ax.set_xlabel("r")
        rems = np.array([3, 5, 9.3])
    else:
        ax.set_xlabel("s")
        rems = np.array([3, 5, 9.3]) * np.sqrt(P) / 1000
    ax.set_xlim(0, R / 1000)
    ax.plot(x, y, color=color)
    for r in rems:
        ax.axvline(r, ls="--", color=color)
    ax.set_xticks(xticks)
    ax.set_ylabel(r"$\sigma$")
    ax.set_xlim(0, R / 1000)
    return


plot_sigma(
    axes["E"],
    small_radial_f,
    P_small,
    6000,
    np.array([0, 5, 10, 15, 20, 25]),
    "brown",
    scale=False,
)
plot_mesh(
    axes["S"],
    small_mesh,
    6000,
    small_rem,
    small_center,
    np.array([0, 5]),
    P_small,
    "brown",
)

plot_sigma(
    axes["E"], large_radial_f, P_large, 25000, np.array([0, 5, 10, 15, 20, 25]), "green"
)
plot_mesh(
    axes["L"],
    large_mesh,
    25000,
    large_rem,
    large_center,
    np.array([0, 5, 10, 15, 20, 25]),
    P_large,
    "green",
)

plot_sigma(
    axes["H"], large_radial_f, P_large, 25000, np.array([0, 5, 10]), "green", scale=True
)
plot_mesh(
    axes["l"],
    large_mesh,
    25000,
    large_rem,
    large_center,
    np.array([0, 5, 10]),
    P_large,
    "green",
    scale=True,
)

plot_sigma(
    axes["H"], small_radial_f, P_small, 6000, np.array([0, 5, 10]), "brown", scale=True
)
plot_mesh(
    axes["s"],
    small_mesh,
    6000,
    small_rem,
    small_center,
    np.array([]),
    P_small,
    "brown",
    scale=True,
)

with plt.rc_context({"mathtext.fontset": "cm"}):
    fig.text(
        0,
        0.86,
        "Small city of size\n      $P_s < P_0$\n     so that\n   $2\\pi\\int_0^{S_s} s\\sigma(s)ds =  P_s$",
        fontsize=12,
    )
    fig.text(
        0.06,
        0.69,
        "Large city of size $P_l > P_0$ so that $2\\pi\\int_0^{S_l} s\\sigma(s)ds = P_l$",
        fontsize=12,
    )
    fig.text(
        0.7,
        0.57,
        "Equivalent (scaled) city of size $P_0$\n so that $2\\pi\\int_0^{R} r\\sigma(r)dr = P_0$",
        fontsize=12,
    )
    fig.text(0.5, 0.08, r"$\sigma$ values preserved", fontsize=12)
    fig.text(0.55, 0.4, r"$r=\sqrt{\frac{P_0}{P_s}}s$", fontsize=12)
    fig.text(0.55, 0.8, r"$r=\sqrt{\frac{P_0}{P_l}}s$", fontsize=12)

plt.savefig("figures/demo_rem.pdf")
plt.close()
