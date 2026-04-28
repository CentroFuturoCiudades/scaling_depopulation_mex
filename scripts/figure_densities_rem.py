import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from depopulation.radial_f import load_radial_f, plot_delta_density, plot_density

# Get colors for example CVE
with open("scripts/cve_colors.json", "r", encoding="utf-8") as f:
    cve_bold = json.load(f)

# Load data
years = (1990, 2000, 2010, 2020)
agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg").set_index("CVE_MET")
cve_list = list(agg_all_df.index.values)
cve_names = agg_all_df.NOM_MET.to_list()
radial_f = load_radial_f(cve_list, Path("outputs/radial_f/"), core=True)

N_c = np.array(
    [[radial_f[cve][f"cumpop_{year}"][-1] for year in years] for cve in cve_list]
)

beta = 0.5

with plt.rc_context(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titleweight": "bold",
    }
):
    fig, ax = plt.subplot_mosaic(
        """
        AB
        """,
        figsize=(6.4 * 2, 5.2),
        layout="constrained",
    )

    # Top lines plot
    plot_delta_density(
        ax["A"],
        cve_list,
        radial_f,
        N_c,
        cve_bold,
        cve_names,
        beta=beta,
        scale=True,
        xlim=10,
        agg=True,
        avg=False,
        rem_year=2020,
        fontsize=12,
    )
    ax["A"].legend()
    sns.move_legend(
        ax["A"],
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=False,
    )
    ax["A"].set_title(" a)", loc="left", y=0.92)

    # Inset
    inset = ax["A"].inset_axes((0.45, 0.15, 0.5, 0.45))
    plot_density(
        inset,
        cve_list,
        radial_f,
        N_c,
        beta=beta,
        scale=True,
        xlim=10,
        agg=True,
        avg=False,
        rem_year=2020,
        fontsize=12,
    )
    inset.legend(loc=None, frameon=False, fontsize=11)

    # Bottom lines plot
    plot_delta_density(
        ax["B"],
        cve_list,
        radial_f,
        N_c,
        cve_bold,
        cve_names,
        beta=beta,
        scale=True,
        xlim=10,
        agg=True,
        avg=True,
        rem_year=2020,
        fontsize=12,
    )

    ax["B"].set_title(" b)", loc="left", y=0.92)

    # Inset
    inset = ax["B"].inset_axes((0.45, 0.15, 0.5, 0.45))
    plot_density(
        inset,
        cve_list,
        radial_f,
        N_c,
        beta=beta,
        scale=True,
        xlim=10,
        agg=True,
        avg=True,
        rem_year=2020,
        fontsize=12,
    )
    inset.legend(loc=None, frameon=False, fontsize=11)

    plt.savefig("figures/densities_rem.pdf")
    plt.close()
