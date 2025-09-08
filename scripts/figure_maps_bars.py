"""Generates figure with 5 population change maps and bars with population per
remoteness brackets."""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from depopulation.radial_f import gen_pop_ar, load_radial_f, pop_change_map

# Get colors for example CVE
with open("scripts/cve_colors.json", "r", encoding="utf-8") as f:
    cve_bold = json.load(f)

# Load data
years = (1990, 2000, 2010, 2020)
mesh_gdf = gpd.read_parquet("outputs/mesh.geoparquet")
agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg").set_index("CVE_MET")
cve_list = list(agg_all_df.index.values)
cve_names = agg_all_df.NOM_MET.to_list()
rem_brackets = pd.read_csv("outputs/remoteness_brackets.csv", index_col=0)
rem_brackets_2 = pd.read_csv("outputs/rem_brackets_agg.csv")
radial_f = load_radial_f(cve_list, Path("outputs/radial_f/"), core=True)
pop_ar = gen_pop_ar(cve_list, Path("outputs/radial_f/"))

fig, axes = plt.subplot_mosaic(
    """
    aaabbbccc
    aaabbbccc
    aaabbbccc
    aaabbbccc
    aaabbbccc
    aaabbbccc
    aaabbbccc
    .........
    dddeeefff
    dddeeefff
    dddeeefff
    dddeeeggg
    dddeeeggg
    dddeeeggg
    """,
    figsize=(6.4 * 3, 6.4 * 2),
    layout="constrained",
)

with plt.rc_context(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titleweight": "bold",
    }
):
    for cve, letter in zip(cve_bold, ["a", "b", "c", "d", "e"]):
        pop_change_map(
            mesh_gdf,
            agg_all_df,
            cve,
            axes[letter],
            rem_brackets.loc[cve].drop("CVE_NAME"),
            radial_f[cve]["r_disk"][-1] / 1000,
            adjust_vmax=True,
        )
        axes[letter].set_title(
            f"{letter}) {cve_names[cve_list.index(cve)]}",
            loc="left",
            backgroundcolor=cve_bold[cve],
            color="white",
            # pad=-20,
            y=0.93,
        )

    # Top barplot
    sns.barplot(
        data=rem_brackets_2,
        x="bracket",
        y="population",
        hue="year",
        ax=axes["f"],
    )
    axes["f"].set_xlabel("")
    axes["f"].set_ylabel("Total population\n(millions)", fontsize=14)
    axes["f"].set_xticks(
        []
        # [0, 1, 2, 3],
        # [r"$r < 3$", r"$3 < r < 5$", r"$5 < r < 9.3$", r"$r > 9.3$"]
    )
    axes["f"].tick_params(axis="y", labelsize=14)
    sns.move_legend(
        axes["f"],
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=4,
        title=None,
        frameon=False,
        fontsize=14,
    )
    axes["f"].set_title(" f)", loc="left", y=0.88)

    # Bottom barplot
    sns.barplot(
        rem_brackets_2,
        x="bracket",
        y="p_fraction",
        hue="year",
        ax=axes["g"],
    )
    axes["g"].set_xlabel("")
    axes["g"].set_ylabel("Fraction of population", fontsize=14)
    axes["g"].set_xticks(
        [0, 1, 2, 3],
        [r"$r < 3$", r"$3 < r < 5$", r"$5 < r < 9.3$", r"$r > 9.3$"],
        fontsize=14,
    )
    axes["g"].tick_params(axis="y", labelsize=14)
    axes["g"].get_legend().remove()
    axes["g"].set_title(" g)", loc="left", y=0.88)
    axes["g"].set_ylim(0, 0.51)

    plt.savefig("figures/maps_bars.pdf")
    plt.close()
