"""Generates a population change map for each zone."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from depopulation.radial_f import load_radial_f, pop_change_map

# Load data
mesh_gdf = gpd.read_parquet("outputs/mesh.geoparquet")
agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg").set_index("CVE_MET")
cve_list = list(agg_all_df.index.values)
rem_brackets = pd.read_csv("outputs/remoteness_brackets.csv", index_col=0)
radial_f = load_radial_f(cve_list, Path("outputs/radial_f/"), core=True)

opath = Path("figures/maps/")
opath.mkdir(parents=True, exist_ok=True)

for i, cve in enumerate(cve_list):
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8))
    pop_change_map(
        mesh_gdf,
        agg_all_df,
        cve,
        ax,
        rem_vals=rem_brackets.loc[cve].drop("CVE_NAME"),
        rmax=radial_f[cve]["r_disk"][-1] / 1000,
    )
    # ax.set_title(f"{cve} {cve_names[i]}", loc="left", pad=-20)
    plt.tight_layout()
    plt.savefig(opath / f"{cve}.pdf")
    plt.close()
