"""Prints statistics for some metropolitan zones for population at 5km."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from depopulation.radial_f import load_radial_f

agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg").set_index("CVE_MET")
cve_list = list(agg_all_df.index.values)
radial_f = load_radial_f(cve_list, Path("outputs/radial_f/"), core=True)
years = (1990, 2000, 2010, 2020)

pop_at_5_dict = {}
for cve in cve_list:
    r_disk = radial_f[cve]["r_disk"]
    cumpop = {y: radial_f[cve][f"cumpop_{y}"] for y in years}

    max_r = r_disk[-1]
    if max_r < 5000:
        idx = -1
    else:
        idx = np.where(r_disk == 5000)[0][0]

    pop_at_5 = {y: cumpop[y][idx] for y in years}
    pop_at_5_dict[cve] = pop_at_5
pop_at_5_df = pd.DataFrame(pop_at_5_dict).T

# Monterrey
for name, cve in zip(
    ["CDMX", "Monterrey", "Guadalajara"], ["09.1.01", "19.1.01", "14.1.01"]
):
    df = pop_at_5_df.loc[cve]
    print(f"{name}. Pop at 5 km from center:")
    print(f"    1990: {round(df.loc[1990])}")
    print(f"    2020: {round(df.loc[2020])}")
    print(f"    Difference: {round(df.loc[2020] - df.loc[1990])}")
    print(
        f"    Population growth: {round((agg_all_df.loc[cve].POB_URB_2020 - agg_all_df.loc[cve].POB_URB_1990) / agg_all_df.loc[cve].POB_URB_1990 * 100, 2)}"
    )
