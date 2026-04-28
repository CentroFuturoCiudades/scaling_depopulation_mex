from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from depopulation.r_scaling import (
    estimates_slopes,
    find_max_p,
    get_quantiles,
    radial_f_collapse_single,
)
from depopulation.radial_f import gen_pop_ar, load_radial_f

cve_bold = {
    "09.1.01": "tab:blue",
    "19.1.01": "tab:orange",
    "14.1.01": "tab:green",
    "02.2.03": "tab:red",
    "06.1.01": "tab:purple",
}
beta = 0.5
func = "rho"


agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg")
years = (1990, 2000, 2010, 2020)
cve_list = list(agg_all_df.CVE_MET.values)
cve_names = agg_all_df.NOM_MET.to_list()

radial_f = load_radial_f(cve_list, Path("outputs/radial_f/"), core=True)
N_c = gen_pop_ar(cve_list, Path("outputs/radial_f/"))

pgrid = np.linspace(0.01, 1, 100)
q_arr_c = get_quantiles(pgrid, cve_list, years, radial_f)
# Finding the maximum quantile at which density loss is observed
row_max = find_max_p(cve_list, cve_names, N_c, radial_f).iloc[0]
p_max = int(row_max.max_p * 100) / 100
idx_max = np.where(pgrid == p_max)[0][0]
print(f"Maximum p observed for {row_max.NOM_MET} at p={p_max} at pgrid index {idx_max}")

# Get scaling factors for all cities
slopes_df = estimates_slopes(cve_list, q_arr_c, N_c, i0=idx_max)

_, axes = plt.subplot_mosaic(
    """
    AA
    BB
    CC
    DD
    EE
    """,
    layout="constrained",
    figsize=(6.4, 4.8 * 5 / 3),
)

axi = 0
cve_ax = {
    "09.1.01": "A",
    "19.1.01": "B",
    "14.1.01": "C",
    "02.2.03": "D",
    "06.1.01": "E",
}

for i, cve in enumerate(cve_list):
    if cve not in cve_bold.keys():
        continue

    axi = cve_ax[cve]
    radial_f_collapse_single(
        axes[axi],
        i,
        cve,
        radial_f,
        slopes_df,
        N_c,
        cve_bold[cve],
        beta=beta,
        func=func,
    )

    axes[axi].set_title(
        f"{axi.lower()}) {cve_names[i]}", loc="left", fontsize=12, fontweight="bold"
    )

axes["D"].set_xlim(0, 25)
axes["E"].set_xlim(0, 15)
axes["A"].set_xlim(0, 10)
axes["C"].set_xlim(0, 10)
axes["B"].set_xlim(0, 12)
plt.savefig("figures/rescaled_rhos.pdf")
