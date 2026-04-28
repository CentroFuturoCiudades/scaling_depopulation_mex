from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from depopulation.r_scaling import (
    estimates_slopes,
    find_max_p,
    get_quantiles,
    plot_qq_scaling_eval,
    radial_f_collapse_single,
)
from depopulation.radial_f import gen_pop_ar, load_radial_f

agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg")
years = (1990, 2000, 2010, 2020)
cve_list = list(agg_all_df.CVE_MET.values)
cve_names = agg_all_df.NOM_MET.to_list()

radial_f = load_radial_f(cve_list, Path("outputs/radial_f/"), core=True)
N_c = gen_pop_ar(cve_list, Path("outputs/radial_f/"))

pgrid = np.linspace(0.01, 1, 100)
q_arr_c = get_quantiles(pgrid, cve_list, years, radial_f)

# Cities to color in plots
cve_bold = {
    "09.1.01": "tab:blue",
    "19.1.01": "tab:orange",
    "14.1.01": "tab:green",
    "02.2.03": "tab:red",
    "06.1.01": "tab:purple",
}

# Finding the maximum quantile at which density loss is observed
row_max = find_max_p(cve_list, cve_names, N_c, radial_f).iloc[0]
p_max = int(row_max.max_p * 100) / 100
idx_max = np.where(pgrid == p_max)[0][0]
print(f"Maximum p observed for {row_max.NOM_MET} at p={p_max} at pgrid index {idx_max}")

# Get scaling factors for all cities
slopes_df = estimates_slopes(cve_list, q_arr_c, N_c, i0=idx_max)
slopes_df.to_csv("outputs/scaling_factors.csv")

# Generate all qq-plots
qq_plots_dir = Path("figures/qq_plots/")
if not qq_plots_dir.exists():
    qq_plots_dir.mkdir()
plot_qq_scaling_eval(slopes_df, q_arr_c, qq_plots_dir)

# Generat all sigma plots
sigma_plots_dir = Path("figures/sigmas/")
if not sigma_plots_dir.exists():
    sigma_plots_dir.mkdir()
for i, cve in enumerate(cve_list):
    fig, ax = plt.subplots(figsize=(6.4, 4.8 / 3), layout="constrained")
    radial_f_collapse_single(
        ax,
        i,
        cve,
        radial_f,
        slopes_df,
        N_c,
        "tab:blue",
        func="sigma",
        legend=False,
    )
    plt.savefig(sigma_plots_dir / f"{cve}.pdf")
    plt.close()

# Generat all barsigma plots
barsigma_plots_dir = Path("figures/barsigmas/")
if not barsigma_plots_dir.exists():
    barsigma_plots_dir.mkdir()
for i, cve in enumerate(cve_list):
    fig, ax = plt.subplots(figsize=(6.4, 4.8 / 3), layout="constrained")
    radial_f_collapse_single(
        ax,
        i,
        cve,
        radial_f,
        slopes_df,
        N_c,
        "tab:blue",
        func="sigma_cum",
        legend=False,
    )
    plt.savefig(barsigma_plots_dir / f"{cve}.pdf")
    plt.close()


# Generat all rho plots
rho_plots_dir = Path("figures/rhos/")
if not rho_plots_dir.exists():
    rho_plots_dir.mkdir()
for i, cve in enumerate(cve_list):
    fig, ax = plt.subplots(figsize=(6.4, 4.8 / 3), layout="constrained")
    radial_f_collapse_single(
        ax, i, cve, radial_f, slopes_df, N_c, "tab:blue", legend=False
    )
    plt.savefig(rho_plots_dir / f"{cve}.pdf")
    plt.close()
