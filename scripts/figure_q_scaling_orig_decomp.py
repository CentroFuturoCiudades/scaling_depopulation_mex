"""Generates figure comparing scaling of median distance uncentered and centered."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from depopulation.extensive_scaling import (
    get_factors,
    scaling_analysis,
    scaling_plot_all_years,
)
from depopulation.r_scaling import get_quantiles
from depopulation.radial_f import load_radial_f

years = (1990, 2000, 2010, 2020)
agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg").set_index("CVE_MET")
cve_list = list(agg_all_df.index.values)
N = agg_all_df[[f"POB_URB_{year}" for year in years]].to_numpy()
radial_f = load_radial_f(cve_list, Path("outputs/radial_f/"), core=False)

mg = gpd.read_file("data/mg_2020_ent.gpkg").to_crs(agg_all_df.crs)

# Quantiles for distance to center
pgrid = np.linspace(0.01, 1, 100)
q_arr = get_quantiles(pgrid, cve_list, years, radial_f)
factors = get_factors(0.84, cve_list, years, radial_f, agg_all_df, mg)

fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8 * 2), layout="constrained")

q_dict_corrected = {}
for i, p in enumerate(pgrid):
    # Extract multi-year data for p
    q_p = q_arr[:, :, i]

    # Perform scaling analysis
    # Tuple includes scaling_df, res, res_pooled, logY_pooled
    q_dict_corrected[p] = scaling_analysis(N * factors, q_p, pooled=True, index=years)

betas_p = np.zeros((4, len(pgrid)))
betas_pooled_p = np.zeros(len(pgrid))
log_Y0_p = np.zeros((4, len(pgrid)))
r2_p = np.zeros((4, len(pgrid)))
cis_p = np.zeros((4, len(pgrid)))
r2_pooled_p = np.zeros(len(pgrid))
cis_pooled_p = np.zeros(len(pgrid))
log_Y0_pooled_p = np.zeros((4, len(pgrid)))
for i, (p, (df_scaling, residuals, residuals_pooled, log_Y0_pooled)) in enumerate(
    q_dict_corrected.items()
):
    betas = df_scaling.beta.to_numpy()[:-1]
    beta_pooled = df_scaling.loc["pooled", "beta"]
    log_Y0s = df_scaling.log_Y0.to_numpy()[:-1]
    cis = df_scaling.beta_ci.to_numpy()[:-1]
    r2 = df_scaling.R2.to_numpy()[:-1]
    r2_pooled = df_scaling.loc["pooled", "R2"]
    cis_pooled = df_scaling.loc["pooled", "beta_ci"]

    betas_p[:, i] = betas
    log_Y0_p[:, i] = log_Y0s
    r2_p[:, i] = r2
    cis_p[:, i] = cis

    betas_pooled_p[i] = beta_pooled
    r2_pooled_p[i] = r2_pooled
    cis_pooled_p[i] = cis_pooled

    log_Y0_pooled_p[:, i] = log_Y0_pooled

df_scaling_temp, residuals_temp, _, _ = scaling_analysis(
    N * factors,
    (q_arr[:, :, 49] / np.exp(q_dict_corrected[pgrid[49]][3])),
    pooled=False,
    transposed=True,
    index=agg_all_df.index,
)
slopes_t = df_scaling_temp.beta.to_numpy()
intercepts_t = df_scaling_temp.log_Y0.to_numpy()


scaling_plot_all_years(
    N * factors,
    (q_arr[:, :, 49] / np.exp(q_dict_corrected[pgrid[49]][3])),
    axes[1],
    slopes=[betas_pooled_p[49]],
    intercepts=[0],
    lines=False,
    slopes_t=slopes_t,
    intercepts_t=intercepts_t,
)
axes[1].set_xlabel(r"$\ln(N)$")
axes[1].set_ylabel(r"$\ln(q_{0.50}) - \ln(q_{0.50, 0}(t))$")

axD_inset = axes[1].inset_axes([0.01, 0.74, 0.25, 0.25])
axD_inset.set_axis_off()
# axD_inset.set_facecolor((0.1, 0.2, 0.5, 0.3))
for s in slopes_t:
    theta = np.arctan(s)
    x = [-np.cos(theta), np.cos(theta)]
    y = [-np.sin(theta), np.sin(theta)]
    axD_inset.plot(x, y, color="grey", alpha=0.3)
theta = [betas_pooled_p[49]]
axD_inset.plot(
    [-np.cos(theta), np.cos(theta)], [-np.sin(theta), np.sin(theta)], color="black"
)

axD_inset_2 = axes[1].inset_axes([0.65, 0.13, 0.3, 0.3])
axD_inset_2.hist(slopes_t)
axD_inset_2.set_xlabel(r"$\alpha_{0.50,i}^{dec}$")
axD_inset_2.axvline(betas_pooled_p[49], color="black")

df_scaling_temp, residuals_temp, _, _ = scaling_analysis(
    N * factors, q_arr[:, :, 49], pooled=False, transposed=True, index=agg_all_df.index
)
slopes_t = df_scaling_temp.beta.to_numpy()
intercepts_t = df_scaling_temp.log_Y0.to_numpy()


scaling_plot_all_years(
    N * factors,
    q_arr[:, :, 49] / 1000,
    axes[0],
    slopes=betas_p[:, 49],
    intercepts=log_Y0_p[:, 49] - np.log(1000),
    lines=False,
    slopes_t=slopes_t,
    intercepts_t=intercepts_t - np.log(1000),
)
axes[0].set_xlabel(r"$\ln(N)$")
axes[0].set_ylabel(r"$\ln(q_{0.5})$ (km)")

axC_inset = axes[0].inset_axes([0.01, 0.74, 0.25, 0.25])
axC_inset.set_axis_off()
# axD_inset.set_facecolor((0.1, 0.2, 0.5, 0.3))
for s in slopes_t:
    theta = np.arctan(s)
    x = [-np.cos(theta), np.cos(theta)]
    y = [-np.sin(theta), np.sin(theta)]
    axC_inset.plot(x, y, color="grey", alpha=0.3)
theta = [betas_pooled_p[49]]
axC_inset.plot(
    [-np.cos(theta), np.cos(theta)], [-np.sin(theta), np.sin(theta)], color="black"
)

axC_inset_2 = axes[0].inset_axes([0.65, 0.13, 0.3, 0.3])
axC_inset_2.hist(slopes_t)
axC_inset_2.set_xlabel(r"$\alpha_{0.50,i}$")
axC_inset_2.axvline(betas_pooled_p[49], color="black")

plt.savefig("figures/scaling_q_orig_vs_decomposed.pdf")
