"""Composed figure showing the  stability of βq and effect of the
correction to coastal/border cities to scaling"""

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

cdict = {1990: "tab:blue", 2000: "tab:orange", 2010: "tab:green", 2020: "tab:red"}

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

coastal = [
    "02.1.01",
    "02.2.02",
    "03.2.01",
    "03.2.02",
    "04.2.01",
    "12.2.02",
    "14.1.02",
    "23.1.01",
    "23.2.02",
    "25.2.02",
    "25.2.03",
    "26.1.01",
    "28.1.02",
    "30.1.01",
    "30.1.06",
]
border = [
    "02.1.01",
    "02.2.03",
    "05.1.03",
    "08.2.03",
    "26.2.04",
    "28.1.01",
    "28.2.04",
    "28.2.05",
]
cb_zones = sorted(list(set(coastal).union(set(border))))

# Index of inland and border/coastal cities
idx_cb = [cve_list.index(c) for c in cb_zones]
idx_land = sorted(list(set(range(len(cve_list))) - set(idx_cb)))

q_dict = {}
for i, p in enumerate(pgrid):
    # Extract multi-year data for p
    q_p = q_arr[:, :, i]

    # Perform scaling analysis
    # Tuple includes scaling_df, res, res_pooled, logY_pooled
    q_dict[p] = scaling_analysis(N, q_p, pooled=True, index=years)

q_dict_corrected = {}
for i, p in enumerate(pgrid):
    # Extract multi-year data for p
    q_p = q_arr[:, :, i]

    # Perform scaling analysis
    # Tuple includes scaling_df, res, res_pooled, logY_pooled
    q_dict_corrected[p] = scaling_analysis(N * factors, q_p, pooled=True, index=years)

betas_p = np.zeros((4, len(pgrid)))
betas_pooled_p = np.zeros(len(pgrid))
r2_p = np.zeros((4, len(pgrid)))
cis_p = np.zeros((4, len(pgrid)))
r2_pooled_p = np.zeros(len(pgrid))
cis_pooled_p = np.zeros(len(pgrid))
for i, (p, (df_scaling, residuals, residuals_pooled, log_Y0_pooled)) in enumerate(
    q_dict_corrected.items()
):
    betas = df_scaling.beta.to_numpy()[:-1]
    beta_pooled = df_scaling.loc["pooled", "beta"]
    cis = df_scaling.beta_ci.to_numpy()[:-1]
    r2 = df_scaling.R2.to_numpy()[:-1]
    r2_pooled = df_scaling.loc["pooled", "R2"]
    cis_pooled = df_scaling.loc["pooled", "beta_ci"]

    betas_p[:, i] = betas
    r2_p[:, i] = r2
    cis_p[:, i] = cis

    betas_pooled_p[i] = beta_pooled
    r2_pooled_p[i] = r2_pooled
    cis_pooled_p[i] = cis_pooled

betas_p_raw = np.zeros((4, len(pgrid)))
betas_pooled_p_raw = np.zeros(len(pgrid))
r2_p_raw = np.zeros((4, len(pgrid)))
cis_p_raw = np.zeros((4, len(pgrid)))
r2_pooled_p_raw = np.zeros(len(pgrid))
cis_pooled_p_raw = np.zeros(len(pgrid))
for i, (p, (df_scaling, residuals, residuals_pooled, log_Y0_pooled)) in enumerate(
    q_dict.items()
):
    betas = df_scaling.beta.to_numpy()[:-1]
    beta_pooled = df_scaling.loc["pooled", "beta"]
    cis = df_scaling.beta_ci.to_numpy()[:-1]
    r2 = df_scaling.R2.to_numpy()[:-1]
    r2_pooled = df_scaling.loc["pooled", "R2"]
    cis_pooled = df_scaling.loc["pooled", "beta_ci"]

    betas_p_raw[:, i] = betas
    r2_p_raw[:, i] = r2
    cis_p_raw[:, i] = cis

    betas_pooled_p_raw[i] = beta_pooled
    r2_pooled_p_raw[i] = r2_pooled
    cis_pooled_p_raw[i] = cis_pooled

fig, axes = plt.subplots(3, 2, figsize=(6.4 * 2, 4.8 * 3), layout="constrained")

for i, year in enumerate(years):
    axes[0, 0].plot(pgrid, betas_p[i], ls="-", label=year, color=cdict[year])
    axes[0, 0].fill_between(
        pgrid,
        betas_p[i] - cis_p[i],
        betas_p[i] + cis_p[i],
        alpha=0.3,
        color=cdict[year],
    )
axes[0, 0].set_xlabel("p")
axes[0, 0].set_ylabel(r"$\beta_p$")
axes[0, 0].legend(loc="lower left")

axes[0, 1].plot(pgrid, betas_pooled_p, ls="-", label="corrected")
axes[0, 1].fill_between(
    pgrid, betas_pooled_p - cis_pooled_p, betas_pooled_p + cis_pooled_p, alpha=0.3
)

axes[0, 1].plot(pgrid, betas_pooled_p_raw, ls="-", label="original")
axes[0, 1].fill_between(
    pgrid,
    betas_pooled_p_raw - cis_pooled_p_raw,
    betas_pooled_p_raw + cis_pooled_p_raw,
    alpha=0.3,
)

axes[0, 1].set_xlabel("p")
axes[0, 1].set_ylabel(r"$\beta_p$")
axes[0, 1].legend()

axB_inset = axes[0, 1].inset_axes([0.13, 0.12, 0.5, 0.3])
axB_inset.plot(pgrid, r2_pooled_p)
axB_inset.plot(pgrid, r2_pooled_p_raw)
axB_inset.set_xlabel("p")
axB_inset.set_ylabel(r"$R^2$")

scaling_plot_all_years(
    N * factors,
    (q_arr[:, :, 24] / np.exp(q_dict_corrected[pgrid[24]][3])),
    axes[1, 0],
    slopes=[betas_pooled_p[24]],
    intercepts=[0],
    lines=True,
    mark_idx=idx_cb,
)
axes[1, 0].set_xlabel(r"$N_{cor}$")
axes[1, 0].set_ylabel(r"$q_{0.25}$")

scaling_plot_all_years(
    N,
    (q_arr[:, :, 24] / np.exp(q_dict[pgrid[24]][3])),
    axes[1, 1],
    slopes=[betas_pooled_p_raw[24]],
    intercepts=[0],
    lines=True,
    mark_idx=idx_cb,
)
axes[1, 1].set_xlabel(r"$N$")
axes[1, 1].set_ylabel(r"$q_{0.25}$")

scaling_plot_all_years(
    N * factors,
    (q_arr[:, :, 49] / np.exp(q_dict_corrected[pgrid[49]][3])),
    axes[2, 0],
    slopes=[betas_pooled_p[49]],
    intercepts=[0],
    lines=True,
    mark_idx=idx_cb,
)
axes[2, 0].set_xlabel(r"$N_{cor}$")
axes[2, 0].set_ylabel(r"$q_{0.5}$")

scaling_plot_all_years(
    N,
    (q_arr[:, :, 49] / np.exp(q_dict[pgrid[49]][3])),
    axes[2, 1],
    slopes=[betas_pooled_p_raw[49]],
    intercepts=[0],
    lines=True,
    mark_idx=idx_cb,
)
axes[2, 1].set_xlabel(r"$N$")
axes[2, 1].set_ylabel(r"$q_{0.5}$")
plt.savefig("figures/scaling_q_stability_compared_factors.pdf")
