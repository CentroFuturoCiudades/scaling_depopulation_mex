"""Generates figure of transversal scaling exponent for q vs p,
and of the time evolution of the prefactors."""

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from depopulation.extensive_scaling import get_factors, scaling_analysis
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

axes[0].plot(pgrid, betas_pooled_p, ls="-", color="black")
axes[0].fill_between(
    pgrid,
    betas_pooled_p - cis_pooled_p,
    betas_pooled_p + cis_pooled_p,
    alpha=0.3,
    color="grey",
)
axes[0].axhline(0.42, ls="--", color="black")
axes[0].axvline(0.5, ls="--", color="black")

axes[0].set_xlabel("p")
axes[0].set_ylabel(r"$\beta_p$")

axA_inset = axes[0].inset_axes([0.13, 0.12, 0.5, 0.3])
axA_inset.plot(pgrid, r2_pooled_p, color="black")
axA_inset.axvline(0.5, color="black", ls="--")
axA_inset.set_xlabel("p")
axA_inset.set_ylabel(r"$R^2$")


cmap = mpl.colormaps["viridis"]
for i, p in enumerate(pgrid[:50]):
    y = (
        (np.exp(log_Y0_pooled_p[:, i]) - np.exp(log_Y0_pooled_p[0, i]))
        * 1e6 ** betas_pooled_p[i]
        / 1000
    )
    # if i == 49:
    #    axes[1].plot(years, y, color="tab:red", lw=3)
    # else:
    axes[1].plot(years, y, color=cmap(i / 50))
plt.colorbar(
    mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=0.5)),
    ax=axes[1],
    label="p",
)
axes[1].set_xlabel("year")
axes[1].set_xticks(years)
axes[1].set_ylabel(
    r"$\left<q_{0, p}(t)\right>_{1e6} - \left<q_{0, p}(1990)\right>_{1e6}$ km"
)

axB_inset = axes[1].inset_axes([0.15, 0.63, 0.35, 0.3])
axB_inset.plot(
    pgrid[:50],
    (np.exp(log_Y0_pooled_p[0, :]) * 1e6**betas_pooled_p / 1000)[:50],
    color="black",
)
axB_inset.set_xlabel("p")
axB_inset.set_ylabel(r"$\left<q_{0, p}(1990)\right>_{1e6}$ (km)")

plt.savefig("figures/scaling_q_beta_vs_p.pdf")
