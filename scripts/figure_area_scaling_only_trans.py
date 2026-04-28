"""Single figure with scaling analysis of urban area vs population.
Results are for complete urban area of metropolitan zones.
Uncentered, an exponent for each year.
Insets: transversal exponent time evolution.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from depopulation.extensive_scaling import (
    perform_trans_temp_scaling,
    scaling_plot_all_years,
)

# Get zone variables
years = (1990, 2000, 2010, 2020)
agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg").set_index("CVE_MET")
cve_list = list(agg_all_df.index.values)
N = agg_all_df[[f"POB_URB_{year}" for year in years]].to_numpy()
A = agg_all_df[[f"AREA_URB_{year}" for year in years]].to_numpy()

# Perform scaling analysis
(
    df_scaling,
    df_scaling_temp,
    df_scaling_temp_dec,
    beta_pooled,
    beta_pooled_R2,
    beta_pooled_ci,
    log_Y0_pooled,
    betas,
    betas_ci,
    log_Y0s,
    alphas,
    log_Y0is,
    alphas_dec,
    log_Y0is_dec,
    residuals_pooled,
    alphas_dec_ci,
    alphas_dec_R2,
    alphas_dec_p,
) = perform_trans_temp_scaling(N, A, years, cve_list)

# Some statistics
print(
    f"A_0(1990)/A_0(2020) = {np.exp(log_Y0_pooled)[0] / np.exp(log_Y0_pooled)[3]:0.2f}"
)
print(
    f"Y_0(t)(1e6)^β = {np.round(np.exp(log_Y0_pooled) * (1e6**beta_pooled)).astype(int)}"
)
print(f"{beta_pooled:0.2f} +- {beta_pooled_ci:0.2f} ({beta_pooled_R2:0.2f})")

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), layout="constrained")

# Main plot
scaling_plot_all_years(
    N,
    A,
    ax,
    slopes=betas,
    intercepts=log_Y0s,
    lines=False,
    slopes_t=None,
    intercepts_t=None,
    color_city=False,
)
ax.set_xlabel(r"$\ln P$")
ax.set_ylabel(r"$\ln A$")

# Inset A1
axA_inset = ax.inset_axes([0.65, 0.13, 0.3, 0.3])
axA_inset.errorbar(years, betas, betas_ci)
axA_inset.axhline(5 / 6, color="black", ls="--")
# axA_inset.axhline(beta_pooled, color="black", ls="-")
axA_inset.set_xlabel("year")
axA_inset.set_ylabel(r"$\gamma$")


plt.savefig("figures/scaling_area_trans_only.pdf")
