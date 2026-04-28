"""Composed figure with scaling analysis of urban area vs population.
Results are for complete urban area of metropolitan zones.
Panel A: Uncentered, Panel B: centered.
Insets: histogram of temporal exponents, original and decomposed,
transversal exponent time evolution, and prefactor time evolution.
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
print(f"A_0(1990)/A_0(2020) = {np.exp(log_Y0_pooled)[0]/np.exp(log_Y0_pooled)[3]:0.2f}")
print(
    f"Y_0(t)(1e6)^β = {np.round(np.exp(log_Y0_pooled)*(1e6**beta_pooled)).astype(int)}"
)
print(f"{beta_pooled:0.2f} +- {beta_pooled_ci:0.2f} ({beta_pooled_R2:0.2f})")

# Prepare figure
fig, axes = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8), layout="constrained")
for ax, label in zip(axes, ["A", "B", "C"]):
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(+0.5, -0.5),
        textcoords="offset fontsize",
        fontsize="medium",
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0),
    )

#### Panel A
# Main plot
scaling_plot_all_years(
    N,
    A,
    axes[0],
    slopes=betas,
    intercepts=log_Y0s,
    lines=False,
    slopes_t=alphas,
    intercepts_t=log_Y0is,
)
axes[0].set_xlabel(r"$\ln N$")
axes[0].set_ylabel(r"$\ln A$")

# Inset A1
axA_inset = axes[0].inset_axes([0.65, 0.13, 0.3, 0.3])
axA_inset.errorbar(years, betas, betas_ci)
axA_inset.axhline(5 / 6, color="black", ls="--")
axA_inset.axhline(beta_pooled, color="black", ls="-")
axA_inset.set_xlabel("year")
axA_inset.set_ylabel(r"$\beta_A(t)$")

# Inset A2
axA_inset_2 = axes[0].inset_axes([0.06, 0.65, 0.3, 0.3])
axA_inset_2.hist(alphas, bins=np.arange(0.14, 1.4, 0.1))
axA_inset_2.set_xlabel(r"$\alpha_i$")
axA_inset_2.axvline(beta_pooled, color="black")

### Panel B
scaling_plot_all_years(
    N,
    (A / np.exp(log_Y0_pooled)),
    axes[1],
    slopes=[beta_pooled],
    intercepts=[0],
    slopes_t=alphas_dec,
    intercepts_t=log_Y0is_dec,
)
axes[1].set_xlabel(r"$\ln N$")
axes[1].set_ylabel(r"$\ln A - \ln A_0 $")

# Inset B1
axB_inset = axes[1].inset_axes([0.65, 0.13, 0.3, 0.3])
axB_inset.plot(years, np.exp(log_Y0_pooled), "o-")
axB_inset.set_xlabel("year")
axB_inset.set_ylabel(r"$Y_0(t)$")

# Inset B2
axB_inset_2 = axes[1].inset_axes([0.06, 0.65, 0.3, 0.3])
axB_inset_2.hist(alphas_dec, bins=np.arange(0.24, 1.4, 0.1))
axB_inset_2.set_xlabel(r"$\alpha_i^{dec}$")
axB_inset_2.axvline(beta_pooled, color="black")
plt.savefig("figures/scaling_area.pdf")
