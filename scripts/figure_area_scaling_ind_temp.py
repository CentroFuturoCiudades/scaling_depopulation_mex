"""Individual urba area temporal scaling plots for each zone.
Enables evaluation of linearity and comparison with transversal trend
Useful for debugging puposes in geometry alignment"""

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

fig, axes = plt.subplots(9, 8, figsize=(6.4 * 8, 4.8 * 9), layout="constrained")
axes = axes.ravel()
for i, cve in enumerate(cve_list):
    scaling_plot_all_years(
        N[i : i + 1],
        (A / np.exp(log_Y0_pooled))[i : i + 1],
        axes[i],
        slopes=[beta_pooled],
        intercepts=[0],
        slopes_t=alphas_dec[i : i + 1],
        intercepts_t=log_Y0is_dec[i : i + 1],
    )
    axes[i].set_title(
        (
            f"{cve} {agg_all_df.NOM_MET.loc[cve]}, R2:{alphas_dec_R2[i]:.02f}, p:"
            f"{alphas_dec_p[i]:.02f}, a-b:{alphas_dec[i]- beta_pooled:.02f}"
        )
    )
plt.savefig("figures/scaling_area_temp_ind.pdf")
