"""Creates a map that displays deviations from area scaling."""

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from depopulation.extensive_scaling import perform_trans_temp_scaling

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

gdf = agg_all_df.assign(
    deviation=residuals_pooled[:, 3],
    circle=lambda df: df.geometry.buffer((np.abs(df.deviation) + 0.1) * 1e5),
    color=lambda df: ["red" if dev > 0 else "blue" for dev in df.deviation],
).set_geometry("circle")
ax = gdf.plot(alpha=0.6, edgecolor="black", color=gdf.color, figsize=(10, 10))
cx.add_basemap(ax, crs="ESRI:102008")
plt.savefig("figures/scaling_area_deviatons_map.pdf", bbox_inches="tight")
