"""Generates figure for residuals of median distance scaling."""

from pathlib import Path

import geopandas as gpd
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

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), layout="constrained")

q_dict_corrected = {}
for i, p in enumerate(pgrid):
    # Extract multi-year data for p
    q_p = q_arr[:, :, i]

    # Perform scaling analysis
    # Tuple includes scaling_df, res, res_pooled, logY_pooled
    q_dict_corrected[p] = scaling_analysis(N * factors, q_p, pooled=True, index=years)

residuals = q_dict_corrected[pgrid[49]][2]
always_pos = np.all(residuals > 0, axis=1)
always_neg = np.all(residuals < 0, axis=1)
no_cross = np.logical_or(always_pos, always_neg)
oscilates = np.logical_and(abs(residuals).max(axis=1) < 0.1, ~no_cross)
ax.plot(years, residuals[no_cross].T, color="grey")
ax.plot(years, residuals[~no_cross].T, color="black")
ax.plot(years, residuals[oscilates].T, color="red")
ax.set_ylabel(r"deviations for $q_{0.50}$")
ax.set_xlabel("year")

plt.savefig("figures/scaling_q_residulas.pdf")
