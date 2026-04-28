from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.anova as anova

from depopulation.r_scaling import (
    estimates_slopes,
    find_max_p,
    fit_models,
    get_quantiles,
    plot_L_P_models,
    plot_L_P_outliers,
)
from depopulation.radial_f import gen_pop_ar, load_radial_f

outliers = [
    "03.2.02",  # Abnormally large growth factor, Los Cabos
    "14.1.02",  # Abnormally large growth factor, Puerto Vallarta
    "23.1.01",  # Abnormally large growth factor, Cancun
    "29.1.01",  # Center does not conform to scaling, Tlaxcala-Apizaco
]

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

models_df, (model_no_int, model_dt) = fit_models(slopes_df, outliers)

model = models_df.loc["shared_slope_DT_intercept"]
print(f"The complete model has and R2 value of {model.r2:0.2f} and coefficiens:")
print(f"beta = {model.beta:0.2f} ({model.beta_low:0.2f}, {model.beta_high:0.2f})")
print(
    f"alpha = {model.itcp_Dt:0.4f} ({model.itcp_Dt_low:0.4f}, {model.itcp_Dt_high:0.4f})"
)

anova_table = anova.anova_lm(model_no_int, model_dt)
print(
    f"Alternative model is rejected with an F-statistic of {anova_table.loc[1, 'F']:0.2f} and a p-value of {anova_table.loc[1, 'Pr(>F)']:0.2f}"
)

fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(7, 7))

plot_L_P_models(ax, slopes_df, models_df)

x1, x2 = ax.get_xlim()
y1, y2 = ax.get_ylim()

###### INSET ######

slopes_df_filtered = slopes_df.assign(dt=lambda df: df.t2 - df.t1)  # .drop(outliers)

markers = {10: "o", 20: "^", 30: "s"}

ax = ax.inset_axes([0.69, 0.08, 0.3, 0.3])
plot_L_P_outliers(ax, slopes_df, models_df, x1, x2, y1, y2)

plt.savefig("figures/models.pdf")
