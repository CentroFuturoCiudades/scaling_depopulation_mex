from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from depopulation.r_scaling import (
    estimates_slopes,
    find_max_p,
    get_quantiles,
    period_colors,
    plot_L_vs_P,
)
from depopulation.radial_f import gen_pop_ar, load_radial_f

cve_bold = {
    "09.1.01": "tab:blue",
    "19.1.01": "tab:orange",
    "14.1.01": "tab:green",
    "02.2.03": "tab:red",
    "06.1.01": "tab:purple",
}

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

remove_outliers = False

slopes_df_filtered = slopes_df.query(
    "period.isin(['1990-2000', '2000-2010', '2010-2020'])"
)
if remove_outliers:
    slopes_df_filtered = slopes_df_filtered.drop(outliers)

fig, axs = plt.subplot_mosaic(
    [
        [".", "histx", "."],
        ["scatter_all", "scatter_zoom", "histy"],
    ],
    figsize=(12.6, 7),
    width_ratios=(4, 4, 1),
    height_ratios=(1, 4),
    layout="constrained",
)

# Shared plot limits
x1, x2 = 0.9, 1.8
y1, y2 = 0.9, 1.6

x1a, x2a = 0, 3.6
y1a, y2a = 0, 2.2

### SCATTER PLOT ###
plot_L_vs_P(
    axs["scatter_all"],
    slopes_df_filtered,
    cve_bold,
    x1a,
    x2a,
    y1a,
    y2a,
    False,
    False,
)
plot_L_vs_P(axs["scatter_zoom"], slopes_df_filtered, cve_bold, x1, x2, y1, y2, True)

### HISTOGRAM FOR L ###
sns.kdeplot(
    slopes_df_filtered,
    y="L",
    hue="period",
    # element="step",
    # fill=True,
    # kde=True,
    # binwidth=0.05,
    # binrange=(0.90, 2.2),
    ax=axs["histy"],
    legend=False,
    palette=period_colors.values(),
)
axs["histy"].set_ylabel("")
axs["histy"].set_xlabel("")
axs["histy"].set_yticks([], minor=True)
axs["histy"].set_yticks([])
axs["histy"].set_ylim(y1, y2)
axs["histy"].tick_params(axis="x", labelsize=12)

### HISTOGRAM FOR Pj/Pi ###
sns.kdeplot(
    slopes_df_filtered,
    x="N2_N1",
    hue="period",
    # element="step",
    # fill=True,
    # kde=True,
    # binwidth=0.05,
    # binrange=(0.90, 2.2),
    ax=axs["histx"],
    legend=False,
    palette=period_colors.values(),
)
axs["histx"].set_ylabel("")
axs["histx"].set_xlabel("")
axs["histx"].set_xticks([], minor=True)
axs["histx"].set_xticks([])
axs["histx"].set_xlim(x1, x2)
axs["histx"].tick_params(axis="y", labelsize=12)

legend_points = fig.legend(
    loc="lower left",
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(0.05, 0.8),
    title="Periods",
    title_fontproperties={"size": 12, "weight": "bold"},
    alignment="left",
)
fig.add_artist(legend_points)
legend_cities = fig.legend(
    handles=[
        mpatches.Patch(color="tab:blue", label="Mexico City"),
        mpatches.Patch(color="tab:orange", label="Monterrey"),
        mpatches.Patch(color="tab:green", label="Guadalajara"),
        mpatches.Patch(color="tab:red", label="Mexicalli"),
        mpatches.Patch(color="tab:purple", label="Colima"),
    ],
    loc="lower left",
    ncol=2,
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(0.18, 0.8),
    title="Cities",
    title_fontproperties={"size": 12, "weight": "bold"},
    alignment="left",
)


plt.savefig("figures/phase_space_double.pdf")

slopes_df_filtered = (
    slopes_df.assign(dt=lambda df: df.t2 - df.t1)
    .query("dt == 10")
    .merge(agg_all_df[["CVE_MET", "NOM_MET"]], on="CVE_MET")
)

total_points = slopes_df_filtered.shape[0]
points_in_A = slopes_df_filtered[
    (slopes_df_filtered.L >= 1) & (slopes_df_filtered.N2_N1 <= 1)
].shape[0]

points_in_B = slopes_df_filtered[
    (slopes_df_filtered.L >= np.sqrt(slopes_df_filtered.N2_N1))
    & (slopes_df_filtered.N2_N1 > 1)
].shape[0]

points_in_C = slopes_df_filtered[
    (slopes_df_filtered.L < np.sqrt(slopes_df_filtered.N2_N1))
    & (slopes_df_filtered.N2_N1 > 1)
    & (slopes_df_filtered.L >= 1)
].shape[0]

points_in_D = slopes_df_filtered[
    (slopes_df_filtered.N2_N1 > 1) & (slopes_df_filtered.L < 1)
].shape[0]

points_in_E = 0

points_in_F = 0

print(f"Total points: {total_points}")
print(f"Points in A: {points_in_A}")
print(f"Points in B: {points_in_B}")
print(f"Points in C: {points_in_C}")
print(f"Points in D: {points_in_D}")
print(f"Points in E: {points_in_E}")
print(f"Points in F: {points_in_F}")


pdict = {
    "A": round(3 / 207, 3),
    "B": round(189 / 207, 3),
    "C": round(13 / 207, 3),
    "D": round(2 / 207, 3),
    "E": 0,
    "F": 0,
}
print("Fractions:")
print(pdict)
