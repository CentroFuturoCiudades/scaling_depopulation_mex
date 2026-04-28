from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from depopulation.r_scaling import estimates_slopes, find_max_p, get_quantiles
from depopulation.radial_f import gen_pop_ar, load_radial_f

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
# Create conterfactual radial functions
for i, cve in enumerate(cve_list):
    r_ring = radial_f[cve]["r_ring"]
    sigma_1990 = radial_f[cve]["sigma_1990"]
    for j, year in enumerate(years):
        sigma = radial_f[cve][f"sigma_{year}"]
        L = np.sqrt(N_c[i, j] / N_c[i, 0])
        # Position at which to evaluate sigma
        r_scaled = r_ring / L
        # We must interpolate it to evaluate
        sigma_cfactual = np.interp(r_scaled, r_ring, sigma_1990, right=0)
        ring_pop_cfactual = (
            sigma_cfactual * np.pi * ((r_ring + 50) ** 2 - (r_ring - 50) ** 2)
        )
        ring_pop_cfactual = ring_pop_cfactual * N_c[i, j] / ring_pop_cfactual.sum()
        cumpop_cfactual = np.zeros(len(ring_pop_cfactual) + 1)
        cumpop_cfactual[1:] = ring_pop_cfactual.cumsum()

        radial_f[cve][f"sigma_cfactual_{year}"] = sigma_cfactual
        radial_f[cve][f"ring_pop_cfactual_{year}"] = ring_pop_cfactual
        radial_f[cve][f"cumpop_cfactual_{year}"] = cumpop_cfactual


# Find population at each remoteness bracket for both observed and cfactual
beta = 0.5
scale = True
rgrid = np.linspace(0, 20, 1000)
series_list = []
for i, cve in enumerate(cve_list):
    r_ring = radial_f[cve]["r_ring"]

    pop = {y: radial_f[cve][f"ring_pop_{y}"] for y in years}

    # Scaling factos
    factor = {y: N_c[i, j] ** beta for j, y in enumerate(years)}

    # Scaled distances
    r_ring_s = {y: r_ring / factor[y] for y in years}

    pop_inner = {y: pop[y][r_ring_s[2020] < 3].sum() for y in years}
    pop_mid = {
        y: pop[y][np.logical_and(r_ring_s[2020] >= 3, r_ring_s[2020] < 5)].sum()
        for y in years
    }
    pop_distant = {
        y: pop[y][np.logical_and(r_ring_s[2020] >= 5, r_ring_s[2020] < 9.3)].sum()
        for y in years
    }
    pop_outmost = {y: pop[y][r_ring_s[2020] >= 9.3].sum() for y in years}
    cve_series = (
        pd.DataFrame(
            {
                "N_inner": pop_inner,
                "N_mid": pop_mid,
                "N_distant": pop_distant,
                "N_outmost": pop_outmost,
            }
        )
        .T.stack()
        .rename(cve)
    )
    series_list.append(cve_series)
rem_brackets_original = pd.DataFrame(series_list)


series_list = []
for i, cve in enumerate(cve_list):
    r_ring = radial_f[cve]["r_ring"]

    pop = {y: radial_f[cve][f"ring_pop_cfactual_{y}"] for y in years}

    # Scaling factos
    factor = {y: N_c[i, j] ** beta for j, y in enumerate(years)}

    # Scaled distances
    r_ring_s = {y: r_ring / factor[y] for y in years}

    pop_inner = {y: pop[y][r_ring_s[2020] < 3].sum() for y in years}
    pop_mid = {
        y: pop[y][np.logical_and(r_ring_s[2020] >= 3, r_ring_s[2020] < 5)].sum()
        for y in years
    }
    pop_distant = {
        y: pop[y][np.logical_and(r_ring_s[2020] >= 5, r_ring_s[2020] < 9.3)].sum()
        for y in years
    }
    pop_outmost = {y: pop[y][r_ring_s[2020] >= 9.3].sum() for y in years}
    cve_series = (
        pd.DataFrame(
            {
                "N_inner": pop_inner,
                "N_mid": pop_mid,
                "N_distant": pop_distant,
                "N_outmost": pop_outmost,
            }
        )
        .T.stack()
        .rename(cve)
    )
    series_list.append(cve_series)
rem_brackets_cfactual = pd.DataFrame(series_list)

# Get the differences between observed and cfactual
rem_brackets_dif = rem_brackets_original - rem_brackets_cfactual

# Get aggregated differences
rem_brackets_dif_agg = (
    (rem_brackets_dif.sum() / 1e6)
    .rename("N")
    .reset_index()
    .rename(columns={"level_0": "bracket", "level_1": "year"})
    .set_index(["bracket", "year"])
).reset_index()

pd.options.display.float_format = "{:,.2f}".format
print("Population difference per bracket:")
print(rem_brackets_dif_agg)

print()

print(f"The mean expansion factor is: {slopes_df.loc[:, 0, 3].N2_N1.mean():0.2f}")
dist_excess = (2.16) ** (0.6 - 0.5) * np.exp(0.0057 * 30)
print(f"With estiamted beta and alpha, distances are {dist_excess:0.2f} times larger.")

# Create Figure
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8), layout="constrained")

sns.barplot(
    data=(rem_brackets_cfactual.sum() / 1e6).reset_index(),
    x="level_0",
    y=0,
    hue="level_1",
    ax=ax[0],
)
ax[0].set_xlabel("")
ax[0].set_ylabel("Counterfactual population\n (millions)")
ax[0].set_xticks([])

sns.barplot(
    rem_brackets_dif_agg,
    x="bracket",
    y="N",
    hue="year",
    ax=ax[1],
)
ax[1].set_xlabel("")
ax[1].set_ylabel("Population difference \n (millions)")
ax[1].set_xticklabels([r"$r < 3$", r"  3 < $r < 5$", r"$ 5 < r < 9.3$", r"$r > 9.3$"])
ax[1].get_legend().remove()
sns.move_legend(
    ax[0],
    "lower center",
    bbox_to_anchor=(0.5, 1),
    ncol=4,
    title=None,
    frameon=False,
)
plt.savefig("figures/cfactual.pdf")
