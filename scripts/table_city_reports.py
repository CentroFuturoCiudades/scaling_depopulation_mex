from pathlib import Path

import geopandas as gpd
import numpy as np

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

city_template = """
\\subsection{{{city_name}, {cve_met}}}


\\begin{{figure}}[H]
    \\centering
    % ==== LEFT COLUMN ====
    \\begin{{subfigure}}[t]{{0.45\\textwidth}}
        \\centering
        % Fig 1: Map
        \\includegraphics[valign=t, width=\\textwidth]{{FIGURES/maps/{cve_met}.pdf}}
        \\caption{{
        Population difference by grid cell (2020-1990). City centres are denoted as black dots
        }}
        \\vspace{{1em}}
        
        % Fig 2: QQ-Plot
        \\includegraphics[width=\\textwidth]{{FIGURES/qq_plots/{cve_met}.pdf}}
        \\caption{{
        Quantile-quantile plots for the radial population distributions $\\rho(s, t_i)$ and $\\rho(s, t_j)$(coloured curves). Urban expansion factors $\\Phi_{{ij}}$ from $t_i$ to $t_j$ are the estimated slopes (black lines).
        }}
    \\end{{subfigure}}
    \\hfill
    % ==== RIGHT COLUMN ====
    \\begin{{subfigure}}[t]{{0.45\\textwidth}}
        \\centering
        \\includegraphics[valign=t,width=\\textwidth]{{FIGURES/legend.pdf}}
        \\vspace{{1em}}

        \\includegraphics[width=\\textwidth]{{FIGURES/rhos/{cve_met}.pdf}}
        \\caption{{
        Radial population distribution $\\rho(r)$ at remoteness distance $r$ from the city centre.
        }}
        \\vspace{{1em}}
        
        \\includegraphics[width=\\textwidth]{{FIGURES/sigmas/{cve_met}.pdf}} % sigma file
        \\caption{{
        Radial population density $\\sigma(r)$ at remoteness distance $r$ from the city centre.
        }}
        \\vspace{{1em}}

        \\includegraphics[width=\\textwidth]{{FIGURES/barsigmas/{cve_met}.pdf}}
        \\caption{{
        Average population density $\\bar\\sigma(r)$ within disks of remoteness $r$ with the same centre as the city.
        }}
        \\vspace{{1em}}

        \\subfloat[Urban expansion factors and their inter quartile range from the Sein-Theil estimation.]{{
        \\begin{{tabular}}{{c|c|c|c}}
            \\hline
            Period ($t_i$-$t_j$) & $\\frac{{P(t_j)}}{{P(t_i)}}$ & $\\Phi_{{ij}}$ & IQR \\\\
            \\hline
            1990-2000 & {p_1990_2000: 0.2f} & {l_1990_2000: 0.2f} & ({l1_1990_2000: 0.2f}, {l2_1990_2000: 0.2f}) \\\\
            2000-2010 & {p_2000_2010: 0.2f} & {l_2000_2010: 0.2f} & ({l1_2000_2010: 0.2f}, {l2_2000_2010: 0.2f}) \\\\
            2010-2020 & {p_2010_2020: 0.2f} & {l_2010_2020: 0.2f} & ({l1_2010_2020: 0.2f}, {l2_2010_2020: 0.2f}) \\\\
            1990-2010 & {p_1990_2010: 0.2f} & {l_1990_2010: 0.2f} & ({l1_1990_2010: 0.2f}, {l2_1990_2010: 0.2f}) \\\\
            2000-2020 & {p_2000_2020: 0.2f} & {l_2000_2020: 0.2f} & ({l1_2000_2020: 0.2f}, {l2_2000_2020: 0.2f}) \\\\
            1990-2020 & {p_1990_2020: 0.2f} & {l_1990_2020: 0.2f} & ({l1_1990_2020: 0.2f}, {l2_1990_2020: 0.2f}) \\\\
            \\hline
        \\end{{tabular}}
    }}
    \\end{{subfigure}}
    \\caption{{Supplementary data for the metropolitan zone of {city_name} with code {cve_met}. Remoteness values are those of 2020.}}
\\end{{figure}}
"""

idx = 0
cve_name = cve_names[idx]
cve_met = cve_list[idx]
city_data = ""
for i, cve_met in enumerate(cve_list):
    cve_name = cve_names[i]
    if i > 0:
        city_data += "\n\\clearpage\n"
    city_data += city_template.format(
        city_name=cve_name,
        cve_met=cve_met,
        l_1990_2000=slopes_df.loc[cve_met, 0, 1].L,
        l_2000_2010=slopes_df.loc[cve_met, 1, 2].L,
        l_2010_2020=slopes_df.loc[cve_met, 2, 3].L,
        l_1990_2010=slopes_df.loc[cve_met, 0, 2].L,
        l_2000_2020=slopes_df.loc[cve_met, 1, 3].L,
        l_1990_2020=slopes_df.loc[cve_met, 0, 3].L,
        p_1990_2000=slopes_df.loc[cve_met, 0, 1].N2_N1,
        p_2000_2010=slopes_df.loc[cve_met, 1, 2].N2_N1,
        p_2010_2020=slopes_df.loc[cve_met, 2, 3].N2_N1,
        p_1990_2010=slopes_df.loc[cve_met, 0, 2].N2_N1,
        p_2000_2020=slopes_df.loc[cve_met, 1, 3].N2_N1,
        p_1990_2020=slopes_df.loc[cve_met, 0, 3].N2_N1,
        l1_1990_2000=slopes_df.loc[cve_met, 0, 1].q1,
        l1_2000_2010=slopes_df.loc[cve_met, 1, 2].q1,
        l1_2010_2020=slopes_df.loc[cve_met, 2, 3].q1,
        l1_1990_2010=slopes_df.loc[cve_met, 0, 2].q1,
        l1_2000_2020=slopes_df.loc[cve_met, 1, 3].q1,
        l1_1990_2020=slopes_df.loc[cve_met, 0, 3].q1,
        l2_1990_2000=slopes_df.loc[cve_met, 0, 1].q3,
        l2_2000_2010=slopes_df.loc[cve_met, 1, 2].q3,
        l2_2010_2020=slopes_df.loc[cve_met, 2, 3].q3,
        l2_1990_2010=slopes_df.loc[cve_met, 0, 2].q3,
        l2_2000_2020=slopes_df.loc[cve_met, 1, 3].q3,
        l2_1990_2020=slopes_df.loc[cve_met, 0, 3].q3,
    )

with open("sup_cities.tex", "w") as f:
    f.write(city_data)
