import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from depopulation.radial_f import (
    agg_rem_brackets,
    gen_rem_brackets_pop_any_beta,
)

with open("data/cve_code_names.json", "r") as f:
    cve_dict = json.load(f)
CVES = list(cve_dict.keys())
NAMES = list(cve_dict.values())
OUT_DIR = "outputs"
RFUNCS_DIR = OUT_DIR + "/radial_f"

agg_df_list = []
for beta_str, beta in zip(["1/3", "5/12", "1/2"], [1 / 3, 5 / 12, 1 / 2]):
    gen_rem_brackets_pop_any_beta(CVES, Path(RFUNCS_DIR), Path(OUT_DIR), beta=beta)
    df = agg_rem_brackets(
        Path(f"outputs/remoteness_brackets_pop_beta_{beta:0.2f}.csv"), Path(OUT_DIR)
    ).assign(year=lambda df: df.year.astype(int))
    agg_df_list.append(df.assign(beta=beta_str))
    print(f"Population change from 1990->2020 for B={beta_str}")
    print(
        np.round(
            -df.query("year==1990").population.values
            + df.query("year==2020").population.values,
            2,
        )
    )
    print(f"Fraction change from 1990->2020 for B={beta_str}")
    print(
        np.round(
            df.query("year==1990").p_fraction.values
            - df.query("year==2020").p_fraction.values,
            2,
        )
    )
    print()

agg_df = pd.concat(agg_df_list).set_index(["beta", "year", "bracket"])


with plt.rc_context(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titleweight": "bold",
    }
):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4))
    ax = axes[0]
    sns.barplot(
        data=agg_df.loc["1/2"],
        x="bracket",
        y="population",
        hue="year",
        ax=ax,
        legend=True,
    )

    sns.barplot(
        data=agg_df.loc["5/12"],
        x="bracket",
        y="population",
        hue="year",
        ax=ax,
        legend=False,
        palette=["chocolate"] * 4,
        fill=False,
    )

    sns.barplot(
        data=agg_df.loc["1/3"],
        x="bracket",
        y="population",
        hue="year",
        ax=ax,
        legend=False,
        palette=["blue"] * 4,
        fill=False,
    )

    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(
        edgecolor="chocolate", label=r"$\beta=5/12$", facecolor="none"
    )
    patch2 = mpatches.Patch(edgecolor="blue", label=r"$\beta=1/3$", facecolor="none")
    handles.append(patch)
    labels.append(r"$\beta=5/12$")
    handles.append(patch2)
    labels.append(r"$\beta=1/3$")
    ax.legend(handles=handles, labels=labels, loc="upper right")
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=False,
        fontsize=12,
    )

    ax.set_ylabel("Total population (millions)", fontsize=12)
    ax.set_xlabel("")
    ax.set_xticks([])

    ax = axes[1]
    sns.barplot(
        data=agg_df.loc["1/2"],
        x="bracket",
        y="p_fraction",
        hue="year",
        ax=ax,
        legend=False,
    )

    sns.barplot(
        data=agg_df.loc["5/12"],
        x="bracket",
        y="p_fraction",
        hue="year",
        ax=ax,
        legend=False,
        palette=["chocolate"] * 4,
        fill=False,
        alpha=1,
    )
    sns.barplot(
        data=agg_df.loc["1/3"],
        x="bracket",
        y="p_fraction",
        hue="year",
        ax=ax,
        legend=False,
        palette=["blue"] * 4,
        fill=False,
        alpha=1,
    )

    ax.set_ylabel("Fraction of population", fontsize=12)
    ax.set_xlabel("")
    ax.set_xticks(
        [0, 1, 2, 3],
        [r"$r < 3$", r"$3 < r < 5$", r"$5 < r < 9.3$", r"$r > 9.3$"],
        fontsize=12,
    )
plt.savefig("figures/bars_compare_exponents.pdf")
