"""Functions to perform radial scaling analysis for distance to centre variables."""

from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import bootstrap
from sklearn.metrics import r2_score

from .utils import get_adj_idx

period_colors = {
    "1990-2000": "#E6AB04",
    "2000-2010": "#B85A0D",
    "2010-2020": "#878372",
    10: "#E6AB04",
    20: "#B85A0D",
    30: "#878372",
}

colors_regions = [
    "#f6e8c3",
    "#d8b365",
    "#8c510a",
    "#01665e",
    "#5ab4ac",
    "#c7eae5",
]

markers_dt = {10: "o", 20: "^", 30: "s"}

outliers = [
    "03.2.02",  # Abnormally large growth factor, Los Cabos
    "14.1.02",  # Abnormally large growth factor, Puerto Vallarta
    "23.1.01",  # Abnormally large growth factor, Cancun
    "29.1.01",  # Center does not conform to scaling, Tlaxcala-Apizaco
]


def get_quantiles(pgrid, cve_list, years, radial_f, adjust_core=False):
    """Find the quantiles for all zones in cve_list from the radial
    distribution function.

    Parameters
    ----------
    pgrid : np.Array
        Vector of quantiles to calculate, sorted list between 0-1.
    cve_list : List
        List of zones to get the quantiles from. Entries correspond to
        keys in <radial_f>.
    years : List or Tuple
        List of years at which to get the quantiles.
    radial_f : Dict
        Dictionary of dictionaries with radial density functions for each zone.
    adjust_core : bool, optional
        Of True, truncates radial distribution to first zero occurrence,
        by default False.

    Returns
    -------
    np.Array
        Array with quantiles, of shape (# of zones, # of years, # of quantiles).
    """
    n_c = len(cve_list)
    n_y = len(years)
    n_p = len(pgrid)

    q_arr = np.zeros((n_c, n_y, n_p))

    for i, cve in enumerate(cve_list):
        r = radial_f[cve]["r_ring"]
        for j, year in enumerate(years):
            rho = radial_f[cve][f"rho_{year}"]

            idx = len(r)
            if adjust_core:
                idx = get_adj_idx(rho)

            q_arr[i, j] = np.quantile(
                r[:idx], pgrid, weights=rho[:idx], method="inverted_cdf"
            )
    return q_arr


def find_max_p(cve_list, cve_names, N_c, radial_f):
    p_list = []
    r_list = []
    for i, cve in enumerate(cve_list):
        r_ring = radial_f[cve]["r_ring"]
        sigma_1990 = radial_f[cve]["sigma_1990"]
        sigma_2020 = radial_f[cve]["sigma_2020"]
        cdf_2020 = radial_f[cve]["cdf_2020"]

        delta_sigma = sigma_2020 - sigma_1990
        idx_max = np.where(delta_sigma > 0)[0]
        assert len(idx_max) > 0
        idx_max = idx_max[0]
        assert idx_max > 0
        p = cdf_2020[idx_max + 1]
        p_list.append(p)
        r_list.append(r_ring[idx_max])
    df_max_p = pd.DataFrame(
        {
            "CVE": cve_list,
            "NOM_MET": cve_names,
            "max_p": p_list,
            "max_r": r_list,
            "N_2020": N_c[:, 3],
            "max_rem": np.array(r_list) / np.sqrt(N_c[:, 3]),
        }
    )
    df_max_p = df_max_p.sort_values("max_p", ascending=False)

    return df_max_p


def estimates_slopes(cve_list, q_arr, N, years=(1990, 2000, 2010, 2020), i0=37):
    """Estimate median slope without intercept for q-q plots.
    This is the Theil-Sen estimator without intercept."""

    # Slope output dictionary
    slopes_out = {}

    for i, cve in enumerate(cve_list):
        slopes_out[cve] = {}

        for (j1, y1), (j2, y2) in combinations(enumerate(years), 2):
            # Consider 10 year intervals or whole interval
            # if not ((j2 - j1 == 1) or (j2 - j1 == 3)):
            #    continue

            # Get quantiles for current interval
            x = q_arr[i, j1, :]
            y = q_arr[i, j2, :]

            # Find all slopes corresponding from the origin
            slopes = y / x

            # Estimate median slopes up to specified index
            sq1, sq2, sq3 = np.quantile(slopes[: i0 + 1], [0.25, 0.50, 0.75])
            smin, smax = (slopes[: i0 + 1].min(), slopes[: i0 + 1].max())

            # Find R2 values
            y_true = y[: i0 + 1]
            y_pred = sq2 * x[: i0 + 1]
            u = ((y_true - y_pred) ** 2).sum()
            v = ((y_true - y_true.mean()) ** 2).sum()
            r2 = 1 - u / v

            slopes_dict = {
                "q1": sq1,
                "L": sq2,
                "q3": sq3,
                "min": smin,
                "max": smax,
                "idx_max": i0,
                "slopes": slopes,
                "N2_N1": N[i, j2] / N[i, j1],
                "t1": y1,
                "t2": y2,
                "R2": r2,
            }
            slopes_out[cve][(j1, j2)] = slopes_dict

    slopes_df = pd.concat(
        [
            pd.concat({k: pd.DataFrame(v).T}, names=["CVE_MET"])
            for k, v in slopes_out.items()
        ]
    )

    slopes_df = slopes_df.infer_objects()

    slopes_df["period"] = slopes_df.t1.astype(str) + "-" + slopes_df.t2.astype(str)

    return slopes_df


def plot_qq_scaling_eval(slopes_df, q_arr, outdir, years=(1990, 2000, 2010, 2020)):
    """Create a series of plots to evaluate the fit."""
    cve_list = slopes_df.index.get_level_values(0).unique()

    colors = {
        (0, 1): "tab:blue",
        (0, 2): "tab:orange",
        (0, 3): "tab:green",
        (1, 2): "tab:red",
        (1, 3): "tab:purple",
        (2, 3): "tab:brown",
    }

    imax = 90
    for i, cve in enumerate(cve_list):
        fig, ax = plt.subplots()
        df = slopes_df.loc[cve]
        for (j1, j2), row in df.iterrows():
            if not (j2 - j1 == 1):
                continue
            # Get quantiles for current interval
            x = q_arr[i, j1, :]
            y = q_arr[i, j2, :]

            ax.plot(
                x[:imax] / 1000,
                y[:imax] / 1000,
                "-",
                label=rf"{years[j1]}-{years[j2]}",
                color=colors[(j1, j2)],
                lw=3,
            )

            ax.plot(
                np.array([0, x[imax - 1]]) / 1000,
                np.array([0, row.L * x[imax - 1]]) / 1000,
                color="black",
                zorder=2.5,
            )

        ax.legend()
        ax.set_xlabel("quantiles at $t_i$")
        ax.set_ylabel("quantiles at $t_j$")

        plt.savefig(outdir / f"{cve}.pdf")
        plt.close()

    return


def radial_f_collapse_single(
    ax,
    i,
    cve,
    radial_f,
    slopes_df,
    N,
    color,
    beta=0.5,
    func="rho",
    remoteness=True,
    legend=True,
):
    years = [1990, 2000, 2010, 2020]

    r_ring = radial_f[cve]["r_ring"]
    r_disk = radial_f[cve]["r_disk"][1:]

    # The scaling factors
    L = {
        y: l
        for y, l in zip(
            [2020, 2010, 2000, 1990],
            np.cumprod(
                [
                    1,
                    slopes_df.loc[cve, 2, 3].L,
                    slopes_df.loc[cve, 1, 2].L,
                    slopes_df.loc[cve, 0, 1].L,
                ]
            ),
        )
    }

    # Radial functions
    rho = {y: radial_f[cve][f"rho_{y}"] for y in years}
    sigma = {y: radial_f[cve][f"sigma_{y}"] * 1e6 for y in years}
    sigma_cum = {
        y: radial_f[cve][f"cumpop_{y}"][1:] / (np.pi * r_disk**2) * 1e6 for y in years
    }
    # Find the city extents, max rad
    cdf = {y: radial_f[cve][f"cdf_{y}"] for y in years}
    max_idx = {y: np.searchsorted(cdf[y], 1.0) for y in years}
    xlim = 0
    if remoteness:
        rem_s = N[i, 3] ** beta
    else:
        rem_s = 1000

    for j, y in enumerate(years):
        idx = max_idx[y]
        if func == "rho":
            # scaled
            x1 = r_ring[: idx + 1] * L[y] / rem_s
            y1 = rho[y][: idx + 1] / L[y] * rem_s
            # original
            x2 = r_ring[: idx + 1] / rem_s
            y2 = rho[y][: idx + 1] * rem_s
        elif func == "sigma":
            x1 = r_ring[: idx + 1] * L[y] / rem_s
            y1 = (N[i, 3] / N[i, j]) * sigma[y][: idx + 1] / L[y] ** 2
            # original
            x2 = r_ring[: idx + 1] / rem_s
            y2 = sigma[y][: idx + 1]
        elif func == "sigma_cum":
            x1 = r_disk[: idx + 1] * L[y] / rem_s
            y1 = (N[i, 3] / N[i, j]) * sigma_cum[y][: idx + 1] / L[y] ** 2
            # original
            x2 = r_disk[: idx + 1] / rem_s
            y2 = sigma_cum[y][: idx + 1]
        else:
            raise NotImplementedError
        xlim = max(xlim, x2.max())

        ax.plot(
            x1,
            y1,
            color=color,
            alpha=0.25 + j / 4,
            label=f"true_{y}",
        )

        if j == 3:
            continue

        ax.plot(
            x2,
            y2,
            color="grey",
            alpha=0.25 + j / 4,
            ls="-",
            label=f"scaled_{y}",
        )

    if remoteness:
        ax.set_xlabel(r"$r$", fontsize=12)
    else:
        ax.set_xlabel(r"$s$", fontsize=12)
    if func == "rho":
        ax.set_ylabel(r"$\rho$", fontsize=12)
    elif func == "sigma":
        ax.set_ylabel(r"$\sigma$", fontsize=12)
    elif func == "sigma_cum":
        ax.set_ylabel(r"$\bar\sigma$", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    handles, labels = ax.get_legend_handles_labels()

    scaled_handles = [
        h for i, h in enumerate(handles) if labels[i].startswith("scaled")
    ]
    scaled_labels = [l.split("_")[1] for l in labels if l.startswith("scaled")]
    true_handles = [h for i, h in enumerate(handles) if labels[i].startswith("true")]
    true_labels = [l.split("_")[1] for l in labels if l.startswith("true")]

    if legend:
        legend_1 = ax.legend(
            scaled_handles,
            scaled_labels,
            title="Observed",
            frameon=False,
            bbox_to_anchor=(1.05, 0.3),
            loc="lower left",
            borderaxespad=0.0,
            ncols=2,
            alignment="left",
        )
        ax.add_artist(legend_1)
        ax.legend(
            true_handles,
            true_labels,
            title="Scaled",
            frameon=False,
            bbox_to_anchor=(1.05, -0.5),
            loc="lower left",
            borderaxespad=0.0,
            ncols=2,
            alignment="left",
        )
    ax.set_xlim(0, xlim)


def plot_L_vs_P(ax, slopes_df_filtered, cve_bold, x1, x2, y1, y2, zoom, labels=True):
    markers = {"1990-2000": "o", "2000-2010": "^", "2010-2020": "s"}

    # Fill regions
    alpha = 0.3
    zorder = 0.5
    xgrid1 = np.linspace(x1, 1, 1000)
    xgrid2 = np.linspace(1, x2, 1000)
    ax.fill_between(
        xgrid1,
        0,
        np.sqrt(xgrid1),
        alpha=alpha,
        color=colors_regions[2],
        zorder=zorder,
        # label=r"$\bar\sigma$ ⬆, R ⬅, P ⬇",
    )
    ax.fill_between(
        xgrid2,
        0,
        1,
        alpha=alpha,
        color=colors_regions[1],
        zorder=zorder,
        # label=r"$\bar\sigma$ ⬆, R ⬅, P ⬆",
    )
    ax.fill_between(
        xgrid2,
        1,
        np.sqrt(xgrid2),
        alpha=alpha,
        color=colors_regions[0],
        zorder=zorder,
        # label=r"$\bar\sigma$ ⬆, R ➡, P ⬆",
    )
    ax.fill_between(
        xgrid2,
        np.sqrt(xgrid2),
        10,
        alpha=alpha,
        color=colors_regions[5],
        zorder=zorder,
        # label=r"$\bar\sigma$ ⬇, R ➡, P ⬆",
    )
    ax.fill_between(
        xgrid1,
        np.sqrt(xgrid1),
        1,
        alpha=alpha,
        color=colors_regions[3],
        zorder=zorder,
        # label=r"$\bar\sigma$ ⬇, R ⬅, P ⬇",
    )
    ax.fill_between(
        xgrid1,
        1,
        10,
        alpha=alpha,
        color=colors_regions[4],
        zorder=zorder,
        # label=r"$\bar\sigma$ ⬇, R ➡, P ⬇",
    )

    # Dashed lines
    ax.axhline(1, color="k", ls="--")
    ax.axvline(1, color="k", ls="--")
    xgrid = np.linspace(x1, x2, 1000)
    ax.plot(xgrid, np.sqrt(xgrid), ls="--", color="k")
    if not zoom:
        ax.text(
            3,
            np.sqrt(3) + 0.23,
            r"$\Phi_{ij}=\sqrt{P(t_j)/P(t_i)}$",
            fontsize=10,
            va="top",
            ha="center",
            rotation=np.arctan(0.5) / np.pi * 180 + 1,
        )

        ax.text(
            0.01,
            0.85,
            "Density loss\nUrban compression\nPopulation loss",
            fontsize=8,
        )

        ax.text(
            0.05,
            1.95,
            "Density loss\nUrban expansion\nPopulation loss",
            fontsize=8,
        )

        ax.text(
            1.05,
            1.95,
            "Density loss\nUrban expansion\nPopulation growth",
            fontsize=8,
        )

        ax.text(
            3.55,
            1.05,
            "Density gain\nUrban expansion\nPopulation growth",
            fontsize=8,
            ha="right",
        )

        ax.text(
            3.55,
            0.82,
            "Density gain\nUrban compression\nPopulation growth",
            fontsize=8,
            ha="right",
        )

        ax.text(
            0.93,
            0.03,
            "Density gain\nUrban compression\nPopulation loss",
            fontsize=8,
            ha="right",
        )

    # Scatter plots
    for p, color in period_colors.items():
        if p not in slopes_df_filtered.period.unique():
            continue
        ax.scatter(
            slopes_df_filtered.query("period == @p").N2_N1,
            slopes_df_filtered.query("period == @p").L,
            color=color,
            label=(
                f"{p.split('-', maxsplit=1)[0]}-{p.split('-')[1]}" if labels else None
            ),
            # label=p,
            alpha=0.7,
            marker=markers[p],
        )

    ax.set_xlabel(r"Population growth $P(t_j)/P(t_i)$", fontsize=12)
    ax.set_ylabel(r"Urban expansion factor $\Phi_{ij}$", fontsize=12)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    for cve, color in cve_bold.items():
        for p, m in markers.items():
            ax.scatter(
                slopes_df_filtered.loc[cve].query("period == @p").N2_N1,
                slopes_df_filtered.loc[cve].query("period == @p").L,
                marker=m,
                s=80,
                edgecolors="k",
                color=color,
            )

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)


def plot_L_P_outliers(ax, slopes_df, models_df, x1, x2, y1, y2):
    slopes_df_filtered = slopes_df.assign(dt=lambda df: df.t2 - df.t1)

    for dt, marker in markers_dt.items():
        ax.scatter(
            slopes_df_filtered.query("dt == @dt").N2_N1,
            slopes_df_filtered.query("dt == @dt").L,
            color="grey",
            label=rf"$\Delta t = {dt}$",
            alpha=0.6,
            marker=marker,
            s=10,
        )

    for dt, marker in markers_dt.items():
        ax.scatter(
            slopes_df_filtered.loc[outliers].query("dt == @dt").N2_N1,
            slopes_df_filtered.loc[outliers].query("dt == @dt").L,
            color="red",
            label=rf"$\Delta t = {dt}$",
            alpha=0.6,
            marker=marker,
            s=10,
        )

    ax.set_xlabel(r"$P(t_j)/P(t_i)$")
    ax.set_ylabel(r"$\Phi_{ij}$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    alpha = models_df.loc["shared_slope_DT_intercept"].itcp_Dt
    beta = models_df.loc["shared_slope_DT_intercept"].beta

    def model(x):
        return x**beta * np.exp(alpha * 10)

    def model2(x):
        return x**beta * np.exp(alpha * 20)

    def model3(x):
        return x**beta * np.exp(alpha * 30)

    ax.axline((1, model(1)), (1.5, model(1.5)), color="grey", ls="-", lw=1)
    ax.axline((1, model2(1)), (1.5, model2(1.5)), color="grey", ls="-", lw=1)
    ax.axline((1, model3(1)), (1.5, model3(1.5)), color="grey", ls="-", lw=1)

    ax.set_yticks([1, 2, 3, 4], ["1", "2", "3", "4"], minor=True)
    ax.set_yticks([1, 2, 3, 4], ["1", "2", "3", "4"], minor=False)

    ax.set_xticks([1, 10], ["1", "10"], minor=False)

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=1,
        edgecolor="k",
        ls="--",
        facecolor="none",
    )
    ax.add_patch(rect)


def plot_L_P_models(
    ax, slopes_df, models_df, boundary=True, dt_filter=None, add_red_patch=True
):
    slopes_df_filtered = slopes_df.assign(dt=lambda df: df.t2 - df.t1).drop(outliers)

    for dt, marker in markers_dt.items():
        if (dt_filter is not None) and (dt != dt_filter):
            continue
        ax.scatter(
            slopes_df_filtered.query("dt == @dt").N2_N1,
            slopes_df_filtered.query("dt == @dt").L,
            color=period_colors[dt],
            label=rf"$\Delta t = {dt}$",
            alpha=0.6,
            marker=marker,
        )

    ax.set_xlabel(r"Population growth $P(t_j)/P(t_i)$ (log scale)", fontsize=12)
    ax.set_ylabel(r"Urban expansion factor $\Phi_{ij}$ (log scale)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    h, l = ax.get_legend_handles_labels()
    red_patch = mpatches.Patch(color="red", label="The red data")
    if add_red_patch:
        h.append(red_patch)
        l.append("Outliers")
    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.0, 1.0),
        ncol=4,
        handles=h,
        labels=l,
    )

    alpha = models_df.loc["shared_slope_DT_intercept"].itcp_Dt
    beta = models_df.loc["shared_slope_DT_intercept"].beta

    for dt in [10, 20, 30]:
        if (dt_filter is not None) and (dt != dt_filter):
            continue

        def model(x):
            return x**beta * np.exp(alpha * dt)

        ax.axline(
            (1, model(1)), (1.5, model(1.5)), color=period_colors[dt], ls="-", lw=2
        )

    x1, x2 = ax.get_xlim()
    xgrid = np.linspace(x1, x2, 1000)
    if boundary:
        ax.plot(xgrid, np.sqrt(xgrid), ls="--", color="k")
        ax.text(
            3,
            np.sqrt(3) + 0.18,
            r"$\Phi_{ij}=\sqrt{P(t_j)/P(t_i)}$",
            fontsize=10,
            va="top",
            ha="center",
            rotation=np.arctan(0.5) / np.pi * 180 + 2,
        )
    ax.set_xlim(x1, x2)

    ax.set_yticks(
        [1, 1.5, 2, 2.5], ["1.0", "1.5", "2.0", "2.5"], minor=True, fontsize=12
    )
    ax.set_yticks(
        [1, 1.5, 2, 2.5], ["1.0", "1.5", "2.0", "2.5"], minor=False, fontsize=12
    )

    ax.set_xticks([1, 2, 3], ["1.0", "2.0", "3.0"], minor=True, fontsize=12)
    ax.set_xticks([1, 2, 3], ["1.0", "2.0", "3.0"], minor=False, fontsize=12)


def fit_model(N2_N1, L, fit_intercept=True, add_dummies=False, dummies=None):
    assert not (fit_intercept and add_dummies)

    X = np.log(N2_N1)[:, None]

    if fit_intercept:
        X = sm.add_constant(X)
    if add_dummies:
        X = np.column_stack([X, dummies.values])
    y = np.log(L)

    model = sm.OLS(y, X)
    results = model.fit()
    # Compute r2 with sklearn to always get centered version
    yhat = results.predict(X)
    r2_value = r2_score(y, yhat)
    aic_value = results.aic
    bic_value = results.bic
    if fit_intercept:
        beta = results.params[1]
        intercept = [results.params[0]]
    elif add_dummies:
        beta = results.params[0]
        intercept = results.params[1:]
    else:
        beta = results.params[0]
        intercept = [0]

    # We also compute bootstraped confidence intervals on the parameters
    # because errors are not normally distributed
    bs_res = bootstrap(
        (np.arange(len(y)),),  # an index array
        lambda i: sm.OLS(y[i], X[i]).fit().params,
    )
    low, high = bs_res.confidence_interval
    if fit_intercept:
        intercept_low = [low[0]]
        intercept_high = [high[0]]
        beta_low = low[1]
        beta_high = high[1]
    elif add_dummies:
        intercept_low = low[1:]
        intercept_high = high[1:]
        beta_low = low[0]
        beta_high = high[0]
    else:
        intercept_low = [0.0]
        intercept_high = [0.0]
        beta_low = low[0]
        beta_high = high[0]

    if not add_dummies:
        dummies_names = np.arange(len(intercept))
    else:
        dummies_names = dummies.columns
    res = pd.Series(
        {"beta": beta, "beta_low": beta_low, "beta_high": beta_high}
        | {f"itcp_{i}": x for i, x in zip(dummies_names, intercept)}
        | {f"itcp_{i}_low": x for i, x in zip(dummies_names, intercept_low)}
        | {f"itcp_{i}_high": x for i, x in zip(dummies_names, intercept_high)}
        | {"r2": r2_value}
    )

    return res, results


def fit_models(slopes_df, outliers):

    slopes_df_filtered = slopes_df.drop(outliers)

    # Model shared slope and intercept
    res_1, model_1 = fit_model(
        slopes_df_filtered.N2_N1.to_numpy(),
        slopes_df_filtered.L.to_numpy(),
        fit_intercept=True,
    )
    res_1 = res_1.rename("shared_slope_shared_intercept")

    # Model shared slopes no intercepts
    res_2, model_2 = fit_model(
        slopes_df_filtered.N2_N1.to_numpy(),
        slopes_df_filtered.L.to_numpy(),
        fit_intercept=False,
    )
    res_2 = res_2.rename("shared_slope_no_intercept")

    # Model shared slopes, individual intercepts
    res_3, model_3 = fit_model(
        slopes_df_filtered.N2_N1.to_numpy(),
        slopes_df_filtered.L.to_numpy(),
        fit_intercept=False,
        add_dummies=True,
        dummies=pd.get_dummies(slopes_df_filtered.period),
    )
    res_3 = res_3.rename("shared_slope_many_intercept")

    # Model shared slope, DT columns, no intercept
    res_4, model_4 = fit_model(
        slopes_df_filtered.N2_N1.to_numpy(),
        slopes_df_filtered.L.to_numpy(),
        fit_intercept=False,
        add_dummies=True,
        dummies=(slopes_df_filtered.t2 - slopes_df_filtered.t1).rename("Dt").to_frame(),
    )
    res_4 = res_4.rename("shared_slope_DT_intercept")

    # Individual models
    res_list = []
    for i, period in enumerate(slopes_df_filtered.period.unique()):
        x = slopes_df_filtered.query("period == @period").N2_N1.to_numpy()
        y = slopes_df_filtered.query("period == @period").L.to_numpy()
        res_list.append(fit_model(x, y, True)[0].rename(f"model_{period}"))

    models_df = pd.concat([res_1, res_2, res_3, res_4] + res_list, axis=1).T

    return models_df, [model_2, model_4]
