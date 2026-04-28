"""Functions to perform scaling analysis for extensive variables with population.
Used to study scaling of total urban area and specific distance quantiles with P."""

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import linregress, t


def scaling_analysis(x, y, pooled=False, index=None, transposed=False):
    """Scaling analysis regressing x onto y in logarithmic coordinates.

    Parameters
    ----------
    x : numpy.Array
        Array with independent variable. If 1D performs a single regression.
        If 2D and M X N, performs N regressions, one per column.
    y: numpy.Array
        Must be same shape as x. Array with the dependent variable.
    pooled: bool
        If true, perform one additional regression with pooled variables.
        Variables are centered in logarithmic space.
        Each column is centered independently.
    index: pd.Index
        Custom index for returned Data Frame. Labels each independent regression.

    Returns
    -------
    Tuple
        Tuple of result objects.
    """

    x = np.asarray(x).reshape((len(x), -1))
    y = np.asarray(y).reshape((len(y), -1))
    if transposed:
        x = x.T
        y = y.T
    nvars, nregress = x.shape

    # t distribution to calculate 95% confidence intervals
    def tinv(p, df):
        return abs(t.ppf(p / 2, df))

    ts = tinv(0.05, nvars - 2)

    logx = np.log(x)
    logy = np.log(y)

    # Returned arrays for estimated parameters
    # one per year and one for pooled fit

    beta = np.zeros(nregress)
    log_y0 = np.zeros(nregress)
    rvalue = np.zeros(nregress)
    pvalue = np.zeros(nregress)
    beta_stderr = np.zeros(nregress)
    log_y0_stderr = np.zeros(nregress)
    beta_ci = np.zeros(nregress)
    log_y0_ci = np.zeros(nregress)
    residuals = np.zeros((nvars, nregress))
    for i in range(nregress):
        res = linregress(logx[:, i], logy[:, i])
        beta[i] = res.slope
        log_y0[i] = res.intercept
        rvalue[i] = res.rvalue**2
        pvalue[i] = res.pvalue
        beta_stderr[i] = res.stderr
        log_y0_stderr[i] = res.intercept_stderr

        beta_ci[i] = ts * res.stderr
        log_y0_ci[i] = ts * res.intercept_stderr

        residuals[:, i] = logy[:, i] - res.intercept - res.slope * logx[:, i]

    # Create results dataframe
    df = pd.DataFrame(
        {
            "log_Y0": log_y0,
            "beta": beta,
            "log_Y0_stderr": log_y0_stderr,
            "log_Y0_ci": log_y0_ci,
            "beta_stderr": beta_stderr,
            "beta_ci": beta_ci,
            "R2": rvalue,
            "pvalue": pvalue,
        }
    )
    if index is not None:
        df = df.set_index(pd.Index(index))

    if not pooled:
        return df, residuals, None, None

    # Perform pooled regression
    ts = tinv(0.05, nvars * nregress - 2)
    logxc = logx - logx.mean(axis=0)
    logyc = logy - logy.mean(axis=0)
    res = linregress(logxc.ravel(), logyc.ravel())
    residuals_pooled = logyc - res.intercept - res.slope * logxc
    series_pooled = pd.Series(
        {
            "log_Y0": res.intercept,
            "beta": res.slope,
            "log_Y0_stderr": res.intercept_stderr,
            "log_Y0_ci": ts * res.intercept_stderr,
            "beta_stderr": res.stderr,
            "beta_ci": ts * res.stderr,
            "R2": res.rvalue**2,
            "pvalue": res.pvalue,
        }
    )

    # Find intercepts for regressions fix slope fixed at pooled value
    log_y0_pooled = np.mean(logy - res.slope * logx, axis=0)

    df = pd.concat([df, series_pooled.rename("pooled").to_frame().T], axis=0)

    return df, residuals, residuals_pooled, log_y0_pooled


def get_moments(cve_list, years, radial_f, d=1):
    """Find the d non-central moment for all zones in cve_list from the radial
    distribution function.

    Parameters
    ----------
    cve_list : List
        List of zones to get the moments from. Entries correspond to keys in <radial_f>.
    years : List or Tuple
        List of years at which to get the moments.
    radial_f : Dict
        Dictionary of dictionaries with radial density functions for each zone.
    d : int, optional
        Order of the moment to calculate.

    Returns
    -------
    np.Array
        Array with moments of shape (# of zones, # of years)
    """

    n_c = len(cve_list)
    n_y = len(years)
    moments = np.zeros((n_c, n_y))
    for i, cve in enumerate(cve_list):
        r = radial_f[cve]["r_ring"]
        for j, year in enumerate(years):
            # l = radial_f[cve][f"lambda_{year}"]
            rho = radial_f[cve][f"rho_{year}"]
            moments[i, j] = simpson(r**d * rho, r)
    return moments


def get_factors(beta, cve_list, years, radial_f, agg_all_df, mg):
    """Get correction factors for border/coastal cities by correcting the population
    to that expected for a city with an extended urban area to the complete disk.
    The factor is (A_T/A_L)**(1/beta), which equals N2/N1 under the scaling law for
    areas.

    Parameters
    ----------
    beta : float
        Estimated scaling exponent for uban area in the current city system.
    cve_list :  List
        List of zones. Entries correspond to keys in <radial_f>.
    years : List or Tuple
        List of years.
    radial_f : Dict
        Dictionary of dictionaries with radial density functions for each zone.
    agg_all_df : DataFrame
        DataFrame with zone level aggregated data. We obtain city centres from here.
    mg : GeoDataFrame
        Marco Geoestadístico at 2020. Used to obtain national area (non-water) on the
        full disk enclosing each zone. Expected state geometries.

    Returns
    -------
    np.Array
        Array with correction factors of shape (# of zones, # of years)
    """
    n_c = len(cve_list)
    n_y = len(years)

    factors = np.ones((n_c, n_y))

    for i, cve in enumerate(cve_list):
        r = radial_f[cve]["r_disk"]
        for j, _ in enumerate(years):
            maxr = r[-1]
            center = agg_all_df.loc[cve, "geometry"]
            disk = center.buffer(maxr)
            area_total = disk.area
            area_land = disk.intersection(mg[mg.intersects(disk)].union_all()).area
            factors[i, j] = (area_total / area_land) ** (1 / beta)

    return factors


def perform_trans_temp_scaling(x, y, idx_trans, idx_temp):
    """Performs simulatanous transversal and temporal scaling analysis on x an y.

    Parameters
    ----------
    x : numpy.Array
        Array with independent variable. If 1D performs a single regression.
        If 2D and M X N, performs N regressions, one per column.
    y: numpy.Array
        Must be same shape as x. Array with the dependent variable.
    idx_trans : pd.Index
        Custom index for returned Data Frame. Labels each independent regression.
    idx_temp : pd.Index
        Custom index for returned Data Frame. Labels each independent regression.

    Returns
    -------
    Tuple
        Tuple of result objects.
    """
    # Transversal scaling
    df_scaling, _, residuals_pooled, log_y0_pooled = scaling_analysis(
        x,
        y,
        pooled=True,
        index=idx_trans,
    )
    beta_pooled = df_scaling.loc["pooled", "beta"]
    beta_pooled_r2 = df_scaling.loc["pooled", "R2"]
    beta_pooled_ci = df_scaling.loc["pooled", "beta_ci"]
    betas = df_scaling.beta.to_numpy()[:-1]
    betas_ci = df_scaling.beta_ci.to_numpy()[:-1]
    log_y0s = df_scaling.log_Y0.to_numpy()[:-1]

    # Temporal scaling area
    df_scaling_temp, _, _, _ = scaling_analysis(
        x,
        y,
        pooled=False,
        transposed=True,
        index=idx_temp,
    )
    alphas = df_scaling_temp.beta.to_numpy()
    log_y0is = df_scaling_temp.log_Y0.to_numpy()

    # Scaling analysis, decomposed variables
    df_scaling_temp_dec, _, _, _ = scaling_analysis(
        x,
        (y / np.exp(log_y0_pooled)),
        pooled=False,
        transposed=True,
        index=idx_temp,
    )
    alphas_dec = df_scaling_temp_dec.beta.to_numpy()
    log_y0is_dec = df_scaling_temp_dec.log_Y0.to_numpy()
    alphas_dec_r2 = df_scaling_temp_dec.R2.to_numpy()
    alphas_dec_ci = df_scaling_temp_dec.beta_ci.to_numpy()
    alphas_dec_p = df_scaling_temp_dec.pvalue.to_numpy()

    return (
        df_scaling,
        df_scaling_temp,
        df_scaling_temp_dec,
        beta_pooled,
        beta_pooled_r2,
        beta_pooled_ci,
        log_y0_pooled,
        betas,
        betas_ci,
        log_y0s,
        alphas,
        log_y0is,
        alphas_dec,
        log_y0is_dec,
        residuals_pooled,
        alphas_dec_ci,
        alphas_dec_r2,
        alphas_dec_p,
    )


def scaling_plot_all_years(
    x,
    y,
    ax,
    slopes=None,
    intercepts=None,
    slopes_t=None,
    intercepts_t=None,
    mark_idx=(),
    lines=False,
    color_city=True,
):
    """Plots log-log scatter plot ox x and y with scaling fits for each year.

    Parameters
    ----------
    x : np.Array
        Array of x values, of shape (# of zones, # of years)
    y : np.Array
        Array of y values, same shape of x.
    ax : matplotlib.axes._axes.Axes
        Axes to place plot.
    slopes : np.Array, optional
        Array of beta scaling exponents, one per year, by default None.
        If provided, plots linear transversal scaling fits.
        Lenght must be that of the second dimension of x.
    intercepts : np.Array, optional
        Intercepts of the transversal linear scaling fits, by default None.
        Must be provided if slopes is provided.
        Must be of the same shape as slopes.
    slopes_t : np.Array, optional
        Temporal scaling exponents, by default None.
        If provided, there must be one exponent per value of x (zones).
        If provided, draw linear fits to each row of x/y (# of zones).
    intercepts_t : np.Array, optional
        Intercepts of the temporal linear scaling fits, by default None.
        Must be provided if slopes_t is provided.
        Must be of the same shape as slopes_t.
    mark_idx : tuple or np.Array, optional
        Index of points to mark with an x, by default ()
    lines : bool, optional
        Wether to join points of the same zone by lines, by default False
    """
    cmap = mpl.colormaps["tab20"]

    logx = np.log(x)
    logy = np.log(y)
    # Temporal regression lines
    if slopes_t is not None:
        assert intercepts_t.shape == slopes_t.shape
        for xi, yi, slope, intercept in zip(logx, logy, slopes_t, intercepts_t):
            xgrid = np.linspace(xi.min(), xi.max(), 2)
            ax.plot(xgrid, intercept + slope * xgrid, "-", color="black", alpha=0.5)

    if color_city:
        # Scatter plot, a color per city
        for i, (xi, yi) in enumerate(zip(logx, logy)):
            if lines:
                ax.plot(xi, yi, ls="-", color="grey", alpha=0.3)
            ax.scatter(xi, yi, alpha=1, s=10, color=cmap(i % 20))
            if i in mark_idx:
                ax.scatter(xi, yi, marker="x", color="black", alpha=1)
    else:
        # Scatter plot, a color per year
        years = (1990, 2000, 2010, 2020)
        for j, (xj, yj) in enumerate(zip(logx.T, logy.T)):
            ax.scatter(xj, yj, alpha=1, s=10, label=years[j])
        ax.legend()

    # Regression lines
    xgrid = np.linspace(logx.min(), logx.max(), 100)
    for i, (intercept, slope) in enumerate(zip(intercepts, slopes)):
        ax.plot(xgrid, intercept + slope * xgrid)
