"""Functions to perform radial scaling analysis for distance to centre variables."""

import numpy as np

from .utils import get_adj_idx


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
