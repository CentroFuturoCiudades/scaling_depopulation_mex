"""Miscellanous reusable functions."""

import numpy as np


def get_adj_idx(l, thresh=0):
    """Get index in array-like <l> for wich l[i] <= thresh

    Parameters
    ----------
    l: np.array
        Array to truncate, assumed sorted.
    tresh: float
        Threshold value to truncate at.

    Returns
    -------
    int
        Index of l for which l[i] <= thresh.
    """
    idx = np.where(l <= thresh)[0]
    if len(idx) > 0:
        idx = idx[0]
    else:
        idx = len(l)

    return idx
