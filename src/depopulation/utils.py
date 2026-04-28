"""Miscellanous reusable functions."""

import numpy as np
from shapely import geometry


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


def row2cell(row, res_xy):
    res_x, res_y = res_xy  # Extract resolution for each dimension
    # XY Coordinates are centered on the pixel
    minX = row["x"] - (res_x / 2)
    maxX = row["x"] + (res_x / 2)
    minY = row["y"] + (res_y / 2)
    maxY = row["y"] - (res_y / 2)
    poly = geometry.box(minX, minY, maxX, maxY)  # Build squared polygon
    return poly
