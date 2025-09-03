"""Code for building and loading radial functions."""

import numpy as np
import pandas as pd
from shapely import Point

from .utils import get_adj_idx


def build_radial_distributions(
    mesh_gdf, outdir, prefix="", cutoff=False, adjust_area=False, mg=None, cve=None
):
    """Builds all zones radial functions and stores them csv files.
    Generates two files per zone:
    - radial_functions_<zone_code>_rho.csv: stores radial functions estimated at the
      middle point of concentric rings.
    - radial_functions_<zone_code>_cdf.csv: stores cumulative radial functions
      estimated within disks or growing radii.

    Parameters
    ----------
    mesh_gdf : GeoDataFrame
        _description_
    outdir : Path
        Path to directory with output files.
    prefix : str, optional
        An optional prefix for output filenames, by default ""
    cutoff : bool, optional
        If True, distances will be truncated at the firts ring with population 0 in
        2020, by default False. Not recommended, as this truncation is now handled in
        post-processing.
    adjust_area : bool, optional
        If True, population density calculations will ignore area outside national
        boundaries, such as oceans and other countries, by default True.
    mg : GeoDataFrame, optional
        Mexico Marco Geoestadistico, must be provided if adjust_area is True, by
        default None. Expects state geometries in 2020 and same crs as mesh.
    cve : string, optional
        If provided, just build radial functions for the zone with this code, by
        default None.

    Returns
    -------
    Path
        Path to directory with output files.
    """

    # List to store clipped meshes
    # new_mesh = {}

    for cve_met in mesh_gdf.index.get_level_values(0).unique():
        if cve is not None and cve_met != cve:
            continue
        met_mesh = (
            mesh_gdf.loc[cve_met]
            .copy()
            .reset_index()
            .set_index(["YEAR", "codigo"])
            .sort_values("DIST")
        )

        # Find portion of marco geo intersectiong zone
        # We trust geometries for 2020
        if adjust_area:
            mesh_u = met_mesh.loc[2020].union_all()
            mg_u = mg[mg.intersects(mesh_u)].union_all()

        # Find the CDF and cumulative populations
        # We know use rings of 100m width
        cdf_list = []
        rho_list = []
        # Create the disatnce array based in max distance to center
        # Assume extension of 2020 to be the largest
        # This assumption fails for 17.1.01 but its likely 1990 data is wrong
        # units are in meters as that is the unit of the expected crs
        maxd = 0
        for year in [2020]:
            y_mesh = met_mesh.loc[year]

            # Build distance grid to max distance from center
            center = y_mesh.centroid_hist.iloc[0]
            boundary = y_mesh.union_all().convex_hull.boundary
            pts = [Point(p) for p in boundary.coords]
            dists = np.array([center.distance(p) for p in pts])
            maxd_new = dists.max()
            maxd = max(maxd, maxd_new)
        dgrid = np.arange(0, maxd + 100, 100)
        mid_pts = 0.5 * (dgrid[1:] + dgrid[:-1])

        # Add smoothed density for distances in index
        # Use 100 m rings ang get the population within each ring
        # We consider distances up to rho(rmax)=0, or ring with 0 pop
        # We find rmax only for 2020, assuming metzones only grow
        for year in [2020, 2010, 2000, 1990]:
            # Mesh for current year and zone
            y_mesh = met_mesh.loc[year]

            # This is a smoother way of getting cumpop
            # Works for any geometries, can be used directly on AGEBS
            # Assumes pop proportional to area intersection
            # Is also slower, but derivatives are well behaved
            marea = y_mesh.area
            # We have one less ring than edges
            pop = np.zeros(len(dgrid) - 1)
            mean_sigma = np.zeros(len(dgrid) - 1)
            adj_factor = np.ones_like(mean_sigma)
            nrings = len(pop)
            for i in range(nrings):
                ring = center.buffer(dgrid[i + 1]).difference(center.buffer(dgrid[i]))
                pop_curr = (
                    (y_mesh.intersection(ring).area / marea) * y_mesh.POB_URB
                ).sum()
                # Only for 2020 we find rmax and truncate dgrid
                if cutoff and year == 2020 and pop_curr == 0.0:
                    dgrid = dgrid[: i + 1].copy()
                    pop = pop[:i].copy()
                    mean_sigma = mean_sigma[:i].copy()
                    mid_pts = 0.5 * (dgrid[1:] + dgrid[:-1])
                    break
                pop[i] = pop_curr

                # We know estimate sigma over the area of the ring belonging to Mexico
                if adjust_area:
                    # Get area of ring over marco geoestadistico
                    adj_area = ring.intersection(mg_u).area
                    assert adj_area > 0
                    # Sigma over effective ring area
                    sigma = pop_curr / adj_area
                    # Adjustment factor for lambda calculation
                    adj_factor[i] = adj_area / ring.area
                else:
                    # Original density over total area
                    sigma = pop_curr / ring.area
                mean_sigma[i] = sigma
            # Cumulative population and cdf up to last r in dgrid
            cumpop = np.zeros_like(dgrid)
            cumpop[1:] = pop.cumsum()
            cdf = cumpop / cumpop[-1]
            # But now we have missaligned series, the pop sigma series are for rings
            # the cumpop cdf series are aligned with dgrid (len(pop)+1)
            # Tha actual estimated values of denisty are at midring points
            # Which are one less than dgrid points

            # Density is adjsuted by Aadj / Atot of a ring
            # If density correction is no used, adj_factor is all ones
            ldensity = 2 * np.pi * mid_pts * mean_sigma * adj_factor
            rho = ldensity / cumpop[-1]

            # Store them separately
            rho_series = pd.DataFrame(
                data={
                    f"ring_pop_{year}": pop,
                    f"rho_{year}": rho,
                    f"lambda_{year}": ldensity,
                    f"sigma_{year}": mean_sigma,
                },
                index=pd.Index(mid_pts, name="r"),
            )
            cdf_series = pd.DataFrame(
                data={
                    f"cumpop_{year}": cumpop,
                    f"cdf_{year}": cdf,
                },
                index=pd.Index(dgrid, name="r"),
            )
            assert cdf_series.index.is_unique
            assert cdf_series.index.is_monotonic_increasing
            cdf_list.append(cdf_series)
            rho_list.append(rho_series)

        df_cdf = pd.concat(cdf_list, axis=1).sort_index()
        df_rho = pd.concat(rho_list, axis=1).sort_index()
        assert df_cdf.isna().sum().sum() == 0
        assert df_rho.isna().sum().sum() == 0

        df_cdf = df_cdf[sorted(df_cdf.columns, key=lambda s: s.split("_")[-1])]
        df_rho = df_rho[sorted(df_rho.columns, key=lambda s: s.split("_")[-1])]
        df_cdf.to_csv(outdir / f"radial_functions{prefix}_{cve_met}_cdf.csv")
        df_rho.to_csv(outdir / f"radial_functions{prefix}_{cve_met}_rho.csv")

        # We will also clip the mesh to the current extents
        # met_mesh_new = met_mesh[met_mesh.intersects(center.buffer(dgrid[-1]))].copy()
        # new_mesh[cve_met] = met_mesh_new
    # new_mesh = pd.concat(new_mesh, names=["CVE_MET"])

    return outdir


def load_radial_f(cve_list, datadir, core=False):
    """Load radial density functions into a dictionary with zone codes as keys.

    Parameters
    ----------
    cve_list : List
        List of metropolitan zone codes.
    datadir : Path
        Path to directory with csv files.
    core : bool, optional
        If True, truncate distances up to the first zero in rho, by default False.

    Returns
    -------
    Dict
        Dictionary of dictionaries containing radial functions for each zone.
    """
    years = [1990, 2000, 2010, 2020]
    radial_f = {}

    def load_col(
        cname, year, df, idx_max, idx, cve_dict, norm=False, delta_r=None, fillvalue=0.0
    ):
        col = df[f"{cname}_{year}"].to_numpy()

        # Up to idx, not including idx, works with idx = len(r)
        if fillvalue == "last":
            col = col[: idx_max + 1].copy()
            col[idx:] = col[idx]
        else:
            col = col[:idx_max].copy()
            col[idx:] = fillvalue

        if norm == "pdens":
            col = col / np.sum(col * delta_r)
        elif norm == "cdens":
            col = col / col[idx]

        cve_dict[f"{cname}_{year}"] = col

        return col

    for cve in cve_list:
        cve_dict = {}

        df_rho = pd.read_csv(datadir / f"radial_functions_{cve}_rho.csv")
        df_cdf = pd.read_csv(datadir / f"radial_functions_{cve}_cdf.csv")
        r_ring = df_rho["r"].to_numpy()
        r_disk = df_cdf["r"].to_numpy()

        # Assuming equidistant sampling points
        delta_r = r_ring[1] - r_ring[0]

        # We first find idx max since all arrays must be of such lenght
        idx_max = len(r_ring)
        if core:
            idx_max = 0
            for year in years:
                rho = df_rho[f"rho_{year}"].to_numpy()
                idx = get_adj_idx(rho, thresh=1e-8)
                idx_max = max(idx_max, idx)
        # Truncate the r arrays
        r_ring = r_ring[:idx_max].copy()
        r_disk = r_disk[: idx_max + 1].copy()
        cve_dict["r_ring"] = r_ring
        cve_dict["r_disk"] = r_disk

        # Truncate all other arrays, fill zeros above idx up to idx_max
        for year in years:
            rho = df_rho[f"rho_{year}"].to_numpy()
            idx = len(r_ring)
            if core:
                # Gets the first index where rho=0
                idx = get_adj_idx(rho, thresh=1e-8)

            _ = load_col(
                "rho",
                year,
                df_rho,
                idx_max,
                idx,
                cve_dict,
                norm="pdens",
                delta_r=delta_r,
            )
            _ = load_col("ring_pop", year, df_rho, idx_max, idx, cve_dict)
            _ = load_col("lambda", year, df_rho, idx_max, idx, cve_dict)
            _ = load_col("sigma", year, df_rho, idx_max, idx, cve_dict)
            _ = load_col(
                "cdf",
                year,
                df_cdf,
                idx_max,
                idx,
                cve_dict,
                norm="cdens",
                fillvalue="last",
            )
            _ = load_col(
                "cumpop", year, df_cdf, idx_max, idx, cve_dict, fillvalue="last"
            )

        radial_f[cve] = cve_dict

    return radial_f
