"""Code for building and loading radial functions."""

from pathlib import Path

import cmcrameri.cm as cmc
import contextily as cx
import numpy as np
import pandas as pd
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


def gen_pop_ar(cve_list, rf_dir_path):
    """Generate array with population values.

    Parameters
    ----------
    cve_list : List
        List of zones (codes) for which to evaluate distances
    rf_dir_path : Path
        Path to radial distribution function data.

    Returns
    -------
    np.Array
        Array with population values of shape (# of zones, # of years)
    """
    years = (1990, 2000, 2010, 2020)
    radial_f = load_radial_f(cve_list, Path(rf_dir_path), core=True)
    pop_ar = np.array(
        [[radial_f[cve][f"cumpop_{year}"][-1] for year in years] for cve in cve_list]
    )
    return pop_ar


def get_remoteness_brackets(
    cve_list,
    cve_names,
    rf_dir_path,
    opath,
):
    """Generates csv file with distances and remotenes values equivalance.
    For each provided remoteness value, the equivalen distance in km is evaluated.
    The csv file is indexed by zone codes with a column for each remoteness value.

    Parameters
    ----------
    cve_list : List
        List of zones (codes) for which to evaluate distances
    cve_names : List
        Names of zones corresponfing to codes in cve_list.
    rf_dir_path : Path
        Path to radial distribution function data.
    opath : Path
        Path of the output directory.
    """
    r_brackets = np.array([3, 5, 9.3])
    pop_ar = gen_pop_ar(cve_list, rf_dir_path)

    dict_list = []
    for i, cve in enumerate(cve_list):
        dict_list.append(
            {
                "CVE_MET": cve,
                "CVE_NAME": cve_names[i],
                "brackets": r_brackets * np.sqrt(pop_ar[i, 3]) / 1000,
            }
        )
    rem_brackets = (
        pd.DataFrame(dict_list)
        .pipe(
            lambda df: pd.concat(
                [
                    df,
                    df[["brackets"]].apply(
                        lambda x: x.values[0], axis=1, result_type="expand"
                    ),
                ],
                axis=1,
            )
        )
        .drop(columns="brackets")
        .rename(columns={0: "r_3", 1: "r_5", 2: "r_9.3"})
        .set_index("CVE_MET")
    )

    # To get rmax we need the radial density functions
    radial_f = load_radial_f(cve_list, Path(rf_dir_path), core=True)
    for cve in cve_list:
        rem_brackets["rmax"] = [radial_f[cve]["r_disk"][-1] / 1000 for cve in cve_list]

    rem_brackets.to_csv(opath / "remoteness_brackets.csv")


def gen_rem_brackets_pop(cve_list, rf_dir_path, opath):
    """Generates csv file with population within specified remotenes values.
    The csv file is indexed by zone codes with a column for each remoteness value and
    year combination.
    Four remoteness brackets are considered:
    - Inner: 0<r<3
    - Mid: 3<r<5
    - Distant: 5<r<9.3
    - Outmost: r>9.3

    Parameters
    ----------
    cve_list : List
        List of zones (codes) for which to evaluate distances
    rf_dir_path : Path
        Path to radial distribution function data.
    opath : Path
        Path of the output directory.
    """
    years = (1990, 2000, 2010, 2020)
    pop_ar = gen_pop_ar(cve_list, rf_dir_path)
    radial_f = load_radial_f(cve_list, Path(rf_dir_path), core=True)
    series_list = []
    for i, cve in enumerate(cve_list):
        r_ring = radial_f[cve]["r_ring"]

        pop = {y: radial_f[cve][f"ring_pop_{y}"] for y in years}

        # Scaling factos
        factor = {y: np.sqrt(pop_ar[i, j]) for j, y in enumerate(years)}

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
    rem_brackets_2 = pd.DataFrame(series_list).rename_axis(index="CVE_MET")
    rem_brackets_2.to_csv(opath / "remoteness_brackets_pop.csv")


def gen_rem_brackets_pop_any_beta(cve_list, rf_dir_path, opath, beta=0.5):
    """Generates csv file with population within specified remotenes values.
    The csv file is indexed by zone codes with a column for each remoteness value and
    year combination.
    Four remoteness brackets are considered:
    - Inner: 0<r<3
    - Mid: 3<r<5
    - Distant: 5<r<9.3
    - Outmost: r>9.3

    Parameters
    ----------
    cve_list : List
        List of zones (codes) for which to evaluate distances
    rf_dir_path : Path
        Path to radial distribution function data.
    opath : Path
        Path of the output directory.
    """
    years = (1990, 2000, 2010, 2020)
    pop_ar = gen_pop_ar(cve_list, rf_dir_path)
    radial_f = load_radial_f(cve_list, Path(rf_dir_path), core=True)
    series_list = []
    for i, cve in enumerate(cve_list):
        # Get distance in km
        r_ring = radial_f[cve]["r_ring"] / 1e3

        pop = {y: radial_f[cve][f"ring_pop_{y}"] for y in years}

        # Scaling factos
        factor = {y: (pop_ar[i, j] / 1e6) ** beta for j, y in enumerate(years)}

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
    rem_brackets_2 = pd.DataFrame(series_list).rename_axis(index="CVE_MET")
    rem_brackets_2.to_csv(opath / f"remoteness_brackets_pop_beta_{beta:0.2f}.csv")


def agg_rem_brackets(brackets_path, opath):
    """Aggregate national population counts at remoteness brackets.

    Parameters
    ----------
    brackets_path : Path
        Path to population counts per remoteness bracket for each zone.
    opath : Path
        Path to output directory.
    """
    rem_df = pd.read_csv(brackets_path, index_col=0, header=[0, 1])

    agg_df = (
        (rem_df.sum() / 1e6)
        .reset_index()
        .rename(columns={"level_0": "bracket", "level_1": "year", 0: "population"})
        .assign(
            p_fraction=lambda df: (
                df.population / df.groupby("year").population.transform("sum")
            )
        )
    )

    # agg_df.to_csv(opath / "rem_brackets_agg.csv", index=False)
    return agg_df

    # beta = 0.5
    # series_list = []
    # for i, cve in enumerate(cve_list):
    #     r_ring = radial_f_core[cve]["r_ring"]

    #     pop = {y: radial_f_core[cve][f"ring_pop_{y}"] for y in years}

    #     # Scaling factos
    #     factor = {y: N_c[i, j] ** beta for j, y in enumerate(years)}

    #     # Scaled distances
    #     r_ring_s = {y: r_ring / factor[y] for y in years}

    #     pop_inner = {y: pop[y][r_ring_s[2020] < 3].sum() for y in years}
    #     pop_mid = {
    #         y: pop[y][np.logical_and(r_ring_s[2020] >= 3, r_ring_s[2020] < 5)].sum()
    #         for y in years
    #     }
    #     pop_distant = {
    #         y: pop[y][np.logical_and(r_ring_s[2020] >= 5, r_ring_s[2020] < 9.3)].sum()
    #         for y in years
    #     }
    #     pop_outmost = {y: pop[y][r_ring_s[2020] >= 9.3].sum() for y in years}
    #     cve_series = (
    #         pd.DataFrame(
    #             {
    #                 "N_inner": pop_inner,
    #                 "N_mid": pop_mid,
    #                 "N_distant": pop_distant,
    #                 "N_outmost": pop_outmost,
    #             }
    #         )
    #         .T.stack()
    #         .rename(cve)
    #     )
    #     series_list.append(cve_series)
    # rem_brackets = pd.DataFrame(series_list).rename_axis(index="CVE_MET")

    # # Get cumulative pop for density calculations
    # rem_brackets_cum = rem_brackets.T.groupby(level=1).transform("cumsum").T

    # r_brackets = np.array([3, 5, 9.3])

    # dict_list = []
    # for i, cve in enumerate(cve_list):
    #     dict_list.append(
    #         {"CVE_MET": cve, "brackets": r_brackets * np.sqrt(N_c[i, 3]) / 1000}
    #     )
    # rem_values = (
    #     pd.DataFrame(dict_list)
    #     .pipe(
    #         lambda df: pd.concat(
    #             [
    #                 df,
    #                 df[["brackets"]].apply(
    #                     lambda x: x.values[0], axis=1, result_type="expand"
    #                 ),
    #             ],
    #             axis=1,
    #         )
    #     )
    #     .drop(columns="brackets")
    #     .rename(columns={0: "r_3", 1: "r_5", 2: "r_9.3"})
    #     .set_index("CVE_MET")
    # )
    # rem_values["rmax"] = [radial_f_core[cve]["r_disk"][-1] / 1000 for cve in cve_list]

    # # Area of each annulus
    # disk_area = rem_values**2 * np.pi
    # annulus_area = disk_area.copy()
    # annulus_area["r_5"] = disk_area.r_5 - disk_area.r_3
    # annulus_area["r_9.3"] = disk_area["r_9.3"] - disk_area.r_5
    # annulus_area["rmax"] = disk_area["rmax"] - disk_area["r_9.3"]
    # annulus_area = annulus_area.rename(
    #     columns={
    #         "r_3": "N_inner",
    #         "r_5": "N_mid",
    #         "r_9.3": "N_distant",
    #         "rmax": "N_outmost",
    #     }
    # )

    # # Find overall density as all the people over all the area
    # density_brackets = (
    #     rem_brackets.sum()
    #     .groupby(level=1)
    #     .transform(lambda s: s / annulus_area.sum().values)
    #     .to_frame("density")
    # )


def pop_change_map(mesh, agg, cve_met, ax, rem_vals=None, rmax=None, adjust_vmax=False):
    """Generates a population change map, where cell values are population
    difference between 2020 and 1990.

    Parameters
    ----------
    mesh : GeoDataFrame
        Multi-temporal population mesh.
    agg : GeoDataFrame
        Table of aggregated zone statistics with geometry centres.
    cve_met : str
        Code of the zone to plot.
    ax : matplotlib.axes.Axes
        Ax to place figure.
    rem_vals : List, optional
        Remoteness values where to draw circumferences at, by default None
    rmax : float, optional
        Values of rem above this are filtered, by default None
    adjust_vmax : bool, optional
        If true, adjust color range to predefined values, by default False
    """
    mesh_met = mesh.loc[cve_met, ["POB_URB", "geometry"]]
    if rmax is not None:
        mask = mesh.loc[cve_met, "DIST"] < rmax
        mesh_met = mesh_met[mask]

    mesh_met = mesh_met.unstack(level=0)
    mesh_met.loc[:, ("POB_URB")] = mesh_met.loc[:, ("POB_URB")].fillna(0).values
    mesh_met.loc[:, ("geometry")] = (
        mesh_met.loc[:, ("geometry")].bfill(axis=1).ffill(axis=1).values
    )
    mesh_met = mesh_met.drop(
        columns=[("geometry", 1990), ("geometry", 2000), ("geometry", 2010)]
    )
    mesh_met.columns = [f"{a}_{b}" for a, b in mesh_met.columns]
    mesh_met = mesh_met.rename(columns={"geometry_2020": "geometry"}).set_geometry(
        "geometry"
    )
    mesh_met = mesh_met.assign(
        DIFF_POB_2020_1990=lambda df: df.POB_URB_2020 - df.POB_URB_1990
    )

    vmax = abs(mesh_met.DIFF_POB_2020_1990).max()
    vmax_tick = vmax
    if adjust_vmax:
        if vmax < 4000:
            vmax_tick = 2000
            vmax = 2300
        else:
            vmax_tick = 4000
            vmax = 4500

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0)
    # cax = ax.inset_axes([0.05, 0, 0.9, 0.05])

    mesh_met.plot(
        column="DIFF_POB_2020_1990",
        ax=ax,
        cax=cax,
        # pylint: disable-next=E1101
        cmap=cmc.vik.reversed(),
        edgecolor="none",
        lw=0.25,
        legend=True,
        legend_kwds={
            # "label": "Population change between 1990 and 2020",
            "orientation": "horizontal",
        },
        vmin=-vmax,
        vmax=vmax,
    )

    cax.set_xticks([-vmax_tick, 0, vmax_tick])

    center = agg.loc[[cve_met], ["geometry"]]
    center.plot(ax=ax, color="black", markersize=10)

    # chull = mesh_met.union_all().convex_hull
    if rem_vals is not None:
        for d in rem_vals.iloc[:3]:
            # center.buffer(d).boundary.intersection(chull).to_frame().plot(
            center.buffer(d * 1000).boundary.to_frame().plot(
                ax=ax, facecolor="none", ls="--", edgecolor="black"
            )

    ax.set_axis_off()

    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    xmid = (x1 + x2) / 2
    ymid = (y1 + y2) / 2
    delta = max(x2 - x1, y2 - y1)
    ax.set_xlim(xmid - delta / 2, xmid + delta / 2)
    ax.set_ylim(ymid - delta / 2, ymid + delta / 2)

    cx.add_basemap(
        ax,
        crs=mesh_met.crs,
        source=cx.providers.CartoDB.PositronNoLabels,
        zoom_adjust=0,
    )

    ax.add_artist(ScaleBar(1))


def plot_delta_density(
    ax,
    cve_list,
    radial_f,
    pop_ar,
    cve_bold,
    cve_names,
    beta=0.5,
    xlim=20,
    scale=True,
    agg=False,
    avg=False,
    rem_year=None,
    fontsize=14,
):
    """Plot delta (2020 - 1990) radial density values against distance from the center.
    Optionally scales radial density functions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Ax to place figure.
    cve_list : List
        List of zones (codes) to plot. A lines per zone.
    radial_f : Dict
        Dictionary of radial distributions functions, keys are zone codes.
    pop_ar : np.Array
        Array of population counts, of shape (# zones, # years)
    cve_bold : Dict
        Dictionay with codes and colors of example zones.
    cve_names : List
        List of zone names corresponding to codes in cve_list.
    beta : float, optional
        The scaling exponent, by default 0.5
    xlim : int, optional
        Limit of the x-axis, by default 20
    scale : bool, optional
        If true, scale radial density functions to a common remoteness scale, by
        default True
    agg : bool, optional
        If True, include aggregated national curve, by default False
    avg : bool, optional
        If true, use average population density, else use point population density,
        by default False
    rem_year : int, optional
        If provided, use this same year for all remoteness calculations, else use
        each years remoteness values, by default None.
    fontsize : int, optional
        Font size for the figure, by default 14
    """

    years = (1990, 2000, 2010, 2020)

    # Population of reference cities
    pop_ref = 1e6

    r_grid = np.linspace(0.0, xlim, 100)
    area_agg = {y: np.zeros_like(r_grid) for y in years}
    cumpop_agg = {y: np.zeros_like(r_grid) for y in years}
    cve_bold_l = cve_bold.keys()

    for i, cve in enumerate(cve_list):
        # Scaling factos, L and one for densities
        if scale and (rem_year is None):
            l_factor = {
                y: (pop_ref / pop_ar[i, j]) ** beta for j, y in enumerate(years)
            }
            s_factor = {
                y: pop_ref / pop_ar[i, j] / l_factor[y] ** 2
                for j, y in enumerate(years)
            }
        elif scale and (rem_year is not None):
            j = years.index(rem_year)
            # Use the same factor for all years
            l_factor = {y: (pop_ref / pop_ar[i, j]) ** beta for y in years}
            s_factor = {y: pop_ref / pop_ar[i, j] / l_factor[y] ** 2 for y in years}
        else:
            l_factor = {y: 1.0 for y in years}
            s_factor = {y: 1.0 for y in years}

        # Load ring/disk variables
        # remove disk variables initial zero point with zero area disk
        # Convert to km
        r_ring = {y: l_factor[y] * radial_f[cve]["r_ring"] / 1e3 for y in years}
        r_disk = {y: l_factor[y] * radial_f[cve]["r_disk"] / 1e3 for y in years}

        # I need to find the minumum r as to not try to interpolate below this
        # r_disk_min = {y: r_disk[y][1:].min() for y in years}
        # Find first index in r_grid that so that r is larger than rmin
        # r_disk_argmin = {y: np.searchsorted(r_grid, r_disk_min[y]) for y in years}

        # We remove the first zeroth element of cumpop so interpolation is taken over
        # non zero population
        sigma = {
            y: np.interp(r_grid, r_ring[y], s_factor[y] * radial_f[cve][f"sigma_{y}"])
            * 1e6
            for y in years
        }
        # Area in km of the disk at remoteness r_grid
        # While it is possible to obtain analytic expression
        # we must match the linear interpolation of cumpop, or pop would
        # alwasy be larger than corresponding area in the interpolating region
        # resulting in oscillations for small r
        # area_disk = {y: np.pi * (r_grid / l_factor[y]) ** 2 for y in years}
        # Make sure interpolation do not rach zero, minumum pop value is second cumpop
        area_disk = {
            y: np.interp(
                r_grid,
                r_disk[y][1:],
                np.pi * (radial_f[cve]["r_disk"][1:] / 1000) ** 2,
            )
            for y in years
        }
        cumpop = {
            y: np.interp(r_grid, r_disk[y][1:], radial_f[cve][f"cumpop_{y}"][1:])
            for y in years
        }
        # We should replace area values before rmin with values at rmin
        # for y in years:
        #    area_disk[y][: r_disk_argmin[y]] = area_disk[y][r_disk_argmin[y]]
        avg_sigma = {y: cumpop[y][1:] / area_disk[y][1:] for y in years}

        delta_sigma = sigma[2020] - sigma[1990]
        delta_avg_sigma = avg_sigma[2020] - avg_sigma[1990]

        if avg:
            y = delta_avg_sigma[:]
            x = r_grid[1:]
        else:
            y = delta_sigma
            x = r_grid

        ax.plot(
            x,
            y,
            color="grey" if cve not in cve_bold_l else cve_bold[cve],
            alpha=0.3 if cve not in cve_bold_l else 1,
            zorder=2 if cve not in cve_bold_l else 2.5,
            label=None if cve not in cve_bold_l else cve_names[i].split("-")[0],
            lw=1 if cve not in cve_bold else 2,
        )

        # Aggregated national density
        for y in years:
            cumpop_agg[y] += cumpop[y]
            area_agg[y] += area_disk[y]

    delta_r = r_grid[1] - r_grid[0]
    area_ring = {y: area_agg[y][1:] - area_agg[y][:-1] for y in years}
    pop_ring = {y: cumpop_agg[y][1:] - cumpop_agg[y][:-1] for y in years}
    avg_sigma_agg = {y: cumpop_agg[y][1:] / area_agg[y][1:] for y in years}
    if avg:
        x_agg = r_grid[1:]
        y_agg = avg_sigma_agg[2020] - avg_sigma_agg[1990]
    else:
        x_agg = r_grid[1:] - delta_r / 2
        y_agg = pop_ring[2020] / area_ring[2020] - pop_ring[1990] / area_ring[1990]
    if agg:
        ax.plot(
            x_agg,
            y_agg,
            color="black",
            lw=3,
            zorder=2.6,
            label="National",
        )

    if scale:
        ax.set_xlabel(r"$r$ (km)", fontsize=fontsize)
    else:
        ax.set_xlabel(r"$s$ (km)", fontsize=fontsize)

    if avg:
        ax.set_ylabel(r"$\Delta \bar{\sigma}(r)$ (people/km$^2$)", fontsize=fontsize)
    else:
        ax.set_ylabel(r"$\Delta \sigma$ (people/km$^2$)", fontsize=fontsize)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    # ax.legend(loc="lower right", frameon=False, ncols=2)
    ax.axhline(0, color="black", ls="--")
    ax.set_xlim(0, xlim)


def plot_density(
    ax,
    cve_list,
    radial_f,
    pop_ar,
    beta=0.5,
    xlim=20,
    scale=True,
    agg=False,
    avg=False,
    rem_year=None,
    fontsize=14,
):
    """Plots national aggregated radial density functions for all years.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Ax to place figure.
    cve_list : List
        List of zones (codes) to plot. A lines per zone.
    radial_f : Dict
        Dictionary of radial distributions functions, keys are zone codes.
    pop_ar : np.Array
        Array of population counts, of shape (# zones, # years)
    beta : float, optional
        The scaling exponent, by default 0.5
    xlim : int, optional
        Limit of the x-axis, by default 20
    scale : bool, optional
        If true, scale radial density functions to a common remoteness scale, by
        default True
    agg : bool, optional
        If True, include aggregated national curve, by default False
    avg : bool, optional
        If true, use average population density, else use point population density,
        by default False
    rem_year : int, optional
        If provided, use this same year for all remoteness calculations, else use
        each years remoteness values, by default None.
    fontsize : int, optional
        Font size for the figure, by default 14
    """

    years = (1990, 2000, 2010, 2020)

    # Population of reference cities
    pop_ref = 1e6

    r_grid = np.linspace(0.0, xlim, 100)
    area_agg = {y: np.zeros_like(r_grid) for y in years}
    cumpop_agg = {y: np.zeros_like(r_grid) for y in years}

    for i, cve in enumerate(cve_list):
        # Scaling factos, L and one for densities
        if scale and (rem_year is None):
            l_factor = {
                y: (pop_ref / pop_ar[i, j]) ** beta for j, y in enumerate(years)
            }
            s_factor = {
                y: pop_ref / pop_ar[i, j] / l_factor[y] ** 2
                for j, y in enumerate(years)
            }
        elif scale and (rem_year is not None):
            j = years.index(rem_year)
            # Use the same factor for all years
            l_factor = {y: (pop_ref / pop_ar[i, j]) ** beta for y in years}
            s_factor = {y: pop_ref / pop_ar[i, j] / l_factor[y] ** 2 for y in years}
        else:
            l_factor = {y: 1.0 for y in years}
            s_factor = {y: 1.0 for y in years}

        # Load ring/disk variables
        # remove disk variables initial zero point with zero area disk
        # Convert to km
        r_ring = {y: l_factor[y] * radial_f[cve]["r_ring"] / 1e3 for y in years}
        r_disk = {y: l_factor[y] * radial_f[cve]["r_disk"] / 1e3 for y in years}

        # We remove the first zeroth element of cumpop so interpolation is taken over
        # non zero population
        sigma = {
            y: np.interp(r_grid, r_ring[y], s_factor[y] * radial_f[cve][f"sigma_{y}"])
            * 1e6
            for y in years
        }
        # Area in km of the disk at remoteness r_grid
        # While it is possible to obtain analytic expression
        # we must match the linear interpolation of cumpop, or pop would
        # alwasy be larger than corresponding area in the interpolating region
        # resulting in oscillations for small r
        # area_disk = {y: np.pi * (r_grid / l_factor[y]) ** 2 for y in years}
        # Make sure interpolation do not rach zero, minumum pop value is second cumpop
        area_disk = {
            y: np.interp(
                r_grid,
                r_disk[y][1:],
                np.pi * (radial_f[cve]["r_disk"][1:] / 1000) ** 2,
            )
            for y in years
        }
        cumpop = {
            y: np.interp(r_grid, r_disk[y][1:], radial_f[cve][f"cumpop_{y}"][1:])
            for y in years
        }
        # We should replace area values before rmin with values at rmin
        # for y in years:
        #    area_disk[y][: r_disk_argmin[y]] = area_disk[y][r_disk_argmin[y]]
        avg_sigma = {y: cumpop[y][1:] / area_disk[y][1:] for y in years}

        delta_sigma = sigma[2020] - sigma[1990]
        delta_avg_sigma = avg_sigma[2020] - avg_sigma[1990]

        if avg:
            y = delta_avg_sigma[:]
            # x = r_grid[1:]
        else:
            y = delta_sigma
            # x = r_grid

        # ax.plot(
        #     x,
        #     y,
        #     color="grey" if cve not in cve_bold_l else cve_bold[cve],
        #     alpha=0.3 if cve not in cve_bold_l else 1,
        #     zorder=2 if cve not in cve_bold_l else 2.5,
        #     label=None if cve not in cve_bold_l else cve_names[i],
        # )

        # Aggregated national density
        for y in years:
            cumpop_agg[y] += cumpop[y]
            area_agg[y] += area_disk[y]

    delta_r = r_grid[1] - r_grid[0]
    area_ring = {y: area_agg[y][1:] - area_agg[y][:-1] for y in years}
    pop_ring = {y: cumpop_agg[y][1:] - cumpop_agg[y][:-1] for y in years}
    avg_sigma_agg = {y: cumpop_agg[y][1:] / area_agg[y][1:] for y in years}
    sigma_agg = {y: pop_ring[y] / area_ring[y] for y in years}
    # if avg:
    #     x_agg = r_grid[1:]
    #     y_agg = avg_sigma_agg[2020] - avg_sigma_agg[1990]
    # else:
    #     x_agg = r_grid[1:] - delta_r / 2
    #     y_agg = sigma_agg[2020] - sigma_agg[1990]
    # if agg:
    #     ax.plot(
    #         x_agg,
    #         y_agg,
    #         color="black",
    #         lw=2,
    #         zorder=2.5,
    #         label="National",
    #     )
    if avg:
        x_agg = r_grid[1:]
        y_agg = avg_sigma_agg
    else:
        x_agg = r_grid[1:] - delta_r / 2
        y_agg = sigma_agg
    cdict = {
        1990: "#e8d5d1",
        2000: "#bf8fa1",
        2010: "#7c5479",
        2020: "#2B213A",
    }
    if agg:
        for y in years:
            ax.plot(x_agg, y_agg[y], color=cdict[y], lw=2, label=y)

    if scale:
        ax.set_xlabel(r"$r$ (km)", fontsize=fontsize)
    else:
        ax.set_xlabel(r"$s$ (km)", fontsize=fontsize)

    if avg:
        ax.set_ylabel(r"$\bar{\sigma}$ (people/km$^2$)", fontsize=fontsize)
    else:
        ax.set_ylabel(r"$\sigma$ (people/km$^2$)", fontsize=fontsize)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    ax.legend(loc=None, frameon=False)
    ax.set_xlim(0, xlim)
