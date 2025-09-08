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
    cve_list, cve_names, rf_dir_path, opath, r_brackets=np.array([3, 5, 9.3])
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
    r_brackets : np.Array, optional
        Array of remoteness brackets for which to find equivalen distances,
        by default np.array([3, 5, 9.3]).
    """
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
            p_fraction=lambda df: df.population
            / df.groupby("year").population.transform("sum")
        )
    )
    agg_df.to_csv(opath / "rem_brackets_agg.csv", index=False)


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
        for d in rem_vals:
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
