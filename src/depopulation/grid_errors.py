"""Functions for estimating errors in global population grids."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr

from depopulation.radial_f import gen_pop_ar
from depopulation.utils import row2cell


def get_centers_gdf(
    rem_path="outputs/remoteness_brackets.csv",
    agg_path="outputs/zones_agg_from_mesh.csv",
):
    # A list of zones and their centers
    centers = (
        pd.merge(
            pd.read_csv(rem_path).set_index("CVE_MET"),
            pd.read_csv(agg_path, index_col=0, header=[0, 1])["Center"],
            left_index=True,
            right_index=True,
        )
        .pipe(
            lambda df: gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326",
            )
        )
        .to_crs("ESRI:54009")
    )
    return centers


def disag_pop_into_agebs(
    rem_path="outputs/remoteness_brackets.csv",
    agg_path="outputs/zones_agg_from_mesh.csv",
    ghs_pop_dir="data/ghs_pop/",
    agebs_geom_dir="data/AGEB_DATA_OUT/zone_agebs/shaped/2020",
    radial_f_dir="outputs/radial_f/",
    output_dir="outputs/",
):
    centers = get_centers_gdf(rem_path, agg_path)
    df_list = []
    cve_list = list(centers.index)
    for cve in cve_list:
        # We need the administrative units, urban agebs from 2020. Inegi MG
        agebs = gpd.read_file(f"{agebs_geom_dir}/{cve}.gpkg").to_crs("ESRI:54009")

        # We need GHS-POP 2020 population grid for Mexico.
        ghs_raster = rxr.open_rasterio(f"{ghs_pop_dir}{cve}POP_2020_54009_100.tif")
        ghs = (
            ghs_raster.to_series()
            .reset_index()
            .rename(columns={0: "Pop"})
            .assign(
                geometry=lambda df: df.apply(
                    row2cell, res_xy=ghs_raster.rio.resolution(), axis=1
                ).pipe(gpd.GeoSeries, crs=ghs_raster.rio.crs)
            )
            .pipe(gpd.GeoDataFrame)
            .assign(Area=lambda df: df.area)
        )

        # Agregate over agebs
        agebs = (
            agebs.set_index("CVEGEO")
            .assign(
                pop_ghs=gpd.overlay(agebs, ghs)
                .assign(Pop_w=lambda df: df.Pop * df.area / df.Area)
                .groupby("CVEGEO")
                .Pop_w.sum(),
                diff=lambda df: df.pop_ghs - df.POBTOT,
                abs_diff=lambda df: abs(df["diff"]),
                sqrd_diff=lambda df: df["diff"] ** 2,
                dens_diff=lambda df: df["diff"] / (df.area / 1e6),
                dens_abs_diff=lambda df: df.abs_diff / (df.area / 1e6),
                d_center=lambda df: (
                    df.centroid.distance(centers.loc[cve, "geometry"]) / 1e3
                ),
                region=lambda df: pd.cut(
                    df.d_center,
                    [0] + centers.loc[cve, ["r_3", "r_5", "r_9.3"]].to_list() + [1e6],
                    labels=["central", "mid", "peri", "outskirts"],
                ),
                CVE_MET=cve,
            )
            .reset_index()
        )
        df_list.append(agebs)
    agebs = pd.concat(df_list, axis=0)

    cve_list = agebs.CVE_MET.unique()
    pop_ar = gen_pop_ar(cve_list, Path(radial_f_dir))
    agebs = agebs.merge(
        pd.Series(pop_ar[:, 3], index=cve_list).rename("P_2020"),
        left_on="CVE_MET",
        right_index=True,
    )
    agebs["remoteness"] = agebs["d_center"] * 1000 / np.sqrt(agebs["P_2020"])

    agebs["re_diff"] = (agebs["diff"] / agebs["POBTOT"]).replace(np.inf, np.nan)
    agebs["re_abs_diff"] = (agebs["abs_diff"] / agebs["POBTOT"]).replace(np.inf, np.nan)

    # Classifie cells according to relative error magnitude
    bins = [-np.inf, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, np.inf]
    labels = [
        # "Negative outlier",
        "Greatly underestimated",
        "Underestimated",
        "Slightly underestimated",
        "Accurately estimated",
        "Slightly overestimated",
        "Overestimated",
        "Greatly overestimated",
        # "Positive outlier"
    ]
    labels = [
        # "Negative outlier",
        "GU",
        "U",
        "SU",
        "AE",
        "SO",
        "O",
        "GO",
        # "Positive outlier"
    ]
    agebs["e_class"] = pd.cut(agebs["re_diff"], bins, labels=labels)
    agebs["e_class"].value_counts(dropna=False)

    agebs.to_file(f"{output_dir}/agebs_ghs.gpkg")
