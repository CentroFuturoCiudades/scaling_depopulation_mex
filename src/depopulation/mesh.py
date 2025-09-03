"""Code to generate and process multi-temporal population mesh from individual
mesh files."""

import geopandas as gpd
import numpy as np
import pandas as pd


def build_mesh_gdf(
    poly_path, met_xlsx_path, agebs_path_shaped, agebs_path_trans, cent_xlsx_path, opath
):
    """Generates a GeoDataFrame with cells from all urban meshes,
    correctly indexed and clipped to the agebs extents.
    Saves GeoDataFrame to disk to <opath>.

    Parameters
    ----------
    poly_path : Path
        Path to directory containing geopackages with population grids previously
        computed.
    met_xlsx_path: Path
        Path to official excel file with aggregated information for Mexican
        Metropilitan Zones. Oficial data from CONAPO:
        https://www.gob.mx/conapo/documentos/las-metropolis-de-mexico-2020
    agebs_path_shaped: Path
        Path to directory with ageb geometries to cookie cut the mesh.
        Contains geometries for agebs of years 2010 and 2020, corrected with mapshaper.
    agebs_path_trans: Path
        Path to directory with ageb geometries to cookie cut the mesh.
        Contains geometries for translated agebs of years 1990 and 2000.
    center_xlsx_path: Path
        Path to excel file with location of urban centres.
    opath: Path
        Path of generated file on disk.

    Returns
    -------
    Path
        Path to geopackage generated on disk.
    """

    # a dataframe with official data for each zone
    metropoli_df = build_zones_df(met_xlsx_path)
    # Get list of cve, drop zona zonurbada and playa del carmen (missing in 1990)
    cve_list = metropoli_df.query("TIPO_MET != 'Zona conurbada'").drop("23.2.03").index

    # We load all mesh cells into a single geodataframe
    mesh_gdf = load_meshes(cve_list, poly_path, agebs_path_shaped, agebs_path_trans)

    # Add city centers and regions from xslx file
    centroids_df = (
        get_cent_region(cent_xlsx_path)
        .merge(
            metropoli_df.reset_index()[["CVE_MET", "NOM_MET"]].assign(
                NOM_MET=lambda x: x.NOM_MET.str.lower()
            ),
            on="NOM_MET",
            how="left",
        )
        .set_index("CVE_MET")
        .sort_index()
        .drop(columns="NOM_MET")
    ).to_crs(mesh_gdf.crs)

    # Add centers to mesh gdf
    mesh_gdf = mesh_gdf.join(centroids_df, how="left")

    # Add columns for cluster labeling
    mesh_gdf = mesh_gdf.assign(
        CENTROID=lambda df: df.centroid,
        REP_P=lambda df: df.representative_point(),
    )

    # Add distance to center
    # Population weighted columns are already normalized by total population
    # Useful to calculate averages as a simple sum
    # Distances are in km, converted from m
    mesh_gdf = mesh_gdf.assign(
        DIST=lambda df: df.CENTROID.distance(df.centroid_hist) / 1e3,
        # PWDIST=lambda df: df.DIST * df.POB_URB / df.POB_URB_TOT,
    )

    mesh_gdf.to_parquet(opath)

    return opath


def load_meshes(cve_list, poly_path, agebs_path_shaped, agebs_path_trans):
    """Generates a GeoDataFrame with cells from all urban meshes,
    correctly indexed and clipped to the agebs extents.

    Parameters
    ----------
    cve_list : List
        List of zone codes to identify data files to load.
    poly_path : Path
        Path to directory containing geopackages with population grids.
    agebs_path_shape : Path
        Path to directory with ageb geometries to cookie cut the mesh.
        Contains geometries for agebs of years 2010 and 2020, corrected with mapshaper.
    agebs_path_trans : Path
        Path to directory with ageb geometries to cookie cut the mesh.
        Contains geometries for translated agebs of years 1990 and 2000.
    Returns
    -------
    GeoDataFrame
        GeoDataFrame with the aggregated cells.
    """
    # We load all mesh cells into a single geodataframe
    poly_list = []
    for cve_met in cve_list:
        for year in ["1990", "2000", "2010", "2020"]:
            mesh = (
                gpd.read_file(poly_path / f"{year}/{cve_met}.gpkg")
                .rename(columns={"pop_fraction": "POBTOT"})
                .to_crs("ESRI:102008")
            )

            # Cut the grid to avoid inflating areas on the urban edge
            # This is a non negligible effect for the grid size we are using
            if year in ["1990", "2000"]:
                agebs_path = agebs_path_trans
            else:
                agebs_path = agebs_path_shaped
            agebs = gpd.read_file(agebs_path / f"{year}/{cve_met}.gpkg").to_crs(
                "ESRI:102008"
            )
            mesh_c = mesh.clip(agebs.make_valid(), keep_geom_type=True)
            # We can loose some cells that bare touch agebs
            # but we should not loose population
            assert np.isclose(mesh.POBTOT.sum(), mesh_c.POBTOT.sum()), (
                cve_met,
                year,
            )
            assert np.isclose(agebs.POBTOT.sum(), mesh_c.POBTOT.sum()), (
                cve_met,
                year,
                agebs.POBTOT.sum(),
                mesh_c.POBTOT.sum(),
                agebs.POBTOT.sum() - mesh_c.POBTOT.sum(),
            )
            mesh = mesh_c.assign(
                AREA_URB=lambda x: x.area / 1e6,
                DENS_URB=lambda x: x.POBTOT / x.AREA_URB,
                PWDENSITY_URB=lambda x: x.DENS_URB * x.POBTOT,
                YEAR=int(year),
                CVE_MET=cve_met,
            ).rename(columns={"POBTOT": "POB_URB"})

            poly_list.append(mesh)
    mesh_gdf = (
        pd.concat(poly_list).set_index(["CVE_MET", "YEAR", "codigo"]).sort_index()
    )

    return mesh_gdf


def aggregate_mesh(mesh_gdf, met_xlsx_path, opath=None):
    """Aggregates mesh data into a zone level table.

    Parameters
    ----------
    mesh_gdf : GeoDataFrame
        Multi-temporal mesh GeoDataFrame, the output of <build_mesh_gdf>
    met_xlsx_path : Path
        Path to official excel file with aggregated information for Mexican
        Metropilitan Zones. Oficial data from CONAPO:
        https://www.gob.mx/conapo/documentos/las-metropolis-de-mexico-2020
    ofile : Path
        If provided, stores GeoDataFrame on disk at <opath>, defaults to None.
    Returns
    -------
    GeoDataFrame
        DataFrame with aggregated urban data indexed by zone code and year.
    """

    metropoli_df = build_zones_df(met_xlsx_path)[["NOM_MET", "TIPO_MET"]].sort_index()

    agg_df = (
        mesh_gdf.assign(NUM_CELLS=1)
        .assign(
            PWDIST=lambda df: df.DIST * df.POB_URB,
        )
        .groupby(["CVE_MET", "YEAR"])
        .agg(
            {
                "POB_URB": "sum",
                "AREA_URB": "sum",
                "PWDENSITY_URB": "sum",
                "NUM_CELLS": "sum",
                "PWDIST": "sum",
                "DIST": "max",
            }
        )
        .rename(
            columns={
                "PWDIST": "DIST_MEAN",
                "DIST": "DIST_MAX",
            }
        )
        .assign(
            DIST_MEAN=lambda x: x.DIST_MEAN / x.POB_URB,
            DENSITY_URB=lambda x: x.POB_URB / x.AREA_URB,
            PWDENSITY_URB=lambda x: x.PWDENSITY_URB / x.POB_URB,
        )[
            [
                "POB_URB",
                "AREA_URB",
                "DENSITY_URB",
                "PWDENSITY_URB",
                "DIST_MEAN",
                "DIST_MAX",
                "NUM_CELLS",
            ]
        ]
        .unstack(level=1)
        .sort_index()
    )
    agg_df.columns = [f"{a}_{b}" for a, b in agg_df.columns]

    per_zone = (
        mesh_gdf.assign(NUM_CELLS=1)
        .groupby(["CVE_MET"])
        .agg(
            {
                "region": "first",
                "centroid_hist": "first",
            }
        )
        .sort_index()
    )

    agg_df = agg_df.join(per_zone, how="left")
    agg_df = metropoli_df.join(agg_df, how="right")

    # Add rank
    agg_df = agg_df.assign(
        rank_2020=lambda df: np.argsort(np.argsort(-df.POB_URB_2020.values)) + 1,
        rank_2010=lambda df: np.argsort(np.argsort(-df.POB_URB_2010.values)) + 1,
        rank_2000=lambda df: np.argsort(np.argsort(-df.POB_URB_2000.values)) + 1,
        rank_1990=lambda df: np.argsort(np.argsort(-df.POB_URB_1990.values)) + 1,
    )

    agg_df = gpd.GeoDataFrame(
        agg_df, geometry="centroid_hist", crs=mesh_gdf.centroid_hist.crs
    )

    if opath is not None:
        agg_df.to_file(opath / "zones_agg_from_mesh.gpkg")

    return agg_df


def build_zones_df(xlsx_path):
    """Generates a DataFrame with information for the Mexican Metropolitan zones.

    Data is gathered from

    - Oficial data from CONAPO
    https://www.gob.mx/conapo/documentos/las-metropolis-de-mexico-2020

    Data is merged from the sheets of the excel file.

    Parameters
    xlsx_path : Path
        Path to excel file Cuadros_MM2020.xlsx

    Returns
    -------
    DataFrame
        DataFrame with the merged data.
    """

    metropoli_zm = pd.read_excel(xlsx_path, sheet_name="92 MET")
    metropoli_cuadro_met = pd.read_excel(xlsx_path, sheet_name="Cuadro A_MET")

    metropoli_df = (
        metropoli_zm.rename(columns={"Clave de metrópoli": "CVE_MET"})
        .set_index("CVE_MET")
        .join(
            metropoli_cuadro_met.rename(columns={"Clave de metrópoli": "CVE_MET"})
            .set_index("CVE_MET")
            .drop(columns=["Tipo de metrópoli", "Nombre de la metrópoli"])
        )
    )

    metropoli_df = (
        metropoli_df.rename(
            columns={
                "Tipo de metrópoli": "TIPO_MET",
                "Nombre de la metrópoli": "NOM_MET",
                "Tasa de crecimiento medio anual 2010- 2020 (%)": "TCMA_URB_2010_2020",
                "Población 1990": "POB_TOT_1990",
                "Población 2000": "POB_TOT_2000",
                "Población 2010": "POB_TOT_2010",
                "Población 2020": "POB_TOT_2020",
                "Producto Interno Bruto": "PIB",
                "Población ocupada": "POB_OCU_2020",
                "Población urbana": "POB_URB_2020",
                "Población rural": "POB_RUR_2020",
                "Densidad media urbana": "PWDENSITY_URB_2020",
            }
        )
        .assign(
            AREA_TOT=lambda x: x["Superficie total (ha)"] / 100,
            AREA_OCU_2020=lambda x: x["Superficie ocupada (ha)"] / 100,
            AREA_URB_2020=lambda x: x["Superficie urbana (ha)"] / 100,
            AREA_RUR_2020=lambda x: x["Superficie rural (ha)"] / 100,
            DENS_OCU_2020=lambda x: x["POB_TOT_2020"] / x["AREA_OCU_2020"],
            DENS_URB_2020=lambda x: x["POB_URB_2020"] / x["AREA_URB_2020"],
            DENS_RUR_2020=lambda x: x["POB_RUR_2020"] / x["AREA_RUR_2020"],
            POB_TOT_1990=lambda x: x["POB_TOT_1990"]
            .replace("N.A.", None)
            .astype(pd.Int64Dtype()),
            TCMA_TOT_1990_2000=lambda x: get_tcma(
                x["POB_TOT_1990"], x["POB_TOT_2000"], 10
            ),
            TCMA_TOT_2000_2010=lambda x: get_tcma(
                x["POB_TOT_2000"], x["POB_TOT_2010"], 10
            ),
            TCMA_TOT_2010_2020=lambda x: get_tcma(
                x["POB_TOT_2010"], x["POB_TOT_2020"], 10
            ),
        )
        .drop(
            columns=[
                "Posición",
                "Superficie total (ha)",
                "Superficie km2",
                "Superficie ocupada (ha)",
                "Superficie urbana (ha)",
                "Superficie rural (ha)",
                "Porcentaje de superficie ocupada respecto al total",
                "Porcentaje de superficie urbana respecto al total",
                "Porcentaje de superficie rural respecto al total",
                "Porcentaje de población urbana de la metrópoli 2020",
                "Porcentaje de población rural de la metrópoli 2020",
                "Población",
                "Densidad de población (hab/ha)",
                "Densidad de población urbana (hab/ha)",
                "Densidad de población rural (hab/ha)",
                "Tasa de crecimiento medio anual 1990-2000",
                "Tasa de crecimiento medio anual 2000-2010",
                "Tasa de crecimiento medio anual 2010-2020",
            ]
        )
    )

    metropoli_df = metropoli_df[
        [
            "NOM_MET",
            "TIPO_MET",
            "AREA_TOT",
            "POB_TOT_1990",
            "POB_TOT_2000",
            "POB_TOT_2010",
            "POB_TOT_2020",
            "POB_URB_2020",
            "AREA_URB_2020",
            "DENS_URB_2020",
            "PWDENSITY_URB_2020",
            "TCMA_TOT_1990_2000",
            "TCMA_TOT_2000_2010",
            "TCMA_TOT_2010_2020",
            "TCMA_URB_2010_2020",
        ]
    ].sort_index()

    return metropoli_df


def get_cent_region(xlsx_path):
    """Load each metropolitan zone centre from file.

    Parameters
    ----------
    xlsx_path : Path
        Path to excel file with center coordinates.

    Returns
    -------
    DataFrame
        DataFrame with center coordinates.
    """
    centroids_df = pd.read_excel(
        xlsx_path,
        sheet_name="final_clean",
        usecols=["City", "region", "Centroide Google Maps/OSM", "Centroide historico"],
    ).rename(
        columns={
            "Centroide Google Maps/OSM": "centroid_osm",
            "Centroide historico": "centroid_hist",
            "City": "NOM_MET",
        }
    )
    centroids_df[["osm_lat", "osm_lon"]] = centroids_df.centroid_osm.str.extract(
        r"\((?P<osm_lat>[+-]?[0-9]*[.]?[0-9]+), (?P<osm_lon>[+-]?[0-9]*[.]?[0-9]+)\)"
    )
    centroids_df[["hist_lon", "hist_lat"]] = centroids_df.centroid_hist.str.extract(
        r"\( (?P<osm_lat>[+-]?[0-9]*[.]?[0-9]+), (?P<osm_lon>[+-]?[0-9]*[.]?[0-9]+)\)"
    )
    # centroids_df["centroid_osm"] = gpd.points_from_xy(
    #     centroids_df.osm_lon, centroids_df.osm_lat, crs=4326
    # )
    centroids_df["centroid_hist"] = gpd.points_from_xy(
        centroids_df.hist_lon, centroids_df.hist_lat, crs=4326
    )
    centroids_df = centroids_df.drop(
        columns=["osm_lat", "osm_lon", "hist_lon", "hist_lat", "centroid_osm"]
    )
    centroids_df = gpd.GeoDataFrame(centroids_df, geometry="centroid_hist").to_crs(
        "ESRI:102008"
    )

    return centroids_df


def get_tcma(x0, x1, num_years):
    """Calculates the mean anual growth rate.

    Parameters
    ----------
    x0 : numeric or array like
        Values at the starting year.
    x1 : numeric or array like
        Values at the end of the interval.
    num_years : int
        Interval lenght in years.

    Returns
    -------
    int or array like
        The mean anual growth rate.
    """
    return ((x1 / x0) ** (1 / num_years) - 1) * 100
