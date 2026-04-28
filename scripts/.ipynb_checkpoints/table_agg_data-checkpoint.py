"""Produces a pretty table with aggregated data for all zones."""

import geopandas as gpd
import numpy as np
import pandas as pd

agg_all_df = gpd.read_file("outputs/zones_agg_from_mesh.gpkg").set_index("CVE_MET")

table_sup = pd.concat(
    [
        agg_all_df[["NOM_MET", "TIPO_MET"]]
        .set_axis(
            pd.MultiIndex.from_arrays(
                [
                    [
                        "ID",
                        "ID",
                    ],
                    ["Name", "Type"],
                ]
            ),
            axis=1,
        )
        .replace(["Zona metropolitana", "Metrópoli municipal"], ["MA", "MM"]),
        np.round(
            agg_all_df[["POB_URB_1990", "POB_URB_2000", "POB_URB_2010", "POB_URB_2020"]]
            / 1e6,
            2,
        ).set_axis(
            pd.MultiIndex.from_arrays(
                [["Population (millions of people)"] * 4, [1990, 2000, 2010, 2020]]
            ),
            axis=1,
        ),
        np.round(
            agg_all_df[
                ["AREA_URB_1990", "AREA_URB_2000", "AREA_URB_2010", "AREA_URB_2020"]
            ],
            2,
        ).set_axis(
            pd.MultiIndex.from_arrays([["Area (km2)"] * 4, [1990, 2000, 2010, 2020]]),
            axis=1,
        ),
        np.round(
            agg_all_df[
                [
                    "DENSITY_URB_1990",
                    "DENSITY_URB_2000",
                    "DENSITY_URB_2010",
                    "DENSITY_URB_2020",
                ]
            ]
        )
        .astype(int)
        .set_axis(
            pd.MultiIndex.from_arrays(
                [["Density (people/km2)"] * 4, [1990, 2000, 2010, 2020]]
            ),
            axis=1,
        ),
        gpd.GeoSeries(agg_all_df.geometry, crs="ESRI:102008")
        .to_crs("4326")
        .get_coordinates()
        .set_axis(
            pd.MultiIndex.from_arrays(
                [["Center", "Center"], ["longitude", "latitude"]]
            ),
            axis=1,
        ),
    ],
    axis=1,
)
table_sup.index.name = "Code"

# Generate pretty csv
table_sup.to_csv("outputs/zones_agg_from_mesh.csv")

# Generate latex table
with open("outputs/zones_agg_from_mesh.tex", "w", encoding="utf-8") as f:
    table_sup.drop(columns="Center").to_latex(
        f,
        float_format="%.2f",
        longtable=True,
        formatters=[
            lambda x: x,
            lambda x: x,
            lambda x: x,
            lambda x: round(x, 2),
            lambda x: round(x, 2),
            lambda x: round(x, 2),
            lambda x: round(x, 2),
            lambda x: round(x, 2),
            lambda x: round(x, 2),
            lambda x: round(x, 2),
            lambda x: round(x, 2),
            lambda x: x,
            lambda x: x,
            lambda x: x,
            lambda x: x,
        ],
        caption=(
            "Data for Mexican metropolises. "
            "MA: Metropolitan Area. MM: Metropolitan Municipality."
        ),
    )

with open("outputs/zones_centers.tex", "w", encoding="utf-8") as f:
    table_sup[["ID", "Center"]].to_latex(
        f,
        float_format="%.6f",
        longtable=True,
        caption=(
            "Location of urban centers for Mexican metropolises. "
            "MA: Metropolitan Area. MM: Metropolitan Municipality."
        ),
    )
