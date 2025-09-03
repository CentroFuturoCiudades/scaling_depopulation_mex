
# Codes of zones to process
CVES = [
    '01.1.01', '02.1.01', '02.2.02', '02.2.03', '03.2.01', '03.2.02', '04.2.01', 
    '05.1.01', '05.1.02', '05.1.03', '05.1.04', '06.1.01', '07.1.01', '07.1.02',
    '08.1.01', '08.1.02', '08.2.03', '09.1.01', '10.2.01', '11.1.01', '11.1.02',
    '11.2.03', '11.2.04', '12.1.01', '12.2.02', '13.1.01', '13.1.02', '14.1.01',
    '14.1.02', '15.1.01', '16.1.01', '16.1.02', '16.1.03', '16.2.04', '17.1.01',
    '17.1.02', '18.1.01', '19.1.01', '20.1.01', '21.1.01', '21.1.02', '21.1.03',
    '22.1.01', '23.1.01', '23.2.02', '24.1.01', '25.2.01', '25.2.02', '25.2.03',
    '26.1.01', '26.2.02', '26.2.03', '26.2.04', '27.1.01', '28.1.01', '28.1.02',
    '28.2.03', '28.2.04', '28.2.05', '29.1.01', '30.1.01', '30.1.02', '30.1.03',
    '30.1.04', '30.1.05', '30.1.06', '30.1.07', '31.1.01', '32.1.01'
]

OUT_DIR = "outputs"
AGEBS_DIR_B = "data/AGEB_DATA_OUT/reprojected/base"
AGEBS_DIR_S = "data/AGEB_DATA_OUT/zone_agebs/shaped"
AGEBS_DIR_T = "data/AGEB_DATA_OUT/zone_agebs/translated"
RFUNCS_DIR = OUT_DIR + "/radial_f"

rule uzip_agebs:
    input:
        "data/AGEB_DATA_OUT.zip"
    output:
        expand(
            "{dir}/{year}/{cve}.gpkg",
             dir=AGEBS_DIR_B, cve=CVES, year=["1990", "2000", "2010", "2020"]
        ),
        expand(
            "{dir}/{year}/{cve}.gpkg",
            dir=AGEBS_DIR_S, cve=CVES, year=["2010", "2020"]
        ),
        expand(
            "{dir}/{year}/{cve}.gpkg",
            dir=AGEBS_DIR_T, cve=CVES, year=["1990", "2000"]
        ),
    run:
        import zipfile
        with zipfile.ZipFile(input[0], 'r') as zip_ref:
            zip_ref.extractall("data/")

rule gen_mesh:
    input:
        "data/Cuadros_MM2020.xlsx",
        "data/centroid_overrides_ER_clean.xlsx",
        rules.uzip_agebs.output
    output:
        f"{OUT_DIR}/mesh.geoparquet"
    run:
        from depopulation import build_mesh_gdf
        from pathlib import Path
        met_xlsx_path = Path(input[0])
        cent_xlsx_path = Path(input[1])
        poly_path = Path(AGEBS_DIR_B)
        ageb_path_shaped = Path(AGEBS_DIR_S)
        ageb_path_trans = Path(AGEBS_DIR_T)
        opath = Path(output[0])
        build_mesh_gdf(
            poly_path, met_xlsx_path, ageb_path_shaped, ageb_path_trans, 
            cent_xlsx_path, opath
        )

rule gen_rfuncs:
    input:
        rules.gen_mesh.output
    output:
        expand(
            "{dir}/radial_functions_{cve}_{sfix}.csv",
            dir=RFUNCS_DIR, cve=CVES, sfix=["rho", "cdf"]
        )
    run:
        from depopulation import build_radial_distributions
        from pathlib import Path
        import geopandas as gpd
        mesh_gdf = gpd.read_parquet(input[0])
        build_radial_distributions(mesh_gdf, Path(RFUNCS_DIR))

rule agg_mesh:
    input:
        rules.gen_mesh.output,
        "data/Cuadros_MM2020.xlsx"
    output:
        f"{OUT_DIR}/zones_agg_from_mesh.gpkg"
    run:
        from depopulation import aggregate_mesh
        from pathlib import Path
        import geopandas as gpd
        mesh_gdf = gpd.read_parquet(input[0])
        met_xlsx_path = Path(input[1])
        agg_all_df = aggregate_mesh(mesh_gdf, met_xlsx_path, Path(OUT_DIR))