import json
from pathlib import Path

# Codes of zones to process
with open('data/cve_code_names.json', 'r') as f:
        cve_dict = json.load(f)
CVES  = list(cve_dict.keys())
NAMES = list(cve_dict.values())

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
        import geopandas as gpd
        mesh_gdf = gpd.read_parquet(input[0])
        met_xlsx_path = Path(input[1])
        agg_all_df = aggregate_mesh(mesh_gdf, met_xlsx_path, Path(OUT_DIR))

rule table_agg_mesh_pretty:
    input:
        rules.agg_mesh.output
    output:
        f"{OUT_DIR}/zones_agg_from_mesh.csv",
        f"{OUT_DIR}/zones_agg_from_mesh.tex",
        f"{OUT_DIR}/zones_centers.tex"
    script:
        "scripts/table_agg_data.py"

rule gen_rem_brackets:
    input:
        rules.gen_rfuncs.output
    output:
        f"{OUT_DIR}/remoteness_brackets.csv"
    run:
        from depopulation.radial_f import get_remoteness_brackets
        get_remoteness_brackets(CVES, NAMES, Path(RFUNCS_DIR), Path(OUT_DIR))

rule gen_rem_brackets_pop:
    input:
        rules.gen_rfuncs.output
    output:
        f"{OUT_DIR}/remoteness_brackets_pop.csv"
    run:
        from depopulation.radial_f import gen_rem_brackets_pop
        gen_rem_brackets_pop(CVES, Path(RFUNCS_DIR), Path(OUT_DIR))

rule agg_rem_brackets:
    input:
        rules.gen_rem_brackets_pop.output
    output:
        f"{OUT_DIR}/rem_brackets_agg.csv"
    run:
        from depopulation.radial_f import agg_rem_brackets
        agg_rem_brackets(Path(input[0]), Path(OUT_DIR))