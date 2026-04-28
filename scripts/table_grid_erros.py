import geopandas as gpd
import pandas as pd

agebs = gpd.read_file("outputs/agebs_ghs.gpkg").assign(
    e_class=lambda df: pd.Categorical(
        df.e_class, ["GU", "U", "SU", "AE", "SO", "O", "GO"], ordered=True
    ),
    region=lambda df: pd.Categorical(
        df.region, ["central", "mid", "peri", "outskirts"], ordered=True
    ),
)

mae = agebs["abs_diff"].mean()
me = agebs["diff"].mean()
mre = agebs["re_diff"].mean()
mare = agebs["re_abs_diff"].mean()

print(f"Mean error: {me:0.2f}, MAE: {mae:0.2f}, MRE: {mre:0.2f}, MARE: {mare:0.2f}")

print(
    agebs.groupby("region", observed=True)[
        ["diff", "abs_diff", "re_diff", "re_abs_diff"]
    ]
    .agg(["median", "mean", "std"])
    .sort_index()
)
