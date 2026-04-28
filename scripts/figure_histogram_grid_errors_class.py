import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

agebs = gpd.read_file("outputs/agebs_ghs.gpkg").assign(
    e_class=lambda df: pd.Categorical(
        df.e_class, ["GU", "U", "SU", "AE", "SO", "O", "GO"], ordered=True
    ),
    region=lambda df: pd.Categorical(
        df.region, ["central", "mid", "peri", "outskirts"], ordered=True
    ),
)

regions = sorted(agebs.region.unique())

fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
for ax, region in zip(axes, regions):
    sns.histplot(
        agebs[agebs.region == region].dropna(), x="e_class", ax=ax, color="grey"
    )
axes[0].text(6.5, 2500, "Central zone", horizontalalignment="right")
axes[1].text(6.5, 2500, "Intermediate zone", horizontalalignment="right")
axes[2].text(6.5, 2500, "Distant zone", horizontalalignment="right")
axes[3].text(6.5, 2500, "Peri-urban zone", horizontalalignment="right")
axes[3].set_xlabel(r"Relative error class.")
fig.subplots_adjust(hspace=0)
