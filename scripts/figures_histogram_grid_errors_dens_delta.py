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

# Create histograms
bins = [-100000, -10000, -1000, -100, -10, 0, 10, 100, 1000, 10000, 100000]
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
for ax, region in zip(axes, regions):
    sns.histplot(
        agebs[agebs.region == region], x="dens_diff", bins=bins, ax=ax, color="grey"
    )
axes[0].text(8e4, 3000, "Central zone", horizontalalignment="right")
axes[1].text(8e4, 3000, "Intermediate zone", horizontalalignment="right")
axes[2].text(8e4, 3000, "Distant zone", horizontalalignment="right")
axes[3].text(8e4, 3000, "Peri-urban zone", horizontalalignment="right")
# axes[3].set_ylim(0, 4000)
axes[3].set_xscale("symlog", linthresh=10)
# axes[3].set_xlim(0, None)
fig.subplots_adjust(hspace=0)
