import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

agebs = gpd.read_file("outputs/agebs_ghs.gpkg").assign(
    e_class=lambda df: pd.Categorical(
        df.e_class, ["GU", "U", "SU", "AE", "SO", "O", "GO"], ordered=True
    ),
    region=lambda df: pd.Categorical(
        df.region, ["central", "mid", "peri", "outskirts"], ordered=True
    ),
)

agebs_2 = agebs.query("remoteness <= 12").dropna().sort_values("remoteness")
xarr = agebs_2["remoteness"].to_numpy()
epsilon = agebs_2["diff"].to_numpy()
abs_epsilon = agebs_2["abs_diff"].to_numpy()
eta = agebs_2["re_diff"].to_numpy()
abs_eta = agebs_2["re_abs_diff"].to_numpy()

frac = 0.2
epsilon_mean = sm.nonparametric.lowess(exog=xarr, endog=epsilon, frac=0.2)[:, 1]
abs_epsilon_mean = sm.nonparametric.lowess(exog=xarr, endog=abs_epsilon, frac=0.2)[:, 1]
eta_mean = sm.nonparametric.lowess(exog=xarr, endog=eta, frac=0.2)[:, 1]
abs_eta_mean = sm.nonparametric.lowess(exog=xarr, endog=abs_eta, frac=0.2)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))

axes[0].plot(xarr, epsilon_mean, label=r"$\left<\epsilon\right>$")
axes[0].plot(xarr, abs_epsilon_mean, label=r"$\left<\left|\epsilon\right|\right>$")
axes[0].legend()

axes[1].plot(xarr, eta_mean, label=r"$\left<\eta\right>$")
axes[1].plot(xarr, abs_eta_mean, label=r"$\left<\left|\eta\right|\right>$")
axes[1].set_xlabel("remoteness r")
axes[1].set_ylabel("relative error")
axes[1].legend()
