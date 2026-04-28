"""Generates log-log plot of PIB vs Population."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression import linear_model
from statsmodels.tools import add_constant

# PIB scaling
cuadro_met = (
    pd.read_excel(
        "./data/Cuadros_MM2020.xlsx",
        usecols=[
            "Clave de metrópoli",
            "Tipo de metrópoli",
            "Nombre de la metrópoli",
            "Población",
            "Población urbana",
            "Producto Interno Bruto",
        ],
    )
    .rename(
        columns={
            "Clave de metrópoli": "CVE_MET",
            "Tipo de metrópoli": "TIPO",
            "Nombre de la metrópoli": "NOM_MET",
            "Población": "POB_TOT_2020",
            "Población urbana": "POB_URB_2020",
            "Producto Interno Bruto": "PIB",
        }
    )
    .set_index("CVE_MET")
)


fig = plt.subplots(figsize=(10, 10))

plt.plot(cuadro_met.POB_TOT_2020, cuadro_met.PIB, "o")

model = linear_model.OLS(
    np.log(cuadro_met.PIB.values), add_constant(np.log(cuadro_met.POB_TOT_2020.values))
).fit()
C, alpha = model.params
R2 = model.rsquared
print(C, alpha, R2)

plt.plot(
    np.linspace(cuadro_met.POB_TOT_2020.min(), cuadro_met.POB_TOT_2020.max()),
    np.exp(C)
    * np.linspace(cuadro_met.POB_TOT_2020.min(), cuadro_met.POB_TOT_2020.max())
    ** alpha,
    c="red",
    ls="-",
    lw=2,
    label=rf"$\alpha={alpha:0.3f}$",
)

model = linear_model.OLS(
    np.log(cuadro_met.PIB.values) - np.log(cuadro_met.POB_TOT_2020.values),
    np.ones(len(cuadro_met)),
).fit()
C = model.params[0]
R2 = model.rsquared
print(C, R2)

plt.plot(
    np.linspace(cuadro_met.POB_TOT_2020.min(), cuadro_met.POB_TOT_2020.max()),
    np.exp(C)
    * np.linspace(cuadro_met.POB_TOT_2020.min(), cuadro_met.POB_TOT_2020.max()),
    c="black",
    ls="-",
    lw=2,
    label=r"$\alpha=1$",
)

model = linear_model.OLS(
    np.log(cuadro_met.PIB.values) - (7 / 6) * np.log(cuadro_met.POB_TOT_2020.values),
    np.ones(len(cuadro_met)),
).fit()
C = model.params[0]
R2 = model.rsquared
print(C, R2)

plt.plot(
    np.linspace(cuadro_met.POB_TOT_2020.min(), cuadro_met.POB_TOT_2020.max()),
    np.exp(C)
    * np.linspace(cuadro_met.POB_TOT_2020.min(), cuadro_met.POB_TOT_2020.max())
    ** (7 / 6),
    c="orange",
    ls="-",
    lw=2,
    label=r"$\alpha=7/6$",
)

plt.yscale("log")
plt.xscale("log")

plt.xlabel("Población")
plt.ylabel("PIB")

idxs = [0, 1, 2, 5, 10, 17, 19, 24, 49, 51, 66, -4]
for x, y, s in zip(
    cuadro_met.POB_TOT_2020.iloc[idxs],
    cuadro_met.PIB.iloc[idxs],
    cuadro_met.NOM_MET.iloc[idxs],
):
    plt.text(x, y, s)

plt.legend()

plt.savefig("figures/pib_scaling.pdf")
