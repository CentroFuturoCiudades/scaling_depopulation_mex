import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout="constrained")

ax.set_xlim(0.1, 10)
ax.set_ylim(0.1, 10)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([0.1, 1, 10], [0.1, 1, 10], fontsize=12)
ax.set_yticks([0.1, 1, 10], [0.1, 1, 10], fontsize=12)

ax.set_xlabel(r"Population growth $P(t_j)/P(t_i)$ (log scale)", fontsize=12)
ax.set_ylabel(r"Urban expansion factor $\Phi_{ij}$ (log scale)", fontsize=12)
ax.axhline(1, ls="-", color="black")
ax.axvline(1, ls="-", color="black")
ax.axline((1, 1), (1.4, np.sqrt(1.4)), color="k", ls="-")
ax.text(
    0.6,
    np.sqrt(0.6) + 0.15,
    r"$\Phi_{ij}=\sqrt{P(t_j)/P(t_i)}$",
    fontsize=12,
    va="top",
    ha="center",
    rotation=np.arctan(0.5) / np.pi * 180 + 0,
)

xgrid = np.linspace(0, 10, 1000)

ax.fill_between(xgrid, 0, np.sqrt(xgrid), alpha=0.5, color="grey")

ax.text(1.1, 0.2, "Density gain", fontsize=14)

ax.text(0.2, 3, "Density loss", fontsize=14)

e_adt = 1.6

ax.axline((1, e_adt), (1.5, 1.5**0.5 * e_adt), color="red", ls="-")
ax.axline((1, e_adt), (1.5, 1.5**0.9 * e_adt), color="red", ls="-")
ax.axline((1, e_adt), (1.5, 1.5**0.25 * e_adt), color="red", ls="-")

ax.axline((1, 1 / e_adt), (1.5, 1.5**0.5 * 1 / e_adt), color="red", ls="--")

ax.text(1.1, 1.35, r"$e^{\alpha \Delta t}$", color="red", fontsize=12)
bracket = FancyArrowPatch(
    (1.05, 1.28), (1.1, 1.28), arrowstyle="]-", mutation_scale=20, color="red"
)
ax.add_patch(bracket)

ax.text(
    3,
    3**0.5 * e_adt + 0.9,
    r"$\beta=0.5$",
    color="red",
    fontsize=12,
    va="top",
    ha="center",
    rotation=np.arctan(0.5) / np.pi * 180 + 0,
)
ax.text(
    3.3,
    3.3**0.5 * e_adt - 1.4,
    r"$\beta=0.5$",
    color="red",
    fontsize=12,
    va="top",
    ha="center",
    rotation=np.arctan(0.5) / np.pi * 180 + 0,
)
ax.text(
    3,
    3**0.9 * e_adt + 1.7,
    r"$\beta>0.5$",
    color="red",
    fontsize=12,
    va="top",
    ha="center",
    rotation=np.arctan(0.9) / np.pi * 180 + 0,
)
ax.text(
    3.5,
    3.5**0.25 * e_adt + 0.5,
    r"$\beta<0.5$",
    color="red",
    fontsize=12,
    va="top",
    ha="center",
    rotation=np.arctan(0.25) / np.pi * 180 + 0,
)

mid_x1 = np.exp((np.log(1) + np.log(0.1)) / 2)
mid_x2 = np.exp((np.log(10) + np.log(1)) / 2)

bracket = FancyArrowPatch(
    (mid_x1, 10.3),
    (mid_x1, 11),
    arrowstyle="]-",
    mutation_scale=109,
    mutation_aspect=0.1,
    color="black",
    clip_on=False,
)
ax.add_patch(bracket)
ax.text(mid_x1, 11, "Population loss", ha="center")
bracket = FancyArrowPatch(
    (mid_x2, 10.3),
    (mid_x2, 11),
    arrowstyle="]-",
    mutation_scale=109,
    mutation_aspect=0.1,
    color="black",
    clip_on=False,
)
ax.add_patch(bracket)
ax.text(mid_x2, 11, "Population growth", ha="center")

bracket = FancyArrowPatch(
    (10.3, mid_x1),
    (11, mid_x1),
    arrowstyle="]-",
    mutation_scale=10,
    mutation_aspect=10.8,
    color="black",
    clip_on=False,
)
ax.add_patch(bracket)
ax.text(11, mid_x1, "Urban compression", va="center", rotation=-90)
bracket = FancyArrowPatch(
    (10.3, mid_x2),
    (11, mid_x2),
    arrowstyle="]-",
    mutation_scale=10,
    mutation_aspect=10.8,
    color="black",
    clip_on=False,
)
ax.text(11, mid_x2, "Urban expansion", va="center", rotation=-90)
ax.add_patch(bracket)

e_adt = 1.6
adt = np.log(e_adt)

# Point A, we enter expansion region from below
# b*log(g) + a*Dt = 0 -> log(g) = ( - a*Dt)/0.5
ax.scatter([1 / np.exp((-adt) / 0.5)], [1], color="red")
ax.text(1 / np.exp((-adt) / 0.5), 1 - 0.15, "A", color="red", fontsize=14)

# Point B, we leave/enter the densification region
# b*log(g) + a*Dt = 0.5*log(g) -> log(g) = (- a*Dt)/(b - 0.5)
ax.scatter(
    [np.exp((-adt) / (0.9 - 0.5))],
    [np.exp((-adt) / (0.9 - 0.5)) ** 0.9 * e_adt],
    color="red",
)
ax.text(
    np.exp((-adt) / (0.9 - 0.5)),
    np.exp((-adt) / (0.9 - 0.5)) ** 0.9 * e_adt + 0.06,
    "B",
    color="red",
    fontsize=14,
)

ax.scatter(
    [np.exp((-adt) / (0.25 - 0.5))],
    [np.exp((-adt) / (0.25 - 0.5)) ** 0.25 * e_adt],
    color="red",
)
ax.text(
    np.exp((-adt) / (0.25 - 0.5)),
    np.exp((-adt) / (0.25 - 0.5)) ** 0.25 * e_adt + 0.2,
    "C",
    color="red",
    fontsize=14,
)

plt.savefig("figures/model_dynamics.pdf")
