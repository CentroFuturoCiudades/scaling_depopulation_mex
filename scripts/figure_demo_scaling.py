from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox

from depopulation.r_scaling import get_quantiles
from depopulation.radial_f import gen_pop_ar, load_radial_f


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """

    def __init__(
        self,
        xy,
        p1,
        p2,
        size=75,
        unit="points",
        ax=None,
        text="",
        textposition="inside",
        text_kw=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(
            self._xydata,
            size,
            size,
            angle=0.0,
            theta1=self.theta1,
            theta2=self.theta2,
            **kwargs,
        )

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(
            ha="center",
            va="center",
            xycoords=IdentityTransform(),
            xytext=(0, 0),
            textcoords="offset points",
            annotation_clip=True,
        )
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.0
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.0
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {
                "max": max(b.width, b.height),
                "min": min(b.width, b.height),
                "width": b.width,
                "height": b.height,
            }
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180], [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":

            def R90(a, r, w, h):
                if a < np.arctan(h / 2 / (r + w / 2)):
                    return np.sqrt((r + w / 2) ** 2 + (np.tan(a) * (r + w / 2)) ** 2)
                else:
                    c = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    T = np.arcsin(c * np.cos(np.pi / 2 - a + np.arcsin(h / 2 / c)) / r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w / 2, h / 2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi / 4)) * ((a % (np.pi / 2)) <= np.pi / 4) + (
                    np.pi / 4 - (a % (np.pi / 4))
                ) * ((a % (np.pi / 2)) >= np.pi / 4)
                return R90(aa, r, *[w, h][:: int(np.sign(np.cos(2 * a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X - s / 2), 0))[0] * 72
            self.text.set_position([offs * np.cos(angle), offs * np.sin(angle)])


font_path = Path("data/Symbol-Signsdata/symbol-signs.otf")
custom_font_prop = fm.FontProperties(fname=font_path)

width_pt = 469.0
inches_per_pt = 1.0 / 72.27
figure_width = width_pt * inches_per_pt
figure_height = figure_width

cve = "06.1.01"
t1 = 1990
t2 = 2020
# Max radius in t2
R = 6

plt.rcParams["font.size"] = 10
fs = 12

# Load radial densities
radial_f = load_radial_f([cve], Path("outputs/radial_f/"), core=True)[cve]

# Load scaling factors
slopes_df = pd.read_csv(
    "../depopulation_mexico_cities/outputs/scaling_factors.csv", index_col=[0, 1, 2]
).loc[cve]

# Load urban scaling factor
phi = slopes_df.query("t1==@t1 & t2==@t2").L.item()

# Load quantiles array
pgrid = np.linspace(0.01, 1, 100)
q1, q2 = get_quantiles(pgrid, [cve], (t1, t2), {cve: radial_f})[0] / 1000
max_idx = np.searchsorted(q2, R)
q1, q2 = q1[:max_idx], q2[:max_idx]

fig, axes = plt.subplot_mosaic(
    """
    QRS
    """,
    figsize=(12, 4),
    # figsize=(figure_width, figure_height),
    layout="constrained",
)

#### Q, plot quantile-quantile plot ####
ax = axes["Q"]
ax.plot(q1, q2, color="grey")

ax.set_xlabel(f"distance to city center s (km)\nQuantiles for $\\rho$ at $t_1={t1}$")
ax.set_ylabel(f"Quantiles for $\\rho$ at $t_2={t2}$\ndistance to city center s (km)")
# Show quartiles
for idx in [24, 49, 74]:
    ax.plot([q1[idx], q1[idx], 0], [0, q2[idx], q2[idx]], color="grey", ls="--")
    ax.text(0.3, q2[idx] + 0.2, f"p={(idx + 1) / 100}", color="grey")
# Add scaling factor
ax.plot(
    [0, q1[-1]],
    [0, phi * q1[-1]],
    color="black",
)
AngleAnnotation(
    [0, 0],
    [q1[10], 0],
    [q1[10], phi * q1[10]],
    ax=ax,
    size=190,
    textposition="inside",
    text=r"$\theta$",
)
ax.text(2.6, 1.2, "$\\Phi=\\tan(\\theta)$")
ax.set_title(
    "Estimation of the Urban Expansion Factor $\\Phi$\nfor Colima from 1990 to 2020.",
    fontsize=10,
)
ax.set_xlim(0, None)
ax.set_ylim(0, None)


######## B, the rho scaling
ax = axes["R"]
s = radial_f["r_ring"] / 1000
rho1 = 1000 * radial_f[f"rho_{t1}"]
rho2 = 1000 * radial_f[f"rho_{t2}"]
ax.plot(s, rho1, color="grey", ls="--", label=r"$\rho(s, t_1)$")
ax.plot(s, rho2, color="tab:purple", ls="-", label=r"$\rho(s, t_2)$")
ax.plot(
    s * phi,
    rho1 / phi,
    color="tab:purple",
    ls="--",
    label=r"$\frac{1}{\Phi}\rho\left(\frac{s}{\Phi}, t_1\right)$",
)  # remoteness is similar to r=s*phi
ax.set_xlabel("distance to city center s (km)")
ax.set_ylabel(r"probability density $\rho$")
# Add maximum reference point to track it
idx_max = np.argmax(rho1)
p1 = (s[idx_max], rho1[idx_max])
p2 = (p1[0] * phi, p1[1] / phi)
ax.scatter([p1[0]], [p1[1]], color="grey")
ax.scatter([p2[0]], [p2[1]], color="tab:purple")

ax.plot([p1[0], 5.8], [p1[1], p1[1]], color="black", ls=":", zorder=0)
ax.plot([p2[0], 5.8], [p2[1], p2[1]], color="black", ls=":", zorder=0)
ax.text(6, p1[1], r"$\rho$")
ax.text(6, p2[1], r"$\frac{\rho}{\Phi}$")
ax.annotate(
    "",
    xytext=(6.15, p1[1] - 0.01),
    xy=(6.15, p2[1] + 0.035),
    arrowprops=dict(arrowstyle="->"),
)

ax.plot([p1[0], p1[0]], [p1[1], 0.54], color="black", ls=":", zorder=0)
ax.plot([p2[0], p2[0]], [p2[1], 0.54], color="black", ls=":", zorder=0)
ax.text(p1[0] - 0.15, 0.55, r"$s$")
ax.text(p2[0] - 0.15, 0.55, r"$\Phi s$")
ax.annotate(
    "",
    xytext=(p1[0] + 0.1, 0.56),
    xy=(p2[0] - 0.05, 0.56),
    arrowprops=dict(arrowstyle="->"),
)

ax.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncols=3,
    mode="expand",
    borderaxespad=0.0,
)


style = "Simple, tail_width=0.5, head_width=4, head_length=5"
kw = dict(arrowstyle=style, color="grey")
arrow = patches.FancyArrowPatch(
    (3.3, 0.02), (4.0, 0.05), connectionstyle="arc3,rad=.2", **kw
)
ax.add_patch(arrow)
kw = dict(arrowstyle=style, color="tab:purple")
arrow = patches.FancyArrowPatch(
    (5.4, 0.15), (7, 0.1), connectionstyle="arc3,rad=.1", **kw
)
ax.add_patch(arrow)
ax.text(0.6, 0.01, "Colima 1990", color="grey")
ax.text(7, 0.09, "Colima 2020", color="tab:purple")

ax.set_xlim(0, 10)
ax.set_ylim(0, 0.6)


######## C, the sigma scaling
s = radial_f["r_ring"] / 1000
sigma1 = 1e6 * radial_f[f"sigma_{t1}"]
sigma2 = 1e6 * radial_f[f"sigma_{t2}"]
N1, _, _, N2 = gen_pop_ar([cve], Path("outputs/radial_f/"))[0]

ax = axes["S"]
ax.plot(s, sigma1, color="grey", ls="--", label=r"$\sigma(s, t_1)$")
ax.plot(s, sigma2, color="tab:purple", ls="-", label=r"$\sigma(s, t_2)$")
ax.plot(
    s * phi,
    N2 / N1 * sigma1 / phi**2,
    color="tab:purple",
    ls="--",
    label=r"$\frac{P(t_2)}{P(t_1)}\frac{1}{\Phi^2}\sigma\left(\frac{s}{\Phi}, t_1\right)$",
)
ax.set_xlabel("distance to city center s (km)")
ax.set_ylabel(r"population density $\sigma$ (people/km$^2$)")

ax.text(1, 6000, r"$\Delta\sigma$")
ax.annotate("", xytext=(1, 7800), xy=(1, 4600), arrowprops=dict(arrowstyle="<->"))


ax.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncols=3,
    mode="expand",
    borderaxespad=0.0,
)
ax.set_xlim(0, 10)
ax.set_ylim(0, None)


max_sigma = max(sigma1.max(), sigma2.max())
min_sigma = min(sigma1[sigma1 > 0].min(), sigma2[sigma2 > 0].min())
min_inc = 0.15
max_inc = 1


def inc(s):
    b = (np.log(max_inc) - np.log(min_inc)) / max_sigma
    return max_inc * np.exp(-b * s)


px = [0.1]
while px[-1] < 10:
    sigma = sigma1[np.argmin(np.abs(s - px[-1]))]
    px.append(px[-1] + inc(sigma))
px = [x for x in px if x < 7]
for x in px:
    ax.text(x, 700, "M", fontproperties=custom_font_prop, size=18, color="grey")

px = [0.1]
while px[-1] < 10:
    sigma = 1.1 * sigma2[np.argmin(np.abs(s - px[-1]))]
    px.append(px[-1] + inc(sigma))
px = [x for x in px if x < 7]
for x in px:
    ax.text(x, 100, "M", fontproperties=custom_font_prop, size=18, color="tab:purple")

style = "Simple, tail_width=0.5, head_width=4, head_length=5"
kw = dict(arrowstyle=style, color="grey")
arrow = patches.FancyArrowPatch(
    (6, 6500), (1.6, 7000), connectionstyle="arc3,rad=.2", **kw
)
ax.add_patch(arrow)
kw = dict(arrowstyle=style, color="tab:purple")
arrow = patches.FancyArrowPatch(
    (6, 4000), (4.2, 3000), connectionstyle="arc3,rad=-.1", **kw
)
ax.add_patch(arrow)
ax.text(6, 6300, "Colima 1990", color="grey")
ax.text(6, 4000, "Colima 2020", color="tab:purple")


plt.savefig("figures/demo_scaling.pdf")
# plt.savefig("figures/demo_scaling.svg")
plt.close()
