"""Hand-drawn schematics in the shared PRF style: setup + pair
electrocoalescence + cascade (Fig. 1) and the prediction protocol (Fig. 2).

Thin black line art, STIX serif labels with italic math, red annotation
accents, white background. No icons, no rounded boxes, no color panels.
"""

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

import prf_style as st

OUTS = ["/home/ubuntu/agents/electrostatic demulsification/paper-src/figs/",
        "/home/ubuntu/agents/electrostatic demulsification/deploy/"
        "demulsification-prediction/assets/"]


def save(fig, name):
    for out in OUTS:
        fig.savefig(out + name, bbox_inches="tight")
    print("saved", name)


def blank(ax, xlim=(0, 10), ylim=(0, 10)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def droplet(ax, x, y, r, fc="#e3ecf5", lw=0.7):
    ax.add_patch(mp.Circle((x, y), r, fc=fc, ec=st.INK, lw=lw, zorder=4))


def charges(ax, x, y, r, fs=7):
    ax.text(x, y + 0.62 * r, "$+$", ha="center", va="center", fontsize=fs,
            zorder=6)
    ax.text(x, y - 0.62 * r, "$-$", ha="center", va="center", fontsize=fs,
            zorder=6)


def arrow(ax, xy0, xy1, color=st.RED, lw=0.8, style="-|>", ms=7, ls="-"):
    ax.add_patch(mp.FancyArrowPatch(
        xy0, xy1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
        linestyle=ls, zorder=6, shrinkA=0, shrinkB=0))


def dim_arrow(ax, xy0, xy1, text, offset=(0, 0), fontsize=7.5):
    ax.add_patch(mp.FancyArrowPatch(
        xy0, xy1, arrowstyle="<|-|>", mutation_scale=5, lw=0.6,
        color=st.INK, zorder=6, shrinkA=0, shrinkB=0))
    xm = (xy0[0] + xy1[0]) / 2 + offset[0]
    ym = (xy0[1] + xy1[1]) / 2 + offset[1]
    ax.text(xm, ym, text, fontsize=fontsize, ha="center", va="center")


# ---------------------------------------------------------------- panel (a)
def panel_setup(ax):
    blank(ax, (0, 10), (0, 12))
    # HV bus bar and needle emitters
    ax.plot([1.6, 8.4], [10.8, 10.8], color=st.INK, lw=1.2)
    for x in np.linspace(2.2, 7.8, 5):
        ax.add_patch(mp.Polygon([[x - 0.22, 10.8], [x + 0.22, 10.8],
                                 [x, 10.0]], closed=True, fc=st.INK))
    ax.text(8.7, 10.75, "$+V$", fontsize=9, va="center")
    ax.text(5.0, 11.5, "needle emitter array", fontsize=7.5, ha="center")
    # corona / ion drift in the air gap
    for x in np.linspace(2.4, 7.6, 4):
        arrow(ax, (x, 9.7), (x, 8.6), color=st.GRAY, lw=0.6, ms=5)
    ax.text(9.0, 9.1, "air gap\n(space charge)", fontsize=7, ha="center")
    # open-top emulsion cell: walls only
    ax.plot([1.5, 1.5], [1.9, 7.4], color=st.INK, lw=0.9)
    ax.plot([8.5, 8.5], [1.9, 7.4], color=st.INK, lw=0.9)
    ax.add_patch(mp.Rectangle((1.5, 2.2), 7.0, 4.3, fc="#eef3f8", ec="none"))
    rng = np.random.default_rng(3)
    for _ in range(70):
        x = rng.uniform(1.75, 8.25)
        y = rng.uniform(2.45, 6.25)
        ax.add_patch(mp.Circle((x, y), 0.055, fc=st.BLUE, ec="none",
                               alpha=0.65))
    ax.text(5.0, 0.15, "water-in-oil nanoemulsion "
            r"($2a\simeq175$ nm, $\phi=2\%$)", fontsize=7.5, ha="center")
    # oil surface line
    ax.plot([1.5, 8.5], [6.5, 6.5], color=st.INK, lw=0.6)
    ax.text(7.9, 6.85, "oil surface", fontsize=6.5, ha="center")
    # ground plate
    ax.add_patch(mp.Rectangle((1.5, 1.9), 7.0, 0.3, fc="#d8d8d8",
                              ec=st.INK, lw=0.7))
    for x in np.linspace(1.7, 8.3, 12):
        ax.plot([x, x - 0.3], [1.9, 1.55], color=st.INK, lw=0.5)
    ax.text(5.0, 1.0, "ground plate", fontsize=7.5, ha="center")
    # field arrow in emulsion
    arrow(ax, (4.4, 6.1), (4.4, 3.4), color=st.RED, lw=1.0, ms=8)
    ax.text(4.75, 4.55, r"$E\lesssim7.5$ kV/cm", fontsize=8, color=st.RED)
    # dimensions
    dim_arrow(ax, (0.85, 10.2), (0.85, 6.5), "", offset=(0, 0))
    ax.text(0.42, 8.4, r"$d$", fontsize=8.5)
    dim_arrow(ax, (9.15, 6.5), (9.15, 2.2), "", offset=(0, 0))
    ax.text(9.6, 4.4, r"$H$", fontsize=8.5)


# ---------------------------------------------------------------- panel (b)
def panel_pair(ax):
    blank(ax, (0, 16), (0, 12))
    y0 = 6.2
    # faint vertical field lines, kept clear of the caption zone
    for x in np.linspace(1.2, 15.2, 8):
        ax.plot([x, x], [4.0, 10.6], color="#c9d4e2", lw=0.5, ls=(0, (2, 3)),
                zorder=1)
    ax.text(1.35, 11.35, r"$E$", fontsize=9)
    arrow(ax, (1.75, 11.6), (1.75, 10.7), color=st.INK, lw=0.7, ms=6)

    # stage 1: polarization + attraction
    x1 = 2.2
    y0 = 7.0
    droplet(ax, x1, y0 + 1.6, 0.95)
    droplet(ax, x1, y0 - 1.6, 0.95)
    charges(ax, x1, y0 + 1.6, 0.95)
    charges(ax, x1, y0 - 1.6, 0.95)
    arrow(ax, (x1, y0 + 0.45), (x1, y0), lw=0.9)
    arrow(ax, (x1, y0 - 0.45), (x1, y0), lw=0.9)
    ax.text(x1, 2.85, "polarization,\ndipolar attraction", fontsize=7,
            ha="center", va="top")
    ax.text(x1, 1.35, r"$U\sim\Lambda\,k_BT$", fontsize=7.5, ha="center")

    # stage 2: chaining
    x2 = 6.1
    for dy in (1.95, 0.0, -1.95):
        droplet(ax, x2, y0 + dy, 0.95)
    ax.text(x2, 2.85, "chaining along $E$", fontsize=7, ha="center",
            va="top")

    # stage 3: film drainage (two flattened drops + zoom)
    x3 = 10.2
    droplet(ax, x3, y0 + 1.05, 0.95)
    droplet(ax, x3, y0 - 1.05, 0.95)
    ax.plot([x3 - 0.62, x3 + 0.62], [y0, y0], color=st.INK, lw=0.5)
    arrow(ax, (x3 - 1.5, y0 + 0.35), (x3 - 0.85, y0 + 0.06), lw=0.7)
    arrow(ax, (x3 + 1.5, y0 - 0.35), (x3 + 0.85, y0 - 0.06), lw=0.7)
    # zoom circle
    zx, zy = x3 + 1.7, y0 + 3.1
    ax.add_patch(mp.Circle((zx, zy), 1.0, fill=False, ec=st.GRAY, lw=0.6))
    ax.plot([x3 + 0.45, zx - 0.7], [y0 + 0.1, zy - 0.65], color=st.GRAY,
            lw=0.5)
    ax.plot([zx - 0.75, zx + 0.75], [zy + 0.22, zy + 0.22], color=st.INK,
            lw=0.7)
    ax.plot([zx - 0.75, zx + 0.75], [zy - 0.22, zy - 0.22], color=st.INK,
            lw=0.7)
    ax.text(zx + 1.15, zy, r"$h$", fontsize=7.5, va="center")
    ax.text(x3, 2.85, "oil-film drainage\n"
            r"$\Delta G\!\approx\!20$–$30\,k_BT$"
            "\n(rate-limiting)", fontsize=7, ha="center", va="top")

    # stage 4: coalesced
    x4 = 14.3
    droplet(ax, x4, y0, 1.35)
    ax.text(x4, 2.85, "coalescence\n(volume doubles)", fontsize=7,
            ha="center", va="top")

    for xa, xb in ((3.55, 4.75), (7.5, 8.6), (12.3, 12.75)):
        arrow(ax, (xa, y0), (xb, y0), color=st.INK, lw=0.7, ms=7)


# ---------------------------------------------------------------- panel (c)
def panel_cascade(ax):
    blank(ax, (0, 22), (0, 8))
    rng = np.random.default_rng(5)
    # fog of fines
    for _ in range(90):
        x = rng.uniform(0.6, 3.6)
        y = rng.uniform(2.2, 6.4)
        ax.add_patch(mp.Circle((x, y), 0.05, fc=st.BLUE, ec="none",
                               alpha=0.6))
    ax.text(2.1, 1.15, r"$2a=175$ nm", fontsize=7.5, ha="center")
    # growing sizes
    for x, r in ((5.3, 0.16), (6.4, 0.30), (7.8, 0.55), (9.6, 0.95)):
        droplet(ax, x, 4.3, r)
    ax.text(7.4, 1.15, r"$\sim$25 volume doublings", fontsize=7.5,
            ha="center")
    arrow(ax, (4.1, 4.3), (4.9, 4.3), color=st.INK, lw=0.7, ms=7)
    # sweeping collector with capture cone
    xs, ys = 12.9, 5.3
    cone = mp.Polygon([[xs, ys], [xs - 1.5, ys - 3.6], [xs + 1.5, ys - 3.6]],
                      closed=True, fc=st.FILL_RED, ec="none", zorder=2)
    ax.add_patch(cone)
    droplet(ax, xs, ys, 1.0)
    for dx, dy in ((-0.55, -1.7), (0.4, -2.3), (-0.15, -2.9)):
        ax.add_patch(mp.Circle((xs + dx, ys + dy), 0.09, fc=st.BLUE,
                               ec="none"))
    arrow(ax, (xs, ys - 1.15), (xs, ys - 3.3), color=st.RED, lw=0.9, ms=8)
    ax.text(xs + 1.75, ys - 1.9, r"$v_{\mathrm{St}}\propto a^{2}$",
            fontsize=8, color=st.RED)
    ax.text(xs, 1.15, "gravitational sweep", fontsize=7.5, ha="center")
    # beaker endgame
    bx = 17.2
    ax.add_patch(mp.Rectangle((bx, 2.2), 3.6, 4.4, fill=False, ec=st.INK,
                              lw=0.9))
    ax.add_patch(mp.Rectangle((bx, 2.2), 3.6, 1.1, fc="#c9dcee", ec="none"))
    ax.plot([bx, bx + 3.6], [3.3, 3.3], color=st.INK, lw=0.5)
    for _ in range(12):
        x = rng.uniform(bx + 0.25, bx + 3.35)
        y = rng.uniform(3.6, 6.3)
        ax.add_patch(mp.Circle((x, y), 0.045, fc=st.BLUE, ec="none",
                               alpha=0.5))
    ax.text(bx + 1.8, 1.15, "settled layer + residual fog", fontsize=7.5,
            ha="center")
    # two clocks
    ax.text(1.7, 7.55, "mass clock: 2–5 min", fontsize=7.5)
    ax.plot([1.7, 4.1], [7.15, 7.15], color=st.INK, lw=1.6,
            solid_capstyle="butt")
    ax.text(9.6, 7.55, r"optical clock: $T_{\mathrm{rel}}=95\%$ "
            "after 1.5–2 h (fines-limited)", fontsize=7.5)
    ax.plot([9.6, 20.8], [7.15, 7.15], color=st.RED, lw=1.6,
            solid_capstyle="butt")


def fig1():
    fig = plt.figure(figsize=(st.FULL_W, 5.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.55, 1.0], hspace=0.02,
                          wspace=0.10)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:5])
    ax_c = fig.add_subplot(gs[1, :])
    panel_setup(ax_a)
    panel_pair(ax_b)
    panel_cascade(ax_c)
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        ax.text(0.01, 0.99, f"({letter})", transform=ax.transAxes,
                fontsize=9.5, va="top")
    save(fig, "fig1_mechanism.png")


# ------------------------------------------------------------------- fig 2
def box(ax, x, y, w, h, lines, fontsize=7.2, ec=st.INK, lw=0.7,
        title=None, tfs=8):
    ax.add_patch(mp.Rectangle((x, y), w, h, fc="white", ec=ec, lw=lw,
                              zorder=3))
    lh = 0.52
    block = (0.62 if title else 0) + lh * len(lines)
    ty = y + h / 2 + block / 2
    if title:
        ty -= 0.31
        ax.text(x + w / 2, ty, title, fontsize=tfs, ha="center",
                va="center", weight="bold", zorder=4)
        ty -= 0.62
    else:
        ty -= lh / 2
    for ln in lines:
        ax.text(x + w / 2, ty, ln, fontsize=fontsize, ha="center",
                va="center", zorder=4)
        ty -= lh


def fig2():
    fig, ax = plt.subplots(figsize=(st.FULL_W, 3.0))
    blank(ax, (0, 24.5), (0, 10.5))

    box(ax, 0.3, 3.4, 5.4, 6.6, [
        "material properties",
        "cell geometry ($H=3.4$ cm)",
        r"device $I$–$V$ ($E\leq 7.5$ kV/cm)",
        "DLS size ($2a=175$ nm)",
        r"water cut ($\phi=2\%$)",
        r"shelf life $\rightarrow$ barrier",
        r"$\Delta G\approx20$–$30\,k_BT$",
    ], title="setup inputs", tfs=8.2, fontsize=7.0)

    routes = [
        ("route 1: analytic cascade",
         ["ignition + growth ladder", "+ gravity endgame"]),
        ("route 2: population balance",
         ["Smoluchowski, 123 bins,", "settling + Mie endpoint"]),
        ("route 3: learning agent",
         ["super-droplet Monte Carlo,", "value function = time"]),
    ]
    entry = [7.15, 6.35, 5.55]
    for i, (title, lines) in enumerate(routes):
        y = 7.9 - 2.75 * i
        box(ax, 8.4, y, 6.9, 2.05, lines, title=title, tfs=7.8)
        arrow(ax, (5.7, 6.7), (8.4, y + 1.0), color=st.INK, lw=0.6, ms=6)
        arrow(ax, (15.3, y + 1.0), (17.4, entry[i]), color=st.INK, lw=0.6,
              ms=6)
    ax.text(11.85, 2.05, "three integration schemes, one shared and "
            "independently verified kernel layer", fontsize=6.9,
            ha="center", va="center", style="italic", color=st.GRAY)

    box(ax, 17.4, 4.7, 3.3, 3.2, [
        "optical clearing",
        r"$t_{95}=90$–$125$ min",
        "(range of central",
        "estimates)"], title="prediction", tfs=8.0)
    arrow(ax, (20.7, 6.3), (21.4, 6.3), color=st.INK, lw=0.7, ms=7)
    box(ax, 21.4, 4.7, 2.7, 3.2, [
        r"$\sim$60 min", "measured;", "overpredicts",
        r"by 1.5–2$\times$"], title="reveal", tfs=8.0)

    ax.add_patch(mp.Rectangle((0.5, 0.6), 5.0, 1.8, fill=False, ec=st.RED,
                              lw=0.8, hatch="///"))
    ax.add_patch(mp.Rectangle((1.0, 0.85), 4.0, 1.3, fc="white", ec="none",
                              zorder=5))
    ax.text(3.0, 1.5, "measured kinetics:\nquarantined", fontsize=7.3,
            ha="center", va="center", color=st.RED, zorder=6)
    ax.plot([5.7, 22.75, 22.75], [1.15, 1.15, 4.7], color=st.RED, lw=0.6,
            ls=(0, (4, 3)), zorder=2)
    ax.text(14.1, 0.62, "opened only after predictions are frozen",
            fontsize=6.9, color=st.RED, ha="center")
    save(fig, "fig2_protocol.png")


if __name__ == "__main__":
    st.apply()
    fig1()
    fig2()
