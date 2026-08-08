"""Feasibility-map figure (field vs droplet size, Lambda* boundary band)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#16130d"
MUTED = "#6d665a"
WARM = "#c0641a"
GOOD = "#1c7a55"
LAM_REF = 5.223   # Lambda at 175 nm diameter, 7.5 kV/cm
D_REF = 175.0
E_REF = 7.5

# operating points (diameter nm, field kV/cm, cleared?) from Panat 2025:
# reference batch; top and bottom of the Fig 4C sweep (V/d, d = 5.5 cm);
# conventional immersed-electrode limit; 20% coarse emulsion
POINTS = [(175, 7.5, 1), (175, 4.1, 1), (175, 1.36, 1),
          (175, 1.0, 0), (600, 2.4, 1)]


def ecrit(d, lam_star):
    return E_REF * np.sqrt(lam_star / LAM_REF) * (D_REF / d) ** 1.5


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK})
    d = np.geomspace(100, 1000, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.fill_between(d, ecrit(d, 2.0), 10, color="#e4ede6", zorder=0)
    ax.fill_between(d, 0.4, ecrit(d, 0.3), color="#efece3", zorder=0)
    ax.fill_between(d, ecrit(d, 0.3), ecrit(d, 2.0), color=WARM,
                    alpha=0.15, zorder=1)
    ax.plot(d, ecrit(d, 0.8), color=WARM, lw=2)
    for dd, ee, ok in POINTS:
        ax.scatter([dd], [ee], s=70, facecolor=GOOD if ok else "white",
                   edgecolor=GOOD if ok else MUTED, linewidth=2, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(100, 1000)
    ax.set_ylim(0.4, 10)
    ax.set_xticks([100, 200, 400, 800])
    ax.set_xticklabels([100, 200, 400, 800])
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_yticklabels([0.5, 1, 2, 4, 8])
    ax.set_xlabel("droplet diameter (nm)")
    ax.set_ylabel("field in oil (kV/cm)")
    ax.text(480, 6.0, "clears ($\\Lambda>\\Lambda^*$)", color=GOOD,
            fontsize=11, weight="bold")
    ax.text(112, 0.60, "stable ($\\Lambda<\\Lambda^*$)", color=MUTED,
            fontsize=11, weight="bold")
    ax.text(120, 3.1, "$\\Lambda^*$ boundary, $E\\sim a^{-3/2}$",
            color=WARM, fontsize=9.5, rotation=-28)
    fig.tight_layout()
    fig.savefig("/home/ubuntu/agents/electrostatic demulsification/"
                "paper-src/figs/fig_map.png", dpi=200)
    print("saved fig_map.png")


if __name__ == "__main__":
    main()
