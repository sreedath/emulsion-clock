"""Site figures in the editorial style: warm paper, serif, restrained color.

1. fig_transients: optical depth + resolved water vs time (PBE).
2. fig_cascade: volume-weighted size distribution marching right (PBE).
3. fig_lambda: measured Fig 4C times vs the electrocoalescence number.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from params import BARRIER_KT_BAND, E_FIELDS, PHI_WATER, T_REL_CLEAR, kT
from params import DROP_DIAM_MEDIAN, EPS0, EPSR_OIL
from pbe_sim import PBEModel
from salinity import BRINE_05M, DI_WATER

PAPER = "#fcfbf7"
INK = "#16130d"
MUTED = "#6d665a"
LINE = "#d9d3c3"
COOL = "#155e8c"
WARM = "#c0641a"
GOOD = "#1c7a55"
OUT = ("/home/ubuntu/agents/electrostatic demulsification/deploy/"
       "demulsification-prediction/assets/")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": LINE,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "font.size": 11,
})

E_PRIMARY = E_FIELDS["max_safe_7.5kV_cm"]
DG = BARRIER_KT_BAND[1]


def run_models():
    snaps = [0.0, 5.0, 15.0, 40.0, 120.0, 600.0]
    brine = PBEModel(E_PRIMARY, BRINE_05M, DG).run(
        log_every=30.0, snapshot_times=snaps)
    di = PBEModel(E_PRIMARY, DI_WATER, DG).run(log_every=30.0)
    return brine, di


def fig_transients(brine, di):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for res, color, name in ((brine, WARM, "0.5 M brine"),
                             (di, COOL, "DI water")):
        h = np.array(res["history"])
        ax1.semilogy(h[:, 0] / 60, h[:, 1], color=color, lw=2, label=name)
        ax2.plot(h[:, 0] / 60, 100 * h[:, 2], color=color, lw=2, label=name)
    thr = -np.log(T_REL_CLEAR)
    ax1.axhline(thr, ls="--", c=MUTED, lw=1)
    ax1.text(2, thr * 1.25, "95% transmittance", fontsize=9, color=MUTED)
    ax1.set_xlabel("time (min)")
    ax1.set_ylabel("droplet optical depth, 1 cm path")
    ax1.set_title("The slow clock: optical clearing", fontsize=12, pad=10)
    ax1.legend(frameon=False)
    ax2.set_xlabel("time (min)")
    ax2.set_ylabel("resolved water (%)")
    ax2.set_ylim(88, 100.5)
    ax2.set_title("The fast clock: bulk separation", fontsize=12, pad=10)
    ax2.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT + "fig_transients.png", dpi=170)


def fig_cascade(brine):
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    model = PBEModel(E_PRIMARY, BRINE_05M, DG)
    radii = model.radii
    shades = ["#a9c2d4", "#7fa5c0", "#5d8bab", "#3f7396", "#255e85", "#155e8c"]
    labels = ["start", "5 s", "15 s", "40 s", "2 min", "10 min"]
    for (t, n), c, lab in zip(brine["snapshots"], shades, labels):
        dv = n * model.masses * radii            # ~ dV/dln r
        dv = dv / (PHI_WATER * 0.35)
        ax.plot(radii * 1e6, dv / dv.max() if dv.max() > 0 else dv,
                color=c, lw=2)
        k = int(np.argmax(dv))
        ax.annotate(lab, (radii[k] * 1e6, 1.02), color=c, fontsize=9.5,
                    ha="center", annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 400)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    ax.set_xlabel("droplet radius (μm)")
    ax.set_ylabel("where the water sits (scaled)")
    ax.set_title("The coalescence cascade: 25 doublings in a few minutes "
                 "(population balance, 7.5 kV/cm)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(OUT + "fig_cascade.png", dpi=170)


def fig_lambda():
    # Fig 4C digitized: applied voltage (kV) vs demulsification time (min);
    # electrode gap 5.5 cm (Methods), field read as V/d (Fig 4B, slope ~1)
    volts = np.array([7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5])
    times = np.array([95.0, 50.0, 30.0, 22.0, 17.0, 13.0, 11.0])
    d_gap = 5.5e-2
    a = 100e-9  # 200 nm mean diameter at 2% water cut
    lam = (np.pi * EPS0 * EPSR_OIL * (volts * 1e3 / d_gap) ** 2 * a ** 3
           / kT())
    c_fit = float(np.mean(times * lam))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    xs = np.geomspace(0.1, 12, 100)
    ax1.plot(xs, c_fit / xs, color=INK, lw=1.4, ls="--",
             label=f"t = {c_fit:.0f} min / Λ")
    ax1.scatter(lam, times, s=55, color=COOL, zorder=5,
                label="batch sweep (Fig. 4C, digitized)")
    ax1.scatter([7.76], [7.5], s=70, facecolor="none", edgecolor=GOOD,
                linewidth=2, zorder=5, label="flow cell, 7.5 kV/cm")
    ax1.axvspan(0.1, 0.2, color=MUTED, alpha=0.15, lw=0)
    ax1.text(0.14, 14, "no clearing\nobserved\n(below corona\nonset)",
             fontsize=8.5, color=MUTED, ha="center")
    ax1.axvline(0.8, color=WARM, lw=1.2, ls=":")
    ax1.text(0.84, 150, "predicted Λ*", fontsize=9, color=WARM,
             rotation=90, va="top")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.1, 12)
    ax1.set_ylim(2, 300)
    ax1.set_xlabel("Λ, electrocoalescence number")
    ax1.set_ylabel("time to 95% transmittance (min)")
    ax1.set_title("Every voltage point on one line", fontsize=12, pad=10)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")

    ax2.axhspan(c_fit * 0.9, c_fit * 1.1, color=COOL, alpha=0.12, lw=0)
    ax2.scatter(lam, times * lam, s=55, color=COOL, zorder=5)
    ax2.axhline(c_fit, color=INK, lw=1.2, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlim(0.2, 6)
    ax2.set_ylim(0, c_fit * 1.6)
    ax2.set_xlabel("Λ")
    ax2.set_ylabel("t × Λ (min)")
    ax2.set_title("The collapse: t·Λ constant to ±10%",
                  fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(OUT + "fig_lambda.png", dpi=170)
    print(f"lambda collapse constant C = {c_fit:.1f} min, "
          f"spread {np.std(times*lam)/c_fit:.1%}")


if __name__ == "__main__":
    brine, di = run_models()
    fig_transients(brine, di)
    fig_cascade(brine)
    fig_lambda()
    print("figures written to", OUT)
