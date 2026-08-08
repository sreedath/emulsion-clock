"""All data figures in the shared PRF style (see prf_style.py).

Outputs (paper-src/figs/ and site assets/): fig_transients, fig_cascade,
fig_lambda, fig_map.
"""

import matplotlib.pyplot as plt
import numpy as np

import prf_style as st
from params import BARRIER_KT_BAND, E_FIELDS, EPS0, EPSR_OIL, T_REL_CLEAR, kT
from pbe_sim import PBEModel
from salinity import BRINE_05M, DI_WATER

OUTS = ["/home/ubuntu/agents/electrostatic demulsification/paper-src/figs/",
        "/home/ubuntu/agents/electrostatic demulsification/deploy/"
        "demulsification-prediction/assets/"]
E_PRIMARY = E_FIELDS["max_safe_7.5kV_cm"]
DG = BARRIER_KT_BAND[1]


def save(fig, name):
    for out in OUTS:
        fig.savefig(out + name, bbox_inches="tight")
    print("saved", name)


def run_models():
    snaps = [0.0, 3.0, 8.0, 15.0, 30.0, 600.0]
    brine = PBEModel(E_PRIMARY, BRINE_05M, DG).run(
        log_every=30.0, snapshot_times=snaps)
    di = PBEModel(E_PRIMARY, DI_WATER, DG).run(log_every=30.0)
    return brine, di


def fig_transients(brine, di):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(st.FULL_W, 2.6))
    for res, color, name in ((brine, st.RED, "0.5 M brine"),
                             (di, st.BLUE, "DI water")):
        h = np.array(res["history"])
        ax1.semilogy(h[:, 0] / 60, h[:, 1], color=color, lw=1.1, label=name)
        ax2.plot(h[:, 0] / 60, 100 * h[:, 2], color=color, lw=1.1,
                 label=name)
    thr = -np.log(T_REL_CLEAR)
    ax1.axhline(thr, ls="--", c=st.INK, lw=0.6)
    ax1.text(4, thr * 1.30, r"$T_{\mathrm{rel}}=95\%$", fontsize=8)
    ax1.set_xlabel(r"$t$ (min)")
    ax1.set_ylabel(r"optical depth $\tau L$")
    ax1.set_xlim(0, 115)
    ax1.legend(loc="upper right")
    st.panel_label(ax1, "a")
    ax2.set_xlabel(r"$t$ (min)")
    ax2.set_ylabel(r"resolved water (%)")
    ax2.set_xlim(0, 115)
    ax2.set_ylim(88, 100.5)
    ax2.legend(loc="lower right")
    st.panel_label(ax2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_transients.png")


def fig_cascade(brine):
    fig, ax = plt.subplots(figsize=(st.FULL_W, 2.7))
    model = PBEModel(E_PRIMARY, BRINE_05M, DG)
    radii = model.radii
    grow = brine["snapshots"][:5]
    late = brine["snapshots"][5]
    blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(grow)))
    labels = [r"$t=0$", r"$3\,$s", r"$8\,$s", r"$15\,$s", r"$30\,$s"]
    for (t, n), c, lab in zip(grow, blues, labels):
        dv = n * model.masses * radii
        if dv.max() <= 0:
            continue
        dv = dv / dv.max()
        ax.plot(radii * 1e6, dv, color=c, lw=1.0)
        k = int(np.argmax(dv))
        ax.annotate(lab, (radii[k] * 1e6, 1.03), color=st.INK, fontsize=8,
                    ha="center", annotation_clip=False)
    dv = late[1] * model.masses * radii
    if dv.max() > 0:
        ax.plot(radii * 1e6, dv / dv.max(), color=st.RED, lw=0.9, ls="--")
        ax.annotate(r"$10\,$min: residual fog + last suspended drops",
                    (0.4, 0.55), color=st.RED, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 400)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel(r"droplet radius $a$ ($\mu$m)")
    ax.set_ylabel(r"$dV/d\ln a$ (normalized)")
    fig.tight_layout()
    save(fig, "fig_cascade.png")


def fig_lambda():
    volts = np.array([7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5])
    times = np.array([95.0, 50.0, 30.0, 22.0, 17.0, 13.0, 11.0])
    d_gap, a = 5.5e-2, 100e-9
    lam = (np.pi * EPS0 * EPSR_OIL * (volts * 1e3 / d_gap) ** 2 * a ** 3
           / kT())
    c_fit = float(np.mean(times * lam))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(st.FULL_W, 2.9), gridspec_kw={"width_ratios": [3, 2]})
    xs = np.geomspace(0.1, 12, 100)
    ax1.plot(xs, c_fit / xs, color=st.RED, lw=0.8,
             label=r"$t=23.5\,\mathrm{min}/\Lambda$")
    ax1.plot(lam, times, "o", color=st.BLUE, ms=4.5, mew=0,
             label="batch sweep (digitized)")
    ax1.plot([7.76], [7.5], "s", mfc="none", mec=st.GREEN, ms=5, mew=1.0,
             label="flow cell")
    ax1.plot([7.76], [60.0], "x", color=st.INK, ms=6, mew=1.2,
             label="batch reference")
    ax1.fill_between([6.5, 9.3], 90, 125, color=st.FILL_GRAY, lw=0)
    ax1.text(7.8, 103, "predicted\n90–125", fontsize=7, ha="center")
    ax1.axvspan(0.1, 0.2, color=st.FILL_GRAY, lw=0)
    ax1.text(0.142, 4.5, "below\nonset", fontsize=7, ha="center")
    ax1.axvline(0.8, color=st.RED, lw=0.6, ls=":")
    ax1.text(0.86, 190, r"$\Lambda^{*}$", fontsize=9, color=st.RED)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.1, 12)
    ax1.set_ylim(2, 300)
    ax1.set_xlabel(r"$\Lambda$")
    ax1.set_ylabel(r"$t_{95}$ (min)")
    ax1.legend(loc="upper right", handlelength=1.4)
    st.panel_label(ax1, "a")

    ax2.axhspan(c_fit * 0.948, c_fit * 1.052, color=st.FILL_BLUE, lw=0)
    ax2.plot(lam, times * lam, "o", color=st.BLUE, ms=4.5, mew=0)
    ax2.axhline(c_fit, color=st.RED, lw=0.8, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlim(0.2, 6)
    ax2.set_ylim(0, 40)
    ax2.set_xlabel(r"$\Lambda$")
    ax2.set_ylabel(r"$t_{95}\,\Lambda$ (min)")
    ax2.text(0.25, c_fit + 2.5, r"$23.5$ min, 5% RMS", fontsize=8)
    st.panel_label(ax2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_lambda.png")


def fig_map():
    lam_ref, d_ref, e_ref = 5.223, 175.0, 7.5

    def ecrit(d, ls):
        return e_ref * np.sqrt(ls / lam_ref) * (d_ref / d) ** 1.5

    d = np.geomspace(100, 1000, 200)
    fig, ax = plt.subplots(figsize=(st.COL_W, 2.9))
    ax.fill_between(d, ecrit(d, 0.3), ecrit(d, 2.0), color=st.FILL_RED,
                    lw=0, zorder=1)
    ax.plot(d, ecrit(d, 0.8), color=st.RED, lw=0.9, zorder=2)
    for dd, ee, ok in [(175, 7.5, 1), (175, 4.1, 1), (175, 1.36, 1),
                       (175, 1.0, 0), (600, 2.4, 1)]:
        ax.plot([dd], [ee], "o", ms=5,
                mfc=st.INK if ok else "white", mec=st.INK, mew=0.8, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(100, 1000)
    ax.set_ylim(0.4, 10)
    ax.set_xticks([100, 200, 400, 800])
    ax.set_xticklabels([100, 200, 400, 800])
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_yticklabels([0.5, 1, 2, 4, 8])
    ax.set_xlabel(r"droplet diameter $2a$ (nm)")
    ax.set_ylabel(r"$E$ (kV/cm)")
    ax.text(430, 6.3, "clears", fontsize=9)
    ax.text(115, 0.55, "stable", fontsize=9)
    ax.text(122, 2.9, r"$\Lambda^{*}$, $E\sim a^{-3/2}$", fontsize=8,
            color=st.RED, rotation=-33)
    fig.tight_layout()
    save(fig, "fig_map.png")


if __name__ == "__main__":
    st.apply()
    brine, di = run_models()
    fig_transients(brine, di)
    fig_cascade(brine)
    fig_lambda()
    fig_map()
