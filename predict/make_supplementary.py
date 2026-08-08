"""Supplementary robustness figure (2x2, PRF style): barrier band,
endpoint-threshold stiffness, grid convergence, onset-offset profile.
Values computed by verify.py / pbe_sim.py runs documented in the repo."""

import matplotlib.pyplot as plt

import prf_style as st

OUTS = ["/home/ubuntu/agents/electrostatic demulsification/paper-src/figs/"]


def main():
    st.apply()
    fig, axs = plt.subplots(2, 2, figsize=(st.FULL_W, 4.6))

    ax = axs[0, 0]
    dg = [20, 25, 30]
    ax.plot(dg, [78.2, 110.1, 198.0], "o-", color=st.RED, lw=1.0,
            label="0.5 M brine")
    ax.plot(dg, [76.6, 90.4, 118.7], "s-", color=st.BLUE, lw=1.0,
            label="DI water")
    ax.set_xlabel(r"film barrier $\Delta G$ ($k_BT$)")
    ax.set_ylabel(r"$t_{95}$ (min)")
    ax.set_xticks(dg)
    ax.legend()
    st.panel_label(ax, "a")

    ax = axs[0, 1]
    ax.plot([94, 95, 96], [80.6, 110.1, 160.6], "o-", color=st.RED, lw=1.0)
    ax.set_xlabel(r"transmittance threshold (%)")
    ax.set_ylabel(r"$t$ (min)")
    ax.set_xticks([94, 95, 96])
    st.panel_label(ax, "b")

    ax = axs[1, 0]
    ax.plot([123, 165, 185, 246], [110.1, 112.9, 112.5, 114.3], "o-",
            color=st.BLUE, lw=1.0)
    ax.set_xlabel("number of size bins")
    ax.set_ylabel(r"$t_{95}$ (min)")
    ax.set_ylim(100, 125)
    st.panel_label(ax, "c")

    ax = axs[1, 1]
    ax.plot([0, 1, 2, 4, 6.3], [5.2, 8.9, 15.6, 33.0, 57.5], "o-",
            color=st.RED, lw=1.0)
    ax.set_xlabel(r"assumed onset offset $V_0$ (kV)")
    ax.set_ylabel(r"RMS spread of $t\,\Lambda$ (%)")
    st.panel_label(ax, "d")

    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    for out in OUTS:
        fig.savefig(out + "fig_supp.png", bbox_inches="tight")
    print("saved fig_supp.png")


if __name__ == "__main__":
    main()


def fig_optics():
    """S: why optical clearing is the hard observable (Mie physics)."""
    import numpy as np
    import prf_style as st
    from optics import mie_qext_qsca
    from pbe_sim import PBEModel
    from params import BARRIER_KT_BAND, E_FIELDS, PATH_LENGTH
    from salinity import BRINE_05M

    st.apply()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(st.FULL_W, 2.8))

    radii = np.geomspace(30e-9, 300e-6, 160)
    per_vol = []
    for a in radii:
        _, qs = mie_qext_qsca(a)
        per_vol.append(3.0 * qs / (4.0 * a))       # turbidity per unit
    per_vol = np.array(per_vol)                    # water volume (1/m)
    ax1.loglog(radii * 1e6, per_vol, color=st.BLUE, lw=1.2)
    ax1.axvspan(0.03, 0.0875, color=st.FILL_GRAY, lw=0)
    a0 = 0.0875
    k0 = int(np.argmin(abs(radii * 1e6 - a0)))
    ax1.plot([a0], [per_vol[k0]], "o", color=st.RED, ms=5)
    ax1.annotate("initial emulsion\n($a=87.5$ nm)",
                 (a0, per_vol[k0]), (0.24, 2.2e6), fontsize=7.5,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=st.GRAY))
    ax1.annotate("residual fog:\nstill milky",
                 (0.05, 6e5), (0.032, 2.3e4), fontsize=7.5, color=st.GRAY)
    kmax = int(np.argmax(per_vol))
    ax1.annotate("Mie peak near the\nwavelength of light",
                 (radii[kmax] * 1e6, per_vol[kmax]),
                 (1.6, 3.5e6), fontsize=7.5,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=st.GRAY))
    ax1.annotate("coarse drops:\n$10^{3}\\times$ less scattering\nper unit water",
                 (60, per_vol[-25]), (6, 2.5e3), fontsize=7.5,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=st.GRAY))
    ax1.set_xlabel(r"droplet radius $a$ ($\mu$m)")
    ax1.set_ylabel(r"scattering per unit water volume (m$^{-1}$)")
    st.panel_label(ax1, "a")

    model = PBEModel(E_FIELDS["max_safe_7.5kV_cm"], BRINE_05M,
                     BARRIER_KT_BAND[1])
    res = model.run(snapshot_times=[0.0, 30.0, 600.0])
    labels = [r"$t=0$", r"$30$ s", r"$10$ min"]
    colors = ["#a9c6dd", st.BLUE, st.RED]
    for (t, n), c, lab in zip(res["snapshots"], colors, labels):
        dod = n * model.csca * PATH_LENGTH * 2.303   # per ln a
        ax2.semilogx(model.radii * 1e6, dod, color=c, lw=1.1, label=lab)
    ax2.set_xlim(0.03, 400)
    ax2.set_xlabel(r"droplet radius $a$ ($\mu$m)")
    ax2.set_ylabel(r"optical-depth density $d(\tau L)/d\ln a$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 60)
    ax2.legend(loc="upper right")
    ax2.annotate("all remaining opacity\nlives in the fog",
                 (0.06, 0.25), (0.35, 1.5e-2), fontsize=7.5, color=st.RED,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=st.RED))
    st.panel_label(ax2, "b")
    fig.tight_layout(w_pad=2.0)
    for out in OUTS:
        fig.savefig(out + "fig_supp_optics.png", bbox_inches="tight")
    print("saved fig_supp_optics.png")


def fig_ladder():
    """S: anatomy of the cascade clock (per-doubling time vs size)."""
    import numpy as np
    import prf_style as st
    from kinetics import (NEAR_CONTACT_AMP, brownian_kernel,
                          coalescence_probability, field_enhancement,
                          gravitational_kernel, pair_contact_energy_kt,
                          settling_velocity)
    from params import BARRIER_KT_BAND, CELL_DEPTH, E_FIELDS, PHI_WATER
    from salinity import BRINE_05M, delta_rho

    st.apply()
    import matplotlib.pyplot as plt
    e_field = E_FIELDS["max_safe_7.5kV_cm"]
    dg = BARRIER_KT_BAND[1] + BRINE_05M.barrier_increment_kt
    drho = delta_rho(BRINE_05M)
    radii = np.geomspace(0.09e-6, 200e-6, 220)
    tau_b, tau_s, tau_tot, tau_settle = [], [], [], []
    for a in radii:
        n_a = PHI_WATER / ((4 / 3) * np.pi * a ** 3)
        lam = float(pair_contact_energy_kt(a, a, e_field))
        p = float(coalescence_probability(NEAR_CONTACT_AMP * lam, dg))
        kb = float(brownian_kernel(a, a)) * float(field_enhancement(lam)) * p
        kg = float(gravitational_kernel(a, 0.5 * a, e_field, drho)) * p
        tau_b.append(2 / (kb * n_a) if kb > 0 else np.inf)
        tau_s.append(2 / (kg * n_a) if kg > 0 else np.inf)
        tau_tot.append(2 / ((kb + kg) * n_a) if kb + kg > 0 else np.inf)
        tau_settle.append(0.5 * CELL_DEPTH / float(settling_velocity(a, drho)))
    fig, ax = plt.subplots(figsize=(st.FULL_W, 3.0))
    ax.loglog(radii * 1e6, tau_b, color=st.BLUE, lw=1.0, ls="--",
              label="Brownian capture alone")
    ax.loglog(radii * 1e6, tau_s, color=st.GREEN, lw=1.0, ls="--",
              label="gravitational sweep alone")
    ax.loglog(radii * 1e6, tau_tot, color=st.INK, lw=1.4,
              label="doubling time (both channels)")
    ax.loglog(radii * 1e6, tau_settle, color=st.RED, lw=1.0,
              label=r"settling time $H/2v_{\mathrm{St}}$")
    tt = np.array(tau_tot); ts = np.array(tau_settle)
    kx = int(np.argmin(abs(np.log(tt[100:] / ts[100:]))) + 100)
    ax.plot([radii[kx] * 1e6], [tt[kx]], "o", color=st.RED, ms=5, zorder=5)
    ax.annotate("gravity takes over:\nhandoff radius",
                (radii[kx] * 1e6, tt[kx]), (30, 3e3), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=st.GRAY))
    kb_ = int(np.argmax(tt[:140]))
    ax.annotate("the bottleneck:\nBrownian motion fading,\nsweep not yet awake",
                (radii[kb_] * 1e6, tt[kb_]), (0.13, 1.2e4), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=st.GRAY))
    ax.set_xlim(0.09, 200)
    ax.set_ylim(1e-2, 1e5)
    ax.set_xlabel(r"droplet radius $a$ ($\mu$m)")
    ax.set_ylabel(r"time (s)")
    ax.legend(loc="lower left", ncols=2)
    fig.tight_layout()
    for out in OUTS:
        fig.savefig(out + "fig_supp_ladder.png", bbox_inches="tight")
    print("saved fig_supp_ladder.png")


def fig_pair_tutorial():
    """S: droplet-droplet electrocoalescence, step by step for a beginner."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    import prf_style as st
    from make_schematics_prf import arrow, blank, droplet

    st.apply()
    fig, axs = plt.subplots(1, 5, figsize=(st.FULL_W, 3.1))
    for ax in axs:
        blank(ax, (0, 10), (0, 13))

    def brush(ax, x, y, r, n=14):
        for k in range(n):
            th = 2 * np.pi * k / n
            ax.plot([x + r * np.cos(th), x + (r + 0.45) * np.cos(th)],
                    [y + r * np.sin(th), y + (r + 0.45) * np.sin(th)],
                    color=st.GRAY, lw=0.5)

    def fieldlines(ax):
        for xx in np.linspace(1.2, 8.8, 5):
            ax.plot([xx, xx], [2.6, 11.6], color="#c9d4e2", lw=0.5,
                    ls=(0, (2, 3)), zorder=1)

    def dipole(ax, x, y, r):
        arrow(ax, (x, y - r * 0.45), (x, y + r * 0.45), color=st.INK,
              lw=0.8, ms=6)
        ax.text(x + r * 0.55, y + r * 0.55, "$+$", fontsize=7)
        ax.text(x + r * 0.55, y - r * 0.8, "$-$", fontsize=7)

    # 1. stable without field
    ax = axs[0]
    for (x, y) in ((5, 9.3), (5, 5.0)):
        droplet(ax, x, y, 1.5)
        brush(ax, x, y, 1.5)
    for (x0, y0, x1, y1) in ((3.0, 7.6, 3.9, 8.3), (7.0, 6.6, 6.2, 6.0)):
        arrow(ax, (x0, y0), (x1, y1), color=st.GRAY, lw=0.6, ms=5)
    ax.text(5, 1.9, "no field: surfactant\nfilms + Brownian kicks;\n"
            "stable for months", fontsize=7, ha="center", va="top")

    # 2. field polarizes
    ax = axs[1]
    fieldlines(ax)
    for (x, y) in ((5, 9.3), (5, 5.0)):
        droplet(ax, x, y, 1.5)
        dipole(ax, x, y, 1.5)
    ax.text(5, 1.9, "field on: each drop\nbecomes an induced\ndipole in ns",
            fontsize=7, ha="center", va="top")

    # 3. dipoles attract and chain
    ax = axs[2]
    fieldlines(ax)
    droplet(ax, 5, 9.4, 1.5)
    droplet(ax, 5, 4.9, 1.5)
    arrow(ax, (5, 7.75), (5, 7.35), lw=1.0, ms=8)
    arrow(ax, (5, 6.55), (5, 6.95), lw=1.0, ms=8)
    ax.text(5, 1.9, "aligned dipoles attract:\n"
            r"bond $U\sim\Lambda\,k_BT$" "\nmust beat thermal noise",
            fontsize=7, ha="center", va="top")

    # 4. film drainage over the barrier (with energy inset)
    ax = axs[3]
    fieldlines(ax)
    droplet(ax, 5, 7.0, 1.5)
    droplet(ax, 5, 3.8, 1.5)
    ax.plot([3.9, 6.1], [5.4, 5.4], color=st.INK, lw=0.6)
    arrow(ax, (2.6, 6.3), (3.7, 5.6), lw=0.7)
    arrow(ax, (7.4, 4.55), (6.3, 5.2), lw=0.7)
    ax.text(6.6, 5.75, "$h$", fontsize=7.5)
    # energy landscape inset
    ins = ax.inset_axes([0.09, 0.68, 0.82, 0.235])
    hh = np.linspace(0, 1, 100)
    barrier = 1.15 * np.exp(-((hh - 0.42) / 0.16) ** 2)
    ins.plot(hh, barrier, color=st.INK, lw=0.9)
    ins.plot(hh, barrier - 0.5 * (1 - hh), color=st.RED, lw=0.9, ls="--")
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_linewidth(0.5)
    ins.set_xlabel(r"film $h\ \rightarrow$", fontsize=5.5, labelpad=1)
    ins.set_ylabel("energy", fontsize=5.5, labelpad=1)
    ins.set_ylim(-0.6, 1.62)
    ins.text(0.44, 1.30, r"$\Delta G$", fontsize=6, ha="center")
    ins.text(0.97, 0.42, "field tilts\nthe barrier", fontsize=5.2,
             color=st.RED, ha="right")
    ax.text(5, 1.9, "the oil film drains;\nthe dipole squeeze tilts\n"
            "the barrier (rate-limiting)", fontsize=7, ha="center", va="top")

    # 5. rupture and merger
    ax = axs[4]
    fieldlines(ax)
    w = 1.10
    ax.add_patch(mp.Circle((0, 0), 0, fill=False))
    ell = mp.Ellipse((5, 6.75), 2 * 1.9 * w, 2 * 1.9 / w,
                     fc="#e3ecf5", ec=st.INK, lw=0.7)
    ax.add_patch(ell)
    hg = mp.Ellipse((4.35, 7.45), 0.75, 0.42, angle=-30,
                    fc="white", alpha=0.55, ec="none")
    ax.add_patch(hg)
    ax.text(5, 1.9, "film ruptures in ns;\nvolumes add, the merged\n"
            "drop rings down", fontsize=7, ha="center", va="top")

    for i, ax in enumerate(axs):
        ax.text(0.4, 12.6, f"({'abcde'[i]})", fontsize=8.5)
    fig.tight_layout(w_pad=0.4)
    for out in OUTS:
        fig.savefig(out + "fig_supp_pair.png", bbox_inches="tight")
    print("saved fig_supp_pair.png")


def fig_clearing_tutorial():
    """S: how the whole emulsion clears, as a beaker timeline."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    import prf_style as st
    from make_schematics_prf import blank

    st.apply()
    fig = plt.figure(figsize=(st.FULL_W, 3.3))
    gs = fig.add_gridspec(2, 5, height_ratios=[3.1, 0.9], hspace=0.30)
    times = [r"$t=0$", r"seconds", r"$\sim$1 min", r"$\sim$10 min",
             r"$\sim$2 h"]
    notes = ["opaque, $T\\approx0\\%$\nnanodroplet fog",
             "chains form,\nfirst merges",
             "drops rain out,\nlayer grows",
             "bulk water resolved\n(mass clock)",
             "optically clear,\n$T=95\\%$ (optical clock)"]
    milky = [0.88, 0.72, 0.45, 0.18, 0.03]
    layer = [0.0, 0.05, 0.13, 0.20, 0.225]
    rng = np.random.default_rng(11)
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        blank(ax, (0, 10), (0, 12))
        ax.plot([1.2, 1.2], [1.0, 10.6], color=st.INK, lw=0.9)
        ax.plot([8.8, 8.8], [1.0, 10.6], color=st.INK, lw=0.9)
        ax.plot([1.2, 8.8], [1.0, 1.0], color=st.INK, lw=0.9)
        emu_top = 9.6
        lh = layer[i] * 8.6
        ax.add_patch(mp.Rectangle((1.2, 1.0 + lh), 7.6,
                                  emu_top - 1.0 - lh,
                                  fc="#b9c6d2", alpha=milky[i], ec="none"))
        if lh > 0:
            ax.add_patch(mp.Rectangle((1.2, 1.0), 7.6, lh, fc="#9cbcd8",
                                      ec="none"))
            ax.plot([1.2, 8.8], [1.0 + lh, 1.0 + lh], color=st.INK, lw=0.5)
        ax.plot([1.2, 8.8], [emu_top, emu_top], color=st.INK, lw=0.5)
        n_dots = [70, 46, 22, 10, 0][i]
        r_dot = [0.07, 0.11, 0.2, 0.1, 0][i]
        for _ in range(n_dots):
            x = rng.uniform(1.5, 8.5)
            y = rng.uniform(1.4 + lh, emu_top - 0.4)
            ax.add_patch(mp.Circle((x, y), r_dot * rng.uniform(0.7, 1.5),
                                   fc="#46769c", ec="none", alpha=0.8))
        if i == 2:
            for xx in (3.2, 6.4):
                ax.annotate("", (xx, 2.2 + lh), (xx, 4.6),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7,
                                            color=st.RED))
        ax.text(5, 11.4, times[i], fontsize=8.5, ha="center")
        ax.text(5, 0.15, notes[i], fontsize=6.8, ha="center", va="top")

    axt = fig.add_subplot(gs[1, :])
    axt.set_xscale("log")
    axt.set_xlim(0.05, 200)
    axt.set_ylim(0, 1)
    axt.set_yticks([])
    axt.spines[["left", "right", "top"]].set_visible(False)
    axt.tick_params(top=False, which="both")
    axt.set_xticks([0.1, 1, 10, 60, 120])
    axt.set_xticklabels(["0.1", "1", "10", "60", "120"])
    axt.set_xlabel("time (min)", fontsize=8)
    axt.plot([0.05, 3.5], [0.62, 0.62], color=st.INK, lw=2.2,
             solid_capstyle="butt")
    axt.text(0.45, 0.83, "mass clock: bulk water separates", fontsize=7)
    axt.plot([0.05, 110], [0.22, 0.22], color=st.RED, lw=2.2,
             solid_capstyle="butt")
    axt.text(9, 0.38, "optical clock: the fog is ground away", fontsize=7,
             color=st.RED)
    for out in OUTS:
        fig.savefig(out + "fig_supp_clearing.png", bbox_inches="tight")
    print("saved fig_supp_clearing.png")
