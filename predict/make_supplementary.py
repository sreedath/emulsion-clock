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
