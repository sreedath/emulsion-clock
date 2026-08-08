"""Transient figure: optical depth and resolved-water fraction vs time for
brine and DI water at the primary field, from the PBE simulation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from params import BARRIER_KT_BAND, E_FIELDS, T_REL_CLEAR
from pbe_sim import PBEModel
from salinity import BRINE_05M, DI_WATER

E_PRIMARY = E_FIELDS["max_safe_7.5kV_cm"]
COLORS = {"0.5 M brine": "#c2571a", "DI water": "#1a6ec2"}


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for phase in (BRINE_05M, DI_WATER):
        model = PBEModel(E_PRIMARY, phase, BARRIER_KT_BAND[1])
        res = model.run(log_every=30.0)
        hist = np.array(res["history"])
        if len(hist) == 0:
            continue
        t_min = hist[:, 0] / 60.0
        ax1.semilogy(t_min, hist[:, 1], color=COLORS[phase.name],
                     label=phase.name)
        ax2.plot(t_min, 100 * hist[:, 2], color=COLORS[phase.name],
                 label=phase.name)
    ax1.axhline(-np.log(T_REL_CLEAR), ls="--", c="gray", lw=1)
    ax1.text(0.02, -np.log(T_REL_CLEAR) * 1.3, "95% transmittance",
             fontsize=8, color="gray", transform=ax1.get_yaxis_transform())
    ax1.set_xlabel("time [min]")
    ax1.set_ylabel("droplet optical depth (1 cm path)")
    ax1.set_title("Optical clearing, 7.5 kV/cm, 2%, 175 nm")
    ax1.legend()
    ax2.set_xlabel("time [min]")
    ax2.set_ylabel("resolved water [%]")
    ax2.set_title("Bulk separation")
    ax2.legend()
    fig.tight_layout()
    fig.savefig("figures/fig_blind_prediction_transients.png", dpi=160)
    print("saved figures/fig_blind_prediction_transients.png")


if __name__ == "__main__":
    main()
