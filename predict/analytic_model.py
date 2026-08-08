"""Method 1: staged analytic estimate of the demulsification time.

Stages (all closed-form; quadrature over the initial lognormal only):
  A. Brownian gate: dipole contact energy lambda(a) = pi eps K^2 E^2 a^3 / kT
     must exceed ~1 somewhere in the distribution or nothing happens.
  B. Ignition: barrier-limited early coalescence. Encounter rate x Boltzmann
     success averaged over the initial size distribution. Ends when drops
     reach the scavenging size a_act where lambda-type pair energy beats the
     barrier (large-small contact energy is 8x the equal-pair value, so
     a_act = a(dG/8)).
  C. Growth ladder: barrier open (P = 1); transport-limited coarsening with
     capture-enhanced Brownian kernel K = K_B * 1.12 lambda^(1/3). Per-octave
     time scales as a^2, so the ladder is dominated by the top rungs. Partial
     coalescence multiplies the rung time by 1/(1 - 0.5 p_pc) and stalls
     growth where p_pc -> 1.
  D. Gravity endgame: settle half the cell depth at the handoff size. The
     handoff size minimizes C + D unless the partial-coalescence stall caps it.

Prints a table over (water phase, E field, barrier dG).
"""

import numpy as np

from kinetics import (
    NEAR_CONTACT_AMP,
    brownian_kernel,
    coalescence_probability,
    field_enhancement,
    gravitational_kernel,
    pair_contact_energy_kt,
    settling_velocity,
)
from params import (
    BARRIER_KT_BAND,
    CELL_DEPTH,
    DROP_DIAM_MEDIAN,
    E_FIELDS,
    LOGNORM_SIGMA_G,
    PHI_WATER,
)
from salinity import BRINE_05M, DI_WATER, delta_rho, partial_coalescence_prob

A_MEDIAN = DROP_DIAM_MEDIAN / 2.0
STALL_P = 0.95          # partial-coalescence probability treated as a stall
OCTAVE_SUM = 2.70       # sum of geometric ladder a^2 weights, top-rung units
N_QUAD = 240


def lognormal_number_weights(n_pts: int = N_QUAD):
    """Radii and number weights for the initial distribution at fixed phi."""
    ln_sig = np.log(LOGNORM_SIGMA_G)
    z = np.linspace(-3.5, 3.5, n_pts)
    radii = A_MEDIAN * np.exp(z * ln_sig)
    pdf = np.exp(-0.5 * z ** 2)
    weights = pdf / pdf.sum()
    vol_mean = np.sum(weights * (4.0 / 3.0) * np.pi * radii ** 3)
    n_total = PHI_WATER / vol_mean
    return radii, weights * n_total


def ignition_time(e_field: float, barrier_kt: float) -> float:
    """Stage B: mean time for a drop to suffer its first barrier crossing."""
    radii, n_w = lognormal_number_weights()
    ai = radii[:, None]
    aj = radii[None, :]
    lam = pair_contact_energy_kt(ai, aj, e_field)
    kern = (
        brownian_kernel(ai, aj)
        * field_enhancement(lam)
        * coalescence_probability(NEAR_CONTACT_AMP * lam, barrier_kt)
    )
    # per-drop coalescence frequency, number-weighted over partners and self
    freq_per_drop = kern @ n_w
    mean_freq = np.sum(freq_per_drop * n_w) / n_w.sum()
    if mean_freq <= 0:
        return np.inf
    return 1.0 / mean_freq


def scavenging_onset_radius(e_field: float, barrier_kt: float) -> float:
    """Radius where a large drop opens the barrier on any partner (8x rule)."""
    lam_eq_median = float(
        pair_contact_energy_kt(A_MEDIAN, A_MEDIAN, e_field)
    )
    if lam_eq_median <= 0:
        return np.inf
    return A_MEDIAN * (
        barrier_kt / (8.0 * NEAR_CONTACT_AMP * lam_eq_median)
    ) ** (1.0 / 3.0)


def octave_time(a: float, e_field: float, phase, barrier_kt: float) -> float:
    """Mean-field time for one volume doubling at size a. Transport is
    Brownian capture plus sweep coagulation (evaluated for the natural
    a vs a/2 polydispersity of a coarsening population)."""
    n_a = PHI_WATER / ((4.0 / 3.0) * np.pi * a ** 3)
    lam = float(pair_contact_energy_kt(a, a, e_field))
    transport = float(brownian_kernel(a, a)) * float(field_enhancement(lam))
    transport += float(
        gravitational_kernel(a, 0.5 * a, e_field, delta_rho(phase)))
    kern = transport * float(
        coalescence_probability(NEAR_CONTACT_AMP * lam, barrier_kt))
    p_pc = float(partial_coalescence_prob(a, e_field, phase))
    if p_pc >= STALL_P:
        return np.inf
    eff = 1.0 - 0.5 * p_pc
    t_equal_pair = 2.0 / (kern * n_a * eff)

    # scavenging-accretion channel: a seed of size a eats median fines; the
    # large-small contact energy (8x rule) opens the barrier much earlier
    lam_sc = float(pair_contact_energy_kt(a, A_MEDIAN, e_field))
    k_sc = (
        float(brownian_kernel(a, A_MEDIAN))
        * float(field_enhancement(lam_sc))
        + float(gravitational_kernel(a, A_MEDIAN, e_field, delta_rho(phase)))
    ) * float(coalescence_probability(NEAR_CONTACT_AMP * lam_sc, barrier_kt))
    v_a = (4.0 / 3.0) * np.pi * a ** 3
    if k_sc > 0:
        t_scavenge = np.log(2.0) * v_a / (k_sc * PHI_WATER * eff)
    else:
        t_scavenge = np.inf
    return min(t_equal_pair, t_scavenge)


def growth_plus_settle(e_field: float, phase, barrier_kt: float):
    """Stages C+D: sum the octave ladder explicitly from the starting size
    and pick the handoff radius minimizing ladder + settling. (The ladder is
    top-dominated in the Brownian regime but bottom-dominated once sweep
    coagulation takes over, so no single-rung shortcut is valid.)"""
    drho = delta_rho(phase)
    rungs = np.geomspace(A_MEDIAN, 300e-6, 121)
    t_oct = np.array([
        octave_time(a, e_field, phase, barrier_kt) for a in rungs])
    # a volume doubling advances radius by 2^(1/3); spread each doubling's
    # dwell time over the grid rungs it spans
    rungs_per_octave = np.log(2.0) / (3.0 * np.log(rungs[1] / rungs[0]))
    dwell = t_oct / rungs_per_octave
    cum_growth = np.cumsum(dwell)
    t_settle = 0.5 * CELL_DEPTH / settling_velocity(rungs, drho)
    total = cum_growth + t_settle
    k = int(np.nanargmin(np.where(np.isfinite(total), total, np.inf)))
    return total[k], rungs[k], cum_growth[k], t_settle[k]


def predict(e_field: float, phase, barrier_kt: float) -> dict:
    barrier_kt = barrier_kt + phase.barrier_increment_kt
    lam0 = float(pair_contact_energy_kt(A_MEDIAN, A_MEDIAN, e_field))
    t_ign = ignition_time(e_field, barrier_kt)
    a_act = scavenging_onset_radius(e_field, barrier_kt)
    total_cd, a_handoff, t_growth, t_settle = growth_plus_settle(
        e_field, phase, barrier_kt
    )
    t_total = t_ign + total_cd
    return {
        "lambda_median": lam0,
        "t_ignition_s": t_ign,
        "a_scavenge_um": a_act * 1e6,
        "a_handoff_um": a_handoff * 1e6,
        "t_growth_s": t_growth,
        "t_settle_s": t_settle,
        "t_total_s": t_total,
    }


def fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "never"
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def main() -> None:
    print("Analytic staged model, blind prediction")
    print(f"scenario: d0 = {DROP_DIAM_MEDIAN*1e9:.0f} nm (median), "
          f"sigma_g = {LOGNORM_SIGMA_G}, phi = {PHI_WATER:.0%}, "
          f"cell depth = {CELL_DEPTH*100:.1f} cm")
    header = (
        f"{'phase':<12} {'E [kV/cm]':>9} {'dG [kT]':>7} {'lam0':>6} "
        f"{'t_ign':>9} {'a_hand':>8} {'t_grow':>9} {'t_settle':>9} "
        f"{'TOTAL':>9}"
    )
    print(header)
    print("-" * len(header))
    for phase in (BRINE_05M, DI_WATER):
        for e_name, e_field in E_FIELDS.items():
            for dg in BARRIER_KT_BAND:
                r = predict(e_field, phase, dg)
                print(
                    f"{phase.name:<12} {e_field/1e5:>9.1f} {dg:>7.0f} "
                    f"{r['lambda_median']:>6.1f} "
                    f"{fmt_time(r['t_ignition_s']):>9} "
                    f"{r['a_handoff_um']:>7.1f}u "
                    f"{fmt_time(r['t_growth_s']):>9} "
                    f"{fmt_time(r['t_settle_s']):>9} "
                    f"{fmt_time(r['t_total_s']):>9}"
                )
        print()


if __name__ == "__main__":
    main()
