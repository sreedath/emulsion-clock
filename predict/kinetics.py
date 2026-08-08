"""Pair-level coalescence physics shared by the predictors.

Model chain (all first-principles, no fit to the paper's kinetics):
  1. Induced-dipole pair energy at contact for conductor drops in oil.
  2. Transport kernel: Brownian (Smoluchowski) collision rate enhanced by
     dipolar capture (Fuchs / capture-radius factor ~ lambda^(1/3)).
  3. Coalescence probability per encounter: Boltzmann barrier crossing.
     The zero-field barrier height is bounded from the emulsion's stated
     shelf life (months with no field), which is setup information.
  4. Stokes settling for gravity removal of coarsened drops.
"""

import numpy as np

from params import (
    CM_FACTOR,
    EPS0,
    EPSR_OIL,
    G_GRAV,
    MU_OIL,
    PHI_WATER,
    RHO_OIL,
    RHO_WATER,
    SHELF_LIFE_S,
    kT,
)

EPS_OIL = EPSR_OIL * EPS0
FUCHS_C = 1.12  # 1/Gamma(4/3), capture-radius enhancement prefactor

# Near-contact amplification of the conductor-pair attraction over the
# point-dipole law (Davis 1964 exact two-sphere solution: the gap field is
# strongly amplified for conducting drops). Applies to the barrier tilt at
# contact, NOT to the far-field capture radius. Central 4, plausible 2-8.
NEAR_CONTACT_AMP = 4.0


def pair_contact_energy_kt(a_i, a_j, e_field: float):
    """|U| at contact, in kT, for aligned induced dipoles (conductor limit).

    U(s) = -8 pi eps_oil K^2 E^2 a_i^3 a_j^3 / s^3, evaluated at s = a_i+a_j.
    Equal spheres reduce to the standard pi eps_oil a^3 (KE)^2.
    """
    a_i = np.asarray(a_i, dtype=float)
    a_j = np.asarray(a_j, dtype=float)
    num = 8.0 * np.pi * EPS_OIL * (CM_FACTOR * e_field) ** 2 * a_i ** 3 * a_j ** 3
    return num / ((a_i + a_j) ** 3 * kT())


def brownian_kernel(a_i, a_j):
    """Smoluchowski Brownian collision kernel (m^3/s)."""
    a_i = np.asarray(a_i, dtype=float)
    a_j = np.asarray(a_j, dtype=float)
    return (2.0 * kT() / (3.0 * MU_OIL)) * (a_i + a_j) * (1.0 / a_i + 1.0 / a_j)


def field_enhancement(lambda_kt):
    """Transport enhancement from dipolar capture: max(1, 1.12 lambda^(1/3))."""
    lam = np.asarray(lambda_kt, dtype=float)
    return np.maximum(1.0, FUCHS_C * np.cbrt(np.maximum(lam, 0.0)))


def coalescence_probability(lambda_kt, barrier_kt: float):
    """P(merge | encounter): exp(lambda - barrier), capped at 1.

    The dipole contact energy tilts the film-drainage barrier; when the
    dipolar well is deeper than the barrier the encounter always merges.
    """
    lam = np.asarray(lambda_kt, dtype=float)
    return np.exp(np.minimum(lam - barrier_kt, 0.0))


def gravitational_kernel(a_i, a_j, e_field: float, drho: float):
    """Sweep coagulation: a settling drop collects partners within the
    dipolar capture radius. Two bounds apply jointly:
      energy:   attraction must beat kT out to R_c
                (R_e = lambda^(1/3) * contact radius);
      kinetics: the partner must drift in before the collector passes
                (drift time r^5/5A < transit 2r/dv => R_k = (10A/dv)^(1/4),
                 A = 4 eps K^2 E^2 a_i^3 a_j^3 / (mu a_small), from the
                 point-dipole force with Stokes mobility of the smaller drop).
    """
    a_i = np.asarray(a_i, dtype=float)
    a_j = np.asarray(a_j, dtype=float)
    lam = pair_contact_energy_kt(a_i, a_j, e_field)
    s_con = a_i + a_j
    r_energy = np.cbrt(np.maximum(lam, 1.0)) * s_con
    dv = np.abs(settling_velocity(a_i, drho) - settling_velocity(a_j, drho))
    a_small = np.minimum(a_i, a_j)
    drift_a = (4.0 * EPS_OIL * (CM_FACTOR * e_field) ** 2
               * a_i ** 3 * a_j ** 3 / (MU_OIL * a_small))
    with np.errstate(divide="ignore"):
        r_kinetic = np.where(
            dv > 0, (10.0 * drift_a / np.maximum(dv, 1e-300)) ** 0.25,
            np.inf)
    r_c = np.clip(np.minimum(r_energy, r_kinetic), s_con, 100.0 * s_con)
    return np.pi * r_c ** 2 * dv


def effective_kernel(a_i, a_j, e_field: float, barrier_kt: float,
                     drho: float | None = None):
    """Full pair coalescence kernel (m^3/s): (Brownian-capture + sweep)
    transport, gated by barrier crossing. The barrier tilt uses the
    near-contact-amplified attraction; transport uses the far-field law."""
    lam = pair_contact_energy_kt(a_i, a_j, e_field)
    transport = brownian_kernel(a_i, a_j) * field_enhancement(lam)
    if drho is not None:
        transport = transport + gravitational_kernel(a_i, a_j, e_field, drho)
    return transport * coalescence_probability(
        NEAR_CONTACT_AMP * lam, barrier_kt)


def settling_velocity(a, drho: float | None = None):
    """Stokes settling speed (m/s) of a water drop in hexadecane."""
    a = np.asarray(a, dtype=float)
    d_rho = (RHO_WATER - RHO_OIL) if drho is None else drho
    return 2.0 * d_rho * G_GRAV * a ** 2 / (9.0 * MU_OIL)


def barrier_from_shelf_life(a_median: float) -> float:
    """Lower bound on the zero-field barrier (kT) from months-scale stability.

    With no field the encounter rate is Brownian; stability over the shelf
    life requires the per-encounter probability exp(-dG) to be small enough
    that fewer than ~one coalescence per drop happens: K_B N0 exp(-dG) * t < 1.
    """
    n0 = PHI_WATER / ((4.0 / 3.0) * np.pi * a_median ** 3)
    rate0 = float(brownian_kernel(a_median, a_median)) * n0
    return float(np.log(rate0 * SHELF_LIFE_S))
