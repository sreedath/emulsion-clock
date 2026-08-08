"""Salinity-dependent physics: settling contrast and field-induced
partial/non-coalescence of conducting drops.

Literature basis (no fit to any demulsification measurement):
  - Ristenpart et al., Nature 461 (2009): charged-drop non-coalescence in
    strong fields; requires fast charge relaxation inside the drop.
  - Bird, Ristenpart, Belmonte, Stone, PRL 103 (2009): critical-cone-angle
    criterion; merge aborts when electric stress at the bridge competes with
    capillarity, i.e. electric Bond number Bo_E = eps_oil E_loc^2 a / gamma
    of order 0.1.
  - Mousavichoubeh & Ghadiri; Aryafar & Kavehpour: partial coalescence under
    DC fields ejects secondary droplets, worse at high field, high drop
    conductivity, low interfacial tension.

Model: an encounter that would merge instead goes "partial" with probability
  p_pc = g_sigma * Bo^2 / (Bo^2 + Bo_crit^2),
  g_sigma = tau_bridge / (tau_bridge + tau_relax)  (charge-supply gate),
recycling half of the smaller drop's volume into daughters of half its radius.
"""

from dataclasses import dataclass

import numpy as np

from params import EPS0, EPSR_OIL, GAMMA_OW, RHO_OIL

EPS_OIL_ABS = EPSR_OIL * EPS0
EPSR_WATER = 80.0
BO_CRIT = 0.1          # critical electric Bond number, order-of-magnitude
FIELD_AMPLIFICATION = 3.0  # local field at chain tips / contact vs mean field
DAUGHTER_VOL_FRACTION = 0.5   # of the smaller drop, recycled on partial event
DAUGHTER_RADIUS_RATIO = 0.5   # daughter radius relative to the smaller drop


@dataclass(frozen=True)
class WaterPhase:
    name: str
    conductivity: float       # S/m
    density: float            # kg/m^3
    barrier_increment_kt: float  # added film barrier from salting-out


# Barrier increment: electrolytes strengthen Span-surfactant interfacial
# films (Opawale & Burgess, JCIS 197 (1998) 142): NaCl dehydrates sorbitan
# headgroups, tightening monolayer packing. Order-of-magnitude +4 kT for
# 0.5 M NaCl; an assumption from film-rigidity literature, not a fit.
BRINE_05M = WaterPhase(
    name="0.5 M brine", conductivity=4.6, density=1017.0,
    barrier_increment_kt=4.0,
)
DI_WATER = WaterPhase(
    name="DI water", conductivity=1.0e-5, density=997.0,
    barrier_increment_kt=0.0,
)


def charge_relaxation_time(phase: WaterPhase) -> float:
    return EPSR_WATER * EPS0 / phase.conductivity


RHO_WATER_NOMINAL = 1000.0


def bridge_time(a_small) -> np.ndarray:
    """Inertial-capillary time of the coalescence bridge."""
    a = np.asarray(a_small, dtype=float)
    return np.sqrt(RHO_WATER_NOMINAL * a ** 3 / GAMMA_OW)


def partial_coalescence_prob(a_small, e_field: float, phase: WaterPhase):
    """Probability that a merge event aborts into partial coalescence."""
    a = np.asarray(a_small, dtype=float)
    e_loc = FIELD_AMPLIFICATION * e_field
    bo = EPS_OIL_ABS * e_loc ** 2 * a / GAMMA_OW
    gate = bridge_time(a) / (bridge_time(a) + charge_relaxation_time(phase))
    return gate * bo ** 2 / (bo ** 2 + BO_CRIT ** 2)


def delta_rho(phase: WaterPhase) -> float:
    return phase.density - RHO_OIL
