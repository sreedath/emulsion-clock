"""Blind-prediction scenario parameters.

Every number here is SETUP information from Panat et al., Sci. Adv. 11 (2025)
eadz6233 (materials, geometry, device characterization) or standard literature
property data. NO measured demulsification time or t(V) fit from the paper is
used anywhere in this package. The one calibration-like input is the paper's
statement that the emulsion shows "no visible phase separation for several
months" WITHOUT a field, which is a property of the starting emulsion, not a
result of the demulsification experiment; it pins the zero-field coalescence
barrier (see kinetics.barrier_from_shelf_life).
"""

from dataclasses import dataclass

# Physical constants
KB = 1.380649e-23          # J/K
EPS0 = 8.8541878128e-12    # F/m
G_GRAV = 9.81              # m/s^2

# Materials (literature values for the paper's stated materials)
T_KELVIN = 298.0
MU_OIL = 3.03e-3           # Pa s, hexadecane at 25 C
RHO_OIL = 773.0            # kg/m^3, hexadecane
RHO_WATER = 1017.0         # kg/m^3, 0.5 M NaCl brine
EPSR_OIL = 2.05            # hexadecane relative permittivity
N_OIL = 1.434              # refractive index, hexadecane
N_WATER = 1.338            # refractive index, 0.5 M brine
SIGMA_OIL = 2.0e-8         # S/m, hexadecane + 1% Span 80 (paper, from lit.)
GAMMA_OW = 4.0e-3          # N/m, brine/hexadecane with Span 80 (lit. 3-5 mN/m)
CM_FACTOR = 1.0            # Clausius-Mossotti factor, conductor-in-oil limit

# Optical readout (paper's definition of demulsification)
LAMBDA_VAC = 550e-9        # m, mid-visible
PATH_LENGTH = 1.0e-2       # m, 1 cm cuvette (paper's DLS transmittance)
T_REL_CLEAR = 0.95         # relative transmittance defining "demulsified"

# Cell geometry (75 ml cubic beaker, 60 ml emulsion)
BEAKER_SIDE = 0.75e-4 ** (1.0 / 3.0) * 1e-0  # placeholder, set below
_SIDE = (75e-6) ** (1.0 / 3.0)               # m, cube side = 4.217 cm
CELL_AREA = _SIDE ** 2                        # m^2
FILL_VOLUME = 60e-6                           # m^3
CELL_DEPTH = FILL_VOLUME / CELL_AREA          # m, = 3.37 cm

# Emulsion scenario requested by the user: 150-200 nm, low water fraction
DROP_DIAM_MEDIAN = 175e-9  # m
LOGNORM_SIGMA_G = 1.35     # geometric std dev (DLS polydispersity, PDI ~ 0.2)
PHI_WATER = 0.02           # 2% v/v, the fraction that yields 150-200 nm drops

# Field scenarios in the oil (V/m). 7.5 kV/cm is the paper's max safe
# operating field; 2.4 and 1.4 kV/cm bracket the 12 kV / 5 cm batch condition
# under the two field-split readings (V/d vs corona-onset-corrected).
E_FIELDS = {
    "max_safe_7.5kV_cm": 7.5e5,
    "batch_Vd_2.4kV_cm": 2.4e5,
    "batch_corona_1.4kV_cm": 1.4e5,
}
E_PRIMARY = E_FIELDS["max_safe_7.5kV_cm"]

# Coalescence barrier band (kT units). Central value from the shelf-life
# bound computed in kinetics.py; band reflects Span 80 steric-barrier
# literature (roughly 20-30 kT for months-stable W/O nanoemulsions).
BARRIER_KT_BAND = (20.0, 25.0, 30.0)

SHELF_LIFE_S = 3 * 30 * 86400.0   # "several months" stability, no field


def kT() -> float:
    return KB * T_KELVIN


def validate() -> None:
    checks = [
        0.0 < PHI_WATER < 0.5,
        50e-9 < DROP_DIAM_MEDIAN < 1e-6,
        1.0 < LOGNORM_SIGMA_G < 2.0,
        all(e > 0 for e in E_FIELDS.values()),
        CELL_DEPTH > 0.01,
    ]
    if not all(checks):
        raise ValueError("scenario parameters out of physical range")


validate()
