"""Turbidity of the emulsion: Mie scattering by water droplets in hexadecane.

Water in hexadecane is a "soft" scatterer (relative index m = 0.933), so we
provide both a full Mie series (used by the PBE simulation on its fixed grid)
and the van de Hulst anomalous-diffraction approximation (used by the RL
environment; cheap and accurate for |m-1| << 1). The two implementations are
deliberately separate so the methods do not share numerical code paths.
"""

import numpy as np

from params import LAMBDA_VAC, N_OIL, N_WATER

M_REL = N_WATER / N_OIL
LAMBDA_MED = LAMBDA_VAC / N_OIL


def size_parameter(radius_m: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(radius_m) / LAMBDA_MED


def mie_qext_qsca(radius_m: float) -> tuple[float, float]:
    """Full Mie efficiencies for one sphere (Bohren-Huffman algorithm)."""
    x = float(size_parameter(radius_m))
    if x <= 0:
        raise ValueError("radius must be positive")
    m = complex(M_REL, 0.0)
    mx = m * x
    nstop = int(x + 4.05 * x ** (1.0 / 3.0) + 2.0) + 1
    nmx = int(max(nstop, abs(mx)) + 16)

    d = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        d[n - 1] = n / mx - 1.0 / (d[n] + n / mx)

    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = complex(psi1, -chi1)
    qext = 0.0
    qsca = 0.0
    for n in range(1, nstop + 1):
        psi = (2 * n - 1) / x * psi1 - psi0
        chi = (2 * n - 1) / x * chi1 - chi0
        xi = complex(psi, -chi)
        dn = d[n]
        an = ((dn / m + n / x) * psi - psi1) / ((dn / m + n / x) * xi - xi1)
        bn = ((dn * m + n / x) * psi - psi1) / ((dn * m + n / x) * xi - xi1)
        qext += (2 * n + 1) * (an.real + bn.real)
        qsca += (2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2)
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = xi
    qext *= 2.0 / (x * x)
    qsca *= 2.0 / (x * x)
    return qext, qsca


def mie_csca_grid(radii_m: np.ndarray) -> np.ndarray:
    """Scattering cross sections (m^2) for a fixed radius grid."""
    out = np.empty(len(radii_m))
    for i, a in enumerate(radii_m):
        _, qsca = mie_qext_qsca(a)
        out[i] = qsca * np.pi * a * a
    return out


def ada_csca(radii_m: np.ndarray) -> np.ndarray:
    """Anomalous-diffraction Q_ext (~Q_sca, non-absorbing) cross sections.

    Q = 2 - (4/rho) sin(rho) + (4/rho^2)(1 - cos(rho)), rho = 2 x |m - 1|.
    Blended into a Rayleigh-Gans tail at small rho where ADA underestimates.
    """
    a = np.asarray(radii_m, dtype=float)
    x = size_parameter(a)
    rho = 2.0 * x * abs(M_REL - 1.0)
    rho = np.where(rho < 1e-6, 1e-6, rho)
    q_ada = 2.0 - (4.0 / rho) * np.sin(rho) + (4.0 / rho ** 2) * (1.0 - np.cos(rho))
    # Rayleigh limit for x << 1 keeps the nanodroplet turbidity honest
    q_ray = (8.0 / 3.0) * x ** 4 * ((M_REL ** 2 - 1.0) / (M_REL ** 2 + 2.0)) ** 2
    q = np.where(x < 1.0, np.maximum(q_ray, q_ada * (x / 1.0) ** 0), q_ada)
    q = np.where(x < 0.3, q_ray, q)
    return q * np.pi * a * a


def transmittance_rel(optical_depth: np.ndarray) -> np.ndarray:
    """Relative transmittance given optical depth tau*L of the droplets."""
    return np.exp(-np.asarray(optical_depth))
