"""Verification pass: independent checks of every load-bearing result.

Each check recomputes from raw constants (never through the model code being
checked) or tests a known limit. Output: pass/fail table for VERIFICATION.md.
"""

import subprocess

import numpy as np

from kinetics import (
    brownian_kernel,
    pair_contact_energy_kt,
    settling_velocity,
)
from optics import ada_csca, mie_qext_qsca
from params import (
    DROP_DIAM_MEDIAN,
    EPS0,
    EPSR_OIL,
    MU_OIL,
    PHI_WATER,
    RHO_OIL,
    RHO_WATER,
    kT,
)

CHECKS = []


def check(name, ok, detail):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main():
    # 1. Lambda arithmetic, recomputed from raw constants
    a = DROP_DIAM_MEDIAN / 2
    lam_raw = np.pi * 8.8541878128e-12 * 2.05 * (7.5e5) ** 2 * a ** 3 / (
        1.380649e-23 * 298.0)
    lam_model = float(pair_contact_energy_kt(a, a, 7.5e5))
    check("Lambda(175 nm, 7.5 kV/cm)",
          abs(lam_raw - lam_model) / lam_raw < 1e-9 and 5.1 < lam_raw < 5.3,
          f"raw {lam_raw:.3f} vs model {lam_model:.3f}")

    # 2. Brownian kernel vs the standard 8kT/3mu result
    kb_expected = 8 * kT() / (3 * MU_OIL)
    kb_model = float(brownian_kernel(a, a))
    check("Brownian kernel equal spheres",
          abs(kb_model - kb_expected) / kb_expected < 1e-9,
          f"{kb_model:.3e} m^3/s (8kT/3mu = {kb_expected:.3e})")

    # 3. Stokes settling, hand value for 10 um drop
    v10 = float(settling_velocity(10e-6))
    v_hand = 2 * (RHO_WATER - RHO_OIL) * 9.81 * (10e-6) ** 2 / (9 * MU_OIL)
    check("Stokes velocity 10 um",
          abs(v10 - v_hand) / v_hand < 1e-9 and 1.5e-5 < v10 < 2.1e-5,
          f"{v10*1e6:.1f} um/s")

    # 4. Mie optics limits: Rayleigh at small x, extinction -> 2 at large x
    m = 1.338 / 1.434
    _, qs_small = mie_qext_qsca(20e-9)
    x = 2 * np.pi * 20e-9 / (550e-9 / 1.434)
    q_ray = (8.0 / 3.0) * x ** 4 * ((m * m - 1) / (m * m + 2)) ** 2
    qe_large, _ = mie_qext_qsca(80e-6)
    check("Mie Rayleigh limit (20 nm)",
          abs(qs_small - q_ray) / q_ray < 0.05,
          f"Mie {qs_small:.3e} vs Rayleigh {q_ray:.3e}")
    check("Mie geometric limit (80 um)", 1.7 < qe_large < 2.3,
          f"Q_ext = {qe_large:.2f} (expect ~2)")

    # 5. ADA vs full Mie agreement at the starting size
    c_mie = mie_qext_qsca(a)[1] * np.pi * a * a
    c_ada = float(ada_csca(np.array([a]))[0])
    check("ADA vs Mie at 87.5 nm", 0.5 < c_ada / c_mie < 3.5,
          f"ratio {c_ada/c_mie:.2f} (independent optics, order-1 agreement)")

    # 6. Initial opacity: DLS reported 0% transmittance through 1 cm
    n0 = PHI_WATER / ((4 / 3) * np.pi * a ** 3)
    od0 = n0 * c_mie * 0.01
    check("Initial emulsion opaque", od0 > 5,
          f"OD(1 cm) = {od0:.1f} -> T = {np.exp(-od0)*100:.2g}%")

    # 7. Shelf-life consistency: zero field, months stable
    lam0 = float(pair_contact_energy_kt(a, a, 0.0))
    check("Zero-field gate closed", lam0 == 0.0,
          "Lambda(E=0) = 0; barrier never opens; matches months of stability")

    # 8. Data collapse: recompute digitized Fig 4C in one line
    volts = np.array([7.5, 10, 12.5, 15, 17.5, 20, 22.5])
    times = np.array([95, 50, 30, 22, 17, 13, 11.0])
    lam_pts = (np.pi * EPS0 * EPSR_OIL * (volts * 1e3 / 5.5e-2) ** 2
               * (100e-9) ** 3 / kT())
    c_prod = times * lam_pts
    spread = np.std(c_prod) / np.mean(c_prod)
    check("t*Lambda collapse", spread < 0.06,
          f"C = {np.mean(c_prod):.1f} min, RMS scatter {spread:.1%}, "
          f"range {c_prod.min():.1f}-{c_prod.max():.1f}")
    slope = np.polyfit(np.log(volts), np.log(times), 1)[0]
    check("Digitized points reproduce paper exponent",
          -2.1 < slope < -1.7,
          f"fitted t ~ V^{slope:.2f} (paper: -1.91)")

    # 9. Collapse discriminates the field reading (corona offset destroys it)
    lam_off = (np.pi * EPS0 * EPSR_OIL
               * (np.maximum(volts - 6.3, 0.1) * 1e3 / 4.0e-2) ** 2
               * (100e-9) ** 3 / kT())
    spread_off = np.std(times * lam_off) / np.mean(times * lam_off)
    check("Offset reading fails to collapse", spread_off > 0.4,
          f"corona-offset spread {spread_off:.0%} vs V/d {spread:.0%}")

    # 10. Method independence: AST-based import audit. The routes SHARE the
    # kernel layer (kinetics/params/salinity) by design; what must be
    # disjoint is the population-integration code.
    import ast
    solver_mods = {"pbe_sim": ["analytic_model.py", "rl_env.py",
                               "rl_train.py"],
                   "analytic_model": ["pbe_sim.py", "rl_env.py",
                                      "rl_train.py"],
                   "rl_env": ["pbe_sim.py", "analytic_model.py"]}
    cross = []
    for mod, files in solver_mods.items():
        for fname in files:
            tree = ast.parse(open(fname).read())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if mod in names:
                    cross.append(f"{fname} imports {mod}")
    check("Solver-layer independence (AST)", not cross,
          "; ".join(cross) if cross
          else "no route imports another route's solver "
               "(kernel layer intentionally shared)")

    # 11. PBE grid convergence: 123 vs 165 bins within 5%
    import pbe_sim
    from params import BARRIER_KT_BAND, E_FIELDS
    from salinity import BRINE_05M
    r1 = pbe_sim.PBEModel(E_FIELDS["max_safe_7.5kV_cm"], BRINE_05M,
                          BARRIER_KT_BAND[1]).run()
    pbe_sim.N_BINS = 165
    r2 = pbe_sim.PBEModel(E_FIELDS["max_safe_7.5kV_cm"], BRINE_05M,
                          BARRIER_KT_BAND[1]).run()
    pbe_sim.N_BINS = 123
    shift = abs(r2["t_clear_s"] / r1["t_clear_s"] - 1)
    check("PBE grid convergence", shift < 0.05,
          f"t95 shift {shift:.1%} for 123 -> 165 bins")

    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    import sys
    sys.exit(0 if n_pass == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
