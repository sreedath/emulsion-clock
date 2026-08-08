"""Method 2: sectional population-balance simulation (deterministic).

Fixed-pivot (Kumar-Ramkrishna) coagulation on a geometric mass grid, with:
  - effective pair kernel (Brownian x dipolar capture x barrier crossing),
  - partial-coalescence channel recycling daughters (salinity-gated),
  - Stokes settling removal (well-mixed cell, rate v(a)/H),
  - full Mie turbidity readout and the paper's 95% relative-transmittance
    endpoint through a 1 cm path.

Independent of the analytic model: no staged approximations, the full
polydisperse dynamics are integrated.
"""

import numpy as np

from kinetics import effective_kernel, settling_velocity
from optics import mie_csca_grid
from params import (
    BARRIER_KT_BAND,
    CELL_DEPTH,
    DROP_DIAM_MEDIAN,
    E_FIELDS,
    LOGNORM_SIGMA_G,
    PATH_LENGTH,
    PHI_WATER,
    T_REL_CLEAR,
)
from salinity import (
    BRINE_05M,
    DAUGHTER_RADIUS_RATIO,
    DAUGHTER_VOL_FRACTION,
    DI_WATER,
    delta_rho,
    partial_coalescence_prob,
)

A_MIN, A_MAX, N_BINS = 25e-9, 300e-6, 123
T_MAX_S = 12 * 3600.0


def build_grid():
    radii = np.geomspace(A_MIN, A_MAX, N_BINS)
    masses = (4.0 / 3.0) * np.pi * radii ** 3  # volume as mass proxy
    return radii, masses


def pivot_weights(masses, m_new):
    """Mass-conserving split of m_new onto adjacent pivots (k, k+1)."""
    m_new = np.clip(m_new, masses[0], masses[-1])
    k = np.clip(np.searchsorted(masses, m_new) - 1, 0, N_BINS - 2)
    m_lo, m_hi = masses[k], masses[k + 1]
    w_hi = (m_new - m_lo) / (m_hi - m_lo)
    return k, np.clip(w_hi, 0.0, 1.0)


class PBEModel:
    def __init__(self, e_field, phase, barrier_kt):
        self.radii, self.masses = build_grid()
        self.e_field = e_field
        self.phase = phase
        self.drho = delta_rho(phase)
        barrier_kt = barrier_kt + phase.barrier_increment_kt
        ai = self.radii[:, None]
        aj = self.radii[None, :]
        self.kernel = effective_kernel(ai, aj, e_field, barrier_kt,
                                       drho=self.drho)
        a_small = np.minimum(ai, aj)
        self.p_pc = partial_coalescence_prob(a_small, e_field, phase)
        m_small = np.minimum.outer(self.masses, self.masses)
        m_sum = np.add.outer(self.masses, self.masses)
        # Kernel split by mass ratio: a merge between comparable sizes is a
        # discrete jump (fixed pivot); a big drop eating a much smaller one
        # (mass ratio < 1%) is continuous growth, treated as upwind
        # advection in size space. Without the split, each capture of a fine
        # is a destroy-and-recreate of the big drop and the explicit step
        # collapses to microseconds for dynamics that are actually slow.
        self.acc = (self.masses[None, :] < 1e-2 * self.masses[:, None])
        self.sim = ~self.acc & ~self.acc.T
        # merges beyond the top bin go straight to the resolved pool
        # (drops that large settle out in seconds anyway)
        self.overflow = m_sum > self.masses[-1]
        self.m_sum = m_sum
        self.dm_up = np.append(np.diff(self.masses), self.masses[-1])
        # full-merge deposit
        self.k_full, self.w_full = pivot_weights(self.masses, m_sum)
        # partial merge: survivor keeps all but the recycled daughter volume
        m_partial = m_sum - DAUGHTER_VOL_FRACTION * m_small
        self.k_part, self.w_part = pivot_weights(self.masses, m_partial)
        # daughters: recycled volume at DAUGHTER_RADIUS_RATIO of small drop
        m_daughter = m_small * DAUGHTER_RADIUS_RATIO ** 3
        self.k_dtr, self.w_dtr = pivot_weights(self.masses, m_daughter)
        self.n_daughters = DAUGHTER_VOL_FRACTION * m_small / m_daughter
        self.settle_rate = settling_velocity(self.radii, self.drho) / CELL_DEPTH
        self.csca = mie_csca_grid(self.radii)
        # split kernels: discrete (similar sizes) vs accretion (big eats fine)
        self.kernel_sim = self.kernel * self.sim
        self.kernel_acc = self.kernel * self.acc
        # mass eaten per big drop per fine per second (net of daughters)
        self.acc_eat = (self.kernel_acc
                        * (1.0 - DAUGHTER_VOL_FRACTION * self.p_pc)
                        * self.masses[None, :])
        self.acc_dtr = self.kernel_acc * self.p_pc * self.n_daughters

    def initial_state(self):
        ln_sig = np.log(LOGNORM_SIGMA_G)
        a_med = DROP_DIAM_MEDIAN / 2.0
        z = np.log(self.radii / a_med) / ln_sig
        pdf = np.exp(-0.5 * z ** 2)
        n = pdf * PHI_WATER / (pdf * self.masses).sum()
        return np.concatenate([n, [0.0]])  # last slot: resolved water volume

    def gain_loss(self, n):
        """Coagulation gain/loss and settling terms for state n.
        Similar-size merges: discrete fixed-pivot jumps. Accretion (mass
        ratio < 1%): continuous growth, upwind advection in size space."""
        outer_n = np.outer(n, n)
        pair = 0.5 * self.kernel_sim * outer_n
        loss = pair.sum(axis=1) + pair.sum(axis=0)  # drops i consumed
        gain = np.zeros(N_BINS)
        full = pair * (1.0 - self.p_pc)
        part = pair * self.p_pc
        overflow_mass = float(np.sum(pair[self.overflow]
                                     * self.m_sum[self.overflow]))
        full = np.where(self.overflow, 0.0, full)
        part = np.where(self.overflow, 0.0, part)

        # accretion: fines are consumed; big drops advect up the mass grid
        r_acc = self.kernel_acc * outer_n
        loss = loss + r_acc.sum(axis=0)               # fines eaten
        eat_per_big = self.acc_eat @ n                # kg/s per big drop
        adv = n * eat_per_big / self.dm_up
        loss = loss + adv
        gain[1:] += adv[:-1]
        overflow_mass += adv[-1] * self.masses[-1]
        # daughters from partial-coalescence during accretion
        dtr_acc = self.acc_dtr * outer_n
        gain += np.bincount(
            self.k_dtr.ravel(),
            (dtr_acc * (1.0 - self.w_dtr)).ravel(), minlength=N_BINS)
        gain += np.bincount(
            (self.k_dtr + 1).ravel(),
            (dtr_acc * self.w_dtr).ravel(), minlength=N_BINS)
        gain += np.bincount(
            self.k_full.ravel(),
            (full * (1.0 - self.w_full)).ravel(), minlength=N_BINS)
        gain += np.bincount(
            (self.k_full + 1).ravel(),
            (full * self.w_full).ravel(), minlength=N_BINS)
        gain += np.bincount(
            self.k_part.ravel(),
            (part * (1.0 - self.w_part)).ravel(), minlength=N_BINS)
        gain += np.bincount(
            (self.k_part + 1).ravel(),
            (part * self.w_part).ravel(), minlength=N_BINS)
        dtr = part * self.n_daughters
        gain += np.bincount(
            self.k_dtr.ravel(),
            (dtr * (1.0 - self.w_dtr)).ravel(), minlength=N_BINS)
        gain += np.bincount(
            (self.k_dtr + 1).ravel(),
            (dtr * self.w_dtr).ravel(), minlength=N_BINS)
        return gain, loss, overflow_mass

    def optical_depth(self, n):
        return np.sum(n * self.csca) * PATH_LENGTH

    def run(self, dt_max=5.0, log_every=None, snapshot_times=None):
        """Explicit Euler time stepping. Coagulation gain/loss cancel bin by
        bin, so mass is conserved identically at any dt; the step is limited
        by stability (dt * fastest active per-particle rate <= 0.2)."""
        y = self.initial_state()
        n, resolved = y[:-1].copy(), 0.0
        od_clear = -np.log(T_REL_CLEAR)
        t, t_clear = 0.0, np.inf
        history = []
        snapshots = []
        snap_due = sorted(snapshot_times) if snapshot_times else []
        while t < T_MAX_S:
            if snap_due and t >= snap_due[0]:
                snapshots.append((t, n.copy()))
                snap_due.pop(0)
            gain, loss, overflow_mass = self.gain_loss(n)
            settle_flux = self.settle_rate * n
            with np.errstate(divide="ignore", invalid="ignore"):
                rate_pp = np.where(
                    n > 0,
                    (loss + settle_flux) / np.maximum(n, 1e-30), 0.0)
            active = n * self.masses > 1e-6 * PHI_WATER / N_BINS
            rate_max = rate_pp[active].max() if active.any() else 0.0
            dt = dt_max if rate_max <= 0 else min(dt_max, 0.2 / rate_max)
            n = np.maximum(n + dt * (gain - loss - settle_flux), 0.0)
            resolved += dt * (np.sum(settle_flux * self.masses)
                              + overflow_mass)
            t += dt
            od = self.optical_depth(n)
            if log_every and int(t / log_every) != int((t - dt) / log_every):
                history.append((t, od, resolved / PHI_WATER))
            if od <= od_clear:
                t_clear = t
                break
        return {
            "t_clear_s": t_clear,
            "resolved_frac_at_end": resolved / PHI_WATER,
            "suspended_frac_at_end": float(
                np.sum(n * self.masses) / PHI_WATER),
            "od_end": self.optical_depth(n),
            "history": history,
            "snapshots": snapshots,
        }


def fmt(seconds):
    if not np.isfinite(seconds):
        return ">12 h"
    return f"{seconds / 60:.1f} min" if seconds < 7200 else f"{seconds/3600:.1f} h"


def main():
    print("PBE sectional simulation, blind prediction")
    print(f"endpoint: T_rel = {T_REL_CLEAR:.0%} over {PATH_LENGTH*100:.0f} cm")
    dg_central = BARRIER_KT_BAND[1]
    rows = []
    for phase in (BRINE_05M, DI_WATER):
        for e_name, e_field in E_FIELDS.items():
            res = PBEModel(e_field, phase, dg_central).run()
            rows.append((phase.name, e_field, dg_central, res["t_clear_s"]))
            mass = res["resolved_frac_at_end"] + res["suspended_frac_at_end"]
            print(f"{phase.name:<12} E={e_field/1e5:>4.1f} kV/cm "
                  f"dG={dg_central:.0f}kT  t_clear = {fmt(res['t_clear_s'])} "
                  f"(resolved {res['resolved_frac_at_end']:.0%}, "
                  f"mass check {mass:.2f})", flush=True)
    print("\nbarrier sensitivity at 7.5 kV/cm:")
    for phase in (BRINE_05M, DI_WATER):
        for dg in BARRIER_KT_BAND:
            res = PBEModel(E_FIELDS["max_safe_7.5kV_cm"], phase, dg).run()
            print(f"{phase.name:<12} dG={dg:.0f}kT  t_clear = "
                  f"{fmt(res['t_clear_s'])}")


if __name__ == "__main__":
    main()
