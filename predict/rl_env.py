"""Method 3, part 1: stochastic emulsion environment for an RL agent.

Super-droplet Monte Carlo (Shima-style, cloud-microphysics lineage): a fixed
array of computational droplets with multiplicities represents the population.
Each physics substep samples random pairs and fires coalescence events with
probability set by the same pair-level physics laws as the other methods, but
the population dynamics, event sampling, settling and optics are implemented
independently of the PBE code (ADA optics, not the Mie series).

The AGENT does not see the microstate. It observes (turbidity bin, mean-size
bin, resolved-water-layer quartile), like an experimentalist watching the
beaker, and picks a field level every 30 s. Levels above the paper's ~8 kV/cm safe limit risk an arc that
trips the supply. Reward is minus elapsed minutes, so the optimal value
function at the start state equals minus the expected demulsification time.
"""

import numpy as np

from kinetics import (
    effective_kernel,
    settling_velocity,
)
from optics import ada_csca
from params import (
    CELL_DEPTH,
    DROP_DIAM_MEDIAN,
    LOGNORM_SIGMA_G,
    PATH_LENGTH,
    PHI_WATER,
    T_REL_CLEAR,
)
from salinity import (
    DAUGHTER_RADIUS_RATIO,
    DAUGHTER_VOL_FRACTION,
    delta_rho,
    partial_coalescence_prob,
)

FIELD_LEVELS_KV_CM = (0.0, 2.0, 4.0, 6.0, 7.5, 9.0)
ARC_PROB_PER_STEP = {9.0: 0.30}   # sporadic arcing beyond ~8 kV/cm (paper)
ARC_PENALTY_MIN = 5.0             # supply trip + restart, minutes-equivalent
CONTROL_DT = 30.0                 # s between agent decisions
PHYS_DT = 5.0                     # s physics substep
N_SUPER = 512
BOX_SIZE = 300e-6                 # m, edge of the MC box
EPISODE_CAP_S = 6 * 3600.0

OD_EDGES = (0.0513, 0.2, 1.0, 3.0, 10.0, 30.0)
SIZE_EDGES_M = (2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 3e-5)
RESOLVED_EDGES = (0.25, 0.5, 0.75)   # resolved water layer, quartiles
N_OD_BINS = len(OD_EDGES) + 1
N_SIZE_BINS = len(SIZE_EDGES_M) + 1
N_RESOLVED_BINS = len(RESOLVED_EDGES) + 1


class EmulsionEnv:
    def __init__(self, phase, barrier_kt: float, seed: int):
        self.phase = phase
        self.barrier_kt = barrier_kt + phase.barrier_increment_kt
        self.drho = delta_rho(phase)
        self.rng = np.random.default_rng(seed)
        self.v_box = BOX_SIZE ** 3
        self.od_threshold = -np.log(T_REL_CLEAR)
        self.reset()

    def reset(self):
        ln_sig = np.log(LOGNORM_SIGMA_G)
        a_med = DROP_DIAM_MEDIAN / 2.0
        self.radius = a_med * np.exp(
            self.rng.normal(0.0, ln_sig, N_SUPER))
        vol = (4.0 / 3.0) * np.pi * self.radius ** 3
        n_real = PHI_WATER * self.v_box / vol.mean() / N_SUPER
        self.mult = np.full(N_SUPER, n_real)
        self.resolved_vol = 0.0
        self.t = 0.0
        return self._observe()

    def _optical_depth(self) -> float:
        csca = ada_csca(self.radius)
        return float(np.sum(self.mult * csca) / self.v_box * PATH_LENGTH)

    def _observe(self):
        od = self._optical_depth()
        w = self.mult * self.radius ** 3
        if w.sum() <= 0:
            a_mean = SIZE_EDGES_M[-1]
        else:
            a_mean = float(np.sum(w * self.radius) / w.sum())
        od_bin = int(np.searchsorted(OD_EDGES, od))
        size_bin = int(np.searchsorted(SIZE_EDGES_M, a_mean))
        resolved_frac = self.resolved_vol / (PHI_WATER * self.v_box)
        res_bin = int(np.searchsorted(RESOLVED_EDGES, resolved_frac))
        return od_bin, size_bin, res_bin

    def _physics_substep(self, e_field: float, dt: float):
        idx = self.rng.permutation(N_SUPER)
        half = N_SUPER // 2
        ii, jj = idx[:half], idx[half:2 * half]
        a_i, a_j = self.radius[ii], self.radius[jj]
        kern = effective_kernel(a_i, a_j, e_field, self.barrier_kt,
                                drho=self.drho)
        xi_max = np.maximum(self.mult[ii], self.mult[jj])
        p_evt = kern * xi_max * dt / self.v_box * (N_SUPER - 1)
        gamma = np.floor(p_evt) + (
            self.rng.random(half) < (p_evt - np.floor(p_evt)))
        fire = gamma >= 1.0
        if np.any(fire):
            self._apply_events(ii[fire], jj[fire], gamma[fire], e_field)
        # settling removal (deterministic multiplicity drain)
        frac = np.clip(
            settling_velocity(self.radius, self.drho) * dt / CELL_DEPTH,
            0.0, 1.0)
        removed = self.mult * frac
        self.resolved_vol += float(
            np.sum(removed * (4.0 / 3.0) * np.pi * self.radius ** 3))
        self.mult = self.mult - removed

    def _apply_events(self, ii, jj, gammas, e_field):
        """Shima-style multiple coalescence: the lower-multiplicity slot's
        drops each swallow gamma partners from the higher-multiplicity slot
        (capped by donor exhaustion). Mass-conserving by construction."""
        for i, j, gamma in zip(ii, jj, gammas):
            if self.mult[i] < 1e-6 or self.mult[j] < 1e-6:
                continue
            lo, hi = (i, j) if self.mult[i] <= self.mult[j] else (j, i)
            xi_lo, xi_hi = self.mult[lo], self.mult[hi]
            gamma = min(gamma, np.floor(xi_hi / xi_lo))
            if gamma < 1.0:
                continue
            a_lo, a_hi = self.radius[lo], self.radius[hi]
            v_lo = (4.0 / 3.0) * np.pi * a_lo ** 3
            v_hi = (4.0 / 3.0) * np.pi * a_hi ** 3
            a_small = min(a_lo, a_hi)
            v_small = (4.0 / 3.0) * np.pi * a_small ** 3
            p_pc = float(
                partial_coalescence_prob(a_small, e_field, self.phase))
            partial = self.rng.random() < p_pc
            recycled = DAUGHTER_VOL_FRACTION * v_small if partial else 0.0
            merged = gamma * xi_lo               # real merge count
            # survivor slot (lo): each drop gains gamma partners minus the
            # recycled daughter volume of each merge
            v_lo_new = v_lo + gamma * v_hi - gamma * recycled
            self.radius[lo] = (3.0 * v_lo_new / (4.0 * np.pi)) ** (1.0 / 3.0)
            # donor slot (hi): remainder at a_hi, plus daughters if partial
            xi_hi_left = xi_hi - merged
            if partial:
                v_d = v_small * DAUGHTER_RADIUS_RATIO ** 3
                n_daughters = merged * recycled / v_d
                count = xi_hi_left + n_daughters
                vol_total = xi_hi_left * v_hi + merged * recycled
                self.radius[hi] = (
                    3.0 * vol_total / (4.0 * np.pi * count)) ** (1.0 / 3.0)
                self.mult[hi] = count
            else:
                self.mult[hi] = xi_hi_left
                if self.mult[hi] <= 0:
                    # recycle empty slot: split the largest-volume slot
                    k = int(np.argmax(self.mult * self.radius ** 3))
                    self.mult[k] *= 0.5
                    self.mult[hi] = self.mult[k]
                    self.radius[hi] = self.radius[k]

    def step(self, action: int):
        e_kv = FIELD_LEVELS_KV_CM[action]
        arc_pen = 0.0
        if self.rng.random() < ARC_PROB_PER_STEP.get(e_kv, 0.0):
            e_kv = 0.0
            arc_pen = ARC_PENALTY_MIN
        e_field = e_kv * 1e5
        n_sub = int(CONTROL_DT / PHYS_DT)
        for _ in range(n_sub):
            self._physics_substep(e_field, PHYS_DT)
        self.t += CONTROL_DT
        obs = self._observe()
        done = self._optical_depth() <= self.od_threshold
        truncated = self.t >= EPISODE_CAP_S
        reward = -(CONTROL_DT / 60.0) - arc_pen
        return obs, reward, done, truncated
