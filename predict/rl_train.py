"""Method 3, part 2: tabular Monte Carlo control on the emulsion env.

Undiscounted shortest-path formulation: reward is minus elapsed minutes, so
the learned value of the start state is minus the expected demulsification
time under the learned policy. Monte Carlo control (returns-to-go averaged
over full episodes, epsilon-greedy) is used instead of one-step Q-learning:
with a near-self-transition "field off" action, bootstrapped Q-learning
keeps do-nothing actions spuriously competitive, while MC returns price
them at their true cost. Episodes that time out get a pessimistic terminal
penalty so non-clearing policies are dispreferred.

Outputs, per water phase: learned policy summary, value-function estimate,
clearing-time distribution of greedy rollouts (the Method-3 prediction),
and a fixed max-safe-field baseline.
"""

import numpy as np

from params import BARRIER_KT_BAND
from rl_env import (
    EPISODE_CAP_S,
    FIELD_LEVELS_KV_CM,
    N_OD_BINS,
    N_RESOLVED_BINS,
    N_SIZE_BINS,
    EmulsionEnv,
)
from salinity import BRINE_05M, DI_WATER

N_TRAIN = 800
N_EVAL = 40
ALPHA = 0.10
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_FRAC = 0.6
TRUNCATION_PENALTY_MIN = 360.0
SEED = 7


def state_index(obs) -> int:
    od_bin, size_bin, res_bin = obs
    return (od_bin * N_SIZE_BINS + size_bin) * N_RESOLVED_BINS + res_bin


def masked_greedy(q_row, counts_row, rng):
    """Greedy over actions actually tried in this state. With all-negative
    rewards, untried actions keep Q=0 and would always win a naive argmax."""
    visited = counts_row > 0
    if not visited.any():
        return int(rng.integers(len(FIELD_LEVELS_KV_CM)))
    masked = np.where(visited, q_row, -np.inf)
    return int(np.argmax(masked))


def run_episode(env, q, counts, eps, rng, learn=True):
    """One episode; if learning, update Q with undiscounted returns-to-go."""
    obs = env.reset()
    trajectory = []
    total_reward = 0.0
    while True:
        s = state_index(obs)
        if rng.random() < eps:
            a = int(rng.integers(len(FIELD_LEVELS_KV_CM)))
        else:
            a = masked_greedy(q[s], counts[s], rng)
        obs2, r, done, truncated = env.step(a)
        trajectory.append((s, a, r))
        total_reward += r
        obs = obs2
        if done or truncated:
            break
    if learn:
        g = -TRUNCATION_PENALTY_MIN if (truncated and not done) else 0.0
        for s, a, r in reversed(trajectory):
            g += r
            counts[s, a] += 1
            q[s, a] += ALPHA * (g - q[s, a])
    return total_reward, env.t, done


def train_phase(phase, barrier_kt):
    n_states = N_OD_BINS * N_SIZE_BINS * N_RESOLVED_BINS
    q = np.zeros((n_states, len(FIELD_LEVELS_KV_CM)))
    counts = np.zeros((n_states, len(FIELD_LEVELS_KV_CM)), dtype=int)
    rng = np.random.default_rng(SEED)
    env = EmulsionEnv(phase, barrier_kt, seed=SEED)
    decay_ep = int(N_TRAIN * EPS_DECAY_FRAC)
    for ep in range(N_TRAIN):
        eps = max(EPS_END, EPS_START + (EPS_END - EPS_START) * ep / decay_ep)
        run_episode(env, q, counts, eps, rng, learn=True)

    times, cleared = [], 0
    for k in range(N_EVAL):
        env_eval = EmulsionEnv(phase, barrier_kt, seed=1000 + k)
        _, t_end, done = run_episode(
            env_eval, q, counts, 0.0, rng, learn=False)
        if done:
            times.append(t_end)
            cleared += 1
    times = np.array(times)

    # fixed max-safe-field baseline for comparison with the learned policy
    base_times, base_cleared = [], 0
    for k in range(20):
        env_b = EmulsionEnv(phase, barrier_kt, seed=5000 + k)
        env_b.reset()
        done = trunc = False
        while not (done or trunc):
            _, _, done, trunc = env_b.step(4)
        if done:
            base_times.append(env_b.t)
            base_cleared += 1
    base_times = np.array(base_times)

    s0 = state_index(EmulsionEnv(phase, barrier_kt, seed=SEED).reset())
    vis0 = counts[s0] > 0
    v0_min = -np.max(q[s0][vis0]) if vis0.any() else np.nan

    # policy summary: greedy field level in the most-visited turbid states
    policy_fields = {}
    for od_bin in range(N_OD_BINS):
        for size_bin in range(N_SIZE_BINS):
            for res_bin in range(N_RESOLVED_BINS):
                s = ((od_bin * N_SIZE_BINS + size_bin) * N_RESOLVED_BINS
                     + res_bin)
                if np.any(q[s] != 0.0):
                    policy_fields[(od_bin, size_bin, res_bin)] = (
                        FIELD_LEVELS_KV_CM[int(np.argmax(
                            np.where(q[s] != 0.0, q[s], -np.inf)))])
    return {
        "phase": phase.name,
        "clear_rate": cleared / N_EVAL,
        "t_median_min": float(np.median(times) / 60) if len(times) else np.inf,
        "t_mean_min": float(times.mean() / 60) if len(times) else np.inf,
        "t_iqr_min": (
            float(np.percentile(times, 25) / 60),
            float(np.percentile(times, 75) / 60),
        ) if len(times) else (np.inf, np.inf),
        "v0_estimate_min": float(v0_min),
        "policy": policy_fields,
        "base_clear_rate": base_cleared / 20,
        "base_median_min": float(np.median(base_times) / 60)
        if len(base_times) else np.inf,
    }


def main():
    dg = BARRIER_KT_BAND[1]
    for phase in (BRINE_05M, DI_WATER):
        res = train_phase(phase, dg)
        print(f"\n=== {res['phase']} (dG = {dg:.0f} kT core) ===")
        print(f"eval clear rate: {res['clear_rate']:.0%} "
              f"(cap {EPISODE_CAP_S/3600:.0f} h)")
        print(f"greedy rollout time: median {res['t_median_min']:.1f} min, "
              f"mean {res['t_mean_min']:.1f} min, "
              f"IQR {res['t_iqr_min'][0]:.1f}-{res['t_iqr_min'][1]:.1f} min")
        print(f"value-function estimate at start: {res['v0_estimate_min']:.1f} min")
        print(f"fixed 7.5 kV/cm baseline: {res['base_clear_rate']:.0%} clear, "
              f"median {res['base_median_min']:.1f} min")
        fields = sorted(set(res["policy"].values()))
        print(f"field levels used by greedy policy: {fields}")
        by_od = {}
        for (od_bin, _sz, _rb), f in res["policy"].items():
            by_od.setdefault(od_bin, []).append(f)
        for od_bin in sorted(by_od):
            vals = by_od[od_bin]
            print(f"  turbidity bin {od_bin}: "
                  f"median field {np.median(vals):.1f} kV/cm")


if __name__ == "__main__":
    main()
