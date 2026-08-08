# emulsion-clock

**From pairwise electrocoalescence to optical clearing: blind prediction of
electrostatic demulsification times for water-in-oil nanoemulsions.**

Companion code for: S. Panat, R. Dandekar, R. Dandekar, *Predicting the
optical clearing time of electrostatically demulsified nanoemulsions from
setup information alone* (2026). Interactive story site:
https://demulsification-prediction.vercel.app

## The problem

Single-droplet electrocoalescence physics (polarize, chain, coalesce) is well
understood, but the observable that matters is macroscopic *optical clearing*
of the emulsion, and no single-pair argument yields that time. This package
predicts the 95%-transmittance clearing time of a 175 nm, 2% water-in-hexadecane
emulsion at up to 7.5 kV/cm using only pre-experiment setup information, three
independent ways, with zero calibrated parameters.

## The three routes

| Route | File | Machinery |
|---|---|---|
| 1. Analytic cascade | `predict/analytic_model.py` | staged closed form: gate, ignition, growth ladder, gravity endgame |
| 2. Population balance | `predict/pbe_sim.py` | 123-bin Smoluchowski + settling + daughter source + Mie optics endpoint |
| 3. RL agent | `predict/rl_env.py`, `predict/rl_train.py` | super-droplet Monte Carlo rig + tabular Monte Carlo control |

Shared pair physics: `predict/kinetics.py` (kernels, barrier model, capture radii),
`predict/salinity.py` (film strengthening, partial coalescence),
`predict/optics.py` (Mie and anomalous-diffraction scattering),
`predict/params.py` (setup constants; every number traced to setup facts or literature).

## Reproduce

```bash
python3 predict/verify.py            # 12 independent correctness checks
python3 predict/analytic_model.py    # route 1
python3 predict/pbe_sim.py           # route 2 (few minutes)
python3 predict/rl_train.py          # route 3 (10-20 minutes)
python3 predict/make_site_figures.py # all data figures
```

Requires numpy, scipy, matplotlib.

## Headline results

- Blind prediction, optical clearing at 7.5 kV/cm: 90 to 136 min (measured, read
  only afterwards: ~60 min).
- Bulk (mass) separation is 20 to 50x faster than optical clearing; the optical
  clock is owned by a sub-100-nm fines tail.
- The electrocoalescence number Lambda = pi eps0 eps_oil K^2 E^2 a^3 / kB T
  collapses the published voltage-sweep data to t x Lambda = 23.5 min +/- 5%,
  with a threshold Lambda* ~ 0.8 predicted in advance from shelf-life stability.

## License

MIT
