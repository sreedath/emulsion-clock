# What does "demulsified" mean, quantitatively?

The prediction target must be an observable with a threshold. Candidate
definitions, and how they relate:

## 1. Optical clearing (the paper's definition, adopted here)

Relative transmittance T_rel reaching 95% (through 1 cm; DLS transmittance or
gray-value proxy). Equivalent statement: the droplet optical depth tau*L must
fall below -ln(0.95) = 0.051.

Why it is a good definition:
- It is what an operator sees; it is cheap to measure continuously.
- It is an END-state metric: nanodroplets near the Mie scattering peak
  (diameter comparable to the wavelength in oil, ~385 nm) scatter maximally,
  so T_rel = 95% cannot be reached while any meaningful nanodroplet
  population survives.

Why it is a HARSH definition, and the two traps hidden in it:
- Trap 1, transient whitening: coalescence initially moves the distribution
  toward the Mie peak, so turbidity RISES before it falls (our simulations
  show optical depth roughly doubling in the first minutes). A monotone
  "percent cleared" intuition is wrong; mean gray value is non-monotonic.
- Trap 2, the fines tail: optical depth of leftover fines at radius a scales
  as phi_res * a^3 (Rayleigh regime). A sub-percent volume fraction of
  ~100 nm fines still holds optical depth of order 1, keeping T_rel far
  below 95% even when >90% of the water has separated. The last 5% of
  transmittance is a statement about the smallest decile of the initial
  size distribution, not about the bulk water.

## 2. Resolved-water fraction (mass-based)

Fraction of the dispersed water that has joined the free-water layer, e.g.
95%. This is the metric that matters industrially (crude specs are on water
CONTENT, <0.2%). It is more forgiving than optical clearing: our simulations
routinely reach 90% resolved water well before T_rel = 95%.

## 3. Mean-size threshold (mechanism-based)

Volume-weighted mean radius crossing the size where Stokes settling clears
the cell height in a fixed time (here ~8-20 um for minutes-scale settling
over 3.4 cm). This marks the END OF THE ELECTRIC FIELD'S JOB: past this
size, gravity finishes alone. Useful for separating "coalescence time" from
"settling time" when comparing mechanisms, but not directly observable
without in-situ sizing.

## Relations (from the population-balance runs)

t(mean-size threshold) < t(90% resolved) < t(T_rel = 95%)

The three orderings can differ by large factors precisely when the fines
tail survives (brine, low field, fine initial distribution). When the field
is strong enough to scavenge the whole distribution, all three definitions
converge to within a factor ~2, and the paper's 95%-transmittance time is
then a fair proxy for "the emulsion is gone."

## Recommendation

Report two numbers: t_95T (the paper's optical endpoint) and t_90R (90%
resolved water). The pair (t_90R, t_95T - t_90R) separates "bulk
separation" from "fines cleanup," which respond to different physics
(growth+settling vs scavenging), and which salinity affects differently.
