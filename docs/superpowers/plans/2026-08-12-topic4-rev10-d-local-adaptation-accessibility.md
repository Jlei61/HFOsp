# Topic 4 rev10-D execution plan

## Contract

- [x] Close rev10-R2 as a bounded static-edge negative result.
- [x] Keep the V6.2 Node substrate, exact no-op edges and common detector.
- [x] Freeze local adaptation as the only new mechanism; no K, core, contact,
  shaft or patient-conditioned spatial parameter.
- [x] Freeze `off/local/global`, `tau_a={250,750,2000} ms`, and
  `q_a={0.10,0.25,0.50}` before reading networks 1081-1083.

## Implementation

- [x] Add an E-only adaptation protocol that passes heterogeneous `Vtheta`
  through exactly and records low-rate state summaries.
- [x] Test exact decay, E-only action, local/global mean-dose identity under an
  imposed spike train, spatial heterogeneity and forbidden-input absence.
- [x] Extend the rev10-R worker without changing the exact-off path.
- [x] Freeze a manifest by reusing the already frozen direction classifier;
  do not refit patient labels.
- [x] Add adaptation metadata to worker NPZ/JSON and equal-network summary.

## Execution

- [ ] Commit all runtime modules and config before launching.
- [ ] Run one measured-RSS sentinel, then choose worker count from half of
  available RAM with one numeric thread per worker and 24 GiB cgroup limits.
- [ ] Launch every worker through `systemd-run --user -> nohup -> managed
  command`; controller waits 180 s between checks and sends a desktop
  notification on completion.
- [ ] Complete 19 candidates x 3 networks with common random numbers.

## Decision

- [ ] Compare each local arm with its same-parameter global control and off.
- [ ] Require same-network returned A/B support; KMeans stability is secondary.
- [ ] If local-specific accessibility is observed, freeze one candidate before
  fresh confirmation 1091-1093 and produce the accepted Fig.4 pair.
- [ ] Otherwise close this adaptation family without opening beta, adding
  spatial cores or comparing optimizers.
