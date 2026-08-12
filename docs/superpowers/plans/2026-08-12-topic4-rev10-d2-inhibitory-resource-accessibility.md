# Topic 4 rev10-D2 execution plan

- [x] Freeze the D1 adaptation negative result.
- [x] Keep Node, topology, weights, delays, detector and patient target fixed.
- [x] Specify one q-only continuous local resource and a mean-drive global
  control without adding spatial fitting parameters.
- [x] Implement local/global q resource with exact-off parity and trace audit.
- [x] Reject the initial `k_q>=0.01/ms` bracket after the weakest sentinel hit
  `q_min` and runaway; preserve it as a dose-parameterization failure.
- [x] Freeze D2.1 at `k_q={0.0005,0.0015,0.004}/ms` before fresh networks
  1141-1143 and validate a 1-ms slow-field update against the 0.1-ms reference.
- [ ] Commit runtime code/config, run a measured-RSS sentinel, then launch via
  `systemd-run --user -> nohup` with 180 s waits and memory-bounded workers.
- [ ] Score returned events with equal network weight and adjudicate local
  against matched global and off.
- [ ] Only after a positive canary, freeze a fresh confirmation and produce the
  accepted direct-waveform and KMeans Fig.4 pair.

## D2.1 outcome

- [x] Complete 21/21 workers on 1141-1143 without runaway.
- [x] Record upper-bound local-specific A support in only 1/3 networks; gate
  remains failed despite score `10.245` versus off `11.417`.
- [x] Freeze one final boundary extension at
  `k_q={0.0055,0.0070,0.0085}/ms` before fresh networks 1171-1173.
- [ ] Run D2.2 with the same local/global/off comparison and stop if A/B does
  not occur in at least 2/3 networks.
