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
