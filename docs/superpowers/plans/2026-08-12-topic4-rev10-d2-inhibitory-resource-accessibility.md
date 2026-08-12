# Topic 4 rev10-D2 execution plan

- [x] Freeze the D1 adaptation negative result.
- [x] Keep Node, topology, weights, delays, detector and patient target fixed.
- [x] Specify one q-only continuous local resource and a mean-drive global
  control without adding spatial fitting parameters.
- [x] Implement local/global q resource with exact-off parity and trace audit.
- [x] Freeze 7 candidates before reading networks 1111-1113.
- [ ] Commit runtime code/config, run a measured-RSS sentinel, then launch via
  `systemd-run --user -> nohup` with 180 s waits and memory-bounded workers.
- [ ] Score returned events with equal network weight and adjudicate local
  against matched global and off.
- [ ] Only after a positive canary, freeze a fresh confirmation and produce the
  accepted direct-waveform and KMeans Fig.4 pair.
