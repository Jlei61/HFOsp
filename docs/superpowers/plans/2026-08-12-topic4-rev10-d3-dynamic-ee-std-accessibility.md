# Topic 4 rev10-D3 execution plan

- [x] Freeze the R2, D1 and D2 negative results without replacing the accepted
  Fig.4 pair.
- [x] Define source-specific E->E STD and an exact latent-mean global control.
- [x] Keep Node, static edges, topology, delays, detector and patient target
  fixed; prohibit observation-conditioned inputs.
- [x] Add engine default-parity, local/global application and contract tests.
- [x] Freeze 9 candidates before fresh network seeds 1201-1203.
- [x] Run one non-no-op measured-RSS sentinel through `systemd-run -> nohup`.
- [x] Launch remaining workers with memory-bounded parallelism, 180 s waits and
  completion notification.
- [x] Aggregate returned events with equal network weights and adjudicate local
  against matched global and exact off.
- [x] Do not run confirmation or replace Fig.4: all local arms had mode A and B
  support in 0/3 networks, while exact off retained B in 3/3.
- [x] Only after a positive canary, freeze confirmation seeds and produce the
  direct-waveform and KMeans Fig.4 pair.
- [x] Close D3 as `SOURCE_SPECIFIC_DYNAMIC_EDGE_ACCESS_NOT_OBSERVED`.
- [ ] Map forced route capacity on a uniform spatial source grid before deciding
  whether a new history-dependent directional route family is justified.
