# Topic 4 rev10-D5.1 execution plan

1. Freeze the seven-arm library: off plus three paired local/permuted amplitudes.
2. Run one measured-RSS sentinel under `systemd-run --user -> nohup`, then choose workers from observed RSS while retaining half of available RAM and enforcing per-worker cgroup limits.
3. Run the remaining fit jobs with a `180 s` controller wait interval and completion notification.
4. Aggregate only returned events with the common detector and equal-network mode score.
5. Select the lowest accessible local amplitude by the frozen rule and report activity burden against off and matched permutation.
6. If a survivor exists, freeze off/local/permuted before touching unseen seeds `1291-1296`.
7. Use the fresh confirmation only to decide whether the accepted Fig.4 direct-waveform and KMeans panels can replace the prior negative R2 panels.

No patient held-out data, source-coordinate placement, core-count growth, static edge fitting, beta opening, or optimizer comparison is allowed in D5.1.
