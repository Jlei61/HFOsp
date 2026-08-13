# Shared Z/M baseline handoff plan

1. Freeze `config/topic4_data_driven_snn_baseline_zm_v1.json` and its evidence
   hashes.
2. Build all future continuous-field candidates through
   `apply_data_driven_snn_baseline()` with an explicit runtime mode.
3. Keep the current D6.3 slow-off replication unchanged until it completes.
4. In the next free-field round, use `active_z_plus_m`, at least 20 s per
   candidate, fresh network seeds, and runaway-invalid selection.
5. Carry the same runtime metadata through joint-field interpolation and any
   fit/selection/confirmation manifests.
6. Include `paired_slow_off` only as a matched comparator, never as a source of
   different field coefficients.
7. Accept a frozen candidate only through the standard spatial/readout and
   natural-KMeans Figure 4 panels.
