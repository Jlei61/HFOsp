# rev11-NLC4 pathway mechanism confirmation plan

1. Add an off-by-default 1 ms pathway recorder to `simulate_kick` and verify
   spike parity when it is enabled.
2. Freeze the accepted four-arm candidate library on seeds 1581-1592 without
   changing Node, edge coefficients, detector, spatial OU, Z/M or beta.
3. Run one measured-RSS sentinel, then launch the remaining workers through
   systemd/nohup at memory-bounded concurrency. Emit status files and desktop
   completion notifications.
4. Audit mode-specific rates, OOD, natural direction alignment, shape and
   event-aligned E-to-E/E-to-I/I/GABA traces with paired network bootstrap.
5. Render the compact Fig.4C candidate from the historical exact ablation while
   the new run is active; regenerate it from the frozen new pool on completion.
6. Accept pathway attribution only as an effect-pattern estimate. Do not add
   extra blockers, retune coefficients or open beta based on this run.
