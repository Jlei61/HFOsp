# FCXR-LC4f X-depth closure — IMPLEMENTATION PLAN

1. Add a pure candidate derivation and X1 gate with unit tests.  Candidate derivation must read the
   archived K=3/4/5 probes and D/X termination bracket and fail loudly if their ordering drifts.
2. Add an off-by-default runner wrapper; no blessed engine file or equation changes.  The wrapper
   must set `use_m=False`, preserve the LC4c H threshold, and change only `K_y=3`.
3. Run targeted tests, broad LC4/LC3/MZ regression and blessed hash gate; write a source/artifact lock.
4. Launch X1 as a detached 22 s run.  On a non-positive verdict write STOP and exit.  Do not rescue.
5. Only X1 positive launches the unchanged 70 s nominal lifecycle gate, then conditional exact-D.
6. Generate figures/README and archive only for stages that actually ran.  Never draw a lifecycle
   panel from an X1-negative result.

Resource implementation: one worker; pinned OMP/BLAS threads; preflight MemAvailable/swap check;
`setsid nohup`; stage-scoped `flock`; PID-based monitoring; durable sentinels; no `pgrep -f` waits.

