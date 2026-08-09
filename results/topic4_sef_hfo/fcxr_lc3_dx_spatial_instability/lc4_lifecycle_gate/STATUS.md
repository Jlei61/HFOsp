# FCXR-LC4 functional selectivity status

- Status: **F0 complete; stopped by preregistered gate**
- Machine verdict: `NO_BASELINE_PRESERVING_HILL_CANDIDATE`
- F0 control: 8 returning events in the scored 10 s (`0.80/s`)
- Hill n=6: 5 events (`0.50/s`), event-rate ratio `0.625`, IEI-CV ratio `1.724`; FAIL
- Hill n=8: 11 events (`1.10/s`), event-rate ratio `1.375`; FAIL
- Both candidates: numerically safe, no sustained bout, zero clipping, and maximum outward-current leakage below `0.1%` of the locked recurrent scale
- F1 frozen-D onset surface: **not run**
- F2 70 s lifecycle and exact-D recovery continuation: **not run**
- Scientific boundary: this rejects only the locked smooth Hill family `{n=6,n=8}` at the locked `K`, timescales and force-matched ictal dose. It does not reject cooperative termination in general and does not establish or refute a complete lifecycle.

Primary evidence: `baseline_verdict.json`; diagnostic: `figures/lc4_functional_baseline_gate.png`.
