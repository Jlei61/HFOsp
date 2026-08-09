# FCXR-LC4c entry-offset alignment status

- Status: **C0–C2 nominal complete; stopped by preregistered lifecycle gate**
- C0: `ENTRY_OFFSET_REPAIR_IDENTIFIABLE`
- C1: `C1_ENTRY_ALIGNED` — fresh 15 s no-kick trajectory enters at 11 s after 29 returning events; first 4 s executor current is exactly zero; finite, zero clip, non-refractory
- C2 nominal: `F2_NOMINAL_LIFECYCLE_INCOMPLETE`
- Continuous trajectory: onset 11 s; autonomous offset 66 s; high-density bout 55 s rather than the locked 1–5 s
- Post-offset observation: only 4 s; no event occurs after offset; mean activity is suppressed below the pre-onset mean, but returning-IED distribution and low-state stability are untested
- Final fixed 8 s is not a valid return window because its first 4 s still lie inside the high bout; its 36 events therefore cannot be reported as postictal returning IEDs
- Exact-final-D continuation: **not authorised**, because nominal eligibility failed
- Numerical/resource status: finite, zero clip, non-refractory; one detached worker; peak RSS 50.886 GiB; swap delta 0 MiB; no residual task process
- Scientific boundary: this single development seed establishes aligned cumulative entry and observes a late autonomous offset with short post-offset suppression. It does not establish a 1–5 s ictal carrier, distributional recovery, stable return, robustness, or a complete seizure lifecycle.

Primary evidence: `entry_gate.json`, `nominal_lifecycle.json`, `AUTOPILOT_STOP.json`; diagnostic: `figures/lc4c_entry_offset_alignment_diagnostic.png`.
