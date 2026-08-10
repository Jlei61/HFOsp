# FCXR-LC4e spatially shared terminator status

- Status: **E0–E1 complete; stopped by preregistered architecture gate**
- E0: `SHARED_EXECUTOR_IDENTIFIABLE`
- E1: `SHARED_EXECUTOR_OFFSET_NEGATIVE`
- Causal prefix: local/shared current starts at 11.83 s; rate and activity-fraction prefixes are exactly equal; current is exactly zero before that boundary
- Entry: both arms enter spontaneously at 11 s after 29 returning events; no kick, reset or parameter step
- Failure: both high bouts run to the 18 s record end; no autonomous offset or two-second guard is observed
- Spatial result: sharing removes the archived core-suppression/off-axis-escape pattern, but leaves a broadly active carrier
- Dose result: shared peak current is 20.339 versus 51.417 in the local arm, consistent with closed-loop suppression of its own driving load
- E2 nominal / exact-D: **not authorised**
- Numerical/resource status: finite, zero clip, refractory fraction 0; peak RSS 16.687 GiB; swap delta about 1.1 MiB; no residual task process
- Scientific boundary: this single-seed E1 closes the locked closed-loop spatial-sharing implementation. Because sharing also reduced delivered dose, it does not isolate spatial allocation at matched cumulative dose, reject activity-dependent relay depression as a family, establish recovery, or establish a seizure lifecycle.

Primary evidence: `candidate_lock.json`, `latency_screen.json`, `architecture_verdict.json`, `E1_DONE.json`, `AUTOPILOT_STOP.json`; diagnostic: `figures/lc4e_shared_executor_screen.png`.
