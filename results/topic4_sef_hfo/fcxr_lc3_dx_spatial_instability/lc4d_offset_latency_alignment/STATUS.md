# FCXR-LC4d offset-latency alignment status

- Status: **L0–L1 complete; stopped by preregistered latency gate**
- L0: `OFFSET_LATENCY_REPAIR_IDENTIFIABLE`
- L1: `OFFSET_LATENCY_REPAIR_INSUFFICIENT`
- Fresh trajectory: no kick, reset, state fork or parameter step; spontaneous onset at 11 s after 29 returning events
- Interictal protection: first 4 s executed terminator current exactly zero; finite, zero clip and non-refractory
- Failure: the qualifying bout runs from 11 s to the 18 s record end; no autonomous offset or two-second guard is observed
- Closed-loop calibration failure: the 15 s current is 18.442 rather than the 44.862 target; target is first reached at 15.84 s, after which the carrier still persists
- Spatial diagnostic: at 17.75 s the two-core mean `y`/`H` is approximately zero/0.085 while off-axis `y`/`H` remains 13.30/1.12
- L2 nominal and exact-D confirmation: **not authorised**
- Numerical/resource status: finite, zero clip, refractory fraction 0; peak RSS 16.687 GiB; swap delta 0 MiB; no residual task process
- Scientific boundary: this single-seed L1 rejects the one-point open-loop dose-transfer repair. It does not reject spatially coordinated termination, establish postictal recovery, or establish a complete seizure lifecycle.

Primary evidence: `candidate_lock.json`, `latency_screen.json`, `L1_DONE.json`, `AUTOPILOT_STOP.json`; diagnostic: `figures/lc4d_offset_latency_screen.png`.
