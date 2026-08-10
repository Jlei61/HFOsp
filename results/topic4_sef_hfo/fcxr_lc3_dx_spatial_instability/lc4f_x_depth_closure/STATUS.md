# FCXR-LC4f X-depth closure status

- Status: **X0–X1 complete; stopped by preregistered X1 gate**
- X0: `X_DEPTH_CANDIDATE_IDENTIFIABLE`
- X1: `X_DEPTH_OFFSET_NEGATIVE`
- Fresh trajectory: no kick, reset, state fork or parameter step; onset at 11 s after 29 returning events
- Isolation: LC4 M current is exactly zero throughout
- Failure: the high bout persists from 11 s through the 22 s record end; no autonomous offset
- Mechanistic readout: population-mean X reaches only 0.488 at minimum and ends at 0.501, above the archived termination boundary 0.380
- X2 70 s lifecycle / exact-D: **not authorised**
- Recovery note: simulation and NPZ completed; JSON was reconstructed without rerun after a post-simulation numpy-bool serialization failure. The failure did not change the negative gate.
- Scientific boundary: this one-seed result rejects transfer of the K_y=3 late-bout depth to the current natural-entry loop. It does not reject lower sensor placement or a recruited-area/non-local X coordinate.

Primary evidence: `candidate_lock.json`, `x_depth_screen.json`, `x_depth_screen_traces.npz`, `X1_DONE.json`, `AUTOPILOT_STOP.json`.
