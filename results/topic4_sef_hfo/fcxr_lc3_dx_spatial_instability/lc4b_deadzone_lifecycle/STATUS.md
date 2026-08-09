# FCXR-LC4b exact-dead-zone lifecycle status

- Status: **D0-D3 nominal complete; stopped by preregistered lifecycle gate**
- D0: `DEADZONE_IDENTIFIABLE`
- D1: `DEADZONE_BASELINE_INERT` — candidate and actuator-off population-rate/active-fraction traces are byte-identical; delivered current is exactly zero
- D2: `ONSET_SURFACE_RETAINED` — `D_healthy` remains interictal and the candidate at fixed `D10` departs into a bounded high-density event train
- D3 nominal: `F2_NOMINAL_LIFECYCLE_INCOMPLETE`
- Continuous trajectory: no kick/reset/parameter step; onset at 5 s after 12 returning IEDs, then a numerically bounded non-refractory event train persists to the 70 s record end
- Autonomous offset: **absent**; postictal suppression and distributional recovery therefore cannot pass
- Final fixed 8 s: 72 self-terminating events (`9.0/s`), median duration `27.5 ms`, median participation `0.1371`; all lie outside the frozen returning-IED reference dimensions
- Termination-dose diagnosis: maximum executed dead-zone current `26.297`, versus matched target `44.862` (`58.6%`); mean relay availability never fell below `0.714`
- Exact-final-D continuation: **not authorised**, because nominal lifecycle was ineligible
- Scientific boundary: the exact dead zone solves LC4's baseline-leakage problem and preserves D/Z entry, but this locked candidate does not autonomously terminate or recover. It is not a complete seizure lifecycle.

Primary evidence: `baseline_verdict.json`, `onset_surface_verdict.json`, `nominal_lifecycle.json`; diagnostic: `figures/lc4b_deadzone_lifecycle_diagnostic.png`.
