# FCXR-LC2-Core status

Canonical stage: `R1_IMPLEMENTATION_ACCEPTED`.

Accepted labels:

- `R1_SUSTAINED_CONTROL_SEPARABLE`
- `R1_ACTIVE_BOUT_REANALYSIS_REQUIRED`
- `R1_LONG_REST_GAP_NOT_BRIDGED`
- `R1_CLOSED_LOOP_H_GEOMETRY_UNTESTED`
- `R2_CLOSED_LOOP_EXPLORATION_AUTHORIZED`

The raw `H_SENSOR_NOT_SEPARABLE` value in `r1_sensor/h_sensor_separability.json` is retained as a
strict full-window diagnostic only. It is not the canonical mechanism verdict and does not stop
closed-loop H/X/Z exploration.

See `r1_sensor/r1_stage_acceptance.json` and
`docs/archive/topic4/sef_hfo/fcxr_lc2_core_r1_implementation_accepted_2026-08-02.md`.
