# Topic 4 Stage 0E 状态

- Verdict: `STAGE0E_NUMERICAL_UNRESOLVED`
- Engineering/provenance pass: `true`
- Peak RSS: `0.221 GiB`
- Wall time: `221.65 s`
- Stage 1: `CLOSED`
- Spatial simulation: `CLOSED`

## 固定点结果

- `z=0.85, alpha_G=15`: `periodic_orbit_numerically_unresolved`; failed gates = `floquet_epsilon_dt_or_margin`
- `z=0.85, alpha_G=16`: `periodic_orbit_numerically_unresolved`; failed gates = `floquet_epsilon_dt_or_margin`

本阶段只审计不变的九维 frozen fast system。无论结果为何，都不自动开放 slow lifecycle、空间耦合或 Stage 1。
