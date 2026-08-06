# Topic 4 Stage 0F 状态

- Verdict: `STAGE0F_NUMERICAL_UNRESOLVED`
- Engineering/provenance pass: `true`
- Peak RSS: `0.173 GiB`
- Wall time: `634.99 s`
- Stage 1: `CLOSED`
- Spatial simulation: `CLOSED`

## 固定点结果

- `z=0.85, alpha_G=15`: `periodic_orbit_derivative_unresolved`; failed gates = `base_smooth_shooting`
- `z=0.85, alpha_G=16`: `periodic_orbit_derivative_unresolved`; failed gates = `base_smooth_shooting`

本阶段只解决 Stage 0E 的 transfer derivative / Floquet 数值证书。即使通过，也不证明 slow entry/exit、空间 recruitment 或完整 SNN lifecycle。
