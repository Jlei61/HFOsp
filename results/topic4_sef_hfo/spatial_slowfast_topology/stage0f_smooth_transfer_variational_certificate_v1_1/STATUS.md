# Topic 4 Stage 0F v1.1 状态

- Verdict: `STAGE0F_NUMERICAL_UNRESOLVED`
- Engineering/provenance pass: `true`
- Execution exception: `false`
- Artifact completeness pass: `true`
- Peak RSS: `0.192 GiB`
- Wall time: `187.16 s`
- Stage 1: `CLOSED`
- Spatial simulation: `CLOSED`

## 固定点结果

- `z=0.85, alpha_G=15`: `periodic_orbit_derivative_unresolved`; failed gates = `base_whole_return_jv`
- `z=0.85, alpha_G=16`: `periodic_orbit_derivative_unresolved`; failed gates = `base_whole_return_jv`

本阶段只修复 homogeneous frozen fast orbit 的导数证书；无论结果如何，都不自动开放 slow lifecycle、空间耦合或 Stage 1。
