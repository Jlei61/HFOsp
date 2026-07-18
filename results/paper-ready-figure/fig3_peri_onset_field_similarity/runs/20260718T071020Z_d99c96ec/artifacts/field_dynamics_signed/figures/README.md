# Fig3-B run-scoped diagnostic timecourses

### `<subject>_signed_broadband_...png`

timecourse producer 在当前 immutable run 内生成的诊断图，用于核对跨 seizure 中位数、方差和 coverage；paper-ready 双面板位于同 run 的 `artifacts/figures/`。

**关注点**：本目录 PNG 是 local diagnostic，不进入 canonical paper manifest，也不作为独立科学证据；正式交付只认 per-seizure CSV、aggregate、summary 与 paper-ready figure 的 manifest 条目。
