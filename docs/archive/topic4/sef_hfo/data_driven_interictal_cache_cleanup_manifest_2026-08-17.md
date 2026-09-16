# Data-driven interictal SNN cache cleanup manifest（2026-08-17）

## 删除范围

以下目录均为未被 git 跟踪的 network cache；正式 worker outputs、配置、seed、统计 sidecar、图和
provenance 不在删除范围内。

| path | files | bytes before | reason |
|---|---:|---:|---|
| `results/topic4_sef_hfo/data_driven_core_field/network_cache` | 1 | 526,564,587 | historical flexible-field network cache，可由冻结 seed/config 重建 |
| `results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache` | 15 | 7,899,012,042 | rev9 network cache，可由冻结 seed/config 重建 |
| `results/topic4_sef_hfo/data_driven_snn_cohort_v1/network_cache` | 8 | 4,215,012,298 | formal cohort shared network cache，可由 subject/seed/config 重建 |

**删除状态**：2026-08-17 已完成，三处目录均不存在。

**目录逻辑大小**：12,640,588,927 bytes（约 11.77 GiB）；文件系统可用空间实增
12,638,044,160 bytes（删除前 187,878,739,968，删除后 200,516,784,128 bytes）。

## 明确保留

- `results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/` 全部正式结果；
- `results/topic4_sef_hfo/data_driven_snn_cohort_v1/{canary,formal}/` 的 worker、summary 和 verdict；
- `results/paper-ready-figure/fig4/` 的 panel、metadata、registry 和 Figure 4 候选；
- 全部 `config/`、`docs/`、测试和 provenance；
- 全局 `~/.cache`、`/tmp` 及其他 Topic 4/5 worktree，不做跨任务清理。

## 删除后验收

删除后应满足：上述三个目录不存在；Figure 4 候选 hash 不变；NLC 与 cohort verdict 可读取；
相关单元测试仍通过。

**验收结果**：三个目录均不存在；Figure 4 candidate hash 与原附件一致；NLC/cohort JSON 可解析；
29 个 data-driven field、NLC、cohort 和 Figure 4 相关测试通过。
