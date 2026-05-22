# SEF-ITP Phase 1 Runner Contract（Archived 2026-05-22）

> **归档说明**：从 `docs/topic4_sef_itp_framework.md` §6.2 抽出的 Phase 1 runner 实施细节——4 源 loader 合同关键条款 + 8 个集成测试用例 + 完整 results 目录树。
>
> 主 doc 保留：Runner 入口命令、4 源数据来源表、smoke test 状态、任务顺序、产出图清单。本归档存放实施级 contract / test 列表 / dir 详细 layout，供 PR-T4-1 runner 维护参考。
> 配套实现：`scripts/run_sef_itp_phase1.py` + `src/topic4_attractor_diagnostics.py::load_subject_for_phase1`（2026-05-21 完工）

---

## 合同关键条款（CLAUDE.md §6 落地）

- **通道对齐**：Phase 0a `channel_names` 必须等于 lagPat loader 返回的 `channel_names`；不一致时 raise（非自动 reorder）。
- **端点 name→index**：PR-6 `source/sink` 是通道名，按 Phase 0a 顺序映射；任一通道名找不到时 raise。
- **mm 单位断言**：`require_coords=True`（默认）时 `assert_coord_result_is_mm_for_main_analysis` 把 voxel coord 拒绝在主分析外（v1.0.5 contract）。
- **valid_pool 交集**：`valid_indices = PR-6 valid_mask ∩ coord mapped_mask`；端点 S/K 中未 map 的通道被 drop，drop 数记入 `n_dropped_endpoints_no_coords_per_cluster` 审计字段。
- **forward/reverse pair schema 翻译**：Phase 0a `cluster_a/cluster_b` → `SubjectPhase1Data.cluster_A_id/cluster_B_id`。

## 集成测试 8 用例（`tests/test_run_sef_itp_phase1_integration.py`，全 GREEN 2026-05-21）

1. happy path — 4 源对齐，全 mm，full valid pool
2. lagPat channel 顺序不一致 → raise
3. coord_units=voxel + require_coords=True → raise
4. mapped_mask 部分 False → 端点 drop + audit 字段
5. PR-6 endpoint name 不在 Phase 0a → raise
6. PR-6 valid_mask 部分 False → valid_indices 取交集
7. Phase 0a forward_reverse_pair 引用的 cluster PR-6 未产出 → 该 pair 自动过滤
8. Phase 0a JSON 内 subject ≠ 文件名 subject → raise

## 产出目录完整 layout

```
results/topic4_sef_itp/                            # ← 整个 root 在 Phase 1 启动时新建
├── phase1_spatial_geometry/
│   ├── per_subject/<dataset>_<subject>.json     # H1/H2/H6 per-subject
│   ├── cohort_summary.json                       # H1/H2/H6 cohort verdicts
│   ├── figures/
│   │   ├── README.md                             # 中文
│   │   ├── endpoint_spatial_map_<subject>.png
│   │   ├── cohort_h1_violin.png
│   │   ├── cohort_h2_reversal_scatter.png
│   │   ├── cohort_h6_participation_heatmap.png
│   │   └── cohort_forest.png
│   └── soz_relation/                              # secondary
│       ├── per_subject/...
│       └── cohort_summary.json
```
