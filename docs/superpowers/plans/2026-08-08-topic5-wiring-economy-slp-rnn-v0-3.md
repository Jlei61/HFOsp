# WE-SLP-RNN v0.3 Implementation Plan

> Spec: `docs/superpowers/specs/2026-08-08-topic5-wiring-economy-spatial-latent-rnn-v0-3-design.md` (LOCKED)

**Goal:** 在患者二维组织平面上，用固定稀疏资源 + 距离偏置 SET rewiring + 布线成本训练一个掩码
leaky RNN 做间期 next-rank 预测，并分析涌现出的循环拓扑、功能分工与模块必要性。

**Architecture:** 组织单元承载状态；触点只是局部读出口（tied `Hᵀ`/`H`）；唯一跨单元通道是
`M ⊙ W`；`M` 由 SET prune/regrow 在固定边数下演化。

**Tech Stack:** torch 2.5.1+cu124 (`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`),
numpy, scipy, networkx (Louvain via `networkx.community.louvain_communities`), matplotlib.

## Global Constraints

- 队列 21 人 / 31 拟合；`fit_id = <subject>__<scope>`，`scope ∈ {shared, own_a, own_b}`。
- 密度 `ρ=10%` a priori 冻结；`d_0 = 10 mm`；`λ_STOP=1.0`；`ε=0.1mm`；`NODE_SEED=20260808`。
- 核宽 `σ = max(2mm, 0.5×median NN spacing)`，硬截断 `3σ`；`M = min(64, max(24, 4C))` 长到每触点 ≥3 单元、上限 192。
- batch `= min(1024, ceil(n_train/8))`，**同一拟合所有臂相同**；显存问题只调并发。
- 全队列同一设备，写入 `RUN_CONTRACT.json` 后不换。
- 收敛是进入分析的前置条件；撞 epoch 上限 → `converged=false` → 不进分析。
- 队列统计单位是患者；非共线患者两个拟合先在患者内平均。
- 跨模式面板分母固定 n=11。
- 原子写（`.tmp` → rename），`DONE.json` 最后写。
- **不得编辑正在运行的 bash 脚本**（用 python orchestrator，不用 bash）。

---

## Task 1: cache builder — 几何、事件、A/B 标签接入

**Files:** Create `scripts/build_topic5_we_cache.py`; Test `tests/test_topic5_we_cache.py`

**Produces:** `results/topic5_wiring_economy_slp_rnn_v0_3/cache/<fit_id>/{plane.npz, events.npz, provenance.json}`
- `plane.npz`: `contacts_xy_mm (C,2)`, `nodes_xy_mm (M,2)`, `H (C,M)`, `D_mm (M,M)`, `sigma_mm`, `scale_mm`
- `events.npz`: `ranks (N,C) int16` densified (-1 = absent), `split (N,) int8` (0/1/2/-1), `mode (N,) int8` (0/1/-1)

- [ ] Step 1: 写失败测试 — 平面组三分（11/10）、`len(valid_idx)==len(labels)` 全 21、
      `event_source_index` 无重复且 `<N_all`、选中 block 事件数之和 == 数据集事件数。
- [ ] Step 2: 跑测试确认失败。
- [ ] Step 3: 实现。共线判据读 `template_gradient_fields` 的 `planes.shared` 是否存在且 `status=='ok'`。
      `mode_to_template` 由 `adaptive_cluster.clusters[k].template_rank` 与 artifact `rank_a/rank_b`
      的 Spearman 相关取最大者确定。
- [ ] Step 4: 跑测试确认通过；跑全 21 生成 31 个 cache。
- [ ] Step 5: commit。

## Task 2: 模型 — 掩码 leaky RNN / GRU + SET

**Files:** Create `src/topic5_wiring_economy_rnn.py`; Test `tests/test_topic5_wiring_economy_rnn.py`

**Produces:**
- `class WEConfig` (dataclass): `cell {'rnn','gru'}`, `arm`, `n_contacts`, `n_nodes`, `density`,
  `state_dim=1`, `eta`, `d0_mm=10.0`, `seed`, `H`, `D_mm`
- `class WEModel(nn.Module)`: `.forward(x, recruited, valid) -> (logits, stop_logits)`,
  `.edge_strength() -> Tensor (M,M)`, `.wiring_cost() -> Tensor`, `.rewire(zeta) -> int`,
  `.freeze_mask()`, `.graph_snapshot() -> dict`
- `def next_rank_stop_loss(logits, stop_logits, target, available, valid, is_last, stop_weight)`

- [ ] Step 1: 失败测试 —
      (a) 活跃边数在 rewire 前后恒定；
      (b) `SPATIAL_SET` 的新边平均距离显著短于 `RANDOM_SET`（同种子、同初始图）；
      (c) 无自环、`M` 对三个 GRU 矩阵是同一个；
      (d) 输入层无 dense M×M（关掉 `M⊙W` 后不同单元之间零互信息 — 用 `h` 对单个触点脉冲的响应支持集检验）；
      (e) `wiring_cost` 对 `d0` 线性、对 `scale_mm` 有量纲一致性；
      (f) `state_dim=2` 时掩码按块施加（`M ⊗ 1_{2×2}`）。
- [ ] Step 2: 跑测试确认失败。
- [ ] Step 3: 实现。新边权重初始化为 0。prune 按 `S_ij` 升序。regrow `P ∝ 1/(d+ε)`。
- [ ] Step 4: 跑测试确认通过。
- [ ] Step 5: commit。

## Task 3: 训练单元

**Files:** Create `scripts/train_topic5_we_unit.py`; Test `tests/test_topic5_we_train.py`

**Produces:** `per_subject/<fit_id>/<arm>/seed<k>/{metrics.json, graph.npz, DONE.json}`
`metrics.json` 键：`converged`, `n_epochs`, `hit_ceiling`, `val_nll`, `test_nll`, `test_nll_by_mode`,
`c_wiring`, `edge_count`, `mean_edge_len_mm`, `long_edge_fraction`, `rollout`, `generator_degenerate`,
`fit_scope`, `label_coverage`, `device`, `batch_size`, `config_sha256`

- [ ] Step 1: 失败测试 — (a) `converged=false` 当且仅当撞上限；(b) 掩码冻结前不计早停；
      (c) rollout 用 argmax 不用 0.5 阈值；(d) 生成守卫 `≥15%` 名次不同。
- [ ] Step 2/3/4: 红→绿。
- [ ] Step 5: commit。

## Task 4: orchestrator（python，非 bash）

**Files:** Create `scripts/launch_topic5_we_cohort.py`

- 任务清单 = `fit_id × arm × seed`；按**绝对路径**查 `DONE.json` 与在飞进程双重去重。
- `--max-workers` 可调；`OMP_NUM_THREADS=2`。
- [ ] Step 1–5: 实现 + 干跑（`--dry-run` 打印 420 个单元）+ commit。

## Task 5: 设备 benchmark + η 扫描 + 2 维探针

- [ ] Step 1: 单个代表患者上 CPU vs GPU 计时，写 `RUN_CONTRACT.json`。
- [ ] Step 2: 8 位开发患者 × 5 个 η，只 `SPATIAL_SET`、seed 0，用 validation 选拐点，冻结。
- [ ] Step 3: 8 位患者 `state_dim=2` 探针。
- [ ] Step 4: commit `RUN_CONTRACT.json`。

## Task 6: 主队列 248 + C2 对照 31 + GRU 复核 93

- [ ] 后台跑，`DONE.json` 计数监控。收敛率 <90% 时停下来查。

## Task 7: 图分析模块（拓扑 + 三个对照）

**Files:** Create `src/topic5_we_graph_analysis.py`; Test `tests/test_topic5_we_graph_analysis.py`

**Produces:** `modularity_q`, `clustering`, `small_worldness`, `edge_length_hist`,
`participation_coefficient`, `connector_fraction`, `long_range_fraction`,
`length_preserving_rewire(M, D, n_bins, rng)`, `contiguous_random_lesion(nodes_xy, size, rng)`

- [ ] Step 1: 失败测试 — (a) `length_preserving_rewire` 保住每单元进出度与边长分箱直方图；
      (b) `contiguous_random_lesion` 返回空间连续的一块且大小匹配；
      (c) 二维随机几何图的 modularity 显著高于 Erdős–Rényi（对照本身有效）。
- [ ] Step 2–5: 红→绿→commit。

## Task 8: 队列分析

**Files:** Create `scripts/analyse_topic5_we_cohort.py`

按 §9 五节输出到 `analysis/`：`pareto.json`, `topology.json`, `function.json`, `lesion.json`,
`tendency.json`, `gates.json`。
- 患者内先平均（非共线两个拟合）；跨患者画配对差。
- 跨模式项分母写死 `n_cross_mode_patients=11`。
- 新鲜度检查：输入 `DONE.json` mtime 晚于上游且覆盖当前队列全集。
- [ ] Step 1–5。

## Task 9: 六格主图 + figures/README.md（中文）

**Files:** Create `scripts/plot_topic5_we_figures.py`
- [ ] 先读 `docs/figure_style_guide.md`。
- [ ] 六格按 §12。图生成后再写 README。

## Task 10: 验收 + 收官

**Files:** Create `scripts/accept_topic5_we_v0_3.py`, `results/.../CLOSEOUT.md`
- 门：cache 31/31、单元覆盖、收敛率、A/B 标签覆盖、五个科学门 G1–G5 的判词、禁止措辞扫描。
- [ ] commit。
