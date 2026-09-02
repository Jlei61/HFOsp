# Topic 5.2 ECoG 物理近邻 RNN v0.1 执行计划

对应 spec：`docs/superpowers/specs/2026-08-16-topic5-ecog-physical-neighborhood-rnn-v0-1-design.md`

## Phase 0：数据与可行性

1. 审计 E958/E1084 的 SQL、head、raw、v2 GPU 文件和网格触点全集。
2. 生成 gap-aware block split；只用 train block 冻结 packing contacts。
3. 只用检测时间先构建群体窗口与参与矩阵，输出事件数、触点覆盖、2×2/3×3 patch 覆盖。
4. 若 E958 达到 spec 数值 gate，锁定为 primary；E1084 无论是否达 gate 都保留透明 feasibility 状态。

输出：

- `results/topic5_ecog_physical_neighborhood_rnn_v0_1/feasibility/`
- `BLOCK_SPLIT.csv`
- `GRID_CHANNELS.csv`
- `EVENT_FEASIBILITY.json`
- `PATCH_FEASIBILITY.csv`

## Phase 1：完整网格 rank-set cache

1. 流式读取每个可用 `.data/.head` block。
2. CAR + 80–250 Hz bandpass；只读取网格触点。
3. 对 Phase 0 冻结窗口计算全部网格触点的参与和时频质心。
4. 用 5 ms tie tolerance 形成 dense rank sets，非参与为 `-1`。
5. 保存 per-block cache，再合并为 patient cache；不保存整段滤波信号。
6. 抽取 20 个事件与 legacy 少触点结果做方向/参与 sanity，不要求逐位相等。

输出：

- `cache/<subject>/per_block/*.npz`
- `cache/<subject>/events.npz`
- `cache/<subject>/provenance.json`
- cache 审计 JSON。

## Phase 2：图合同与训练

1. 构建并冻结 4-neighbour `TRUE_GRID`。
2. 预生成 31 个 degree-class `WRONG_GRID` 置换与 31 个 degree-preserving random graphs。
3. 审计每张图的节点数、边数、每节点度、互易性、连通性、真实物理边重合率和 hash。
4. 生成 train-only suffix-shuffled labels并审计边际不变。
5. 先跑 1 个 seed 的 1 个真图 + 1 个错图 smoke；通过后跑完整 3-seed 矩阵。
6. 每个 unit 保存训练历史、最佳 validation epoch、test 决策表和自由生成场。

输出：

- `graphs/<subject>/*.npz`
- `per_unit/<subject>/<arm>/<graph_id>/<seed>/`
- `TRAINING_MANIFEST.csv`

## Phase 3：训练前物理几何统计

1. 汇总 TRUE_GRID 与 31 个 WRONG_GRID 的 test contact NLL exact permutation test。
2. 平行汇总 DEGREE_RANDOM 与 SUFFIX_SHUFFLED。
3. 汇总 top-k、STOP、full/start-removed field、距离分层结果。
4. 运行 8-neighbour sensitivity；不改 primary。

输出：

- patient-level/graph-level CSV；
- `GEOMETRY_SUMMARY.json`；
- 训练前几何结果图。

## Phase 4：训练后连续局部连接削弱

1. 从 train cache 冻结 eligible 2×2/3×3 patches。
2. 为每个 patch 预生成 32 个 matched dispersed edge sets。
3. 在 TRUE_GRID checkpoint 上对 4 个剂量做 teacher-forced test scoring。
4. 对预设 reference prefixes 做 closed-loop rollout；保存连续 logits、STOP 和生成场，不只保存投影标量。
5. 验证所有 checkpoint 参数 hash 干预前后不变。
6. 计算 patch-vs-outside、patch-vs-dispersed 的差中差和剂量趋势。

输出：

- `lesion/<subject>/PATCH_CONTRACT.csv`
- per-decision/per-patch summaries；
- `LOCAL_NECESSITY_SUMMARY.json`；
- 局部削弱结果图。

## Phase 5：收口

1. 完成工程审计、科学 claim ladder 和实际 denominator 表。
2. 图目录同步写中文 `figures/README.md`，逐图说明“画什么”和“科学含义”。
3. 目视核对 PNG/PDF 同状态、字体、裁切和统计标注。
4. 写中文收口报告，按 spec §5 四格解释矩阵给唯一允许结论。
5. 不自动替换 Figure 6；先把候选 panel 给用户审阅。

