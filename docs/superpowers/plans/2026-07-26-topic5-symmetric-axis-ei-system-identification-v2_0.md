# Topic 5 / Figure 6 v2.0 execution plan

> **先读**：
> 1. `docs/archive/topic5/persistent_path_mode_rnn_closeout_and_v2_pivot_2026-07-26.md`
> 2. `docs/superpowers/specs/2026-07-26-topic5-symmetric-axis-ei-system-identification-v2_0.md`
> 3. `docs/paper-draft/figure4_subject_specific_snn.md`
>
> 本计划只实现 v2.0。v0.7/v0.9/v1.0 已冻结，不得继续调参。

## 0. 执行原则

1. 每一阶段先写测试，再实现，再跑最小数据，再进入重任务。
2. 所有结果写入新目录；旧 RNN 和 Topic 4 SNN 产物不覆盖、不删除。
3. 所有 run 保存 config fingerprint、git commit、seed、stdout/stderr、checkpoint、
   metrics、显存峰值和 completion state。
4. 所有图在实际生成后补中文 `figures/README.md`，并做肉眼 QA。
5. 训练只使用 masked interictal rank。A/B、IEI、ictal values 在 Gate 4 前保持
   hard seal。
6. GPU 并行前先做显存探测；单进程上限从 smoke 实测值推算，保留至少 20% 显存与
   32 GiB 系统内存余量。launcher 使用 `nohup`，monitor 独立记录，不用轮询刷屏。

## Milestone A：冻结旧线并建立 provenance

### A1. 旧线 manifest

新增：

- `results/topic5_structured_axis_graph/persistent_path_mode_closeout_v1_0/closeout_manifest.json`
- `.../artifact_inventory.csv`

manifest 只索引：

- v0.7/v0.9/v1.0 configs/specs；
- 510-run formal gate 与关键统计；
- bounded-negative Figure 6；
- `ictal_target_read=false`；
- producer commit 和输入 fingerprint。

不复制 510 个 checkpoint，不改旧 result root。

### A2. 回归测试

运行旧线的最小审计：

```bash
conda run -n cuda_env pytest -q \
  tests/test_topic5_rank_distribution_dataset.py \
  tests/test_topic5_rank_distribution_model.py \
  tests/test_topic5_path_mode_prior.py \
  tests/test_topic5_persistent_path_rnn.py \
  tests/test_topic5_persistent_path_trainer.py \
  tests/test_topic5_persistent_path_formal_analysis.py \
  tests/test_topic5_persistent_path_internal_dynamics.py \
  tests/test_topic5_structured_rank_rnn_figure.py
```

通过标准：全部 green；正式 gate JSON 仍为
`formal_interictal_gate_pass=false`、`ictal_target_read=false`。

## Milestone B：SNN synthetic benchmark 合同

### B1. 只读审计现有 SNN exporter

审计：

- `scripts/run_sef_hfo_subject_snn.py`
- SNN engine 的 spike/current return schema
- 当前 `readout_*.json` 和 `figdata_*.npz`

输出：

`results/topic5_symmetric_axis_system_id/snn_benchmark/schema_audit.json`

必须回答：

- E 与 I 的逐神经元/逐触点 proxy 是否能导出；
- source-left/source-right 是否可在同一 network realization 下 paired；
- axis、AR、kernel scales 和 observation montage 是否有可靠真值；
- 现有 selected Figure 4 artifact 哪些不能进入 benchmark。

若 I proxy 无法无歧义导出，先保留 excitation/restraint 数学状态，并在代码、图和文档中
禁用 biological E/I wording；不得为赶进度伪造 I target。

### B2. 新建 benchmark exporter

建议新增：

- `scripts/export_topic5_snn_system_id_benchmark.py`
- `src/topic5_symmetric_axis_rnn.py`
- `tests/test_topic5_snn_system_id_benchmark.py`

结果目录：

```text
results/topic5_symmetric_axis_system_id/
└── snn_benchmark/
    ├── manifest.json
    ├── schema_audit.json
    ├── per_seed/
    └── figures/
        └── README.md
```

benchmark 四类 generator：

1. `symmetric_anisotropic_left/right`；
2. `symmetric_isotropic_left/right`；
3. `directed_asymmetric_left/right`（优先使用独立轻量 contact-graph generator，
   不为负对照改动 guarded SNN engine）；
4. `restraint_intact/lesioned`。

同一个 paired seed 必须复用 neuron positions、random connectivity realization 和 montage，
只改变预注册的 source/control。不得按“有漂亮双向事件”筛 seed。

### B3. Benchmark 单元测试

必须测试：

- axis sign invariance；
- left/right 配对 fingerprint 相同；
- isotropic 真值 ratio = 1；
- asymmetric control 真值确实非对称；
- event ranks 中未参与触点为 mask，不是 finite phantom rank；
- synthetic latent truth 不进入 training batch；
- fixed seed bitwise/deterministic 到允许精度。

## Milestone C：v2.0 最小模型

### C1. 纯函数先行

在 `src/topic5_symmetric_axis_rnn.py` 实现并测试：

- `symmetric_anisotropic_kernel(coords, axis, l_parallel, l_perp)`；
- sign flip `u -> -u` 输出完全不变；
- patient-centered / median-distance coordinate normalization；
- 99% operator-energy eigentruncation 与 full-kernel equivalence；
- `local_shaft_kernel(...)`；
- topology-only fallback；
- masked set encoder；
- nonnegative excitation/restraint update；
- next-set、STOP、future participation、remaining-rank losses。

测试文件：

- `tests/test_topic5_symmetric_axis_kernel.py`
- `tests/test_topic5_symmetric_axis_losses.py`
- `tests/test_topic5_symmetric_axis_state.py`

P0 invariants：

- `W == W.T`；
- 没有 dense contact-mixing parameter；
- forward/reverse 不产生两套参数；
- lesion 前后 node-bias tensor fingerprint 相同；
- heldout20 不参与 \(W\)、\(b_i\) 或 hyperparameter selection。
- 低秩压缩与完整 kernel 在 SNN 主终点上的相对误差不超过 1%；否则 formal 默认完整
  kernel，不再调 rank。

### C2. Trainer

新增：

- `config/topic5_symmetric_axis_ei_rnn_v2_0.yaml`
- `scripts/train_topic5_symmetric_axis_rnn.py`
- `scripts/analyze_topic5_symmetric_axis_snn_recovery.py`

trainer 分两层：

1. shared dynamics：在 training subjects / synthetic training seeds 上拟合；
2. patient/system identification：只用目标患者 train80 拟合 axis/scales/local offsets；
3. heldout20 只作 evaluation。

所有 loss 权重和范围写入 config。任何从命令行覆盖的参数都必须写回 resolved config。

## Milestone D：Gate 0 SNN 参数恢复

### D1. 3-seed smoke

顺序执行四类 generator × 3 paired seeds，CPU 数据生成与 GPU 训练可并行，但每个输出
目录独占。检查：

- 数据和真值 schema；
- loss 下降；
- 无 NaN/OOM；
- axis recovery 与 source reversal 方向正确；
- lesions 确实改变对应算子。

smoke 只修 bug，不改 spec gate。

### D2. 12-seed confirm

用 `nohup` 启动至少 12 paired seeds，launcher 与 monitor 建议为：

- `scripts/run_topic5_symmetric_axis_snn_confirm.sh`
- `scripts/monitor_topic5_symmetric_axis_snn.py`

confirm 完成后生成：

- `snn_recovery_per_seed.csv`
- `snn_recovery_summary.json`
- axis、anisotropy、source reversal、misspecification、state-recovery 五张诊断图；
- `figures/README.md`。

按 spec Gate 0 一次性判定。失败即停止，不启动人体 pilot。

## Milestone E：人体数据与几何审计

### E1. 复用 v0.4，不重建事件

新增只读 adapter：

`scripts/build_topic5_symmetric_axis_dataset_index.py`

它不得复制上百万事件，只写：

- 34 人 NPZ/JSON fingerprint；
- train80/heldout20 数量；
- contact/geometry/shaft eligibility；
- `geometry_full`、`topology_only` 分层；
- forbidden input audit。

输出：

`results/topic5_symmetric_axis_system_id/human_dataset_index/`

硬断言：

- 恰好 34 人；
- 当前预期 `geometry_full=25`，若真实数据变化则停止并修订合同，不能静默换分母；
- 所有 patient train80/heldout20 非空；
- `candidate_target_patient` 不参与 routing。

### E2. heldout read-back 合同

建立独立函数，在模型冻结后才从 heldout20 计算：

- data-driven unsigned propagation axis；
- source-side 分层；
- optional A/B/template correspondence。

测试必须证明训练阶段无法 import/read 该 artifact。

## Milestone F：Gate 1 三患者 pilot

开发病例固定：

- `epilepsiae_1073`
- `epilepsiae_1146`
- `yuquan_chenziyang`

conditions：

- full；
- node-bias/no-history；
- local-isotropic；
- axis-shuffle；
- no-restraint；
- asymmetric upper bound。

3 subjects × 3 seeds。先单进程测峰值显存，再并发。产物：

- resolved config；
- per-run logs/checkpoints/states；
- patient-seed metrics；
- node-bias fingerprint audit；
- Gate 1 JSON；
- 诊断图和中文 README。

只允许两类修复：

1. 明确代码/shape/determinism bug；
2. 资源设置导致的 OOM。

不得根据三位患者表现改 loss、轴参数化、gate 或选择患者。

## Milestone G：Gate 2/3 全 34 人正式实验

### G1. 冻结与 dry-run

正式运行前写：

- `FORMAL_LOCK.json`：config hash、code commit、34 人、3 seeds、conditions、metrics、
  stats、geometry subset；
- 6 个 dry-run completion states；
- 目标 seal 审计。

### G2. 并行训练

建议每个 fold/seed 共享 full-model training，能从同一 checkpoint 做的 frozen lesion
不要重复训练；nested isotropic/no-history/asymmetric control 必须独立重训。具体并发数由
smoke 峰值显存决定，不写死。

launcher 必须：

- 自动过滤空 subject；
- 断言 34 人 × 3 seeds × frozen condition count；
- 原子写 `run_state.json`；
- 已完成 run 跳过，非完整目录先归档再重跑；
- 每 5 分钟写一次 compact monitor snapshot；
- 系统内存或显存余量低于门限时暂停派发，不杀已运行任务。

### G3. patient-first 分析

新增：

- `scripts/analyze_topic5_symmetric_axis_formal.py`
- `scripts/analyze_topic5_symmetric_axis_bidirectionality.py`

固定输出：

- `run_inventory.csv`
- `conditional_prediction_patient_seed.csv`
- `conditional_prediction_statistics.csv`
- `axis_stability_geometry_full.csv`
- `source_reversal_statistics.csv`
- `state_lesion_statistics.csv`
- `formal_gate_summary.json`
- 31 人 development-excluded sensitivity。

统计顺序：event → patient-seed → patient median → cohort test。禁止把事件或 seed 当独立
患者。

Gate 2/3 任一失败：

- 写 bounded-negative archive；
- 生成 A–E 诊断图，F 标记 target sealed；
- 不建立 ictal value cache，不运行 Gate 4。

## Milestone H：Gate 4 clinical-onset 动态迁移

### H1. 先 inventory，后定分母

新增：

`scripts/audit_topic5_clinical_onset_dynamic_transfer.py`

它先在不读取能量/招募数值的情况下，逐一核对 34 人：

- clinical-onset annotation；
- clinical-onset contacts；
- ictal rank producer 和 contact-name exact join；
- seizure 数与排除原因；
- EEG-onset-only 标记。

输出：

- `target_inventory.csv`
- `target_denominator.json`
- `TARGET_LOCK.json`

只有 inventory 审阅通过后，才读取 target values。不得预设仍是 13 人。

### H2. 冻结 transfer

新增：

- `scripts/build_topic5_clinical_onset_rank_prefixes.py`
- `scripts/run_topic5_symmetric_axis_frozen_ictal_transfer.py`
- `scripts/analyze_topic5_symmetric_axis_ictal_transfer.py`

primary：

- clinical-onset-aligned earliest rank set(s) → later participation；
- clinical-onset-aligned earliest rank set(s) → remaining recruitment rank。

controls 共用完全相同的 node bias 和 target events。所有模型参数只读，加载后
`requires_grad=False`，脚本断言 optimizer 不存在。

secondary：

- clinical-onset `[0,10] s`、`1–150 Hz` static energy compatibility；
- EEG-onset-only sensitivity，单独目录、永不 pool。

这一步是 retrospective cross-state completion，不写 prospective warning。

## Milestone I：Figure 6 与论文收口

### I1. 六块图

producer：

`scripts/paper_figures/plot_fig6_symmetric_axis_system_identification.py`

输出：

```text
results/paper-ready-figure/fig6_symmetric_axis_system_identification/
├── figures/
│   ├── README.md
│   ├── fig6_symmetric_axis_system_identification.png
│   └── fig6_symmetric_axis_system_identification.pdf
└── fig6_symmetric_axis_system_identification_summary.json
```

Panel A–F 严格按 spec §8。若 Gate 4 未开放，F 必须显示“target sealed”，不得以旧静态
field 填空。

### I2. 视觉与科学双验收

科学 QA：

- 每个数字可回溯到 aggregate artifact；
- denominator、seed folding、null、q 值和 sign-invariant axis 全部正确；
- A/B 未进入训练；
- topology-only 病例不混进三维轴结论；
- clinical onset 与 EEG onset 不混池。

视觉 QA：

- 300 dpi PNG + vector PDF；
- 字号、线宽和色盲安全；
- patient-level points 可见；
- 无被图例遮挡、裁切、过密标签；
- `figures/README.md` 逐图解释科学问题与关注点。

### I3. 文稿

更新：

- `docs/paper-draft/figure6_*.md`
- `docs/archive/topic5/INDEX.md`
- `results/FIGURE_INDEX.md`

旧 Figure 6 bounded-negative 保留 supplementary/provenance，不覆盖。

## 交付与停止点

每个 Milestone 都写一个 `*_GATE.json`。只有状态为 `PASS` 才自动进入下一阶段。
`FAIL`、`BLOCKED_INPUT` 或 `TARGET_SEALED` 必须停止 launcher，生成可读报告，不尝试用
下游结果补救上游门。

第一位执行 agent 的实际起点是 **Milestone B1**，不是直接写 RNN，也不是直接跑 34 人。
