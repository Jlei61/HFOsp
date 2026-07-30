# Topic 5 RNN internal-state reduction v0.1 execution plan

## 目标

冻结现有 34 人、3 seed full-history GRU，不以提高 AUC 为目标，定量回答：

1. hidden state 是否低维且跨 seed/时间稳定；
2. 它是否包含超出 static、unordered、last-set 和 rank-shuffle 的顺序信息；
3. 少数 hidden directions 如何改变 contact-level rank distribution；
4. 这些 contact fields 是否能读回严格 clinical-onset early-ictal energy field。

## Milestone A：输入与复现审计

- 核对 34×3×5 checkpoint、dataset fingerprint、contact order 和 split；
- 固定 `train60/validation20/heldout20`；
- 写出 input manifest，记录缺失 cell，但不因局部缺失停止其他 cell；
- 固定 event subsampling，所有 seeds 使用完全相同的 event/prefix IDs。

验收：manifest 可重建每个 subject/seed/control 输入；target 尚未由本阶段代码读取。

## Milestone B：hidden-state inventory

- 对 full-history 与 rank-shuffle GRU teacher forcing；
- 每个 split 最多取 `2048/1024/2048` events，使用等距确定性抽样；
- 保存 prefix-level hidden state、event index、rank step；
- 计算原模型 heldout next-set/STOP NLL，和旧结果交叉核对。

资源：CPU 并行 12 workers，每 worker 1–2 threads；状态以 float32 压缩 NPZ 保存。

## Milestone C：低维性与稳定性

- train60 PCA；
- effective rank、k80/k90/k95；
- `k={1,2,4,8,16,32}` heldout reconstruction/decoder NLL；
- 每个 k 的随机子空间 sensitivity；
- raw/residual linear CKA；
- split-half subspace overlap；
- seed-within-patient collapse。

验收：每个指标均有 patient-level table、cohort median、95% CI 和 n_positive。

## Milestone D：增量信息 probes

固定 probe family：

- node prior；
- last-set + progress；
- unordered recruited set + last-set + progress；
- 上述 observable features + full hidden PCA；
- 上述 observable features + rank-shuffle hidden PCA。

任务：

- next action/STOP；
- future participation；
- remaining-rank weighted score。

正则化只在 validation20 选择，heldout20 只读一次。报告 full-hidden 增量的完整 k 曲线，
不按单个 k 做 post-hoc 选择。

## Milestone E：顺序扰动与 state perturbation

- 对同一 heldout prefix 打乱已观察 rank-set 顺序，保持 prefix contact membership、
  target、candidate mask 和 STOP 不变；
- 对稳定 PCA/output-coupled directions 做 `±0.25/0.5/1 SD`；
- 记录 next-action NLL、JS divergence、STOP shift、contact loading 与 seed stability；
- 代表患者图必须同时画真实 interictal contact distribution 与模型 distribution。

## Milestone F：严格 early-ictal read-back

只有 `INTERICTAL_FREEZE.json` 写出后运行：

- strict clinical-onset 16 人/106 seizure；
- `0–10 s`、`1–150 Hz`；
- fixed participation、fixed endpoint field；
- five-field omnibus sensitivity；
- 5000 all-contact coherent permutations；
- within-shaft sensitivity（可映射患者）；
- full、rank-shuffle、static、perturbation fields 的 paired patient-first comparison。

## Milestone G：图与报告

建议六块：

| Panel | 科学含义 |
|---|---|
| A | 自监督任务、hidden inventory 和三段时间拆分 |
| B | hidden spectrum、effective rank 与 k-NLL fidelity |
| C | raw/residual cross-seed stability |
| D | full hidden 相对 observable/rank-shuffle 的 probe 增量 |
| E | state perturbation 如何改变 contact distribution |
| F | 固定 hidden-derived fields 与 strict early-ictal energy 的 read-back |

图目录同步写中文 `figures/README.md`。报告按 Tier A/B/C 分级，但不使用单项结果提前
停止，不把工程完成等同于机制成立。

## 运行与监控

- 长任务使用 `nohup`；
- 每 60 秒记录进度、CPU/RAM/GPU；
- 单 worker OOM/异常只重试该 cell；
- 先完成 3 subject smoke，再开放 34×3；
- 日志、失败原因、重试次数和最终完整性写入 `RUN_STATUS.json`。
