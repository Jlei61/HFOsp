# Topic 3：Where / SOZ 空间归因

> 状态：**Epilepsiae 基础设施已解锁** — 2026-04-15 更新
> 范围：只讨论慢调制和时序差异在空间上发生在哪里，尤其是 SOZ / non-SOZ 的分离。

---

## 1. 这个 topic 只回答什么问题

本 topic 只回答：

1. lagPat 群体事件框架为什么回答不好 where。
2. per-channel relaxed-refine 框架下，SOZ 与 non-SOZ 是否真的不同。
3. 观察到的差异中，哪部分更像全局调制，哪部分更像 SOZ 的局部短程记忆。
4. **新增**：跨数据集 HFO 检测 SOZ 区分度验证（Raw / Refined AUC）。

它**不**回答：

- `~2 Hz` peak 是不是真的：那是 `docs/topic2_between_event_dynamics.md`
- 单个事件内部传播是否刻板：那是 `docs/topic1_within_event_dynamics.md`

---

## 2. 一句话当前结论

lagPat 群体事件框架中的 SOZ / non-SOZ 对比被结构性选择偏差严重污染。转到 per-channel relaxed-refine 后，raw serial correlation 的 SOZ 优势基本消失；更可信的信号是：**SOZ 通道在全局慢调制之上，可能额外保留了更强的局部短程记忆。**

跨数据集 SOZ-AUC 验证证实新 pipeline 的检测质量与老论文一致：Yuquan refined AUC 0.874（老论文 0.857），Epilepsiae refined AUC 0.952。

---

## 3. 核心证据链

### 3.1 为什么旧 lagPat 框架不适合回答 where

旧框架有三层结构性偏差：

1. refine 先把通道宇宙截成 HFO 高发通道，天然富集 SOZ
2. 群体事件的 SOZ 标注是二值的，只要有一个 SOZ 通道参与就算 SOZ 事件
3. 群体事件本身就是多通道重叠现象，在 SOZ 富集通道宇宙里几乎必然被 SOZ 主导

因此旧 lagPat 结果里看到的 SOZ > non-SOZ，很可能混着：

- 真实空间差异
- 事件率差异
- 通道选择偏差
- group-event 定义本身的偏向

### 3.2 per-channel relaxed-refine 为什么更干净

新路线不是重写整条 pipeline，而是：

- 从 `*_gpu.npz` 加载 per-channel events
- 用 relaxed refine 扩大通道集
- 在通道层面计算 IEI / detrended metrics / event rate
- 再做 SOZ / non-SOZ 对比

这样统计单元从"群体事件类别"变成"subject 内的通道"，这是更像样的数据结构。

### 3.3 Per-channel 时序指标结果（Yuquan-only PR-1）

Yuquan-only PR-1（Epilepsiae `gpu.npz` 当时全坏）得到：

- Yuquan 审计：`11/18` 有效，PASS
- Epilepsiae 审计：`0/20` 有效，FAIL（已通过重跑解决，见 §4）
- relaxed refine 选 `k = 0.0`，中位通道数约 `33`
- 有效 SOZ/non-SOZ 配对：`9` subjects

主要结果：

- raw `iei_lag1_r`：SOZ 与 non-SOZ 无差异，`p = 1.000`
- detrended `iei_detrended_r`：SOZ 稍高，`7/9` 同方向，`p = 0.250`
- `detrend_fraction`：SOZ 更低，方向 `7/9` 指向"SOZ 慢漂移占比更低"
- `iei_median`：SOZ 更短，`p = 0.055`

### 3.4 这意味着什么

最有信息量的不是某个单独的 `p` 值，而是分解后的方向：

- raw serial correlation 无差异
- 但 detrended residual 倾向 SOZ 更高
- `detrend_fraction` 倾向 SOZ 更低

最合理的解释是：

- 全局慢调制对 SOZ / non-SOZ 都在起作用
- SOZ 通道之上还叠加了额外的局部短程记忆

这比"SOZ 整体更强"要精确得多，也比旧 lagPat 结果更可信。

---

## 4. HFO 检测基础设施重建与 SOZ-AUC 验证（2026-04-15）

### 4.1 重跑动机

Topic 3 PR-1 暴露了根本性的上游数据缺口：Epilepsiae 的 legacy `*_gpu.npz` 全部是 216 字节损坏桩，Yuquan 也有 7/18 缺失。新 pipeline 从原始信号（EDF / .data+.head）重跑 HFO 检测，自主产出 `*_gpu.npz` 和 `_refineGpu.npz`。

### 4.2 两数据集关键参数对比

重构前深入分析了 legacy 代码（`epilepsiae_detectHFOs.py` vs `p16_cuda_24h_bipolar.py`），确认以下差异并在 `config/subject_params.json` 中正确反映：


| 参数          | Yuquan (SEEG)   | Epilepsiae (ECoG/SEEG) | 来源                                  |
| ----------- | --------------- | ---------------------- | ----------------------------------- |
| 参考方式        | bipolar (gap=1) | **CAR**                | legacy 代码确认                         |
| side_thresh | 1.5             | **2.0**                | `epilepsiae_detectHFOs.py __main`__ |
| 重采样         | 800 Hz          | **原生 fs**（大部分 1024 Hz） | legacy 代码确认                         |
| rel_thresh  | 2.0             | 2.0                    | 一致                                  |
| abs_thresh  | 2.0             | 2.0                    | 一致                                  |
| band        | [80, 250]       | [80, 250]              | 一致                                  |
| min_sfreq   | —               | 500 Hz (Nyquist 门禁)    | 新增安全措施                              |


Legacy 的一个隐患：`epilepsiae_detectHFOs.py` 模块顶部默认 `rel_thresh=3`，只在 `__main`__ 批跑时才改为 2。新 pipeline 不受此影响，参数统一从 JSON 读取。

### 4.3 当前完成状态

**Yuquan：21/21 有数据 subject 全部完成**（`results/hfo_detection/<name>/`）。

**Epilepsiae：19/20 完成，1/20 进行中**（`results/hfo_detection/<id>/`）：


| Subject | Blocks | Ch (ref)  | 状态         | 备注                              |
| ------- | ------ | --------- | ---------- | ------------------------------- |
| 253     | 268    | 29 (CAR)  | ✅          | fs=512 Hz，通过 Nyquist 门禁         |
| 139     | 130    | 41 (CAR)  | ✅          | 43 blocks 256 Hz 被正确跳过          |
| 384     | 65     | 93 (CAR)  | ✅          | 65 blocks 256 Hz 跳过（mixed fs）   |
| 442     | 178    | 70 (CAR)  | ✅          |                                 |
| 548     | 147    | 83 (CAR)  | ✅          | 论文 E14                          |
| 583     | 63     | 40 (CAR)  | ✅          | 143 blocks 256 Hz 跳过（mixed fs）  |
| 590     | 254    | 95 (CAR)  | ✅          |                                 |
| 620     | 256    | 38 (CAR)  | ✅          |                                 |
| 635     | 123    | 57 (CAR)  | ✅          |                                 |
| 818     | 255    | 46 (CAR)  | ✅          |                                 |
| 916     | 435    | 73 (CAR)  | ✅          |                                 |
| 922     | 114    | 82 (CAR)  | ✅          |                                 |
| 958     | 225    | 96 (CAR)  | ✅          |                                 |
| 1073    | 231    | 71 (CAR)  | ✅          | 首轮 GPU OOM → memory flush 修复后成功 |
| 1077    | 48/189 | 121 (CAR) | 🔄 CPU 运行中 | 121 ch 超 24GB GPU 显存 → CPU 模式   |
| 1084    | 252    | 87 (CAR)  | ✅          |                                 |
| 1096    | 165    | 74 (CAR)  | ✅          |                                 |
| 1125    | 160    | 62 (CAR)  | ✅          |                                 |
| 1146    | 117    | 114 (CAR) | ✅          |                                 |
| 1150    | 161    | 124 (CAR) | ✅          |                                 |


**GPU OOM 修复**：在 `scripts/run_hfo_detection.py` 中添加了 `_flush_gpu_memory()`，每个 block 检测后显式释放 CuPy 内存池（`pool.free_all_blocks()` + `pinned.free_all_blocks()`）。1073 修复后成功。1077 因 121 通道在单 block 内就需要 >24GB，只能 CPU 模式。

**Mixed-fs subjects**（139, 384, 583）：部分 recording 为 256 Hz，被 min_sfreq=500 Hz 门禁正确跳过。Refine 仅基于 ≥500 Hz 的有效 blocks。

### 4.4 SOZ-AUC 跨数据集验证

脚本：`scripts/plot_refine_soz_validation.py --dataset epilepsiae`


|                             | **Yuquan** | **Epilepsiae**               |
| --------------------------- | ---------- | ---------------------------- |
| 有 SOZ + refine 配对           | 20         | 14                           |
| 无 SOZ 信息（排除）                | 1          | 5 (384, 620, 818, 916, 1125) |
| **Raw AUC 均值**              | **0.836**  | **0.935**                    |
| Raw AUC 中位数                 | 0.845      | 0.941                        |
| **Refined AUC 均值**          | **0.874**  | **0.952**                    |
| Refined AUC 中位数             | 0.921      | 0.978                        |
| Wilcoxon p (Raw vs Refined) | **0.018**  | 0.158                        |


**关键观察**：

1. **Epilepsiae Raw AUC 显著高于 Yuquan**（0.935 vs 0.836）。CAR 参考下 SOZ 通道的 HFO 事件率本身就有更好的区分度。
2. **Refine 对 Yuquan 有统计显著提升**（p=0.018），对 Epilepsiae 不显著（p=0.158），原因是 Raw 基线已经很高（天花板效应）。
3. **Yuquan Refined AUC 0.874 与老论文 0.857 吻合**——重构后 pipeline 检测质量可靠。
4. **两个 Epilepsiae 异常 subject**：253（refine 后 AUC 大幅下降 0.872→0.731，只有 29 ch + fs=512 Hz）、1150（轻微下降 0.970→0.926，3 SOZ / 124 total 极端不平衡）。

**Epilepsiae Per-Subject 结果**（15 subjects with SOZ）：


| Subject   | Raw AUC | Refined AUC | Δ      | SOZ/Total |
| --------- | ------- | ----------- | ------ | --------- |
| 1096      | 1.000   | 1.000       | +0.000 | 6/74      |
| 922       | 1.000   | 1.000       | +0.000 | 8/82      |
| 548 (E14) | 0.968   | 1.000       | +0.032 | 4/83      |
| 1146      | 0.963   | 0.994       | +0.031 | 14/114    |
| 1073      | 0.917   | 0.990       | +0.074 | 3/71      |
| 635       | 0.902   | 0.991       | +0.089 | 10/57     |
| 583       | 0.889   | 0.979       | +0.090 | 4/40      |
| 590       | 0.948   | 0.978       | +0.030 | 6/95      |
| 1084      | 0.934   | 0.975       | +0.041 | 7/87      |
| 958       | 0.987   | 0.959       | −0.028 | 6/96      |
| 139       | 0.878   | 0.926       | +0.047 | 4/41      |
| 1150      | 0.970   | 0.926       | −0.044 | 3/124     |
| 442       | 0.868   | 0.877       | +0.009 | 5/70      |
| 253       | 0.872   | 0.731       | −0.141 | 3/29      |
| 1077      | 0.806   | N/A         | —      | 4/121     |


---

## 5. 当前最可信的结果

- **Epilepsiae per-channel 空间归因已解锁**——19/20 subjects 有完整 `gpu.npz` + `_refineGpu.npz`
- 跨数据集 SOZ-AUC 验证通过：Yuquan refined 0.874 ≈ 老论文 0.857，Epilepsiae refined 0.952
- 在 Yuquan-only per-channel 分析里，旧 lagPat 框架下的 SOZ raw-corr 优势并不稳
- 更可信的信号是 detrended 之后的残差方向，而不是 raw serial correlation
- `iei_median` 更短与 SOZ 更高事件率方向一致，但还只是边缘结果

---

## 6. 仍未解决的问题 / 下一步

- **Epilepsiae per-channel 时序指标**：现在上游 gpu.npz 已就绪，可以跑 per-channel relaxed-refine SOZ 对比（PR-2），预期 n 从 9 → 15+ 配对
- **Epilepsiae 三值梯度（i/l/e）**：Epilepsiae 有 `focus_rel` 三值标注（`results/epilepsiae_electrode_focus_rel.json`），可以做 SOZ > lesion > extra-focal 的梯度分析
- 1077 CPU 模式跑完后补充 refine → 补入 SOZ-AUC 表
- SOZ / non-SOZ 的 event-rate 支撑域不完全重叠，仍有混淆，需 mixed-effects 模型控制
- 当前结果最容易被误读成"SOZ 没差异"；更准确的说法是"差异主要出现在去趋势后的局部残差，而不是 raw 总量"

---

## 7. 可选方向：稳定传播模板的空间锚定（Topic 1 × Topic 3）

> 来源：`docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md` §6.2 / §9。
> 性质：可选 / 后续；不替代当前 PR-2 的 Epilepsiae per-channel 工作。
> **状态升级（2026-04-19）**：PR-4C P0 已 CLOSED，propagation pattern 的 seizure-proximity 调制正式封板为 null。该方向已成为 Topic 1 × Topic 3 桥接里 **唯一仍能解释"群体事件刻板 ↔ 病理网络"假设** 的路径，优先级从"可选"上调为"P1 候选"。

**科学问题**：Topic 1 已经稳健建立了刻板传播模板（30/30 stable adaptive solutions、`23/30 strong` + `7/30 moderate` 跨时间复现、`8/9` forward/reverse subject 跨时间分裂可复现）。这套模板反映的是结构性病理网络，按假设它的 source / sink 通道应该在解剖上锚定到 SOZ / lesion / extra-focal，而不是均匀分布。这条路把 Topic 1 的"刻板"直接挂到 Topic 3 的"病理空间"，不依赖发作邻近窗口的功效——后者已被 PR-4C 复跑证伪。

**最小工作合同**：

1. 直接复用 Topic 1 已经验收的 `lagPatRank` cluster templates 与 `lagPatRaw` 衍生的 per-event relative lag（`src/interictal_propagation.py` 已有 `build_cluster_templates` / `assign_events_to_templates`），不重新聚类
2. 对每个 subject 的每个 stable cluster，定义两个空间读数：
   - **source 通道**：cluster centroid 中 rank 最早 / lag 最小的 1–3 个通道
   - **sink 通道**：rank 最晚 / lag 最大的 1–3 个通道
3. 用现有 SOZ / focus_rel 标签（`results/yuquan_soz_core_channels.json`、`results/epilepsiae_soz_core_channels.json`、`results/epilepsiae_electrode_focus_rel.json`）做归因
4. 主统计读数：subject-level paired 比较 source vs sink 中 SOZ 通道占比；cohort-level 用 sign test / Wilcoxon
5. forward/reverse subject 单独看：互逆模板的 source 与 sink 是否在解剖上正好对调
6. 暂不画新主图；在已有 PR-3 cohort 6-panel 图旁边补一栏小 inset 即可

**与本 topic 现有 PR-1/PR-2 的边界**：

- 当前 PR-1（per-channel SOZ serial corr）回答的是"慢调制是否 SOZ-specific"
- 这条可选方向回答的是"传播模板的源/汇通道是否落在 SOZ 解剖上"
- 两者数据不重叠（PR-1 用 per-channel HFO 列；这条用 lagPat cluster centroid），结论可以独立并存

**判定边界**：本分支只回答"刻板模板是否有解剖锚定"。它**不**回答"模板是否随发作邻近变化"——后者由 Topic 1 PR-4C 在 2026-04-19 复跑后正式封板为 null（5 个 propagation pattern metrics 在主+辅助两套配置下 cohort 级均无可复制信号）。换句话说：发作邻近这条门已经关上，剩下能撑起"病理网络"假设的就是模板本身的解剖锚定。

---

## 8. 代码与结果入口

- 主文档：`docs/archive/topic3/spatial_modulation_soz_analysis.md`
- HFO 检测脚本：`scripts/run_hfo_detection.py`（支持 `--dataset yuquan/epilepsiae --all --gpu`）
- 检测参数：`config/subject_params.json`
- SOZ-AUC 验证脚本：`scripts/plot_refine_soz_validation.py`（支持 `--dataset yuquan/epilepsiae`）
- 审计脚本：`scripts/audit_gpu_npz.py`
- Per-channel 主脚本：`scripts/run_spatial_modulation.py`
- Per-channel 作图：`scripts/plot_spatial_modulation.py`
- 相关代码：`src/event_periodicity.py` 中的 per-channel / SOZ helpers，`src/group_event_analysis.py`
- 检测结果：`results/hfo_detection/`（Yuquan + Epilepsiae gpu.npz / refineGpu.npz）
- SOZ-AUC 结果：`results/refine_soz_validation/yuquan/`、`results/refine_soz_validation/epilepsiae/`
- Per-channel 结果：`results/spatial_modulation/`

---

## 9. 与其他 topic 的边界

- 如果问题是"慢调制本身是否存在、是不是 oscillator"，去 `docs/topic2_between_event_dynamics.md`
- 如果问题是"SOZ 传播路径是否更刻板"，去 `docs/topic1_within_event_dynamics.md`
- Topic 3 关注的是 where，不是 whether

---

## 10. 历史文档索引

- `docs/archive/topic3/spatial_modulation_soz_analysis.md`
  - 保留完整的计划、执行过程、基础设施与阶段结果

当前正式口径以本文件为准。