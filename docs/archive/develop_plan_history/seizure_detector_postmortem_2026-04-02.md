# Seizure Detector PR1 + PR1.5 + PR2 + PR2.5 验收与复盘（Archived 2026-05-22）

> **归档说明**：这是 `docs/DEVELOP_PLAN.md` §4 旧 L256–362 的完整原文，合并了 4 个连续的 detector 验收复盘段。
>
> **数字过期警告**：PR2.5 的"v3 最佳候选"参数表（litengsheng FP=16, sunyuanxin FP=39）是 2026-04-02 时点的探索性快照——当前 detector 实现已被 `src/preprocessing.py::detect_seizure_by_spatial_extent[_epilepsiae]` + HFO Detector v2 (2026-05-05 canonical) 替代，不再代表"当前最佳"。引用参数请以 `docs/archive/hfo_detector_v2/v2_specification.md` 为准。
>
> 本归档只保留四段历史发现链：PR1 EDF+ annotation parsing → PR1.5 Epilepsiae 数据契约 → PR2 channel-mean 失败复盘 → PR2.5 空间招募 detector 三大踩坑，供 detector 演化历史溯源。

---

#### PR1 真验收更新（2026-04-01）

- **PR1 现在可以作为真正验收，但验收边界要说清楚**：
  - ✅ 已验收：EDF+ annotation parsing、seizure interval extraction、timezone conversion、header-driven timeline foundation
  - ❌ 不能夸大：它还不是"临床 gold-standard seizure inventory"；EDF 原始标注本身存在重复 onset、孤立 onset、缺失 END 等脏数据
- 全量 Yuquan 审计（21 subjects / 260 EDF）后确认：
  - 原始命中 `32` 个 seizure-bearing EDF、`54` 个原始 intervals
  - 归一化后为 `25` 个 valid interval-bearing EDF、`30` 个 normalized intervals
  - 另外保留 `16` 个 orphan onset markers（有 onset、无可靠 offset）
  - 对 32 个 seizure-labeled EDF 的 offset 审计显示：有效 offset 全部来自后续 `END` 标签配对，`duration` 来源为 0；重复问题来自多 onset 共享同一个 END，而不是 offset 错配
  - 大多数 subject 是连续分段记录，但总时长并不固定在 24h；`litengsheng`、`zhangjiaqi` 存在真实缺口
- 结论：PR1 可作为 **后续 PR2/PR3 的可靠时间轴与人工标注入口**，但不是最终的发作真值来源。

#### PR1.5 Epilepsiae 数据契约调研（2026-04-02）

- 验收结论：
  - ✅ **PR1.5 已验收通过**
  - 验收边界：`Epilepsiae` 数据契约、统一时间轴、manifest、event-level synchrony 主链，以及基于事件边界严格归属的 interval/window 聚合层
  - 不在 PR1.5 验收边界内：PR6 统计建模本身（Friedman / LMM / 终稿图）
- 新增脚本：`scripts/survey_epilepsiae_dataset.py`
- 新增脚本：`scripts/run_epilepsiae_interictal_synchrony.py`
- 新增脚本：`scripts/aggregate_epilepsiae_interictal_synchrony.py`
- 新增正式接口：`src/epilepsiae_dataset.py`
- 新增正式接口：`src/interictal_synchrony_aggregation.py`
- 新增文档：`docs/epilepsiae_dataset_structure.md`
- 已确认：
  - `Epilepsiae` 原始数据合同是 `*.data + *.head + SQL`，不是 EDF
  - 挂载盘全量是 **27** 个 SQL subjects；其中只有 **20** 个有 `all_data_lns` 间期中间产物
  - 时间真值应优先信 SQL `recording / block / seizure`，`.head.start_ts` 只做块级校验
  - 当前挂载数据上，`.head.start_ts` 与 SQL `block.begin` 是 **0s 对齐**
  - 数据连续性不干净：`75` 个 recordings 里只有 `10` 个 block 级连续；`27` 个 subjects 里只有 `5` 个没有明显 inter-recording gap
  - seizure 标注整体可用，但不是每条 EEG interval 都完整；`vigilance` 不能直接当 day/night
  - 当前挂载数据的时区已经钉死为 `UKLFR -> Europe/Berlin`；`src.epilepsiae_dataset` 已内建 override 接口与 `08:00-20:00` day/night 规则
  - 已输出 `results/epilepsiae_sync_subject_manifest.csv`，按 `ready_full_artifacts / ready_partial_artifacts / missing_interictal_artifacts` 分层
  - `interictal_synchrony` 已接上 manifest，并实际跑完 `ready_full_artifacts` 的 `16` 个 subjects / `2962` 个 blocks
  - event-level synchrony 已进一步聚合成 `subject × seizure_interval × window_type` 分析表；聚合规则是严格按事件边界归属，跨 seizure / post-ictal / day-night / gap 边界的 event 直接排除
  - 当前聚合实跑保留：`1903` blocks 能安全落进完整 seizure interval，`1742` blocks 能进入 `phase(post_ictal/interictal)` 聚合，主表产出 `409` 行
- 对后续 PR 的硬约束：
  - 若将 Epilepsiae 纳入同步性分析，必须消费 `results/epilepsiae_*_inventory.*` 形成的统一时间轴与 seizure inventory
  - subject 选择必须优先消费 manifest，而不是手工挑病人
  - 不能把"20 个 artifact subjects"误当成"全量数据集"
  - 不能把 Yuquan 的 EDF 路径直接套到 Epilepsiae 上
  - 不能把 1h artifact block 的来源元数据误当成分析主语；当前聚合层明确以 event-level metric 为主，block summary 只保留兼容派生视图

#### PR2 Streaming Seizure Detector 验收复盘（2026-04-02）

- 已完成（基础设施，可复用）：
  - `src/preprocessing.py`：二进制 EDF 流式读取与 channel-mean LL+RMS 检测主流程
  - `scripts/pr2_seizure_validation.py`：单 EDF 叠图、24h 总览、误差散点、审计 CSV
  - `tests/test_seizure_streaming.py`：`_flag_to_runs` / `_merge_close_runs` / `match_seizure_intervals` 合成数据测试
  - NFS 性能优化：单次顺序读取 + 特征缓存，支持 `n_jobs` 并行批跑
  - 手工标注评估修正：interval 与 onset-only 分开匹配，避免漏算人工标注
- 已验收（litengsheng）：
  - 峰值内存约 `110MB`（< `500MB`）
  - onset 中位误差约 `4.6s`（< `30s`）
  - recall 达到门槛附近（约 `80%+`）
- 未通过（跨 subject 泛化）：
  - 同一参数在 `sunyuanxin` 上出现明显漏检，无法同时兼顾低 FP 与高 recall
  - 根因不是阈值没调好，而是数据结构有损：先对多通道取均值再做特征，抹掉了空间招募信息
- 结论：
  - PR2 基础设施通过，可作为后续检测器与验证框架
  - channel-mean 检测器不再继续加补丁（如 `ignore_initial`）；进入 PR2.5 重构

#### PR2.5 空间招募检测器（第一性原理）（2026-04-02 起，进行中）

- 第一性原理约束：
  - 发作具有通道逐步招募特征（participation 上升）
  - 发作期存在大幅高频振荡（LL 对该特征敏感）
  - 发作有自限性（participation 回落形成 offset）
- 核心顺序修正：
  - 旧：`channels -> mean -> LL/RMS -> threshold`
  - 新：`channels -> bipolar per-channel LL -> per-channel z -> active-channel fraction -> threshold`
- 消除的补丁参数：`combine_mode`, `ignore_initial_sec`, `rms_k`, `AND/OR/SUM`
- 新增 4 个物理可解释参数：`per_channel_k`, `min_active_frac`, `min_duration_sec`, `merge_gap_sec`

**实现过程中踩到的三个坑（按发现顺序）：**

1. **MAD 缩放常数缺失**：`_robust_z` 用 raw MAD 做分母，k=3.5 实际只等价于 2.36σ（P≈0.91%）而非 3.5σ（P≈0.023%）。修复：在 `detect_seizure_by_spatial_extent` 中用 `MAD × 1.4826` 估计 σ。但单独修这个对 FP 几乎无改善，因为 LL 是重尾分布、通道间高度相关。
2. **单极导联共参考噪声（根因）**：`_stream_edf_channel_ll` 读的是 EDF 原始单极通道（A1, A2, ...），它们共享参考电极。参考端的呼吸/心电/体动噪声同时污染 100% 的通道 → participation 虚假飙升 → FP 风暴。无论怎么提高 `min_active_frac` 都没用，因为噪声本身就是共模的。修复：在流式读取器中直接做双极减法（相邻触点，同 shaft），消除共参考后通道之间才真正近独立。
3. **`min_duration_sec` 默认值过短**：初始 10s 对玉泉数据集（发作 >50s）太宽松，放过了大量生理性短暂事件。修正为 30s。

**当前最佳候选（v3 = 双极 LL + MAD×1.4826 + frac=0.40 + dur=30）：**

> ⚠️ 以下参数表是 2026-04-02 时点快照，已被 HFO Detector v2 替代，不再代表当前最佳。

| 指标 | litengsheng (7 seizures) | sunyuanxin (8 seizures) |
|---|---|---|
| TP | 6 | 8 |
| FP | 16 | 39 |
| FN | 1 | 0 |
| Recall | 0.857 | 1.000 |
| Precision | 0.273 | 0.170 |
| Median onset err | 3.9s | 2.5s |
| Peak memory | ~150MB | ~122MB |

对比演进：v1（单极, raw MAD）→ FP=819/528；v2（单极, MAD×1.4826）→ FP=939/499（无改善）；**v3（双极）→ FP=16/39**。

**已消除的参数**：`combine_mode`, `ignore_initial_sec`, `rms_k`, `rms_win_sec`, `rms_step_sec`
**保留的旧接口**：`detect_seizure_streaming()` 标记 deprecated，不删除

**但这还不算 PR2.5 验收通过**：

- 上面结果只证明 `Yuquan` 已标注患者上的 detector 方向正确，不足以证明对未标注患者鲁棒。
- `PR2.5` 的真实验收口径改为：
  - 同一 detector core 同时支持 `Yuquan EDF` 与 `Epilepsiae *.data + *.head`
  - 真实患者验证必须统一输出审计表、时间线图、误差图
  - 主门槛不再是单 cohort 调参，而是 labeled patients 的 held-out subject 验证
  - 目标是 `FP` 压到与手工标注同一量级，并保持 `recall >= 0.80`
- `Epilepsiae` 的角色不是补一份报告，而是作为额外的高质量手工标注训练/验证来源，用来约束跨患者泛化而不是做患者特异补丁
