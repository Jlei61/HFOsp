# Group-Event State v0.1 — 技术报告（2026-09-01）

配套：[科学 spec](group_event_state_v0_1_scientific_spec_2026-09-01.md) ·
[数据合同](group_event_state_v0_1_data_contract_2026-08-31.md) ·
[执行计划](group_event_state_v0_1_execution_plan_2026-09-01.md) ·
[source audit 报告](group_event_state_v0_1_source_audit_report_2026-09-01.md)

机器可读产物根：`results/epi_prssm/group_event_state/v0_1/`
缓存/数据集根：`/data/hfosp_group_event_state_v0_1/`

---

## 1. 逐患者分母

审计候选 41 人 / 1,774,188 事件 → 合格 27 人 / 1,663,394 事件
（条件：waveform pointer == 1.0 且最长连续覆盖段 ≥ 5,000 事件，**选择规则先于任何模型结果**）。
已缓存并一致化 23 人 / **1,506,867 interictal 事件**。

| subject | C | interictal | ictal | sessions | 最长段 | 参与/事件 | 招募步/事件 | 延迟跨度 ms | 事件间隔 s | 背景锚点覆盖 | 几何 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| epilepsiae_922 | 8 | 234,976 | 9,015 | 8 | 67,838 | 5 | 1 | 13.8 | 0.95 | 0.991 | ok |
| epilepsiae_1096 | 7 | 222,254 | 1,093 | 15 | 63,544 | 5 | 2 | 25.9 | 0.82 | 0.997 | ok |
| epilepsiae_1073 | 6 | 194,207 | 314 | 18 | 28,544 | 4 | 2 | 20.5 | 1.48 | 0.995 | partial |
| epilepsiae_958 | 16 | 165,252 | 325 | 18 | 29,839 | 12 | 4 | 90.5 | 1.61 | 1.000 | ok |
| yuquan_huangwanling | 4 | 107,062 | 0 | 1 | 107,062 | 3 | 1 | 5.3 | 0.54 | **0.062** | ok |
| yuquan_zhaojinrui | 4 | 81,102 | 39 | 1 | 81,141 | 3 | 1 | 6.2 | 0.77 | **0.860** | ok |
| epilepsiae_253 | 8 | 74,956 | 106 | 15 | 19,402 | 4 | 2 | 25.3 | 2.70 | 0.998 | ok |
| epilepsiae_1125 | 8 | 69,904 | 803 | 10 | 20,557 | 6 | 2 | 30.7 | 2.03 | 0.990 | partial |
| epilepsiae_1077 | 6 | 56,155 | 372 | 15 | 13,718 | 3 | 2 | 33.0 | 2.92 | 0.997 | ok |
| yuquan_zhangjiaqi | 7 | 48,494 | 0 | 2 | 25,493 | 5 | 1 | 20.3 | 1.01 | 0.989 | ok |
| yuquan_pengzihang | 12 | 45,048 | 1,007 | 1 | 46,055 | 7 | 2 | 27.5 | 1.15 | 0.967 | ok |
| epilepsiae_1146 | 15 | 44,283 | 2,400 | 15 | 25,298 | 12 | 2 | 41.8 | 0.80 | 0.998 | ok |
| epilepsiae_384 | 9 | 42,482 | 51 | 11 | 17,596 | 6 | 4 | 82.1 | 2.08 | 0.981 | ok |
| yuquan_chengshuai | 8 | 27,632 | 0 | 1 | 27,632 | 5 | 2 | 33.8 | 1.67 | 0.995 | ok |
| yuquan_zhourongxuan | 4 | 23,142 | 0 | 1 | 23,142 | 3 | 1 | 10.3 | 2.00 | 0.999 | ok |
| yuquan_zhangkexuan | 26 | 18,157 | 33 | 1 | 18,190 | 19 | 4 | 115.9 | 2.31 | 0.969 | ok |
| yuquan_xuxinyi | 15 | 9,645 | 1 | 1 | 9,646 | 10 | 3 | 60.3 | 3.80 | 0.999 | ok |
| yuquan_chenziyang | 10 | 9,609 | 0 | 1 | 9,609 | 6 | 3 | 75.1 | 3.02 | 0.993 | **unavailable** |
| yuquan_zhangbichen | 52 | 8,371 | 0 | 1 | 8,371 | 33 | 3 | 93.0 | 4.31 | 0.999 | ok |
| yuquan_gaolan | 12 | 7,417 | 34 | 1 | 7,451 | 7 | 3 | 63.2 | 2.26 | 0.993 | **unavailable** |
| yuquan_zhangjinhan | 5 | 6,196 | 0 | 1 | 6,196 | 4 | 1 | 9.2 | 2.57 | 0.995 | ok |
| yuquan_hanyuxuan | 22 | 5,468 | 0 | 1 | 5,468 | 15 | 3 | 80.0 | 1.98 | 0.998 | **unavailable** |
| yuquan_sunyuanxin | 12 | 5,055 | 30 | 1 | 5,055 | 8 | 3 | 53.5 | 7.33 | 1.000 | **unavailable** |

**cohort 中位数**：参与触点 6 / 可分辨招募步 **2** / 事件内延迟跨度 33 ms / 事件间隔 2.0 s。

### 两个必须并报的分母限制

1. **背景 SEEG 对最高发放率的患者几乎不存在**。背景锚点规则是"30 s 网格上取 2 s 窗，
   与任何 packed event core 重叠即丢弃"。`huangwanling` 事件间隔中位 0.54 s，
   于是只有 **6.2%** 的事件能找到一个此前完成的干净背景窗（`zhaojinrui` 86.0%）。
   `a5` 臂在这两人身上**结构性地**几乎拿不到背景信息——这不是模型没学会。
2. **4 位 Yuquan 患者的触点坐标解析不到**（`chenziyang` `gaolan` `hanyuxuan` `sunyuanxin`）。
   他们的几何分支只有 shaft / 序号 + 一个"坐标非真实"的标志位，**不塞 0 冒充坐标**。

## 2. 采样率与频带 mask

| 采样率 | 患者 | fast ripple (150–250 Hz) |
|---|---|---|
| 512 Hz | epilepsiae_253, epilepsiae_139 | **missing**（Nyquist 不支持） |
| 1024 Hz | 其余 Epilepsiae | 支持 |
| 2000 Hz | 全部 Yuquan | 支持 |

不支持的频带在 shard 中带 `band_available=False`，模型侧乘 0 **并附 mask flag**，不填 0。

## 3. 模型与臂

每患者一个模型（触点宇宙、采样率、几何都是患者特异的），patient-first 聚合。

- **event encoder**：waveform（三个参考视图，各带可学习 view embedding）/
  多频带（包络轨迹 + 5 个 band summary + 10 个跨频带互相关滞后）/
  结构（participation、精确连续 delay、tied recruitment group）/ 几何；
  触点 token 过 masked self-attention + FFN，masked mean/max 池化。
- **状态**：`z_fast` 64 维（τ 1 s–1 h）、`z_slow` 32 维（τ 60 s–48 h），
  两者都用 `τ = exp(clamp(log τ))`；事件间按**真实秒**向 bias 弛豫。
- **头**：timing（log-normal）、participation（per-contact Bernoulli）、
  group size（MAE）、delay、band energy、band peak、cross-band lag（Gaussian）。

臂：`a1` 无状态近期历史基线 / `a2` rank+participation / `a3` +tied group+精确 delay /
`a4` +波形+多频带 / `a5` +背景修正；消融 `b1` 无真实 Δt / `b2` 无波形 /
`b3` 无多频带 / `b4` memoryless / `b5` 无几何 / `b6` fast 状态压到 8 维。

## 4. 泄漏防线（每条都有回归测试）

| 防线 | 测试 |
|---|---|
| timing 头读**演化之前**的状态 | `test_state_evolution_is_the_only_place_dt_enters` |
| baseline 的 lag-1 `dt` 特征后移一位 | `test_recent_history_never_contains_the_interval_it_must_predict` |
| 近期历史摘要不含当前事件 | `test_history_summaries_exclude_the_current_event` |
| τ_slow 能真正到小时（不是 softplus 上限） | `test_slow_timescales_can_actually_reach_hours` |
| memoryless 臂真的不传递任何东西 | `test_memoryless_state_carries_nothing_between_events` |
| 全 mask 事件不产生 NaN | `test_fully_masked_event_does_not_produce_nan` |
| packedTimes 按 packer 变体配对 | `test_packed_file_follows_the_lagpat_variant_not_file_existence` |
| 未观测小时不被桥接 | `test_unobserved_recorded_block_breaks_the_session` |
| ictal 剔除 / preictal 保留 | `test_ictal_mask_excludes_overlap_and_keeps_preictal` |
| stream 批处理逐项等价 | `test_stream_batching_reproduces_segment_by_segment_totals` |
| ragged 尾部不贡献观测 | `test_padded_stream_slots_contribute_no_observations` |

另外两条不在测试里但同样是硬约束：normalization 统计量只从 train split 估；
wrong-time 对照同时打乱 content 与 timing 两套状态。

## 5. 资源与 OOM

- 2 × RTX 3090（24 GB）、80 核、251 GB RAM、`/data` 3.3 TB 可用、`/` 仅 ~40 GB。
- **缓存**：107 MB/1k 事件（Epilepsiae）/ 81 MB/1k（Yuquan）；worker 峰值 RSS 0.37–4.9 GB。
  并发数由实测决定：32 workers 比 18 workers **总吞吐低 2.4×**（磁盘寻道竞争），
  最终按物理盘分池 9（sdd）+ 7（sdc）。Yuquan 池 172/172 blocks、403,468 事件、
  30.3 GB、2.70 h、0 失败。
- **训练吞吐**：单流 599 events/s → 8 流并行 **4,053 events/s**（6.8×）。
  profile 显示单流下 recurrence 30% + backward 62%，encoder 只占 2.8%——
  所以并行的是流，不是 batch。
- **GPU 队列**：单一 queue owner，每个 job 独立子进程钉在一张卡上，
  完成判据是 `result.json` 存在 → 重跑即续跑。2 GPU × 4 slot，
  单 job 显存 362–554 MiB，GPU 利用率 98–99%。
- **OOM 降级**：只捕获 `torch.cuda.OutOfMemoryError` → 记录失败配置 → `empty_cache` →
  `chunk_events` 减半 → 重试 3 次；仍失败写 `resource_failed.json` 并**明确标注
  "这不是科学阴性"**。

## 6. 工程事故记录（都已修，都留了防线）

| 事故 | 后果 | 修法 |
|---|---|---|
| `round(t0·fs)` 与 `round(t1·fs)` 分别取整 | 恒定 0.11 s 的 core 在 112/113 samples 间跳变，一个块**静默丢 37% 事件** | 每块只取一个名义宽度 |
| 波形窗越界事件把该事件所有触点标无效 | 注意力整行被 mask → NaN → 从第 0 步毒化整条状态链 | 触点有效性改为 montage 级 + 全 mask 行保护 |
| 原始 µV 波形 / log 能量未归一化 | 第一个 epoch 全 NaN，参数一步没动 | train-split 统计量归一化 + 头偏置按目标位置初始化 |
| wrong-time 对照只打乱 content 状态 | timing 端点数值**完全相同**，对照静默失效 | 同时打乱 timing 状态 |
| `np.savez(path)` 给非 `.npz` 路径补后缀 | 原子 rename 找不到临时文件 | 传已打开的文件对象 |
| `pkill -f <pattern>` 匹配到发起它的 shell | 自杀在 `pkill` 那行，后续 `rm`/patch 全没跑却像成功了 | 按 PID kill |
| 工具调用超时连带杀掉 `nohup setsid` 子进程 | 缓存任务被腰斩 | 用 harness 后台执行；靠 shard 幂等性挽回 |
| H3 state-matched placebo 逐行全扫 | O(n²)，47k 事件的 test 段跑不完 | 固定候选池匹配 |

## 7. H3 探针的先验校准

在**已知答案**的合成数据上验证过判别力（这是本仓库 2026-08-26 复审记录的
"固定 event jump 饱和后变成免费截距"那一失效模式的直接防线）：

| 情形 | real exposure 增益 | intercept-matched 对照 | state-matched placebo |
|---|---|---|---|
| 状态**已经**抓住了慢驱动 | −0.0002（K=100） | −0.0009 | −0.0000 |
| 状态**没有**抓住慢驱动 | **+0.751**（K=100） | −0.0018 | +0.0001 |

同时每个尺度都并报**独立窗口数**：40,000 事件的 test 段在 K=10,000 时
只有 **3** 个独立窗口——滑窗数绝不可当样本数。

---

## 8. 结果

*(H1 / H2a / H2b / H3 的逐端点、patient-first 结果在队列跑完后补入本节；
见 `summary_main.json` / `h2b_transfer.json` / `h3_exposure.json`。)*
