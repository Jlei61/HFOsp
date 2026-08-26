# Raw-SEEG 可演化预测状态模型 R0.1 — 技术报告

**修订**：R0.1 · **合同版本**：`raw_seeg_state_contract_v2_2026-08-21`
**科学合同**：`docs/archive/topic5/raw_seeg_state_scientific_spec_2026-08-21.md`
**执行计划**：`docs/archive/topic5/raw_seeg_state_execution_plan_2026-08-21.md`
**白话报告**：`docs/archive/topic5/raw_seeg_state_r0_1_plain_2026-08-21.md`
**结果根**：`results/epi_prssm/raw_seeg_state/r0_1/`
**接手文件**：`results/epi_prssm/raw_seeg_state/r0_1/CURRENT_HANDOFF.md`

> **状态标记**：§1–§6、§9–§11 为已确定的基础设施与合同事实，均已落盘可核。
> §7（逐患者结果）、§8（队列结果）在训练完成前标 `PENDING`，不得据此下任何结论。

---

## 1. 数据 manifest

产物目录 `results/epi_prssm/raw_seeg_state/r0_1/data/`：

| 文件 | 内容 | 规模 |
|---|---|---|
| `dataset_manifest.parquet` | 记录块区间、采样率、session、split | **3547 块** |
| `contact_metadata.parquet` | bipolar 通道、杆、坐标、有效性 | **3182 通道** |
| `window_index.parquet` | 分钟网格与逐 horizon eligibility | **148 720 分钟** |
| `eligibility_summary.csv` | 逐患者分母 | 34 行 |
| `split_manifest.json` | 继承自上游的墙钟边界 | 34 主体，test SEALED |
| `data_audit.json` | 逐患者 + 队列的 7 条硬性无效条件检查 | 34 + 1 |

**块来源 provenance**：`sql_block_inventory` 3349 块（Epilepsiae）、
`yuquan_block_inventory` 89 块、**`edf_header` 109 块**（9 位在冻结 inventory 中
无行的 Yuquan 患者，时间轴由 EDF 固定头重建）。

**队列**：34 位 = 18 Epilepsiae + 16 Yuquan。

## 2. 训练 / 验证分母

| 量 | 数值 |
|---|---|
| 记录总时长 | 3634 h |
| dev 分区已覆盖时长 | **2393 h**（train 1874 h + validation 520 h） |
| 发作护栏移除 | 339 h |
| dev 内发作条目 | 202（其中 2 条来自标注扫描补充源） |

逐 horizon 可用窗口（全队列合计，**分钟窗口不是独立生物学样本**，仅作工程分母）：

| horizon | train | validation | 有 validation 窗口的患者数 |
|---:|---:|---:|---:|
| 1 min | 85 072 | 24 101 | 34/34 |
| 5 min | 84 391 | 23 843 | 34/34 |
| 10 min | 83 626 | 23 531 | 33/34 |
| **100 min** | 71 518 | 18 775 | **30/34** |

h=100 在 validation 上为 0 的 4 位患者（gaolan / litengsheng / songzishuo /
sunyuanxin）验证段短于 110 分钟，只进 `per_horizon` 次级表，**不进队列 horizon
曲线，并逐名列出**（合同 §EVAL_SET）。

## 3. 患者 eligibility 与已知降级

| 维度 | 结果 |
|---|---|
| 状态 | **27 OK + 7 `DEGRADED_NO_SEIZURE_GUARD`，0 blocked** |
| bipolar 触点数 | 最少 24 / 中位 94 / 最多 183 |
| 有解剖坐标的触点 | 2480 / 3182 = **77.9 %** |
| `coord_mode` | `mm` 29 位；`shaft_index_only` **5 位** |
| 发作护栏来源 | `inventory` 25 位；`annotation_scan_only` 2 位；**`none_found` 7 位** |
| dev 分区受采样率限制 | **仅 `epilepsiae_139`**（256 与 512 Hz 混合） |

### 3.1 两条必须并列写出的局限

1. **5 位 Yuquan 患者没有电极坐标**（chenziyang / gaolan / hanyuxuan /
   sunyuanxin / wangyiyang）。全盘核查确认整个挂载盘上只有原始 MRI/CT，没有导出的
   电极定位。这些患者的位置编码只有杆号与杆上序号，**不携带任何解剖距离信息，
   不能支持任何空间主张**，队列统计须与 `mm` 组分开报。
2. **7 位 Yuquan 患者在两个发作来源里都查不到标注**。**"没有标注"不等于
   "没有发作"**——对这些患者无法保证 ictal 排除。

## 4. 模型公式

$$z_t \in \mathbb{R}^{32},\qquad
z_{t+h} = \mu + B(h)\,(z_t-\mu),\qquad
\hat y_{t+h} = W z_{t+h} + b$$

$B(h)$ 为 16 个二维块的 block diagonal：

$$B_j(h)=e^{-h/\tau_j}\begin{pmatrix}\cos(\omega_j h)&-\sin(\omega_j h)\\ \sin(\omega_j h)&\cos(\omega_j h)\end{pmatrix}$$

- $\tau_j=\exp\big(\mathrm{clamp}(\log\tau_j,\ \log 1,\ \log 2880)\big)$ 分钟，
  log 空间均匀初始化覆盖 1 min–48 h。**不用 `softplus`**（上一版 Epi-PRSSM v0.1
  的 `softplus(log τ)` 把时间常数压到 5.7 秒，模型结构上无法表示慢状态）。
- $\omega_j = \Omega_{\max}\tanh(\cdot)$，$\Omega_{\max}=\pi$ rad/min。
  **上限从 $2\pi$ 改为 $\pi$**：所有 horizon 与一致性步长都是整数分钟，
  $\omega$ 与 $\omega+2\pi$ 在整数 $h$ 上给出逐位相同的预测，原上限的上半区是
  下半区的精确混叠，会让 `dynamics_modes.json` 里 $\omega>\pi$ 的 mode 被误读成
  快 mode。
- 每个 mode 严格稳定；$h=100$ 直接算 $B(100)$，不做 100 步递归。
- decoder 单层线性，四个 horizon **共用**。

损失：$\mathcal{L}=\mathcal{L}_{\text{forecast}}+\lambda_{\text{cons}}\mathcal{L}_{\text{cons}}$，
四个 horizon **等权**（一个 horizon 若在该 batch 内无有效元素，从分母中剔除而不是
计 0）；$\mathcal{L}_{\text{cons}}$ 为 Huber($\delta=1$)。仅比较
$\lambda_{\text{cons}}\in\{0.1,\ 0\}$。

## 5. 超参数

| 项 | 值 |
|---|---|
| 分析采样率 / 频段 | 256 Hz；1–100 Hz，12 个 log bin |
| patch / window / minute | 64 / 1280 / 15360 sample |
| 上下文 | 10 分钟；horizon 1 / 5 / 10 / 100 分钟 |
| d_model / 层数 | 128；temporal 2 + spatial 2 + context 3，4 头 |
| 参数量 | **1 492 496**（C=100，identity 臂逐位相同） |
| latent / modes | 32 / 16 |
| 优化 | AdamW，lr 3e-4，wd 1e-2，grad clip 1.0，warmup 100 步 |
| 训练预算 | 每 epoch **400** 个训练窗口（有种子地重抽）；最多 **20** epoch；patience **5**（2026-08-22 削减，见 §9.4） |
| 验证 | 每 epoch 固定 **200** 窗口；报告用最终全量（上限 **900** 窗口） |
| DataLoader | `num_workers=4`，压缩库内部线程关闭（见 §9.5） |
| batch | 按触点数自动：`batch × 触点数 ≤ 440`，范围 1–8 |
| 精度 | AMP bfloat16，逐步非有限值检查 |

## 6. worker 与资源配置

| 资源 | 实测 |
|---|---|
| GPU | **1 × RTX 3090 24 GB** —— 任务书设想的多 GPU patient-level worker **不可行**，患者作业串行排队 |
| CPU | 80 逻辑核 / 40 物理核 |
| RAM | 251 GB（本任务并发峰值 < 50 GB） |
| 盘 | 全部**机械盘**；仓库卷 `/` 仅 118 GB 可用 |

**缓存放盘（交叉，避免源盘=写盘）**：Epilepsiae 读 `/mnt/epilepsia_data` 写
`/mnt/yuquan_data`；Yuquan 读 `/mnt/yuquan_data` 写 `/mnt/epilepsia_data`。

**四个 worker 的分工与文件所有权**见执行计划 §2。四个 worker 于 2026-08-21 18:00
前后**同时撞到会话额度限制**，全部停在改到一半的状态；主 agent 接手完成收口
（详见 §9）。

**GPU 吞吐实测**（batch 4，bf16，10 分钟上下文）：

| 触点数 | 显存峰值 | 每样本 | 30 epoch 估计 |
|---:|---:|---:|---:|
| 31 | 4.9 GB | 34.5 ms | ~16 min |
| 87 | 13.4 GB | 91.5 ms | ~41 min |
| 139 | 21.4 GB | 137.7 ms | ~62 min |

**削减预算 + 装载优化后的实测**（epilepsiae_620，31 触点，完整臂）：
epoch **95 s**，20 epoch 加最终分析约 **32 分钟/作业**。整夜队列共 **100 个作业**，
按科学价值排序（见 §6.1）。

### 6.1 整夜队列的 100 个作业

| 波次 | 作业数 | 内容 |
|---|---|---|
| 1 | 12 | 3 位差异最大的患者 × 4 个臂 |
| 2 | 12 | 另外 3 位 pilot 患者 × 同样 4 个臂 |
| 3 | 12 | 6 位 × 完整臂 seed 1、2（**换种子噪声地板**） |
| 4 | 8 | `no_consistency` + horizon 阶梯 |
| 5 | 56 | 其余 28 位 × {完整臂, identity 对照} |

四个臂：`full`、`identity`（同编码器、$B(h)=I$）、`ctx_last_minute`（**遮掉 10 分钟
历史里的 9 分钟**——若与完整臂打平，则这个状态不需要历史）、`target_shuffled`
（**训练时打乱"历史↔未来"的配对，评测用真配对**——若仍胜过均值基线则是泄漏）。
后两个臂是 2026-08-22 新增并在看到任何数字之前预注册的，理由见
`raw_seeg_state_r0_1_pilot_readout_prereg_2026-08-21.md`。

## 7. 逐患者结果

`PENDING` — 训练未完成。产物路径见执行计划 §3 阶段 C。

## 8. 队列结果、基线、一致性、state-swap

`PENDING` — 训练未完成。

**预登记的报告纪律**：预测层与一致性层分开陈述；forecast 为正而一致性失败只能
称 "forecastable latent code"；队列统计以患者为单位；每个数字旁必须带该患者的
`n_windows`；主 horizon 曲线只用 `common_all_horizons` 窗口集。

## 9. 失败作业与已修缺陷

### 9.1 三个会静默污染结论的缺陷（均已修 + 均有回归测试）

| # | 缺陷 | 后果 | 处置 |
|---|---|---|---|
| 1 | 跨块分钟被判为"已覆盖" | Epilepsiae 小时块间有 ~1 s 缝隙，跨缝隙的分钟累加覆盖 59/60 s 过判据，但内含录制中断；拼接会在 60 s Welch 里插入阶跃，宽带能量溅进**每一个**频段 | 覆盖判据改用**最长不间断段**；全队列 dev 覆盖 2415 h → 2393 h（−0.9 %）。测试 `test_minute_straddling_a_block_gap_is_not_covered` |
| 2 | CUDA 网格上限 | patch 阶段把 (batch×触点×分钟×窗) 压进注意力的 batch 轴，batch 4 × 139 触点 = 66 720 > 65 535，**硬启动失败**（不是显存不足，减 batch 救不回来）；队列内 12 位患者触点数 ≥137 | 分块启动（`MAX_ATTENTION_ROWS = 32768`）。测试 `test_patch_stage_splits_above_the_cuda_grid_limit` |
| 3 | $\omega$ 上限混叠 | $2\pi/\text{min}$ 是可辨识范围的 2 倍，整数 horizon 下上半区是下半区的精确混叠 | 上限改 $\pi$/min，理由写入 `contract.py` |

### 9.2 一次运维错误（无结果受影响）

判断旧缓存构建"已退出"时查的是 `setsid` 包装器的 PID 而非 worker，导致一段时间里
**两套构建在写同一批 zarr**（执行计划 §4 明令禁止）。全部停止、缓存清空重建；
启动器加了互斥闸（发现任何 cache-build worker 存活即拒绝启动并退 3）。
**没有任何结果基于被双写的缓存。**

### 9.3 一次已撤回的数值

曾报"zstd 压缩比 4.68×、全队列缓存 57 GB"。该测量取自一次**未完成**的构建，
不成立，已撤回；真实比值以完成后的 `BUILD_STATUS.json` 为准。

## 9.4 训练预算的一次削减（2026-08-22 02:50，预注册允许）

第一个真实作业实测 **7.3 分钟/epoch**，30 个 epoch 就是 3.7 小时，而队列有 100 个
作业、只有一块卡。pilot 判读预注册明确把这一类改动列为"属于工程，不属于结果"：
"单作业墙钟 > 90 min → 降 `train_windows_per_epoch` 或 `max_epochs`，并在技术报告
写明改前改后"。

| | 改前 | 改后 |
|---|---|---|
| 每 epoch 训练窗口 | 800 | **400** |
| 最大 epoch | 30 | **20** |
| 早停耐心 | 6 | **5** |
| 每 epoch 验证窗口 | 300 | **200** |
| 最终全验证窗口 | 1200 | **900** |

依据：第一个作业的曲线在第 7 个 epoch 就已经走平（第 7–10 个 epoch 验证损失都在
0.80），20 个 epoch 加耐心 5 损失很小。**第一个作业被丢弃并按新预算重跑**，
以保证队列里每个作业彼此可比。`tests/test_raw_seeg_state_train.py` 里有一条断言
钉住这五个数字，将来任何静默漂移都会红。

## 9.5 数据装载路径的三处优化（结果逐位不变）

削完预算仍是 7.3→2.3 分钟/epoch，其中约 70 % 是数据装载。逐项 profile 后改了三处，
每一处都验证过输出不变：

1. **多进程读取死锁**。压缩库（blosc）在 C 层有自己的线程池，`fork` 出来的
   DataLoader worker 继承了处于任意状态的互斥量，子进程第一次解压就死锁——
   第一个真实作业上一个 worker 变成僵尸、主进程在 futex 上等了 9 分钟、GPU 占用 0 %。
   关掉压缩库内部线程即修复；实测预热后单线程解压 550 MB/s，远高于 GPU 的消耗。
2. **每个样本 14 次 zarr 读取**。频谱目标与伪迹掩码被逐分钟去 zarr 取（持续场 1 次 +
   四个 horizon 各 1 次 + 各自的掩码），而 zarr 3 每次读都要过一遍 async→sync 的
   线程桥，profile 显示 180 ms 的样本里有一半耗在 `_thread.lock.acquire`。
   这两个数组很小（最大植入也只有 123 MB + 2.6 MB），一次性驻留内存后
   **逐位比对与 zarr 源一致**。
3. **归一化的三个临时数组**。`(blk*int16_scale - center)/scale` 每个样本要造三个
   21 MB 临时数组；融合成一次原地 multiply-add 后一个都不造，与原式的最大相对
   偏差 **9.1e-08**。

合计 **180 ms → 111 ms** 每样本，epoch **139 s → 95 s**，单作业约 **32 分钟**。

## 10. OOM 与降级阶梯

阶梯顺序（与执行计划 §4 一致）：
**① 打开 activation checkpointing（batch 不变）→ ② batch 减半、梯度累积翻倍
→ ③ 降低 DataLoader prefetch → ④ 从最近 checkpoint 恢复**；三次减半后仍失败则
该作业标 FAILED，其他患者继续。每次事件写 `logs/oom_events.jsonl`
（患者 / batch / shape / 显存），最终落在哪一档写进 `run_manifest.json`。

先开 checkpointing 而不是先减 batch，是因为它以 ~35 % 时间换 13 倍显存，
**且不改变优化过程**；减 batch 会改变有效优化，只能排在后面。

## 11. checkpoint、代码哈希与复现命令

- 每 200 步 + 每 epoch 末落 checkpoint（临时文件 + rename），含 model /
  optimizer / scheduler / step / epoch / RNG 状态 / 完整配置；`--resume` 全量恢复。
- 所有 manifest 走 `contract.atomic_write_json`。
- `run_manifest.json` 记 `contract.code_revision()`、
  `contract.package_hash(contract.r0_1_source_files())`、GPU 型号、显存峰值、
  墙钟时长、git dirty 文件数、batch 规则、OOM 落档。
- 本报告基础设施部分对应 `code_revision = 7393745c6777`（工作区含用户的无关改动，
  R0.1 的全部文件为新增）。

```bash
source results/epi_prssm/raw_seeg_state/r0_1/ENV.sh && cd $HFOSP_ROOT

# 1) 数据合同（34 患者，约 3 s）
$PY scripts/topic5_raw_seeg_state/build_data_contract.py --subjects all --jobs 12 --force

# 2) 缓存 + 频谱 target（按 jobs/cohort_cache_order.json 的交叉放盘顺序）
results/epi_prssm/raw_seeg_state/r0_1/logs/launch_cache.sh cohort "<subject list>" 6

# 3) 集成闸门（8 项，任一关键项 FAIL 即不得放行）
$PY scripts/topic5_raw_seeg_state/integration_check.py --subject epilepsiae_620

# 4) pilot / 全队列 GPU 队列（单卡串行，可重复执行，DONE 的作业自动跳过）
$PY scripts/topic5_raw_seeg_state/queue_runner.py --jobs .../jobs/pilot_jobs.json
$PY scripts/topic5_raw_seeg_state/queue_runner.py --jobs .../jobs/cohort_jobs.json

# 5) 基线（CPU，不占 GPU）
$PY scripts/topic5_raw_seeg_state/run_baselines.py --subject <subject>

# 6) 汇总与图
$PY scripts/topic5_raw_seeg_state/aggregate_cohort.py
$PY scripts/topic5_raw_seeg_state/make_figures.py --figure all

# 单元测试（116 项）
$PY -m pytest tests/test_raw_seeg_state_*.py -q
```

## 12. 图与数据路径

| 图 | 文件 | 状态 |
|---|---|---|
| R1 结构与数据流 | `figures/r1_model_and_data_flow.{png,pdf}` | ✅ 已生成并目视验收（返工三版） |
| R2 forecast error vs horizon | `figures/r2_forecast_error_vs_horizon.*` | PENDING |
| R3 observed vs open-loop 轨迹 + mode loading | `figures/r3_*` | PENDING |
| R4 matched state-swap 与一致性 | `figures/r4_*` | PENDING |

每张图同时输出 PNG + 矢量 PDF + `<name>_metadata.json`，并在
`figures/README.md` 中用中文写明"展示什么 / 看哪里 / 能支持什么 /
**不能**支持什么"。
