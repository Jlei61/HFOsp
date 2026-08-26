# Raw-SEEG 可演化预测状态模型 — 执行计划 (R0.1)

**建立日期**：2026-08-21
**科学合同**：`docs/archive/topic5/raw_seeg_state_scientific_spec_2026-08-21.md`
**冻结常量**：`src/topic5_raw_seeg_state/contract.py`（主 agent 独占）
**结果根**：`results/epi_prssm/raw_seeg_state/r0_1/`
**缓存根**：`/mnt/yuquan_data/hfosp_cache/raw_seeg_state_r0_1/`（**不在仓库卷上**）
**接手文件**：`results/epi_prssm/raw_seeg_state/r0_1/CURRENT_HANDOFF.md`

---

## 1. 资源实测（2026-08-21 16:50）

| 资源 | 实测 | 约束 |
|---|---|---|
| GPU | **1 × RTX 3090, 24 GB, 空闲** | **只有一块卡**——原计划中的"多 GPU patient-level worker"不可行，患者作业必须串行排队 |
| CPU | 80 逻辑核 / 40 物理核（2×Xeon 5218R） | 缓存构建可 16–24 路并行 |
| RAM | 251 GB 总，134 GB 可用 | 单作业驻留控制在 <16 GB |
| `/`（仓库卷） | 122 GB 可用，86 % 已用 | **禁止**把 raw cache 放这里 |
| `/mnt/yuquan_data` | 834 GB 可用，可写 | raw cache 放这里 |
| `/mnt/epilepsia_data` | 193 GB 可用，可写 | 只读使用 |

**环境注意事项（已实测）**

1. `pandas` 需要 conda env 的 libstdc++：所有脚本/worker 必须设
   `LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:$LD_LIBRARY_PATH`，
   否则 `import pandas` 报 `GLIBCXX_3.4.29 not found`。
2. 本次为 R0.1 安装了 `zarr==3.1.6 / numcodecs==0.16.5 / pyarrow==25.0.1`。
   安装过程曾把 `numpy` 顶到 2.4.6，**已回退到 1.26.4** 并验证
   `numpy/scipy/torch/pandas/zarr/pyarrow/sklearn/matplotlib/mne` 全部可导入、
   zarr 与 parquet 读写往返正常。zarr 3.1.6 的元数据声明 `numpy>=2.0`，实测在
   1.26.4 下工作正常，**不要为了消 pip 警告而升 numpy**。
3. Yuquan EDF 必须用 `encoding='latin1'` 打开（中文医院头）。

**实测 IO 速度**

| 源 | 速度 |
|---|---|
| Yuquan EDF（2000 Hz, 145 ch），`mne.get_data` | 600 s 数据 2.48 s（≈240× 实时），但一次 1.4 GB |
| Epilepsiae `.data`（int16 memmap, 1024 Hz, 97 ch） | 600 s 数据 1.15 s（≈520× 实时），238 MB |

→ 直接在训练时读原始文件不可行（内存 + EDF 打开开销 4 s）。**必须先建 256 Hz
int16 缓存**。

---

## 2. 文件所有权（禁止两个 worker 同时编辑同一文件）

| 所有者 | 文件 |
|---|---|
| **主 agent（独占）** | `src/topic5_raw_seeg_state/contract.py`、`CURRENT_HANDOFF.md`、两份 report、本计划、科学 spec |
| **Worker A** | `src/topic5_raw_seeg_state/data_contract.py`<br>`scripts/topic5_raw_seeg_state/build_data_contract.py`<br>`tests/test_raw_seeg_state_data_contract.py` |
| **Worker B** | `src/topic5_raw_seeg_state/raw_cache.py`、`spectral_target.py`、`windows.py`<br>`scripts/topic5_raw_seeg_state/build_raw_cache.py`、`build_spectral_target.py`、`build_window_index.py`<br>`tests/test_raw_seeg_state_io.py` |
| **Worker C** | `src/topic5_raw_seeg_state/model.py`、`dynamics.py`、`losses.py`<br>`tests/test_raw_seeg_state_model.py` |
| **Worker D** | `src/topic5_raw_seeg_state/train.py`、`baselines.py`、`analysis.py`<br>`scripts/topic5_raw_seeg_state/run_patient.py`、`run_baselines.py`、`aggregate_cohort.py`、`make_figures.py`、`queue_runner.py`<br>`tests/test_raw_seeg_state_train.py` |

公共接口先由主 agent 在 `contract.py` 冻结；worker 只 import，不修改。worker 若
认为某常量有误，**报告**而不是改。

---

## 3. 阶段划分与验收

### 阶段 A — 数据基础设施

| 步骤 | 产物 | 验收 |
|---|---|---|
| A1 连续时间索引 | `data/dataset_manifest.parquet` | 34/34 患者有行；9 位无 inventory 的 Yuquan 患者标 `source_kind=edf_header`；无 `block_end < block_start` |
| A2 触点元数据 | `data/contact_metadata.parquet` | 每患者 bipolar 通道 ≥20；`coord_valid` 比例记录；raw cache 列序 = 本表 `channel_index` 序 |
| A3 分钟网格 + eligibility | `data/window_index.parquet`、`data/eligibility_summary.csv` | 每位患者 train/val 的 h=1/5/10/100 可用窗口数；封条检查通过 |
| A4 split 落地 | `data/split_manifest.json` | 与上游 `SPLIT_MANIFEST.json` 边界逐位一致；`test_status=SEALED` |
| A5 raw cache | `/mnt/.../<subject>/raw_256hz.zarr` | 形状 = (n_minutes·15360, C)；抽样比对原始信号谱 |
| A6 spectral target | `/mnt/.../<subject>/spectral_target.zarr` | 形状 = (n_minutes, C, 12)；有限值率 |
| A7 审计报告 | `data/data_audit.json` | 7 条硬性无效条件逐条 PASS |

### 阶段 B — 4–6 位患者 pilot

选择需覆盖：两个 dataset、高/低触点数、高/低事件密度、高/低连续记录质量、
不同采样率。**pilot 每个配置先跑一个 seed**；只有方向不稳或结果异常才加 seed。

顺序：
1. 仅 1 分钟 horizon；
2. 1/5/10 分钟；
3. 加 stable dynamics；
4. 加 consistency；
5. 加 100 分钟；
6. baselines；
7. state-swap；
8. figures。

### 阶段 C — 34 位患者扩展

模型与超参**冻结**后，以患者为单位排队（单 GPU 串行）。每位患者产出：

```
per_subject/<subject>/
    checkpoint.pt
    config.json
    training_curve.json
    validation_horizon_metrics.json
    latent_trajectory.zarr
    dynamics_modes.json
    decoder_loading.npy
    state_consistency.parquet
    state_swap_results.parquet
    run_manifest.json
```

队列统计**以患者为单位**，不把分钟窗口当独立生物学样本。

### 阶段 D — 后续版本

R0.2 / R0.3 / E0.4a / E0.4b / E0.5 / E0.6，见科学 spec §11。**R0.1 完成且有清楚
产物之前，不得把 E0.4–E0.6 混进同一训练作业。**

---

## 4. 资源与 OOM 管理

**必须采用**：raw streaming；chunked Zarr；patient-specific batch；gradient
accumulation；mixed precision（并监控非有限值）；动态 batch size；
**每 GPU 同时最多一个主要训练作业**；`OMP_NUM_THREADS=1`；`MKL_NUM_THREADS=1`；
受控 DataLoader workers；patient-level job queue。

**禁止**：一次载入完整患者记录到内存/GPU；为并行同时启动全部 34 个 GPU 训练；
多 worker 同时写同一 Zarr chunk（缓存构建按患者分片，一个患者一个进程）；
多作业共用同一 checkpoint 路径；OOM 后无限重试相同配置。

**OOM 处理链**：记录（患者 / batch / shape / 显存）→ batch 减半 →
增加 gradient accumulation → 减少 DataLoader prefetch → 从最近 checkpoint 恢复 →
其他患者继续。全部写入 `logs/oom_events.jsonl`。

**显存预算估算**（d_model=128，C≈100，10 min context）：
每样本约 240 k 个 250 ms patch token，前向+反向约 320 GFLOP，激活约 1.2 GB。
起始 `batch_size=2` + `grad_accum=4`，AMP 开启；OOM 则降到 1 + 8。

---

## 5. 长任务与断点续跑

- tmux session：**`raw_seeg_state_r01`**；长作业一律 `nohup setsid` 起，
  PID 写 `logs/<job>.pid`，命令/cwd/env 写 `jobs/<job>.launch.json`。
- 所有 manifest 用 `contract.atomic_write_json`（临时文件 + rename）。
- 每位患者的训练每 N 步落 checkpoint，`run_patient.py` 支持 `--resume`。
- 日志逐患者分开：`logs/<subject>.log`。
- 不依赖持续网络连接；网络步骤失败不阻断本地工作。
- **不得仅以"作业已启动"宣布任务完成。**

`CURRENT_HANDOFF.md` 必须持续维护，包含：当前阶段 / 已完成患者 / 正在运行患者 /
失败患者及原因 / tmux session / PID+GPU / 日志路径 / 下一条可执行命令 /
预计产物 / 当前科学边界。

---

## 6. 图（阶段 B 起）

| 图 | 内容 |
|---|---|
| **R1** | 模型结构与数据流 |
| **R2** | 1/5/10/100 分钟 forecast error vs horizon，对比 mean / persistence / feature-AR / identity dynamics |
| **R3** | 代表患者 observed vs open-loop spatial-frequency trajectory + latent mode loading |
| **R4** | matched state-swap 与 state consistency |

每张图同时输出 PNG + 矢量 PDF + metadata（生成命令 / 数据来源 / code revision），
并写 `figures/README.md`（中文，逐图说明"展示什么 / 看哪里 / 能支持什么 /
**不能**支持什么"）。**生成后必须实际打开 PNG 目视检查**，不能只因脚本成功就验收。
绘图前先读 `docs/figure_style_guide.md`。

---

## 7. 最终报告

- 白话报告：`docs/archive/topic5/raw_seeg_state_r0_1_plain_2026-08-21.md`
- 技术报告：`docs/archive/topic5/raw_seeg_state_r0_1_technical_2026-08-21.md`

内容清单见用户任务书第十三节，逐项落实。

---

## 8. 需要停下来问用户的情况（仅此六项）

1. 原始数据挂载不存在；
2. 关键通道/时间映射无法由现有合同判断；
3. 需要打开正式检验分区；
4. 需要改变核心科学问题；
5. 需要删除或覆盖已有重要结果；
6. 需要显著扩大模型或改变主要 target。

其余情况（普通探索性阴性结果、batch size、chunk 大小、checkpoint 频率、
图排版、内部模块组织、患者调度顺序）自主决定。

---

## 9. 已知风险登记

| 风险 | 现状 | 缓解 |
|---|---|---|
| 只有 1 块 GPU | 已确认 | 患者串行排队；预估 34 患者 × ~15 min ≈ 9 h |
| 缓存 150–200 GB | 已确认盘位 | 放 `/mnt/yuquan_data`；先建 pilot 6 位（≈30 GB）验证再全量 |
| 9 位 Yuquan 患者无 block inventory | 已确认 | 从 EDF 头重建，manifest 标 provenance |
| 256 Hz recording 压低频率上限 | 已确认（139/384/583） | 全队列锁 1–100 Hz，逐患者标 `nyquist_limited` |
| Yuquan 单患者 dev 仅 ~11–15 h train | 已确认 | 允许；`eligibility_summary` 逐患者报可用窗口数，cohort 统计按患者加权 |
| 100 min horizon 在部分患者窗口数偏少 | 初估 25/25 有 ≥100 窗口 | 逐患者报分母，不合并成"cohort n" |
| Epi-PRSSM v0.1 产物被误覆盖 | 未发生 | R0.1 只读 v0.1 的 SPLIT_MANIFEST，写入全部在新根目录 |

---

## 10. 集成闸门（主 agent 亲自执行，worker 的"代码完成"不算通过）

worker 的单元测试各自只覆盖自己那一段。真正会毁结论的错误发生在**交界处**。
在启动 pilot 训练之前，主 agent 必须在一位真实患者上跑完下面 8 条，全部 PASS 才放行。
脚本：`scripts/topic5_raw_seeg_state/integration_check.py`（主 agent 独占）。

| # | 检查 | 判据 |
|---|---|---|
| 1 | 合同自洽 | 12 个频段每个都有 ≥1 个 FFT bin；`PATCH/WINDOW/MINUTE` 采样数整除关系成立 |
| 2 | 数据合同产物齐全 | 5 个文件存在，34/34 患者有行，7 条硬性无效条件逐条 PASS |
| 3 | **缓存时间对齐** | 从缓存任取一个已缓存分钟，用它自己重算 Welch 频谱，与 `spectral_target.zarr` 中该分钟逐元素相对误差 <1e-4 |
| 4 | **通道序一致** | 缓存列序与 `contact_metadata.channel_index` 一致——用同一分钟的每通道方差排序做指纹比对，不看名字 |
| 5 | Dataset ↔ Model | 一个真实 item 直接喂 `RawSeegStateModel`，四个 horizon 输出形状正确、全有限；`assert_no_forbidden_inputs` 在注入 `soz=` 时抛错 |
| 6 | **封条** | 遍历全部产物（parquet / zarr / json / cache_index）中的每一个时间戳，`max < dev_end_epoch` |
| 7 | **只用 train 归一** | 用 train 分钟独立重算 mean/std，与 `train_stats.json` 逐元素一致；把 validation 数据 ×10 后重算，统计量**不变** |
| 8 | 端到端可续跑 | 5 步训练 → checkpoint → resume → 再 5 步，参数与不间断 10 步逐位一致；loss 全程有限 |

第 3、4、6、7 条是"静默污染"型错误的唯一拦网，任何一条 FAIL 都必须修复后重跑
受影响作业，**不得以"其他检查都过了"放行**。
