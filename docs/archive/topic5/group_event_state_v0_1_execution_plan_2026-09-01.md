# Group-Event State v0.1 — 执行计划（frozen 2026-09-01）

## 1. 队列与患者选择（选择规则先于任何模型结果）

选择条件**只用覆盖度**，与模型好坏无关：

1. `waveform_pointer_fraction == 1.0`（每个 packed event 都能回到原生样本区间）
2. `max_events_in_contiguous_session ≥ 5000`
3. 两个队列都要有、两种采样率都要有、要有带发作的患者供 H2b

audit（41 位候选、1,774,188 事件）结果：**27 位合格**，1,663,394 事件。
名单与逐患者分母见 `results/epi_prssm/group_event_state/v0_1/subject_inventory.csv`。

## 2. 缓存

- shard 单位 = block；每个 shard 原子写 `<record>.npz` + `<record>.manifest.json`；
  重跑跳过已完成 block（可恢复）。
- 缓存根：`/data/hfosp_group_event_state_v0_1/cache`（仓库盘 `/` 只剩 42 GB，**不得**用）。
- 实测（承重的资源数字）：
  - **107 MB / 1000 事件**（Epilepsiae，16 触点 × 768 样本 × 3 视图）
  - **81 MB / 1000 事件**（Yuquan，8 触点 × 2000 样本 × 2 视图）
  - 单 worker 峰值 RSS **0.37–4.9 GB**（宽触点患者在高端）
- **并发数是实测决定的，不是猜的**：32 workers 时单块 283–473 s，18 workers 时同一
  患者同量级块只要 40–90 s → 磁盘寻道竞争，32 并发的总吞吐反而低 ~2.4×。
  最终按**物理盘分池**：Epilepsiae 9 workers（sdd）+ Yuquan 7 workers（sdc）。
- Yuquan 池实测：172/172 blocks、403,468 事件、30.3 GB、2.70 h、0 失败、maxRSS 4.9 GB。

## 3. 一致化（consolidate）

shard → 每患者一组 memmap `.npy` + `index.json`：事件时钟、contiguous session、
真实 Δt、背景锚点对齐、ictal 排除、按时间 70/10/20 切分、静态几何（坐标可得时）。
坐标解析不了的患者保留 `geometry.status` 与一个"坐标是否真实"的标志位，
**不塞 0 冒充真实位置**。

## 4. 训练

- **每患者一个模型**（触点宇宙、采样率、几何都是患者特异的），
  跨患者用 patient-first 聚合，不把事件数或滑窗数当患者数。
- 序列 = contiguous session；chunk 内 encoder 并行、状态循环串行；
  chunk 间 detach（truncated BPTT）。
- `chunk_events` 由 `n_contacts × n_context_samples × n_views` 自动收缩，
  宽触点患者（如 `zhangbichen` 52 触点）自动减小。
- AMP bfloat16 只用于 encoder；**状态演化、Δt、似然累加全部 FP32**。
- 早停看 inner-validation 总 NLL（不含 `group_size`，那是 MAE 不是 NLL）。

### 训练充分性（每个 run 都要落盘）

train/val 曲线、selected epoch、stop reason、grad norm、**逐模块参数更新幅度**
（`encoder` / `state` / `heads` 相对 Frobenius 变化）、state norm、slow update 幅度、
拟合后的 `τ_fast` / `τ_slow` 分位、参数量、wall time、峰值显存。
**epoch-0 或参数没动的结果不得用于判断科学假设。**

## 5. GPU / OOM / nohup 合同

- 硬件实测：2 × RTX 3090（各 24 GB，启动时全空闲）、80 核、251 GB RAM、
  `/data` 3.4 TB 可用、`/` 仅 42 GB。
- 单一 queue owner 进程持有队列；每个 job 是独立子进程，用 `CUDA_VISIBLE_DEVICES` 钉在一张卡上；
  完成判据是 `result.json` 存在 → 重跑即续跑。
- OOM 处理：只捕获 `torch.cuda.OutOfMemoryError` → 记录失败配置 → `empty_cache` →
  `chunk_events` 减半 → 重试（默认 3 次）；仍失败则写 `resource_failed.json`，
  **明确标注"这不是科学阴性"**。绝不无限重启同一个 OOM 作业。
- 长作业用 harness 管理的后台执行；**教训**：在工具调用里 `nohup setsid ... &`
  会被工具超时连带杀掉（本轮已发生一次，24 个 block 的进度靠 shard 幂等性挽回）。
- `pkill -f <pattern>` 会匹配到发起它的那个 shell 自己（本轮已发生一次，
  自杀在 `pkill` 那一行，后面的 `rm`/patch 全部没跑却看起来"成功了"）。
  按 PID kill，或用不会自匹配的正则。

## 6. seed

- 工程 triage 最少 3 seeds；承重对比 5 seeds。
- seed 必须真正改变初始化与采样（`torch.manual_seed` + 独立 `torch.Generator`
  驱动 τ 的 log-uniform 初始化）。
- 交付前必须验证多 seed 的 payload **不是逐字节相同**，不得把 3 份相同结果报成 3 seeds。

## 7. 产出

机器可读：
`source_audit.json` / `subject_inventory.csv` / `block_inventory.csv` /
`event_pointer_audit.json` / `contiguous_session_inventory.csv` / `band_availability.csv` /
`cache_build_*.json` / `dataset_summary.json` / `runs/<tag>/<run_id>/result.json` /
`STATUS.json` / `CURRENT_HANDOFF.md`。

文档：
- `group_event_state_v0_1_scientific_spec_2026-09-01.md`
- `group_event_state_v0_1_data_contract_2026-08-31.md`
- 本执行计划
- source audit 报告 / 资源与 OOM 报告 / 白话结果报告 / 技术结果报告

全部原子写入（先写临时文件再 rename）。

## 8. 已知不做的事（v0.1 边界）

- 不做跨患者共享 encoder（触点宇宙患者特异）
- 不用 `results/lagpat_broad*` 的 top_n=20 重打包（会收窄 large-narrow 患者）
- 不用旧 `dataset_v0_4` 的已筛 definite-interictal 子集
- 背景 SEEG 只修正**内容**端点，不参与 timing 预测（timing 预测发生在
  下一个背景锚点被观测到之前；连续 hazard 形式留给 v0.2）
