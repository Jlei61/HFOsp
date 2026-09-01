# 可直接交给 Agent B 的执行 Prompt：H2b seizure transfer

你接手 `/home/honglab/leijiaxin/HFOsp` 的 Group-Event State v0.2 Agent B。你的职责是冻结并读取间期任务 producer，检验其能否预测发作风险和发作早期空间场。不要用 seizure label 反向训练 state，也不要等待 Agent A 宣布阳性才开始仪器与 support 工作。

## 1. 开始姿态

1. 先读当前 worktree 的 `AGENTS.md` 和其要求的 legacy/data/topic 文档。
2. 完整读：
   - `docs/topic0_methodology_audits.md`
   - `docs/topic5_seizure_subtyping.md`
   - `docs/epilepsiae_dataset_structure.md`
   - `docs/archive/topic5/group_event_state_v0_1_data_contract_2026-08-31.md`
   - `docs/archive/topic5/group_event_state_v0_2_common_contract_2026-09-01.md`
   - `docs/archive/topic5/group_event_state_v0_2_engineering_invariants_2026-09-01.md`
   - `docs/archive/topic5/group_event_state_v0_2_h2b_spec_plan_2026-09-01.md`
3. 从 `codex/topic5-group-event-state-v0-2` 建独立 branch/worktree，例如 `codex/topic5-group-event-state-v02-b`。记录 base commit、dirty state、worktree 和活动队列。core producer 代码视为只读；新增 H2b 模块/adapter，不覆盖 A 的 registry。
4. 旧 `/tmp/hfosp_group_event_state_v01` 队列与结果只读。开始时实时检查 PID/GPU；不得停止或复用其 tag。

## 2. 科学目标

读取 registry 中所有 `B_multiscale`、`P_local`、`P_slow`，并列回答：

1. fixed-time state 是否让我们更早知道离下一次发作还有多久；
2. 同一间期状态是否在 6 h/2 h/30 min/5 min 前预测下一场发作最初 5 s 的 per-contact 能量/募集场和入口路径。

B2 与 B1 是并列主任务。只做 AUC 或只取发作前最后一个 IED 不算完成。

## 3. 必须完成的实现

1. 用 recording code 显式 crosswalk Yuquan/Epilepsiae seizure IDs，逐 onset 核对；输出未匹配与歧义，不允许字符串 inner join 静默丢患者。
2. 每 5 min fixed-grid risk set，离散 bins：0–5、5–15、15–30、30–60 min、1–2 h、2–6 h、far/censoring。
3. early ictal field：onset 后前 5 s 为主、10 s 敏感性；构造 normalized per-contact energy/recruitment vector、first group、laterality、entropy、early axis、IED-to-ictal reuse。
4. 只用 TRAIN seizures 建 route/template/normalization；held-out seizure 不参与定义。
5. baseline 包含 clock/session/coverage、`B_multiscale`、time since previous seizure、postictal/cluster、day since admission；sleep/ASM/stimulation 仅在元数据可靠时加入。
6. state producer 全冻结；同一 risk rows/held-out seizures 比 baseline、recent IED、`P_local`、`P_slow`、state+current event。
7. onset 终止 trajectory；offset 后 60 min 才开新 segment，30/120 min 只敏感性。不得隔着发作静默传播。
8. 主时间 null 为 within-session block circular shift；粗 matched donor 仅敏感性，不能过匹配 repertoire/participation。
9. 评估以 held-out seizure 为基本分母：time-dependent log score/Brier/calibration 和 early-field continuous score；event rows 不冒充样本量。

## 4. 执行顺序

1. B0：先完成 support inventory、crosswalk、risk-set 和 early-field target；目视检查每队列至少 3 场 onset/channel 对齐。
2. B1：用旧 trajectory 标 `plumbing_only` 调通 survival/field/censoring/schema；绝不把该效应写成 v0.2 人体结果。
3. B2：registry 出现 producer 后自动读取全部 producer，不按 A 的效果筛选。缺失 producer 标 `not_available`，不 fallback。
4. B3：先固定 3 位 seizure support 较好患者 × 3 seeds；再扩所有有 held-out seizures 的 development 患者。
5. B4：同时完成 B1/B2 lead-time curves；一项不可估计不阻断另一项。预先指定的主配置可加 5 seeds。

## 5. 并行协作与等待方式

你可以在 A 训练时完成全部 CPU target/support/baseline/visual QA。用只读轮询或文件 watcher 等待 registry producer 条目；不要反复重启队列，不要修改 A 的 producer。需要新增 schema 时写 additive adapter/issue 文件，并在 `CURRENT_HANDOFF.md` 记录。

## 6. 资源与持久运行

- Python 固定 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`；线程环境全部设 1。
- 大 target/预测写 `/data/hfosp_group_event_state_v0_2/agent_b/`；仓库结果根只放索引、小统计和报告。
- 长任务用 nohup+setsid 或 tmux；单一 queue owner、原子结果、幂等 resume、STATUS/PID/log/CURRENT_HANDOFF。
- CPU 先 8 worker，按 p95 RSS、`MemAvailable`、物理盘压力逐级扩到 16；发作 raw-field 读取优先按记录分片，避免随机寻道。
- GPU probe 若需要，先每 GPU 1 job 实测峰值，保留 4 GiB，80% 峰值预算计算 slot；GPU 被旧队列高利用时不叠加。OOM 降单 job batch/chunk，最多 3 次，记 `resource_failed`。
- 原子写 `.../v0_2/shared/resource_leases/agent_b.json`；不得 `pkill -f`，不得抢占 A/C 租约。

## 7. 完成标准和交付

承重图两部分：state 对 survival/Brier 的增量随 lead time；state 对 early ictal field/path 的增量随 lead time。每点明确 held-out seizure 数。生成 PNG+PDF+metadata+中文 `figures/README.md` 并目视验收。

交付 plain/technical 两份报告、逐发作预测、machine JSON/CSV、crosswalk/support inventory、复现命令和 `CURRENT_HANDOFF.md`。分开报告工程完成、assay 可估计、cross-task 支持。不要打开 formal/sealed，不碰 paper-ready Fig1–Fig4。
