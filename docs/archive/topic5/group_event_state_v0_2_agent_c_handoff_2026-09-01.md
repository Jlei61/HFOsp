# 可直接交给 Agent C 的执行 Prompt：H3 event feedback

你接手 `/home/honglab/leijiaxin/HFOsp` 的 Group-Event State v0.2 Agent C。你的职责是区分“IED 只是共同慢状态的读数/observer 信息”与“IED 数量或内容进入后续状态转移”。不要把普通 RNN event update 或 latent 位移写成生理塑形。

## 1. 开始姿态

1. 先读当前 worktree 的 `AGENTS.md` 和其要求的 legacy/data/topic 文档。
2. 完整读：
   - `docs/topic0_methodology_audits.md`
   - `docs/topic5_seizure_subtyping.md`
   - `docs/archive/topic5/group_event_state_v0_1_data_contract_2026-08-31.md`
   - `docs/archive/topic5/group_event_state_v0_2_common_contract_2026-09-01.md`
   - `docs/archive/topic5/group_event_state_v0_2_engineering_invariants_2026-09-01.md`
   - `docs/archive/topic5/group_event_state_v0_2_h3_spec_plan_2026-09-01.md`
3. 从 `codex/topic5-group-event-state-v0-2` 建独立 branch/worktree，例如 `codex/topic5-group-event-state-v02-c`。记录 base commit/dirty/worktrees/活动队列。A 的 core/registry 只读；H3 新模型和 adapter 放独立模块与结果根。
4. 旧 `/tmp/hfosp_group_event_state_v01` 队列与结果只读。实时核实 PID/GPU，不停止、不 patch、不复用 tag。

## 2. 科学目标

先画观察轨迹上的 functional innovation，再显式比较：

- `M0_no_feedback`：common-drive/readout-only；
- `M1_count_rate_feedback`：IED 数量/负荷进入低容量 signed feedback；
- `M2_mark_specific_feedback`：在 count/time 固定后，participation、extent、waveform、multiband 内容提供额外 feedback。

主 estimand 是完全未见 future block 的 held-out log-score 增量，不是 hidden-state 欧氏距离。最高措辞是 `event-feedback-like predictive dependence`。

## 3. 必须完成的实现

1. 从 registry 读取 producer/trajectory/functional-readout，并验证 source/config/checkpoint/hash；缺失不 fallback。
2. functional innovation：对每 event 保存读取事件前后的冻结 future-block 功能读出差值，并关联未来 5/30/120 min 实际变化。
3. M0/M1/M2 共享 observer、base dynamics、decoder、optimizer steps 和 checkpoint selection；新增 feedback 用低容量 adapter，允许正负作用。
4. burden estimand 与 content estimand 分开：burden 不能匹配掉 exposure count；content 必须保持 event count/times，只替换 mark。
5. fixed-time 主 horizon 5/30/120 min；6 h 仅 support 足够时探索。100/1,000/10,000 event 只是时间映射敏感性，不跑全笛卡尔网格。
6. 主 perturbation 只保留 real sequence、no feedback、state-matched mark replacement；rate-preserving shuffle 和 burst thinning secondary。
7. constant/intercept/drift zero-truth 保留为回归测试，不扩成完整人体主臂。
8. exposure/target 不跨 gap、split、seizure；统计以不重叠 physical blocks 为分母。future block 不读取中间真实事件。
9. 输出 count/rate 与 conditional mark 分解，以及 event-type-specific signed impulse response；不预设所有 IED 都促发作。

## 4. 执行顺序

1. C0：先完成 coverage-support inventory、M0/M1/M2 接口、functional-readout 对齐和零真值测试。
2. C1：固定 3 位长患者 × 3 seeds 画 functional innovation trajectory，确认语义和资源；这一步不宣布反馈。
3. C2：训练 M0/M1/M2，checkpoint 只看间期 inner-validation future-block objective；synthetic 只校准可识别性，不作继续 gate。
4. C3：运行 5/30/120 min 人体主比较，burden/content 分开；再扩所有有有效不重叠窗的 development 患者。
5. C4：在冻结模型上做最小 perturbation；预先固定主配置可加 5 seeds。C 首轮总 GPU 预算不得超过 B。

## 5. 与 A/B 协作

你可在 registry producer 完成前并行实现 support、schema、synthetic 和 adapter，但承重人体结果必须绑定 registry 的真实 producer。A 的 H1 阳性不是 gate；B 的 risk readout 可作为 secondary functional output，缺失不阻断 repertoire/field H3。

不要编辑 A 的 core 或整份 registry；需要字段就写 additive adapter/issue 文件。你的独占结果根为 `results/epi_prssm/group_event_state/v0_2/h3/`。

## 6. 资源与持久运行

- Python 固定 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`；所有线程环境设 1。
- 大 replay/target/checkpoint 写 `/data/hfosp_group_event_state_v0_2/agent_c/`。
- 长任务使用 nohup+setsid 或 tmux；单一 queue owner、原子输出、幂等 resume、STATUS/PID/log/CURRENT_HANDOFF。
- CPU 从 8 worker 起，按 p95 RSS、`MemAvailable` 和 I/O 逐步扩到 16。
- GPU 先每卡 1 个代表性 smoke，记录峰值；保留至少 4 GiB，按 80% 峰值预算定并发。旧队列高利用时先做 CPU 工作。OOM 只降当前 job batch/chunk/slot，最多 3 次，记 `resource_failed`，不得删患者。
- 原子写 `.../v0_2/shared/resource_leases/agent_c.json`；不得 `pkill -f`，不得抢 A/B 租约。

## 7. 完成标准和交付

承重图：M0/M1/M2 对未见 future-block 的 score（count 与 conditional mark 分开）以及 event-type-specific signed impulse response。其他 latent、update norm、额外尺度/扰动进入辅助材料。生成 PNG+PDF+metadata+中文 `figures/README.md` 并目视验收。

交付 plain/technical 两份报告、逐 block machine JSON/CSV、support inventory、复现命令和 `CURRENT_HANDOFF.md`。分开报告工程完成、assay 可估计和 H3 支持。不要打开 formal/sealed，不碰 paper-ready Fig1–Fig4。
