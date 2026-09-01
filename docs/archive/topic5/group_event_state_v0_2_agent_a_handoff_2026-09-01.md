# 可直接交给 Agent A 的执行 Prompt：H1/H2a predictive-state identification

你接手 `/home/honglab/leijiaxin/HFOsp` 的 Group-Event State v0.2 Agent A。你的职责是建立三种可复用 producer，并完成 H1/H2a；不要替 B 做 seizure 结果，也不要替 C 做反馈机制结论。

## 1. 开始姿态

1. 先读当前 worktree 的 `AGENTS.md`，并按其中顺序读 legacy/data/topic 合同。
2. 完整读：
   - `docs/topic0_methodology_audits.md`
   - `docs/topic5_seizure_subtyping.md`
   - `docs/archive/topic5/group_event_state_v0_1_data_contract_2026-08-31.md`
   - `docs/archive/topic5/group_event_state_v0_2_common_contract_2026-09-01.md`
   - `docs/archive/topic5/group_event_state_v0_2_engineering_invariants_2026-09-01.md`
   - `docs/archive/topic5/group_event_state_v0_2_h1_h2a_spec_plan_2026-09-01.md`
3. 从共同基线 branch `codex/topic5-group-event-state-v0-2` 建独立 branch/worktree，例如 `codex/topic5-group-event-state-v02-a`。先记录 base commit、dirty state、现有 worktree 和旧队列；保存用户无关修改，不在主 worktree 直接开发。
4. `/tmp/hfosp_group_event_state_v01` 和其 `tag=main` 是只读旧实验。开始时用精确 PID/命令/GPU 查询重新核实；不得停止、patch、覆盖或把 v0.2 job 接到旧 queue。

## 2. 科学目标

一次 RNN timestep 是一次完整间期群体事件，保留 participation、tied groups、`lagPatRaw` 连续延迟、bipolar/CAR waveform、多频带能量/峰时/cross-band lag 和几何/mask。连续背景只作 manifest 明确的辅助 observation。

你必须并列生产：

- `B_multiscale`：1/5/30/120 min 的可解释多尺度 GLM + 低容量 MLP；
- `P_local`：修复后 session-preserving next-event model；
- `P_slow`：直接用 5/30/120 min future-block loss 训练的 multi-horizon state producer。

不要从一步 checkpoint 事后硬找慢状态。慢状态的主定义是固定物理时间上对未见 future block 的功能预测，不是 `z_slow` 名字、tau 范围或 reset 结果。

## 3. 必须完成的实现

1. 按累计 recorded physical time 生成 TRAIN/inner-validation/development-test；target 不跨 split/gap/seizure。
2. 每 5 min fixed-grid anchor；把 last-event state 按真实 `dt` 传播到 grid。
3. 稀疏 future-target builder：5/30/120 min 的 count 与 conditional mark 分开，使用 cumulative sums/prefix counts/sparse participation，禁止物化巨大 dense target。
4. session-preserving trainer：batch 不同 sessions；同 session chunks 严格有序并 carry state，只 detach 不 reset。
5. `P_slow` loss：local + 5/30/120 min；确认每个 head 和上游 state encoder 均收到梯度并实际更新。
6. `checkpoint_registry.json`：显式登记 input、background、feedback、physical_dt、objective、anchor grid 和 source/config/checkpoint hash；采用原子 producer entry，禁止 silent fallback。
7. 主比较：`B_multiscale`、`P_local`、`P_slow`、correct-time、within-session block circular shift；shift 必须大于目标 horizon。
8. H2a same-prefix：匹配首发触点、前两个 tied groups、前 50–100 ms waveform/早期能量，预测 later recruitment、STOP 和后续 multiband expression。
9. clusters 仅作解释；continuous event-embedding distribution 是主稳健 repertoire target。
10. 缩减诊断：reset 仅 1/100/1000/full 与 5/30/120 min/full；fast/slow/memoryless/matched donor 放辅助。

## 4. 实验顺序

1. A0：完成 target/split/session/seizure/registry 回归测试。
2. A1：固定 3 位长患者 × 3 seeds 跑 smoke，验证收敛、梯度、峰值资源和输出语义。
3. A2：原中期 8 位 × 3 seeds，三 producer 同 anchor 比较；这不是阳性 gate。
4. A3：扩预先定义的全部 development 可训练患者，不按结果选人；短 coverage 只缺失长 horizon，不记阴性。
5. A4：预先固定的承重配置可补 5 seeds；不能看到结果后才选配置补 seed。

## 5. 资源与持久运行

- Python 固定 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`。
- 所有线程环境设 1。长训练用 `nohup` + `setsid` 或 tmux；必须单一 queue owner、原子 manifest、幂等 resume、STATUS/PID/log/CURRENT_HANDOFF。
- 大输出写 `/data/hfosp_group_event_state_v0_2/agent_a/`，仓库结果根只放索引、小 JSON/CSV/报告。
- 不要根据“还有显存”盲目多开。先每 GPU 1 个代表性 smoke，记录 peak reserved/allocated 和 wall time；保留至少 4 GiB，按 80% 峰值容量预算定 slot。若 GPU 已被旧队列持续高利用，先做 CPU 工作，不叠加训练。
- 资源允许时用满 worker：GPU 依据实测峰值逐级 1→2→4…；CPU 从 8 起，根据 p95 RSS、`MemAvailable` 和 I/O 扩到 16。OOM 对单 job 降 batch/chunk/slot，最多 3 次，记 `resource_failed`，不得删患者。
- 使用共享 `.../v0_2/shared/resource_leases/agent_a.json` 原子登记 PID/PGID、GPU、slot、峰值和心跳。禁止 `pkill -f`。

## 6. 完成标准和交付

承重图只有一张：future physical horizon 5/30/120 min 上相对 `B_multiscale` 的 score gain，拆 count 与 conditional mark，显示 `P_local/P_slow/correct-time/block-shift`；same-prefix 可做同图小 panel。生成 PNG+PDF+metadata+中文 `figures/README.md` 并目视验收。

交付：plain report、technical report、machine JSON/CSV、registry、完整复现命令和 `CURRENT_HANDOFF.md`。报告分别说明工程是否完成、assay 是否可估计、H1/H2a 是否支持。不要打开 formal/sealed，不碰 paper-ready Fig1–Fig4。
