# Group-Event State v0.3.3 — Workstream B：Persistent Training Laboratory 设计

**日期：** 2026-09-02
**分支 / worktree：** `codex/topic5-ges-v033-training-lab` @ `.worktrees/topic5-ges-v033-training-lab`
**基线 commit：** `233f3ad1`（v0.3.2 closeout；supervisor 尚未指定 release commit，见 §11 假设 A1）
**Release 状态：** `V0_3_3_EXECUTION_RELEASE.json` 全盘不存在 → 本轮只做 harness / 单元测试 / tiny-overfit / synthetic smoke / 资源 sentinel；不启动任何承重人体 search。
**上游合同：** `group_event_state_v0_3_3_dual_view_state_spec_2026-09-02.md` §11（T0–T6）、`..._plan_2026-09-02.md` §4（B0–B8）、`..._agent_b_training_laboratory_handoff_2026-09-02.md`、`..._training_supervisor_handoff_2026-09-02.md` §4–§6。
**状态：** `DESIGN_LOCKED_FOR_IMPLEMENTATION_NO_RELEASE`

## 0. 一句话

Training Laboratory 是一个**与科学 target 无关的持续训练服务**：它接收 Agent C 的原子训练请求（"训练什么"），对 `S_N`、`S_G`、R0、R1 和明确批准的 exploratory 模型执行同一套 T0–T6 合同（"如何公平地训练到位"），交付 `training_card.json`，并只在 tiny overfit、synthetic recovery、blocked inner-validation 三者都成立且 checkpoint 不落在 warm-up / 预算边界时才标 `TRAINING-ADEQUATE`。它不改 target、split、`H_rate/H_mark`、endpoint reduction、H2b label discipline 或患者职责；training adequacy 永远不等于 H1/H2 阳性。

## 1. 角色边界与本轮范围

| 允许（无 release） | 禁止（无 release） | 永远禁止 |
|---|---|---|
| 实现 harness、单元测试 | 人体 development 训练搜索（T1–T4 on human） | 读取 sealed partition |
| toy / synthetic smoke search（T0–T6 全链路） | 读取任何 `dev_test` 数字 | 改 target / split / H / endpoint / H2b 标签 |
| 人体 T0 tiny-slice overfit（只用 `state_train`） | 用人体 inner-val 选配方 | 把 training adequacy 写成 H1/H2 阳性 |
| 资源 sentinel（人体 bundle 只作负载样本） | 新增 horizon / 患者 / seed | `pkill -f`、杀其它 topic 作业 |

人体 T0 / sentinel 只用 v0.3.2 已经作为 development 患者使用过的 `epilepsiae_1146`（fingerprint 已缓存），标 `DIAGNOSTIC`；不触碰任何可能成为 v0.3.3 "untouched replication" 的患者。

## 2. 训练请求接口（B0）

请求文件：`shared/job_requests/*.json`（Agent C 写 `science_*.json`）。13 个必填字段：

```text
request_id            str  唯一；status 文件名 training_<request_id>.json
scientific_target     {family: S_N|S_G|R0|R1|exploratory, objective: <registered name>, bins_seconds|...}
input_view            {kind: toy|synthetic|R0|R1, subject?, synthetic?: {...}}
state_family          fixed_leaky | gated_exploratory（后者需 exploratory_approved=true）
split_hash            str  必须等于 DataView.split_hash
baseline_H            {source: agent2_registry|provisional_local|<agent_a_name>, hash?}
endpoint_and_reduction {selection_phase: inner_val, metric: nb_nll, reduction: mean_per_anchor}
search_budget         {n_configs, max_steps, rung_steps:[...], eta, seeds_low, seeds_mid, seeds_final, n_final}
seed_policy           {base_seed}
resource_ceiling      {max_workers, gpu_ids, vram_gib, ram_gib, threads}
code_commit           str
input_hash            str  必须等于 DataView.input_hash
requested_by          str
```

判定：
- 任一字段缺失 / 空 / 类型错 → `INVALID_REQUEST`（附 `missing_fields`），**不猜**；
- `objective` 未在 trainable registry 注册 → `INVALID_REQUEST`（退回 Agent C）；
- `split_hash` / `input_hash` 与数据视图不一致 → `HELD_MISMATCH`；
- `code_commit` ≠ 当前 HEAD → `HELD_CODE_COMMIT_MISMATCH`；
- `input_view.kind ∈ {R0,R1}` 且 release 文件缺失 → `HELD_NO_RELEASE`；
- `gated_exploratory` 且未 `exploratory_approved` → `INVALID_REQUEST`。

job key = sha256(target, input_view, state_family, subject, seed, split_hash, config_hash, code_commit, input_hash)。

## 3. 数据视图 `DataView`

由 v0.3.2 `SubjectBundle`（真实或 toy）构造，**只暴露** `train`（= nested `state_train`，20–70% 有效记录时间）和 `inner_val`（= nested `dev_val`，70–80%，按时间在 TRAIN 之后）两个 anchor 集；`dev_test` 的 counts 在构造时被抹为 −1 并断言从不进入 loss / selection。
- `bins_seconds`：目标窗 `[(a_0,b_0),…]`，默认 spec §7.1 `[(0,300),(300,900),(900,1800)]`；bin counts 由 `event_times` 直接计数；`(0,300)`/`(0,1800)` 与 bundle `counts` 逐 anchor 一致（测试）。
- `log_mu_h[b]`：每 bin 的显式历史基线。真实 bundle 只能提供与 H registry horizon 重合的累计窗 `(0,h)`；其它 bin 缺 H → `DataView.missing_H_bins` 非空 → 请求 `HELD_MISMATCH(baseline_H)`。toy/synthetic 用 provisional ridge（`_fit_count_ridge` on B_multiscale，state_train 拟合、dev_val 选 ridge）逐 bin 拟合，`h_source="provisional_local"`。
- `split_hash` = sha256(subject, partition boundary epochs, fractions, horizons)；`input_hash` = sha256(feature fingerprint, standardizer 统计)。
- `blocks`：inner-val anchors 按 segment 内连续 `max(h,1800 s)` 时间箱分块，供 blocked bootstrap；`effective_independent_windows` 同 v0.3.2。
- `sample_weights(mode)`：`anchor_balanced` = 全 1；`event_balanced` = 与 anchor 之前 `max(taus)` 秒内事件数 `1+n` 成比例、TRAIN 上归一到均值 1（假设 A4）。
- 输入缩放：`zscore`（v0.3.2 `TrainStandardizer`）或 `robust`（TRAIN median / IQR·1.349，退化列→1），统计只由 TRAIN 事件决定。

## 4. `Trainable` 协议与参考实现

```python
class Trainable(Protocol):
    name: str
    def build(self, cfg: RecipeConfig, view: DataView, seed: int) -> nn.Module
    def param_groups(self, model, cfg) -> list[dict]        # name/params/lr/weight_decay，每个参数恰属一组
    def loss_terms(self, model, view, phase, *, differentiable_statistics) -> LossTerms
    def state_terms(self, model, view, phase) -> StateTerms  # anchor_state (A,D)、modulation (A,B)
    def with_state_override(self, model, view, phase, state) -> Tensor   # per-anchor NLL 用于 shift / mean 臂
    def h_only_nll(self, view, phase) -> np.ndarray
```

参考实现 `ResidualCountTrainable`（`objective="count_profile"`，`S_N` 家族）包装 `FlexibleResidualStateModel`：

| 旋钮 | 取值 | 实现定义 |
|---|---|---|
| encoder depth / width | 1/2/3；32/64/128 | 隐层数 / 隐层宽 |
| activation / dropout | ReLU/GELU/SiLU；0/0.1 | 隐层 |
| hidden norm | none / LayerNorm | 只在 encoder 隐层；**state 不做逐时刻 LayerNorm**（测试：模块树中 LayerNorm 只出现在 encoder） |
| init | xavier / orthogonal | 所有 Linear 权重；bias 0 |
| write_scale | 0.01/0.1/1 | 末层 Linear 初始化权重乘子（初始化旋钮，可被优化器改变） |
| write width | 2/4/8 | `channels_per_tau`；3 τ → 状态维 6/12/24 |
| time bank | {5,30,120} / {10,60,180} min | `taus_seconds` |
| alpha_init | 0.01/0.03/0.1 | 残差 gate 初值；**从 step 1 可训练**（无冻结） |
| gate_bias_init | −1/0/+1（仅 gated） | gated 更新网末层 gate 半部 bias 初值 |
| input scaling | zscore / robust | §3 |
| dispersion | frozen / low_lr | `log r` 固定为 TRAIN H-arm MLE / 可训练 LR=0.1×adapter |
| sampling | anchor_balanced / event_balanced | §3 权重 |
| TBPTT（仅 gated） | 30/60/120 min | chunk 边界 carry+detach，不 reset |

读出：`log μ_{a,b} = log μ_{H,a,b} + α · (W S̃_a)_b`，`W ∈ R^{B×D}` 无 bias（无自由截距），`S̃` = TRAIN 固定 mean/scale（与 v0.3.2 §3.1 相同、可微 TRAIN 统计 + 冻结 buffer）。loss = 每 anchor 对 bins 求和后加权平均（`endpoint_and_reduction.reduction = mean_per_anchor`）。NB 似然复用 `v032_model.readout.nb_log_prob`（内部 float64）。状态骨干复用 `v032_model.state.MarkedLeakyBank`；gated exploratory 新写 `GatedEventState`（同 v0.3.2 repaired 更新规则 + TBPTT chunk detach + gate bias init）。

## 5. Trainer（T1）

全批次（一个 optimizer step = 所有 TRAIN anchors），`fixed_leaky` 每步完整 chronological scan。
- 参数组：`encoder_weights / encoder_bias / state_weights / state_bias / adapter_w / adapter_gate_alpha / adapter_dispersion`，每组独立 LR（log-uniform `[1e-5,3e-3]` 采样）；bias/gate/dispersion 无 weight decay。
- optimizer：AdamW / Adam / RMSprop；schedule：constant / cosine / ReduceLROnPlateau（以 inner-val 为监控量）；全局线性 warm-up 0/5/10% of `max_steps`。
- 每步记录：loss、每组 grad norm、全局 pre/post clip norm、`clipped`、每组 update norm（‖θ_{t+1}−θ_t‖）、`first_active_step[group]`（该组 grad 非零且参数变化非零的首个 step）。
- 每 `validate_every` 步：inner-val NLL（冻结 TRAIN 统计）、TRAIN NLL、modulation RMS；best = inner-val 最小；`selected_step`、`selected_in_warmup`、`selected_at_budget_edge`、`all_groups_active_before_selection`、`plateau`（最近 `patience` 次 validation 改善 < tol）。
- NaN：首个非有限张量名 / step / 参数组 grad 写 `nan_dump.json`，退出码 4。OOM：退出码 3。resume：`last.pt`。
- `learning_curves.parquet` + `result.json`；checkpoint 含 TRAIN 统计 buffer、config_hash、split_hash、input_hash、seed、selected_step、parameter_sha256。

## 6. T0 诊断（每项一个 JSON 段，含判据常量回显）

| 项 | 定义 | pass |
|---|---|---|
| tiny_slice_overfit | 同一 segment 连续 `n_slice`（默认 12）个 TRAIN anchors，无 wd / dropout，训练 `steps`；`gap_closed = (NLL_H − NLL_end)/(NLL_H − NLL_saturated)`，saturated = μ=y 的 NB NLL | `gap_closed ≥ 0.5` |
| oracle_head_fit | 只训 adapter，state 用真值 z（synthetic）→ inner-val gain | gain CI_low > 0（仅 synthetic 视图） |
| state_write_jacobian | bank：`∂S_a/∂u_j` 数值 = `exp(−Δ/τ)`；`∂log μ/∂S̃ = α W`；均有限非零 | 相对误差 < 1e-4 |
| optimizer_membership | 每个 requires_grad 参数恰属一组；buffer 不在组内 | 精确 |
| first_active_step | 每组首个有效更新 step | 全组 ≤ `grace_period` |
| gradient_update_norms | 每组 grad/update norm 轨迹摘要 | 报告 |
| clipping_fraction | `clipped` 步数占比 | 报告（>0.5 flag） |
| amp_small_gradient | 同一 batch：bf16-encoder AMP vs FP32 各组 grad norm 比值 | 比值 ∈ [1e-2, 1e2] 且无零 |
| state_output_modulation | TRAIN/inner-val 上 `modulation` RMS、非零占比、动态 vs TRAIN-mean 状态输出差 | RMS > 0 |

## 7. 搜索（T1–T4）与失败驱动（T5）

- 空间 `SearchSpace.for_family(state_family)`：§4 旋钮 × §5 optimizer/schedule/warm-up/LR；seeded 随机采样（categorical 均匀、LR log-uniform）。
- ASHA：`rung_steps = [r_0, r_0·η, …, max_steps]`；每 rung 保留 top `1/η`（按 inner-val best）；**grace period 规则**：一个 config 只有在 `all_groups_active_step + validate_every ≤ rung` 且该 rung 内至少一次 validation 后才可被裁；否则顺延到下一 rung 并记 `grace_deferred`。
- seed policy：rung 0 单 seed；进入 rung ≥ 1 的 config 3 seeds（同 config 不同 seed 的 inner-val 取中位数比较）；最终 top `n_final ∈ {2,3}` 五 seeds。
- 一个 search batch = 一个 ASHA bracket；incumbent = 迄今最佳 config 的 seed-中位 inner-val gain。**连续两个 batch 无改善**（改善 < `tol`）→ 停止盲搜；**stable plateau**（incumbent 最近两个 batch 变化 < tol 且其 curve `plateau=true`）→ 收口。
- T5 分类表（输入：T0 结果、TRAIN/inner-val 学习情况、random-reservoir delta、support、budget edge）：

| 观察 | 分类 | 下一动作 |
|---|---|---|
| tiny overfit 失败 或 某参数组从未激活 | `gradient_path` | 查路径 / 容量 / LR / normalization |
| TRAIN NLL 未低于 H | `underfit` | 加容量 / LR |
| TRAIN 学会、inner-val 无增量 | `overfit_or_objective` | 正则 / 容量 / 退回 Agent C 查 objective-support |
| synthetic recovery 过、人体 inner-val 无增量、random reservoir 等价 | `objective_or_support`（非优化问题） | 退回 Agent C；分母由 Agent A |
| selected step 在预算末端 | `budget_edge` | 延长预算再判 |
| 两个 batch 无改善 | `search_exhausted` | 停止盲搜 |
| 有效独立窗口 < 请求最小值 | `support_insufficient` | 退回 Agent A/C |

`S_N` 学会而 `S_G` 不学 → 各自分类，不共用结论（分类按 request 独立）。

## 8. 训练卡（T6）

`training_card.json` 字段：request 引用、config + hash、split/input/code hash、curves 摘要（+ parquet 路径）、`best_step`、`plateau`、`seed_dispersion`（best NLL / gain / selected_step 的 std 与范围）、`gradient_update`（每组）、`clipping_fraction`、`state_variance_rank`（TRAIN 标准化状态协方差谱、participation-ratio 有效秩、readout 秩）、`random_reservoir_delta`（learned − random inner-val NLL，CI）、`shift_null`（inner-val shifted − correct，CI）、`output_modulation`、`tiny_overfit`、`synthetic_recovery`、`blocked_inner_val_gain`（H − correct，segment block bootstrap CI）、`selected_in_warmup`、`selected_at_budget_edge`、`all_groups_active_before_selection`、`evidence_label`、`adequacy_rule`。

`TRAINING-ADEQUATE` ⇔ `tiny_overfit.pass ∧ synthetic_recovery.pass ∧ blocked_inner_val_gain.ci_low > 0 ∧ ¬selected_in_warmup ∧ ¬selected_at_budget_edge ∧ all_groups_active_before_selection`；否则 `DIAGNOSTIC`。卡内**不得**出现 dev_test 数字；`selection_metric_is_canonical=false` 直到 Agent A evaluator hash 登记。

## 9. Queue 与资源

目录：
```text
/data/hfosp_group_event_state_v0_3_3/agent_b/{requests_seen,runs,t0,search,cards,sentinels,synthetic,logs,controller}/
/data/hfosp_group_event_state_v0_3_3/shared/{job_requests,job_status,resource_leases}/
results/group_event_state/v0_3_3/training_laboratory/   （小型索引：cards index、sentinel 摘要、报告、CURRENT_HANDOFF.md）
```
状态枚举：`PENDING RUNNING COMPLETE FAILED OOM_RETRYABLE RESOURCE_UNRESOLVED NAN INVALID_REQUEST HELD_NO_RELEASE HELD_MISMATCH HELD_CODE_COMMIT_MISMATCH SKIPPED_EXISTING STALE`。

- sentinel：每类 workload（`cpu_train_fixed_leaky`、`gpu_train_fixed_leaky`、`gpu_train_gated`、`cpu_t0`）先跑一个非空作业，记录 peak allocated/reserved、host RSS 峰值、I/O 字节、wall time、有效 batch。
- 并发上限 = min(pending, ⌊(free_vram − 4 GiB)/(1.25·peak_reserved)⌋, ⌊(MemAvailable − max(20% RAM, 20 GiB))/(1.25·rss)⌋, ⌊(cores − 2 − other_load)/threads⌋, lease.max_workers, ceiling)；disk < 10 GiB 或 iowait 持续高 → 0；稳定两个 heartbeat 周期且有余量时 +1。
- lease：读 `shared/resource_leases/supervisor_grant_agent_b.json`（缺失 → 保守默认 max_workers=2、gpu_ids=[0,1]，`lease_source=default_conservative`）；写 `shared/resource_leases/agent_b.json`（PID/PGID/heartbeat/当前占用）。
- OOM：保存 traceback / peak / config → `OOM_RETRYABLE`，同类并发 −1，退避阶梯 `chunk_seconds ↓ → grad accumulation（anchor 分块）→ checkpointing → 更小 chunk`，最多 3 次 → `RESOURCE_UNRESOLVED`。
- NaN：`nan_dump.json`（首个非有限张量、step、各组 grad）→ `NAN`；自动派生一个诊断 unit（LR×0.5、AMP 关）标 `diagnostic_rerun=true`，不计入 search。
- stale：RUNNING + PID 不存在 + heartbeat > 900 s → 保留旧 log/status，核对原子输出后只恢复未完成 unit。
- controller / worker：`setsid nohup`，stdin `/dev/null`，绝对路径 + 固定 Python，`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`；各自 heartbeat（60 s）、PID 文件、独立 log、原子 status、resume；`agent_b.status.json` ≤5 min 刷新；只用记录的 PID/PGID 管理本线作业。

## 10. 本轮实际运行清单（无 release）

1. 单元测试全绿（`tests/test_group_event_state_v033_training_lab_*.py`）。
2. toy smoke：完整 T0 → ASHA search（tiny budget）→ card，CPU。
3. synthetic smoke（E1146 真实 scaffold + v0.3.2 residual-positive 合成 counts，β=0.7）：T0 + tiny search + card，GPU；标 `synthetic`。
4. 人体 T0：E1146 `state_train` tiny-slice overfit + jacobian + membership + AMP + modulation；标 `DIAGNOSTIC`，不读 inner-val 之外任何分区（T0 本身不读 inner-val）。
5. sentinel：四类 workload 各一次，写 `sentinels/*.json` 与并发规划。
6. controller 干跑：ingest 一个 toy request → 状态文件 → worker 执行 → card；验证 heartbeat / resume / STALE / OOM 模拟路径。

## 11. 明确假设（供 supervisor / 用户否决）

- **A1** 基线 commit = `233f3ad1`；release 文件出现后若 `base_commit` 不同，需 rebase 并重跑 smoke。
- **A2** release 文件路径未定义：依次查 `/data/hfosp_group_event_state_v0_3_3/V0_3_3_EXECUTION_RELEASE.json`、`.../shared/`、`results/group_event_state/v0_3_3/`（主树与本 worktree）。
- **A3** supervisor lease 文件 schema 未定义：本设计定义 `supervisor_grant_agent_b.json`；缺失时保守默认。
- **A4** `event_balanced` sampling 定义为 §3 权重（非 event-anchor 训练集）；event-anchor 训练需 Agent C 提供 anchor set 与其 H。
- **A5** `write_scale` = 末层初始化乘子；`gated bias` = gated 更新网 gate 半部 bias 初值。
- **A6** synthetic recovery 在 Agent A D0–D4 到达前用 v0.3.2 residual-positive 合成（`source="v032_residual_positive_proxy"`），到达后换 D1/D2 且卡内记录来源。
- **A7** inner-validation = nested `dev_val`（70–80%），与 v0.3.2 model side 一致；Agent A 若重定义 split，`split_hash` 会不一致并被 HELD。
- **A8** 人体 T0 / sentinel 只用 E1146（v0.3.2 已触碰）。

## 12. 交付物

`src/topic5_group_event_state/v033_training_lab/`（request / data / models / objective / trainer / diagnostics / search / card / resources / queue / paths）、`scripts/run_group_event_state_v033_training_lab.py`、`tests/test_group_event_state_v033_training_lab_*.py`、training cards + 搜索轨迹 + sentinel 报告（`/data/.../agent_b/` + 小型索引）、`docs/archive/topic5/group_event_state_v0_3_3_training_laboratory_{plain,technical}_2026-09-02.md`、`results/group_event_state/v0_3_3/training_laboratory/CURRENT_HANDOFF.md`、`agent_b.status.json`。
