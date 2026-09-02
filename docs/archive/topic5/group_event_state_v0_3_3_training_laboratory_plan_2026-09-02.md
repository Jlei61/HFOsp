# Group-Event State v0.3.3 Training Laboratory — 实施计划

> **For agentic workers:** 本计划由本 session 内联执行（superpowers:executing-plans 风格，TDD）。每个任务 = 接口 + 失败测试 + 合同条款清单（CLAUDE.md §6 ritual）+ 提交。Steps use checkbox (`- [ ]`) syntax.

**Goal：** 交付 Workstream B 的持续训练服务：训练请求接口、target 无关的 T0–T6 训练合同、ASHA 多保真搜索、失败分类、训练卡、资源 sentinel 与受 lease 约束的 queue/controller/worker；在无 release 的条件下用 toy / synthetic / E1146 T0 全链路验证。

**Architecture：** `request(13 字段) → DataView(TRAIN/inner-val only) → Trainable(参考: FlexibleResidualStateModel + count-profile NB residual) → RecipeTrainer(T1 旋钮, 每组 first-active/update/clip 记录) → T0 diagnostics → ASHA search(T4) + T5 分类 → training_card(T6) → queue/controller/worker(resource-planned)`。复用 v0.3.2 的 `SubjectBundle`、`MarkedLeakyBank`、`nb_log_prob`、`block_circular_donor`、`block_bootstrap_mean_ci`、`make_synthetic_targets`、toy bundle。

**Tech Stack：** Python 3.11（`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`），torch 2.5.1+cu124，numpy，pandas/pyarrow（curves parquet），psutil，pytest。

**Spec：** `docs/archive/topic5/group_event_state_v0_3_3_training_laboratory_design_2026-09-02.md`（§2–§9 为合同；§11 为假设）。

## Global Constraints

- 只写：`src/topic5_group_event_state/v033_training_lab/`、`scripts/run_group_event_state_v033_training_lab.py`、`tests/test_group_event_state_v033_training_lab_*.py`、`docs/archive/topic5/group_event_state_v0_3_3_training_laboratory_*`、`results/group_event_state/v0_3_3/training_laboratory/`、`/data/hfosp_group_event_state_v0_3_3/agent_b/`、`/data/hfosp_group_event_state_v0_3_3/shared/{job_status/training_*.json,resource_leases/agent_b.json}`。
- 不改 v032_model / v032_eval 代码；不改 target / split / H / endpoint / H2b；不读 dev_test；不读 sealed。
- 无 release：人体只做 T0（`state_train`）与 sentinel，且只用 `epilepsiae_1146`。
- 运行：`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`；>10 min 作业 `setsid nohup`；原子写；resume；不 `pkill -f`。
- 所有 NB 似然 / reduction FP32 以上（`nb_log_prob` 内部 float64）；AMP 只可包 encoder。

## 文件结构

```
src/topic5_group_event_state/v033_training_lab/
  __init__.py       PACKAGE_VERSION
  paths.py          根目录、release 查找、原子写（复用 v02.registry）、hash、thread env
  request.py        JobRequest 校验（13 字段）、状态枚举、job key
  data.py           DataView（TRAIN/inner-val、bins、per-bin H、hash、blocks、weights、scaling）
  models.py         FlexibleResidualStateModel、GatedEventState（TBPTT）、CountProfileAdapter、init 方案
  objective.py      Trainable 协议、LossTerms/StateTerms、ResidualCountTrainable、TRAINABLE_REGISTRY
  trainer.py        RecipeConfig、build_optimizer/scheduler、train_recipe（记录/选择/resume/NaN）
  diagnostics.py    T0 九项 + state rank + shift null + random reservoir + synthetic recovery + blocked gain
  search.py         SearchSpace、sample_configs、ASHA、seed policy、batch loop、classify_failure、stop rules
  card.py           build_training_card、adequacy_rule、no-dev-test 断言
  resources.py      snapshot、Sentinel、plan_concurrency、lease read/write
  queue.py          ingest、units、status 写、controller loop、worker entry、OOM/NaN/stale
scripts/run_group_event_state_v033_training_lab.py   CLI（validate-request/t0/search/card/sentinel/controller/worker/status/smoke）
tests/test_group_event_state_v033_training_lab_{request,data,models,trainer,diagnostics,search,card,resources,queue}.py
```

---

### Task 1：paths + request 接口

**Files:** Create `paths.py`, `request.py`, `__init__.py`；Test `tests/test_group_event_state_v033_training_lab_request.py`。

**Interfaces（Produces）：**
```python
# paths.py
V033_ROOT = Path("/data/hfosp_group_event_state_v0_3_3"); AGENT_B_ROOT = V033_ROOT/"agent_b"; SHARED_ROOT = V033_ROOT/"shared"
RESULTS_INDEX = repo_root()/"results/group_event_state/v0_3_3/training_laboratory"
RELEASE_FILENAME = "V0_3_3_EXECUTION_RELEASE.json"
def release_status(candidates: Sequence[Path] | None = None) -> dict   # {"present": bool, "path": str|None, "payload": dict|None}
def set_single_thread_env() -> None
def current_commit() -> str
# request.py
REQUIRED_FIELDS: tuple[str, ...]  # 13
class JobStatus(str, Enum): PENDING RUNNING COMPLETE FAILED OOM_RETRYABLE RESOURCE_UNRESOLVED NAN INVALID_REQUEST HELD_NO_RELEASE HELD_MISMATCH HELD_CODE_COMMIT_MISMATCH SKIPPED_EXISTING STALE
@dataclass(frozen=True) class JobRequest: ...13 fields...; exploratory_approved: bool = False; raw: dict
def parse_request(payload: dict) -> tuple[JobRequest | None, dict]   # (request, verdict{status, reasons, missing_fields})
def validate_request(payload, *, registered_objectives, release_present, head_commit) -> dict   # verdict
def job_key(request, *, subject, seed, config_hash) -> str
def is_human_view(input_view: dict) -> bool   # kind in {"R0","R1"}
```
**合同条款：** [Q1] 13 字段任一缺失/空 → `INVALID_REQUEST` 且 `missing_fields` 列全；[Q2] objective 未注册 → `INVALID_REQUEST`；[Q3] gated 未批准 → `INVALID_REQUEST`；[Q4] 人体视图 + 无 release → `HELD_NO_RELEASE`；toy/synthetic 无 release → `PENDING`；[Q5] commit 不等 → `HELD_CODE_COMMIT_MISMATCH`；[Q6] job_key 对 seed/config/subject 敏感、对字典顺序不敏感；[Q7] `release_status` 在候选路径都缺失时 `present=False`。

- [ ] 写失败测试 Q1–Q7 → 跑确认失败 → 实现 → 通过 → 提交 `feat(topic5): v033 training lab request interface`

### Task 2：DataView

**Files:** Create `data.py`；Test `..._data.py`。

```python
DEFAULT_BINS = ((0.0, 300.0), (300.0, 900.0), (900.0, 1800.0))
@dataclass class DataView:
    subject; bins: tuple[tuple[float,float],...]; horizon: float   # = max(b_hi)
    event_times, event_segment, x_train_scaled (N,D) float32, train_event_mask
    t_anchor, anchor_segment, last_event_pos                       # 全部 anchors（状态轨迹需要）
    phase_index: dict[str, np.ndarray]                              # {"train": idx, "inner_val": idx}
    counts: (A,B) int64  (dev_test 行 = -1)
    log_mu_h: (A,B) float64; log_r_h: (B,) float64; h_source: str; missing_h_bins: list[int]
    split_hash: str; input_hash: str; scaling: str; taus_hint: tuple
    def n(self, phase) -> int
    def blocks(self, phase, block_seconds=None) -> np.ndarray        # block id per anchor of phase
    def sample_weights(self, phase, mode, lookback_seconds) -> np.ndarray
    def effective_independent_windows(self, phase) -> int
    def assert_no_dev_test(self) -> None
def build_view(bundle, *, bins=DEFAULT_BINS, scaling="zscore", allow_provisional_h=True) -> DataView
def bin_counts(event_times, t_anchor, bins) -> np.ndarray
def robust_scale_fit(x, train_mask) -> dict; def robust_scale_apply(x, stats) -> np.ndarray
def provisional_bin_history(bundle, bins) -> tuple[np.ndarray, np.ndarray]   # log_mu_h (A,B), log_r (B,)
```
**合同条款：** [D1] `bin_counts` 对 (0,300)/(0,1800) 与 `bundle.counts` 逐 anchor 相等；[D2] `phase_index` 只含 train/inner_val，`counts[dev_test]==-1`，`assert_no_dev_test` 在 loss 输入含 dev_test 索引时 raise；[D3] `split_hash` 只随 partition/subject/horizons 变，`input_hash` 随特征 fingerprint / scaling 变；[D4] robust 统计只由 TRAIN 事件决定（改 inner-val 行不改统计）；[D5] `event_balanced` 权重 TRAIN 均值 = 1、无事件 anchors 权重最小；[D6] 真实 bundle 的 H 只对 `(0,h)∈registry` bins 可用，其它 bin 进 `missing_h_bins`；toy 视图 provisional per-bin 有限；[D7] `blocks` 不跨 segment、块长 ≥ max(h,1800)。

- [ ] 测试 D1–D7 → 实现 → 提交 `feat(topic5): v033 training lab data view`

### Task 3：models + objective

**Files:** Create `models.py`, `objective.py`；Test `..._models.py`。

```python
@dataclass(frozen=True) class ArchConfig:
    state_family: str = "fixed_leaky"; taus_seconds=(300.,1800.,7200.); write_width: int = 4
    depth: int = 1; width: int = 32; activation: str = "gelu"; dropout: float = 0.0; hidden_norm: str = "none"
    init: str = "xavier"; write_scale: float = 1.0; alpha_init: float = 0.03; gate_bias_init: float = 0.0
    tbptt_seconds: float = 1800.0; rnn_hidden: int = 32; chunk_seconds: float = 3600.0
class FlexibleResidualStateModel(nn.Module):
    def __init__(self, arch: ArchConfig, in_dim: int, n_bins: int, log_r_init: np.ndarray)
    encoder / state / adapter(CountProfileAdapter: W (B,D) no bias, alpha scalar, log_r (B,))
    buffers phi_mean, train_mean_state, train_state_scale
    def writes(x, train_event_mask=None); trajectory(...); anchor_states(...); standardize_state(...); log_mu(log_mu_h (A,B), state, train_state=None) -> (A,B)
    def param_groups(self, lrs: dict[str,float], weight_decay: float) -> list[dict]   # 7 组固定名
    def modulation_jacobian(self) -> Tensor (B,D)
class GatedEventState(nn.Module):   # v0.3.2 repaired 更新规则 + tbptt_seconds chunk detach + gate bias init
class Trainable(Protocol) ...; @dataclass class LossTerms(nll (A,), weights (A,), modulation (A,B), state (A,D), idx); @dataclass class StateTerms
class ResidualCountTrainable: name="count_profile"; build/param_groups/loss_terms/state_terms/with_state_override/h_only_nll
TRAINABLE_REGISTRY = {"count_profile": ResidualCountTrainable}
```
**合同条款：** [M1] LayerNorm 只可能出现在 encoder（`hidden_norm="layernorm"`），state/adapter 模块树永无 LayerNorm；[M2] `alpha` 从构造起 `requires_grad=True`，无冻结 API；[M3] 每个 requires_grad 参数恰属一组，buffers 不在组；bias/gate/dispersion 组 wd=0；[M4] `write_scale` 只改末层初始化幅度（init 后 `‖W_last‖` 随 write_scale 线性）；`init="orthogonal"` 时方阵层正交；[M5] 状态维 = 3×write_width，`taus` 可换 {600,3600,10800}；[M6] gated：`tbptt_seconds` 下 `∂S_last/∂u_first` 跨 chunk = 0、同 chunk ≠ 0，且 chunk 边界不 reset（state 连续）；`gate_bias_init` 改变初始 gate 均值；[M7] `log_mu` 无自由截距：`log_mu − log_mu_h` 对 state 线性、W=0 时恒 0；[M8] `dropout>0` 在 eval 关闭。

- [ ] 测试 M1–M8 → 实现 → 提交 `feat(topic5): v033 flexible residual state model and count-profile trainable`

### Task 4：RecipeTrainer（T1）

**Files:** Create `trainer.py`；Test `..._trainer.py`（toy view）。

```python
@dataclass(frozen=True) class RecipeConfig:
    arch: ArchConfig; optimizer: str = "adamw"; schedule: str = "constant"; warmup_fraction: float = 0.0
    lr: dict[str, float]  # 7 组；weight_decay: float = 1e-4; grad_clip: float = 1.0
    dispersion: str = "frozen"  # frozen|low_lr ; sampling: str = "anchor_balanced"; scaling: str = "zscore"
    max_steps: int = 600; min_steps: int = 50; validate_every: int = 10; patience: int = 10; amp_encoder: bool = False
    def config_hash(self) -> str; def warmup_steps(self) -> int
def train_recipe(trainable, view, cfg, seed, *, device, out_dir, arm="learned", steps_budget=None, overwrite=False, interrupt_after_step=None) -> dict
# result 关键字段: status, selected_step, selected_in_warmup, selected_at_budget_edge, first_active_step{group}, all_groups_active_before_selection,
#   plateau{reached, since_step}, clipping_fraction, best_validation{inner_val_nll, inner_val_nll_h, gain}, history[...], curves_path, checkpoint, checkpoint_sha256, config_hash, split_hash, input_hash, nan{...}?
def load_trained(out_dir, trainable, view, device) -> nn.Module
```
**合同条款：** [T1] optimizer ∈ {adamw, adam, rmsprop} 各可跑；schedule constant/cosine/plateau 的 LR 轨迹符合定义（cosine 末端 LR→0；plateau 在 inner-val 停滞后降 LR）；[T2] warm-up：前 `warmup_steps` 步每组 LR = base·step/warmup；`selected_in_warmup` 正确；[T3] `alpha` 在 step 1 后即变化（无冻结）；[T4] `first_active_step` 每组 = 首个 grad≠0 且 Δθ≠0 的 step，随机 reservoir 臂 encoder 组为 None；[T5] `clipping_fraction` = 被裁步数/总步数；[T6] `dispersion="frozen"` 时 log_r 不变且不在 optimizer 组；[T7] `steps_budget` 覆盖 max_steps（ASHA rung）并可 resume 续到更大预算且参数与一次性训练一致；[T8] NaN → result.status="nan" + `nan_dump.json`（首个非有限张量名、step）且返回码语义由 worker 映射；[T9] curves parquet 行数 = validation 次数、列含每组 grad/update；[T10] 选择只看 inner_val，结果中无任何 dev_test 键（递归检查）。

- [ ] 测试 T1–T10 → 实现 → 提交 `feat(topic5): v033 recipe trainer with per-group activation tracking`

### Task 5：T0 诊断 + 卡片输入

**Files:** Create `diagnostics.py`；Test `..._diagnostics.py`。

```python
def tiny_slice_overfit(trainable, view, cfg, seed, *, device, n_slice=12, steps=300, threshold=0.5) -> dict
def oracle_head_fit(trainable, view, cfg, true_state (A,K), seed, *, device) -> dict
def state_write_jacobian(trainable, view, model, *, device) -> dict
def optimizer_membership(model, groups) -> dict
def amp_small_gradient_check(trainable, view, cfg, seed, *, device) -> dict
def state_output_modulation(trainable, view, model, *, device) -> dict
def state_variance_rank(trainable, view, model, *, device) -> dict
def shift_null(trainable, view, model, *, device, fraction=0.5) -> dict          # inner_val shifted − correct, block CI
def random_reservoir_delta(trainable, view, cfg, seed, *, device, out_dir) -> dict   # learned − random inner_val NLL
def synthetic_recovery(trainable, view, cfg, seed, *, device, out_dir, beta=0.7) -> dict   # v032 residual-positive proxy
def blocked_inner_val_gain(trainable, view, model, *, device) -> dict            # H − correct, block bootstrap CI
def run_t0(trainable, view, cfg, seed, *, device, out_dir) -> dict
```
**合同条款：** [G1] tiny overfit 在 toy 上 `gap_closed ≥ 0.5` pass，把 encoder LR 设 0 且 W=0 时 fail；[G2] jacobian：bank 数值 `∂S_a/∂u_j` 与 `exp(−Δ/τ)` 相对误差 < 1e-4；[G3] membership 对新增未分组参数 fail；[G4] AMP 检查在 CPU 上返回 `skipped`（无 cuda），在 cuda 上各组比值有限；[G5] shift null 只用有效 donor、与 v0.3.2 `block_circular_donor` 同规则；[G6] synthetic recovery 用 toy 视图 β=0.8 恢复 `ci_low>0 ∧ shifted−correct>0`；[G7] 所有 diag JSON 回显阈值常量；[G8] 无 dev_test。

- [ ] 测试 G1–G8 → 实现 → 提交 `feat(topic5): v033 T0 diagnostics and card inputs`

### Task 6：搜索（T1–T5）

**Files:** Create `search.py`；Test `..._search.py`。

```python
LR_LOG_RANGE = (1e-5, 3e-3)
class SearchSpace: @classmethod for_family(state_family, *, gated_approved=False) -> SearchSpace; def sample(self, rng) -> RecipeConfig; def describe() -> dict
@dataclass class SearchBudget: n_configs, max_steps, rung_steps: tuple[int,...], eta: int, seeds_low=1, seeds_mid=3, seeds_final=5, n_final=2, validate_every=10
def asha_promote(rows: list[dict], rung: int, eta) -> list[str]     # config ids 进入下一 rung；grace 规则
def run_search_batch(trainable, view, space, budget, *, base_seed, device, out_dir, batch_index, incumbent=None) -> dict
def run_search(trainable, view, space, budget, *, base_seed, device, out_dir, max_batches=4, tol=1e-3) -> dict   # 停止规则
def classify_failure(observations: dict) -> dict   # {"category", "rule", "next_action"}
```
**合同条款：** [S1] 采样 LR ∈ [1e-5,3e-3] log-uniform（seeded、可复现）、categorical 覆盖全部取值；[S2] ASHA 每 rung 保留 ⌈n/η⌉，`all_groups_active_step + validate_every > rung` 的 config 不被裁并标 `grace_deferred`；[S3] seeds：rung0 1 seed、rung≥1 3 seeds、final n_final×5 seeds，比较用 seed 中位；[S4] 连续两个 batch 无改善 → `stop_reason="no_improvement_two_batches"`；plateau → `stable_plateau`；[S5] classify_failure 覆盖 §7 表全部 7 行且互斥优先级固定；[S6] `search_trace.json` 记录每个 unit 的 config_hash/seed/rung/steps/inner_val；[S7] 无 dev_test。

- [ ] 测试 S1–S7（tiny budget，toy）→ 实现 → 提交 `feat(topic5): v033 ASHA search and failure classification`

### Task 7：训练卡（T6）

**Files:** Create `card.py`；Test `..._card.py`。

```python
ADEQUACY_RULE = "tiny_overfit.pass and synthetic_recovery.pass and blocked_inner_val_gain.ci_low > 0 and not selected_in_warmup and not selected_at_budget_edge and all_groups_active_before_selection"
def build_training_card(*, request, recipe_result, seed_results: list[dict], t0, diagnostics, search_summary=None) -> dict
def adequacy(card) -> tuple[str, dict]   # ("TRAINING-ADEQUATE"|"DIAGNOSTIC", reasons)
def assert_card_has_no_dev_test(card) -> None
```
**合同条款：** [C1] 六条件全真 → `TRAINING-ADEQUATE`，任一为假 → `DIAGNOSTIC` 且 `reasons` 指明；[C2] seed_dispersion 用 ≥2 seeds 计算 std/range，1 seed 时标 `insufficient_seeds`；[C3] 卡含 §8 全部字段名；[C4] 含 "dev_test" 键或 sealed 字样 → assert raise；[C5] `selection_metric_is_canonical=False` 默认。

- [ ] 测试 C1–C5 → 实现 → 提交 `feat(topic5): v033 training card`

### Task 8：资源

**Files:** Create `resources.py`；Test `..._resources.py`。

```python
def snapshot() -> dict   # gpus[{index,total,used,free}], mem{total,available}, load1, cores, iowait_pct, disk_free_gib(root), other_python_load
def run_sentinel(workload_class, fn, *, out_path) -> dict   # peak_allocated/reserved, rss_peak, io_read/write bytes, wall, effective_batch, threads
def plan_concurrency(snap, sentinel, lease, *, pending, threads=1, ceiling=None) -> dict   # {"slots", "limits": {...}, "binding": name}
def read_supervisor_lease(shared_root) -> dict   # 缺失 → default_conservative
def write_agent_lease(shared_root, payload) -> Path
```
**合同条款：** [R1] 并发 = 各限制最小值且 binding 名正确（构造 4 组 snap/sentinel 用例）；[R2] VRAM 预留 4 GiB、需求 ×1.25；RAM 预留 max(20%,20 GiB)、需求 ×1.25；CPU 预留 2 核 + other_load；[R3] disk<10 GiB 或 iowait>阈 → 0；[R4] lease 缺失 → `lease_source="default_conservative"` max_workers=2；[R5] sentinel 在 CPU 上 peak_allocated=0 但 rss/wall/io 有值。

- [ ] 测试 R1–R5 → 实现 → 提交 `feat(topic5): v033 resource sentinel and concurrency planner`

### Task 9：queue / controller / worker + CLI

**Files:** Create `queue.py`, `scripts/run_group_event_state_v033_training_lab.py`；Test `..._queue.py`。

```python
def ingest_requests(shared_root, *, registered, release_present, head_commit) -> list[dict]   # 写 job_status/training_<id>.json
def expand_units(request, view_meta, budget) -> list[Unit]        # t0 / rung train / random / synthetic / card（lazy：由 search driver 生成）
def write_unit_status(path, **fields); def read_status(path)
class Controller: def __init__(shared_root, agent_root, *, lease, poll_seconds=30); def step(self) -> dict; def run(self, stop_file)
def spawn_worker(unit_path, *, gpu, log_path) -> int   # setsid nohup, 返回 pid，写 pid/pgid
def worker_main(unit_path) -> int   # 退出码 0/1/3(OOM)/4(NaN)
def detect_stale(status, *, heartbeat_timeout=900) -> bool
def oom_backoff(unit_cfg, attempt) -> dict | None   # chunk↓ → accumulation → checkpointing → smaller chunk; attempt>3 → None
def write_agent_status(agent_root, results_index, **fields)
```
**合同条款：** [K1] ingest 对 INVALID/HELD/PENDING 各写正确 status，重复 ingest 幂等；[K2] worker 对同 job_key 已 COMPLETE → `SKIPPED_EXISTING`；[K3] OOM 模拟（unit 抛 CUDA OOM 文本）→ `OOM_RETRYABLE` + backoff 序列正确 + 第 4 次 `RESOURCE_UNRESOLVED`；[K4] NaN 模拟 → `NAN` + 派生 diagnostic unit（lr×0.5, amp off）；[K5] stale：RUNNING + PID 死 + heartbeat 旧 → STALE 且只恢复无 result 的 unit；[K6] controller.step 不超过 plan_concurrency slots 与 lease；无 release 时人体 unit 不 spawn；[K7] `agent_b.status.json` 字段齐全（commit, sealed=false, heartbeat, counts, resources, leases, next_batch_rationale）；[K8] 不使用 `pkill`（grep 源码）。

- [ ] 测试 K1–K8 → 实现 → 提交 `feat(topic5): v033 training queue controller and worker`

### Task 10：真实运行（无 release）

1. `pytest tests/test_group_event_state_v033_training_lab_*.py`；
2. toy smoke：`smoke --view toy` → T0 + search(tiny) + card，`/data/.../agent_b/smoke/toy/`；
3. synthetic smoke：`smoke --view synthetic --subject epilepsiae_1146 --beta 0.7 --device cuda:1`；
4. 人体 T0：`t0 --subject epilepsiae_1146 --device cuda:1`（DIAGNOSTIC）；
5. sentinel ×4 类；`plan_concurrency` 输出；
6. controller 干跑：写一个 toy request → `controller --once` → worker → card；STALE/OOM 路径用 `--simulate`。

### Task 11：报告、索引、状态、提交

- `docs/archive/topic5/group_event_state_v0_3_3_training_laboratory_{plain,technical}_2026-09-02.md`；`docs/archive/topic5/INDEX.md` 加行；
- `results/group_event_state/v0_3_3/training_laboratory/{README.md,CURRENT_HANDOFF.md,cards_index.json,sentinel_summary.json}`；
- `agent_b.status.json`；memory 更新；提交（不推送 main）。
