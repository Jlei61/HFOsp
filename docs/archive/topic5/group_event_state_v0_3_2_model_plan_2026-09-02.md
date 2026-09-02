# Group-Event State v0.3.2 模型侧 — 实施计划

> **For agentic workers:** 本计划由本 session 内联执行（executing-plans 风格，TDD）。每个任务 = 接口 + 失败测试 + 合同条款清单 + 提交。

**Goal：** 交付 v0.3.2 模型侧工作包：12 维 marked leaky bank、repaired-RNN 对照、safe residual adapter、30 min NB residual trainer、synthetic 正/空对照、梯度/replay/BPTT 诊断、frozen trajectory exporter、任务队列与报告。

**Architecture：** `bundle(data) → features(TRAIN-only 标准化) → encoder φ → 状态轨迹（闭式 leaky bank 或 顺序 RNN）→ anchor 状态 → residual adapter on log μ_H → NB loss`；全批次 AdamW；评估臂 = H / correct / shifted / mean / random；synthetic 用真实 token 生成 y。

**Tech Stack：** Python 3.11（cuda_env），torch 2.5.1+cu124，numpy，pytest。

**Spec：** `docs/archive/topic5/group_event_state_v0_3_2_model_design_2026-09-02.md`

## Global Constraints

- 只写：`src/topic5_group_event_state/v032_model/`、`scripts/run_group_event_state_v032_model*.py`、`scripts/audit_group_event_state_v032_model*.py`、`config/topic5_group_event_state_v032_model*.yaml`、`tests/test_group_event_state_v032_model*.py`、`docs/archive/topic5/group_event_state_v0_3_2_model_*`、`/data/hfosp_group_event_state_v0_3_2/model/`、`/data/hfosp_group_event_state_v0_3_2/shared/frozen_state_registry.json`。
- 不改 v032_eval、INDEX.md、FIGURE_INDEX.md、paper-ready 图、sealed 分区、H2b/H3/background 主线；不动 v0.3.1 产物。
- τ = (300, 1800, 7200) s × 4 维 = 12；无 state-to-state 混合；无 LayerNorm；TRAIN-only 固定标准化；Δt 只用于衰减。
- `log μ_{H+S} = log μ_H + α wᵀS`；α 初值 0.03 且前 50 步冻结；w 默认随机初始化；bias/gate(α)/dispersion 不做 weight decay；标准化统计为 buffer。
- state 衰减、NB 似然、reduction 均 FP32（轨迹内部 float64）；AMP 只可包 encoder。
- 首轮只训 1800 s NB count residual；300 s 仅诊断；7200 s 仅 eligibility 判定 eligible 的患者作 secondary。
- 运行：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，`setsid nohup`，原子写，可续跑，hash 一致跳过，STATUS.json ≤ 10 min。

## 文件结构

```
src/topic5_group_event_state/v032_model/
  __init__.py        导出
  paths.py           根目录常量、原子写（复用 v02.registry）、hash
  config.py          ModelConfig（frozen dataclass）+ yaml 加载 + config_hash
  features.py        event token 特征（纯 numpy 数组接口）+ TrainStandardizer + 缓存
  history_baseline.py HistoryBaseline、Agent2 registry 读取器、provisional H、eligibility 读取
  data.py            SubjectBundle 装配（timeline+partition+features+anchors+counts+H）
  state.py           leaky_bank_trajectory / anchor_states / MarkedLeakyBank / RepairedRecurrentState
  encoder.py         EventProjection（MLP）
  readout.py         nb_log_prob / fit_nb_log_dispersion / ResidualCountAdapter
  model.py           ResidualStateModel（组装、param_groups、arm 状态构造）
  shift.py           block_circular_donor
  trainer.py         全批次 trainer（checkpoint/progress/resume/选择/审计统计）
  evaluate.py        evaluate_arms + block bootstrap
  synthetic.py       synthetic targets + assay + 判定
  diagnostics.py     16 项诊断
  registry.py        frozen state 导出 + shared registry 原子合并
  queue.py           manifest 构建 / claim / finish / STATUS
  summary.py         triage 决策 + 三患者汇总
scripts/run_group_event_state_v032_model.py         CLI
scripts/run_group_event_state_v032_model_worker.py  worker
scripts/audit_group_event_state_v032_model.py       诊断 CLI
config/topic5_group_event_state_v032_model.yaml
tests/test_group_event_state_v032_model_{state,readout,features,data,evaluate,synthetic,trainer,registry,queue}.py
```

---

### Task 1：paths/config + 状态骨干（leaky bank 精确轨迹）

**Files:** Create `paths.py`, `config.py`, `state.py`, `__init__.py`；Test `tests/test_group_event_state_v032_model_state.py`。

**Interfaces（Produces）：**
```python
# config.py
@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "leaky_bank"          # or "repaired_rnn"
    taus_seconds: tuple[float, ...] = (300.0, 1800.0, 7200.0)
    channels_per_tau: int = 4
    phi_dim: int = 4
    encoder_hidden: int = 32
    rnn_event_dim: int = 16
    alpha_init: float = 0.03
    alpha_freeze_steps: int = 50
    lr_encoder: float = 1e-3
    lr_state: float = 1e-3
    lr_adapter: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_steps: int = 600
    min_steps: int = 100
    validate_every: int = 10
    patience: int = 10
    horizon_seconds: float = 1800.0
    diagnostic_horizons_seconds: tuple[float, ...] = (300.0,)
    secondary_horizon_seconds: float = 7200.0
    amp_encoder: bool = False
    chunk_seconds: float = 3600.0
    detach_chunks: bool = False
    shift_fractions: tuple[float, ...] = (0.5, 0.25, 0.75)
    bootstrap_resamples: int = 1000
    @property
    def state_dim(self) -> int
    def config_hash(self) -> str
def load_config(path: Path | None = None, **overrides) -> ModelConfig

# state.py
def leaky_bank_trajectory(u: Tensor, times: Tensor, segment_ids: Tensor, taus: Tensor, *,
                          chunk_seconds: float, detach_chunks: bool = False) -> tuple[Tensor, Tensor]
    # u (N,K) float32; times (N,) float64 sorted within segment; segment_ids (N,) long non-decreasing
    # returns state_pre, state_post each (N, T*K) float32, tau-major layout
def anchor_states(state_post: Tensor, event_times: Tensor, t_anchor: Tensor,
                  last_event_pos: Tensor, taus_full: Tensor) -> Tensor   # (A, D)
class MarkedLeakyBank(nn.Module):
    def __init__(self, taus_seconds, channels_per_tau, chunk_seconds, detach_chunks=False)
    taus_full: Tensor  # buffer (D,)
    state_dim: int
    def forward(self, u, times, segment_ids) -> tuple[Tensor, Tensor]
    def anchor(self, state_post, event_times, t_anchor, last_event_pos) -> Tensor
class RepairedRecurrentState(nn.Module):
    def __init__(self, taus_seconds, channels_per_tau, event_dim, hidden=32)
    def forward(self, e, times, segment_ids) -> tuple[Tensor, Tensor]
    def anchor(...)  # same as bank (autonomous decay only)
```

**合同条款清单（§6 ritual）：**
- [C1] 精确性：`state_post[e] == Σ_{j≤e, same seg} exp(-(t_e-t_j)/τ) u_j`，与 brute force 在 1e-5 内一致（含跨 chunk）。
- [C2] 无混合：∂state_post[:, i]/∂u[:, k] 仅在 i≡k (mod K) 非零。
- [C3] Δt 只进入衰减：改变 `times` 但保持 u 不变时，state 变化完全由 exp(−Δ/τ) 解释。
- [C4] segment 起点 0：每个 segment 第一个事件 state_pre = 0。
- [C5] 无截断：默认 `detach_chunks=False` 时 ∂state_post[last]/∂u[first] = exp(−(t_last−t_first)/τ)（跨 chunk）；`detach_chunks=True` 时为 0。
- [C6] anchor 衰减：`anchor_states` = post[last]·exp(−(t_a−t_last)/τ)，无 last 时 0。
- [C7] RNN 无 LayerNorm（模块树里没有 LayerNorm），τ 衰减相同，写入依赖 state（有混合，作为对照）。

- [ ] 写失败测试（C1–C7）→ 运行确认失败 → 实现 → 通过 → 提交 `feat(topic5): v032 leaky bank state core`

### Task 2：readout（NB + adapter）

**Files:** Create `readout.py`；Test `tests/test_group_event_state_v032_model_readout.py`。

```python
def nb_log_prob(y: Tensor, mu: Tensor, log_r: Tensor) -> Tensor           # FP32 elementwise
def moment_log_dispersion(y: np.ndarray, mu: np.ndarray) -> float
def fit_nb_log_dispersion(y: np.ndarray, mu: np.ndarray, *, lo=math.log(0.05), hi=math.log(1e5)) -> float
class ResidualCountAdapter(nn.Module):
    def __init__(self, state_dim: int, alpha_init: float, log_r_init: float)
    w: nn.Linear(state_dim, 1, bias=False); alpha: Parameter(()); log_r: Parameter(())
    def modulation(self, S) -> Tensor        # alpha * w(S).squeeze(-1)
    def forward(self, log_mu_h, S) -> Tensor # log_mu_h + modulation
    def set_alpha_trainable(self, flag: bool) -> None
```
条款：[R1] NB 与 scipy.stats.nbinom 逐点一致；[R2] MLE 恢复模拟 r；[R3] 无自由截距（forward 与 log_mu_h 差 = modulation）；[R4] α 初值 = alpha_init 且 `set_alpha_trainable(False)` 使 grad 为 None；[R5] 全 FP32（输入 bf16 也升为 fp32）。

### Task 3：features + standardizer

**Files:** Create `features.py`；Test `tests/test_group_event_state_v032_model_features.py`。

```python
FEATURE_VERSION = "v032_event_token_1"
def event_token_features(*, participation, relative_delay, tied_group_id, band_features,
                         cross_band_lag, contact_valid, coords, core_seconds, has_waveform,
                         band_available) -> tuple[np.ndarray, tuple[str, ...]]
class TrainStandardizer:
    @classmethod
    def fit(cls, x: np.ndarray, train_mask: np.ndarray) -> "TrainStandardizer"
    def transform(self, x) -> np.ndarray
    mean, scale, zero_variance: np.ndarray; def to_dict / from_dict
def build_subject_features(subject, *, dataset_root, out_root, overwrite=False) -> Path  # npz + fingerprint
```
条款：[F1] 列名不含 dt；[F2] 标准化统计只由 train_mask 行决定；[F3] TRAIN 零方差列 transform 后恒 0；[F4] NaN → 0；[F5] participation/leader/extent/tied-group/delay/dispersion/multiband/confidence/coverage 九类都有列；[F6] 单触点事件离散度 0 且 flag=1。

### Task 4：history baseline + data bundle

**Files:** Create `history_baseline.py`, `data.py`；Test `tests/test_group_event_state_v032_model_data.py`。

```python
@dataclass
class HistoryBaseline:
    log_mu: dict[int, np.ndarray]; nb_log_dispersion: dict[int, float | None]; source: str; meta: dict
def load_agent2_history_baseline(registry_path: Path, subject: str, t_anchor: np.ndarray,
                                 horizons: Sequence[float]) -> HistoryBaseline | None
def fit_provisional_history_baseline(timeline, partition, horizons) -> HistoryBaseline
def load_endpoint_eligibility(path: Path, subject: str) -> dict | None
@dataclass
class SubjectBundle:  # 见 design §1/§2
    subject; n_events; event_times; event_segment; event_phase; x_std; feature_names
    t_anchor; anchor_segment; anchor_session; anchor_phase; last_event_pos; eligible; counts; horizons
    history: HistoryBaseline; partition; timeline; fingerprint: dict
    def anchor_mask(self, phase: str, horizon: float) -> np.ndarray
    def train_event_mask(self) -> np.ndarray
def load_subject_bundle(subject, *, features_root, shared_root, horizons, allow_provisional_h=True) -> SubjectBundle
```
条款：[D1] `anchor_mask(phase,h)` = phase 相符 ∧ eligible ∧ 窗口不越 phase 上界（沿用 v0.3 规则）；[D2] provisional H 只在 state_train 拟合、dev_val 选 ridge，且 `source=="provisional_local"`；[D3] Agent2 registry 缺失返回 None + reason，不静默替换；[D4] bundle 的 `train_event_mask` 只覆盖 state_train 事件。

### Task 5：encoder + model 组装 + param groups + shift

**Files:** Create `encoder.py`, `model.py`, `shift.py`；Test `tests/test_group_event_state_v032_model_model.py`, `..._evaluate.py`（shift 部分）。

```python
class EventProjection(nn.Module): def __init__(self, in_dim, hidden, out_dim); forward(x)->Tensor
class ResidualStateModel(nn.Module):
    def __init__(self, cfg, in_dim, log_r_init)
    encoder; state; adapter; buffers: phi_mean (K,), train_mean_state (D,)
    def writes(self, x_std) -> Tensor                # bank: tanh(phi - phi_mean); rnn: embedding
    def trajectory(self, x_std, times, segment_ids) -> tuple[Tensor, Tensor]
    def anchor_states(self, state_post, event_times, t_anchor, last_event_pos) -> Tensor
    @torch.no_grad() def refresh_train_mean(self, x_std, train_event_mask) -> None
    def param_groups(self, cfg) -> list[dict]        # 每组带 name / weight_decay / lr
def block_circular_donor(t_anchor, segment, indices, horizon, fraction) -> np.ndarray  # local donor idx or -1
```
条款：[M1] 每个 requires_grad 参数恰属一组，buffers 不在任何组；[M2] no-decay 组 = encoder biases、state biases、alpha、log_r；[M3] leaky bank 的 state 组为空；[M4] `refresh_train_mean` 只用 train mask 且不产生梯度；[M5] donor 与本 anchor 同 segment、|Δt| ≥ horizon，segment < 3 anchors → 全 −1。

### Task 6：trainer（全批次、α 冻结、选择、resume）

**Files:** Create `trainer.py`；Test `tests/test_group_event_state_v032_model_trainer.py`（toy bundle）。

```python
def anchor_nll(model, bundle, phase, horizon, *, device, state_override=None) -> dict[str, Tensor]
def train_residual_model(bundle, cfg, seed, *, device, out_dir, arm="learned", overwrite=False) -> dict
```
条款：[T1] 前 `alpha_freeze_steps` 步 α 不变；[T2] 选择只看 dev_val（1800 s）；[T3] 记录 `selected_step / selected_first_validation / selected_at_budget_edge`；[T4] checkpoint 含 phi_mean、train_mean_state、standardizer、config_hash、feature fingerprint；[T5] resume：存在 `last.pt` 则从其 step 继续且结果一致；[T6] 记录裁剪前后 grad norm 与分组 norm；[T7] random_reservoir 臂 encoder 冻结（grad None）；[T8] 全部 anchor 张量 FP32。

### Task 7：evaluate arms + bootstrap

**Files:** Create `evaluate.py`；Test 扩展 `..._evaluate.py`。
```python
def block_bootstrap_mean_ci(values, groups, *, block_len, n_boot, seed) -> dict
def evaluate_arms(model, bundle, cfg, *, device, random_model=None, phases=("dev_val","dev_test"), horizons=None) -> dict
```
条款：[E1] 五臂在同一 anchor 集合；[E2] mean 臂 = 常数偏移（modulation 方差 0）；[E3] shifted 臂只用有效 donor 的配对；[E4] 报告 effective independent windows；[E5] H 臂 dispersion 来自 TRAIN MLE 或 registry。

### Task 8：synthetic assays

**Files:** Create `synthetic.py`；Test `..._synthetic.py`。
```python
def make_synthetic_targets(bundle, *, horizon, beta, dispersion_r, generator_seed, noise_seed) -> SyntheticTargets
def run_synthetic_assay(subject, kind, replicate, cfg, *, device, out_root, ...) -> dict
def judge_synthetic(results: list[dict], kind: str) -> dict
```
条款：[S1] positive 的 z 与 H 的线性重建 R² < 0.5（隐藏分量不在 H 里）；[S2] null 的 β=0；[S3] 同 seed 可复现；[S4] 判定阈值写在函数常量里并回显到 JSON。

### Task 9：diagnostics

**Files:** Create `diagnostics.py`, `scripts/audit_group_event_state_v032_model.py`；Test `..._diagnostics.py`（关键函数）。
16 项 → 三个 JSON（design §7）。条款：[G1] 参数覆盖审计对模型任何新增参数都会失败；[G2] ∂S_a/∂u_j 数值 = exp(−Δ/τ)；[G3] replay 与训练轨迹 allclose(1e-6)；[G4] Jacobian = α‖w‖。

### Task 10：registry 导出 + queue + CLI + summary

**Files:** Create `registry.py`, `queue.py`, `summary.py`, `scripts/run_group_event_state_v032_model.py`, `scripts/run_group_event_state_v032_model_worker.py`, `config/topic5_group_event_state_v032_model.yaml`；Test `..._registry.py`, `..._queue.py`。
条款：[X1] registry 条目字段齐全（design §8）；[X2] 原子写 + 按 (subject, seed, architecture) 合并；[X3] 完成且 hash 一致的任务跳过；[X4] STATUS.json 含 updated_epoch / counts / running。

### Task 11：真实运行

features → synthetic → triage → lock → 3×3 → diagnostics → export → summary → 报告 → 提交推送。
