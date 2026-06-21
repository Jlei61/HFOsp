# M3 local-W propagation operator + W-coupled slow permissivity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> **Spec:** `docs/superpowers/specs/2026-06-21-sef-hfo-m3-local-w-propagation-operator-design.md` — read it for the *why*; this plan is the *how*.
> **Supersedes** `docs/superpowers/plans/2026-06-19-sef-hfo-m3-hub-gated-critical-scaffold-plan.md` (hub 版降级为退路；其 worktree 已跑基础设施大量复用，见 Global Constraints「复用」)。

**Goal:** 在 cm-scale 放电网络上，用 small-kick 测出一个可观测的局部传播算子 `W_kicked`，把间期 HFO 群体事件建模成该传播场在低易感度（亚临界）下的自发、**时间自限**（回静息、不持续招募）传播 excursion；再用一个经 `h(W)` 耦合的慢易感度旋钮 `μ`，把同一条场从亚临界（间期）推到超临界（发作样持续招募）——先建 W 测量 + h-耦合基础设施 + 跑相图/basin pilot（不需决定的探索），昂贵的动态 m / 网格 / 消融留到 pilot + 用户讨论之后。

**Architecture:** local-W（无走廊/端点/hub）：(1) `W_kicked` 测量 harness（循环 `simulate_kick(kick_center=bin)` 小幅踢、一步招募窗、多 seed、coarse-bin → 去噪 → `W_0^eff` bundle（`W_resp`/`W_step`/`W_shape`）、主轴(`W_shape`)、`Λ₀=ρ(W_step)`、`h^post(W_resp)`）；(2) μ 经 `h(W)` 耦合（`V_th_eff = V_th0 − Δθ·μ·h`，骑现有 `V_th_per_neuron`，零引擎改动、μ=0 逐比特一致）；(3) `Λ₀×μ` + `recovery×μ` 相图 + `K_min(μ)` basin（机制核心证据）；(4) 复用虚拟 SEEG readout + Task-0 subject-level 等价验收。**复用现有 worktree `.worktrees/topic4-m3` 的 degnorm/branching_ratio/acceptance/hub_diag/runner 接线；hub 专属件（长程边/crossing/hub-θ）留着但不进 local-W 主线。**

**Tech Stack:** Python, NumPy, SciPy sparse/eigs；`src/snn_engine/` LIF substrate（worktree base = M1 recovery + M2 gate + hub infra，全默认关）；现成 readout/geometry 脚本；pytest。

## Global Constraints

- **Worktree（P0，复用）：所有 M3-final 代码在现有 `.worktrees/topic4-m3`（branch `topic4-snn-m3-hub`）之上续建。** 不新开 worktree（用户 2026-06-21 决定）。hub 专属件（`hub_gain` 长程边 / `crossing_path_gain` / `hub_theta_delta`）保留默认关、不进 local-W 主线。只 re-bless 本 worktree 自己的 `engine_versions.json`（本计划主线**不改引擎**，见下）。
- **复用件（worktree 已跑、~45 测试绿）：** `src/topic4_degnorm.py::ee_degree`（→ `W_struct` 路径的**结构先验** h，**非** primary）、`src/topic4_hub_criticality.py::{recruitment_operator, branching_ratio}`（→ `W_struct` + `ρ` 复用）、`src/topic4_m3_acceptance.py::{subject_tolerance_band, layer2_equivalence}`（→ Layer-2 verbatim 复用）、`src/topic4_hub_diag.py::hub_global_recruitment`（→ reach/招募读出）、runner `--degnorm-alpha` 走的 `V_th_per_neuron` 预变换路径（→ h-耦合复用同一条路径）。
- **零引擎改动主线（P0）：** `W_kicked` 用现成 `simulate_kick(kick_center=, r_kick=, KICK_BOOST=, V_th_per_neuron=)` **读取**响应、不改引擎；h-耦合是 runner 预变换（骑 `V_th_per_neuron`）。**μ=0 / h-coupling off 时引擎 spike 与 base 逐比特一致**（base SHA 在 Task 0 锚定）。动态 `m_i(t)`（Phase 2）才需要新引擎件、单独 re-bless。
- **PILOT-FIRST（P0 硬停）：** 不跑动态 m / 网格 / 消融，直到 baseline（Task 5）+ W 预测性（Task 6）+ `Λ₀×μ`（Task 7）+ basin（Task 9）pilot 全过 + 用户讨论。
- **存在性早报、不硬停（用户 2026-06-21 决定）：** Task 5/6 的"亚临界可读自停窗是否存在"**早报作语境**，但**不**作 go/no-go 硬停；继续跑 Task 7/8/9，最终由相图 + basin + 对照**联合判断**。联合图景为负 → 退 hub（spec §10）+ 老实记录（spec §9）。
- **预注册（出任何 SNN 动力学结果之前，Task 4 落盘）：** h 三方案（post/out/hybrid）比较 + 主口径 `h^post`、`A_p` primary、响应窗 `[Δ1,Δ2]` 一步约定、Layer-2 subject-level 容差带、R0–R4 分类阈值。结果出来不许改。
- **Λ₀ 只在 μ=0 测一次（P0，避免 x 轴自漂）：** 每个 `J_EE_scale` 旋钮值下，μ=0 测一次 `W_kicked` → bundle `W_0^eff`（`W_resp`/`W_step`/`W_shape`）/`h(W_resp)`/`Λ₀=ρ(W_step)`；各 μ 用**同一个** baseline bundle 算 `Λ_eff=ρ[D_μ(h)·W_step]`（线代预测）+ 跑 SNN（真判据）。**别在每个 μ 重新踢。**
- **Claim discipline（spec §3 C1–C6）：** 临界 = `ρ(W_step)≈1`（**非**静息 max Re λ；`W_0^eff` 是 μ=0 bundle 名、ρ 只吃 `W_step`）；自限 = **时间自限 / 不持续招募**（**非**空间封闭、**不再要求**沿轴短/L 不变/不贴边）；μ **必须经 `h(W)` 耦合**（均匀 μ / 打乱 h 是对照）；`W_kicked` 按**一步招募算子**构造（C6）；ictal = synthetic feasibility bridge（不解释临床发作）；Layer-2 = subject-level 等价（**非** event-level "不被拒=PASS"）。
- **真实 readout：** 所有 synthetic 结论必须经真实 masked lagPat propagation pipeline，禁止只看 raster。

---

### Task 0: worktree base 复用校验 + 复用件清单 + base SHA 锚定

**Files:** worktree `.worktrees/topic4-m3`；无代码改动。

- [ ] **Step 1: 进 worktree、merge 本计划 + spec（current branch 带新 spec/plan）**
```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3
git merge --no-edit topic4-event-extent-audit   # 带 2026-06-21 local-W spec/plan + superseded 横幅；预期无冲突（不同文件）
```
- [ ] **Step 2: 校验复用件都在 + 可 import**
```bash
python3 -c "from src.topic4_degnorm import ee_degree; \
from src.topic4_hub_criticality import recruitment_operator, branching_ratio; \
from src.topic4_m3_acceptance import subject_tolerance_band, layer2_equivalence; \
from src.topic4_hub_diag import hub_global_recruitment; print('reuse helpers OK')"
grep -q "degnorm_alpha" scripts/run_sef_hfo_snn_cm_spontaneous_readout.py && echo "runner V_th path OK"
grep -q "ee_std_u" scripts/run_sef_hfo_snn_cm_spontaneous_readout.py && echo "M1 recovery OK"
```
- [ ] **Step 3: 锚定 base spike SHA**（μ=0 / h-coupling off bit-parity 锚 — 复用 hub plan Task 0 同一锚）
```bash
python3 - <<'PY'
import sys, hashlib, numpy as np; sys.path.insert(0, "src/snn_engine")
from params import Params; from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot; from kick_probe import simulate_kick
p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
rng = np.random.default_rng(1); pos, labels, NE, NI = place_neurons(p, rng)
net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
net["rng"] = np.random.default_rng(1)
res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE+NI, 18.0))
print("M3_BASE_SHA", hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16])
PY
```
记录输出为 `M3_BASE_SHA`（写进所有 parity 测试 + 本 plan 此处）。应与 hub plan 的 base SHA 一致（同一基底）。
- [ ] **Step 4: 跑现有引擎 + 复用件 smoke**：`python3 -m pytest tests/ -k "snn or degnorm or criticality or acceptance or hub_diag or recovery" -q` → PASS（base 无回归）。

---

### Task 1: `W` 三对象测量算子模块（`src/topic4_propagation_operator.py`，纯/读引擎）

> **审阅 §1 修正（2026-06-21）**：W 拆成三对象——`W_resp`（未归一→h）/ `W_step`（按源活动归一→Λ₀）/ `W_shape`（行/列归一→主轴+顺序）。
> **不可** 用一个 row-normalized 矩阵同时算 `Λ₀` 和 `h`（row-norm 让 ρ≈1 恒成立、h 抹平到 1）。

**Files:**
- Create: `src/topic4_propagation_operator.py`
- Test: `tests/test_topic4_propagation_operator.py`

**Interfaces:**
- Consumes: `simulate_kick`（`src/snn_engine/kick_probe.py`，已存在）；`branching_ratio`（`src/topic4_hub_criticality.py`，复用算 ρ）。
- Produces:
  - `spatial_bins(posE, n_bins_per_axis) -> dict(bin_of_cell, bin_centers)` — E cell 按 2D 位置 coarse-bin 成 `n_bins=n_bins_per_axis²` 格。
  - `build_w_resp(p, net, NE, NI, bins, V_th0, *, kick_boost, r_kick, t_kick, win_ms, seeds) -> dict(W_resp, src_mass)` — 对每个 bin `q`，`simulate_kick(kick_center=bin_centers[q], KICK_BOOST=kick_boost, r_kick=r_kick, t_kick=t_kick, V_th_per_neuron=V_th0)` 与 sham（KICK_BOOST=0）各跑 `seeds` seed；`A_p` = bin `p` 内 E cell 在 `[t_kick+win_ms[0], t_kick+win_ms[1]]` 的 spike 总数。`W_resp[p,q] = clip(mean_seed(A_p|kick) − mean_seed(A_p|sham), 0, inf)`，**对角置 0、不 row-normalize**；`src_mass[q] = W_resp[q,q]`（去对角前的源 q 自身 kick 诱发活动，给 W_step 用）；同时返回 `injected_mass[q]`（注入 q 的预期 spike mass = kick 直接驱动量，作 W_step sensitivity 分母）。`win_ms` = **一个传播代**（C6，由 Task 1.5 标定）。
  - `make_step_operator(W_resp, src_mass, *, eps=1e-6, src_mass_floor=None, injected_mass=None) -> np.ndarray` — `W_step[p,q] = W_resp[p,q] / (normalizer[q] + eps)`，`normalizer = injected_mass if injected_mass is not None else src_mass`（**按源活动列归一**；**不** 行/列和恒 1）。**防爆保护（审阅 P1）**：`src_mass_floor` 非 None 时，`src_mass[q] < src_mass_floor` 的源 bin **整列置 0**（排除低响应源、防小分母放大假增益）；`injected_mass` 提供时走"除以注入期望 spike mass"的 sensitivity 口径。主口径（`src_mass`）与 sensitivity 口径的 `ρ` 相对排序须一致。
  - `make_shape_operator(W_resp) -> np.ndarray` — `W_shape`：去对角 + 行归一化（每行除行和，0 行保 0）。**仅** 给主轴 / 顺序，**不** 算 Λ₀。
  - `h_field(W_resp, scheme) -> np.ndarray(n_bins)` — 从**未归一 `W_resp`** 算：`post`=`norm(行和)`、`out`=`norm(列和)`、`hybrid`=`½(post+out)`；`norm`=除以中位数。
  - `spectral_radius(W_step) -> float` — `branching_ratio(sparse(W_step), idx=all)` 复用（Λ₀，可 >/<1）。
  - `principal_axis(W_shape, bin_centers) -> np.ndarray(2)` — 响应加权位移的主方向单位向量。
  - `ordering_predictivity(W_shape, bin_centers, event_bin_order, *, rates) -> dict(rho_W, rho_dist, rho_rate)` — 给自发事件 bin 激活先后，比较 W_shape 预测顺序 / 纯距离 / 纯放电率 与实测的 Spearman ρ。

- [ ] **Step 1: 失败测试**（合成校验，不跑真 SNN）
```python
import numpy as np
from src.topic4_propagation_operator import (make_step_operator, make_shape_operator,
    spectral_radius, h_field, principal_axis, ordering_predictivity)

def test_h_from_unnormalized_resp_not_flat():
    # W_resp 未归一 -> h_post 反映真实招募强度差异 (不被 row-norm 抹平到 1)
    W = np.array([[0,1.,1.],[0,0,0.],[2.,0,0]])     # 行和 = [2,0,2]
    h = h_field(W, "post")
    assert h[0] != h[1]                              # 不抹平
    assert np.allclose(h, np.array([2,0,2]) / np.median([2,0,2]))

def test_step_operator_scales_with_gain_not_rownorm():
    # W_step 按源活动归一; 整体放大 W_resp -> ρ 增大 (若是 row-norm 则 ρ 不变 = bug)
    W = np.array([[0,2.,0],[0,0,2.],[0,0,0]]); sm = np.array([1.,1,1])
    rho1 = spectral_radius(make_step_operator(W, sm))
    rho2 = spectral_radius(make_step_operator(2*W, sm))
    assert rho2 > rho1 + 1e-6                        # ρ 跟增益走 (非 row-norm 恒定)

def test_shape_operator_rownormalized():
    W = np.array([[0,1.,3.],[0,0,0.],[0,0,0]])
    S = make_shape_operator(W)
    rs = S.sum(1); assert np.allclose(rs[rs > 0], 1.0) and np.allclose(np.diag(S), 0)

def test_ordering_predictivity_W_beats_distance():
    centers = np.array([[0,0],[1,5],[2,0.],[3,5]])
    Wshape = np.array([[0,1.,0,0],[0,0,1.,0],[0,0,0,1.],[0,0,0,0]])
    out = ordering_predictivity(Wshape, centers, [0,1,2,3], rates=np.ones(4))
    assert out["rho_W"] >= out["rho_dist"]

def test_step_operator_excludes_low_src_mass():
    # 防爆 (审阅 P1): 低 src_mass 源 bin -> 整列置 0, 不被小分母放大成假高增益
    W = np.array([[0,2.,0],[0,0,2.],[0,0,0]])
    sm = np.array([1., 1e-9, 1.])                    # bin 1 = 不可靠源
    S = make_step_operator(W, sm, src_mass_floor=1e-3)
    assert np.allclose(S[:, 1], 0.0)                 # 排除, 而非 2/1e-9 爆掉

def test_step_operator_injected_mass_sensitivity():
    # sensitivity 口径: 除以注入期望 spike mass, 不是 src_mass
    W = np.array([[0,2.,0],[0,0,0.],[0,0,0]]); sm = np.array([1.,1,1]); inj = np.array([4.,4,4])
    S = make_step_operator(W, sm, injected_mass=inj)
    assert np.isclose(S[0,1], 0.5)                   # 2/4, 用 injected 分母
```
- [ ] **Step 2: 跑，验证 fail**（`ModuleNotFoundError` / `ImportError`）。
- [ ] **Step 3: 实现** `make_step_operator` / `make_shape_operator` / `spectral_radius`（调 `branching_ratio`）/ `h_field` / `principal_axis` / `ordering_predictivity`（纯线代）；`spatial_bins` / `build_w_resp`（读 `simulate_kick`，循环 bin × seed）。
- [ ] **Step 4: 跑单元测试 pass**：`pytest tests/test_topic4_propagation_operator.py -v`（6+ tests，含 `test_h_from_unnormalized_resp_not_flat` + `test_step_operator_scales_with_gain_not_rownorm` 两道 row-norm 反陷阱测试 + `test_step_operator_excludes_low_src_mass` + `test_step_operator_injected_mass_sensitivity` 两道 src_mass 防爆 / sensitivity 测试）。
- [ ] **Step 5: build_w_resp L=12 smoke**：建 L=12 net、`n_bins_per_axis=4`、kick_boost 小、seeds=2 → `W_resp.shape==(16,16)`、`(W_resp>0).any()`；打印 `spectral_radius(make_step_operator(W_resp, src_mass))` + `h_field(W_resp,'post')`（确认 h 非全等）。
- [ ] **Step 6: Commit** `feat(topic4 M3-final): W three-object propagation operator (resp/step/shape, h/rho/axis)`

---

### Task 1.5: kick 幅度 + 一步窗标定（M3-1.5，C6c；标定值喂 Task 4 预注册冻结）

**Files:** Output `results/topic4_sef_hfo/m3_local_w/kick_calibration/` + `figures/`（README 中文）。
- Create: `scripts/run_m3_kick_calibration.py`

- [ ] **Step 1: 扫**：对 3–5 个代表 bin，扫 `KICK_BOOST ∈ [very_small, small, medium, large] × win_ms ∈ [[2,6],[4,10],[8,16],[12,24]]`，量 (a) 响应 vs KICK_BOOST 近线性区；(b) 直接刺激 artifact 窗 vs first downstream generation peak；(c) kick 是否直接触发 self-sustained wave。
- [ ] **Step 2: 选 + 落盘**：选 quasi-linear local regime 的 `kick_boost` + first-recruitment `[Δ1,Δ2]`（太小=全噪声丢、太大=直接触发全局 event 丢），写 `kick_calibration.json`（含选值 + 理由 + 各候选曲线）。
- [ ] **Step 3: 图（response-vs-boost 近线性 / generation-peak 窗 / self-sustained 边界）+ README + Commit** `analysis(topic4 M3-final): kick amplitude + one-generation window calibration`。**输出喂 Task 4 预注册冻结。**

---

### Task 2: h-耦合阈值变换（`src/topic4_permissivity.py`，纯 helper，零引擎改动）

**Files:**
- Create: `src/topic4_permissivity.py`
- Test: `tests/test_topic4_permissivity.py`

**Interfaces:**
- Consumes: `h_field`（Task 1）。
- Produces: `permissivity_vth_delta(h, bin_of_cell, NE, NI, *, mu, delta_theta, control='none', rng=None) -> np.ndarray(NE+NI)` — 长度 NE+NI 的阈值**增量**向量（I cell=0；E cell = `−delta_theta * mu * h_eff[bin_of_cell[i]]`，**符号为负 = 压低阈值 = 抬高 permissivity**）。`control`：`'none'`=用 h；`'uniform'`=h_eff 全置 1（均匀 μ 对照，C5）；`'shuffle'`=用 `rng` 打乱 h 的 bin→值映射（打乱 h 对照，C5）。调 `simulate_kick` 前 `V_th_per_neuron += delta`。

- [ ] **Step 1: 失败测试**
```python
import numpy as np
from src.topic4_permissivity import permissivity_vth_delta

def test_mu0_zero_delta():
    h = np.array([1.,2.,0.5]); boc = np.array([0,0,1,1,2,2])
    d = permissivity_vth_delta(h, boc, NE=6, NI=2, mu=0.0, delta_theta=3.0)
    assert d.shape == (8,) and np.allclose(d, 0)         # μ=0 -> 全 0 (bit-parity)

def test_sign_negative_and_high_h_lowers_more():
    h = np.array([1.,2.]); boc = np.array([0,1])
    d = permissivity_vth_delta(h, boc, NE=2, NI=1, mu=0.5, delta_theta=4.0)
    assert (d[:2] <= 0).all()                            # 压低阈值
    assert d[1] < d[0]                                   # 高 h -> 压更多
    assert d[2] == 0                                     # I cell 不动

def test_uniform_control_ignores_h_shape():
    h = np.array([1.,9.]); boc = np.array([0,1])
    d = permissivity_vth_delta(h, boc, NE=2, NI=0, mu=0.5, delta_theta=2.0, control='uniform')
    assert np.allclose(d[0], d[1])                       # uniform -> 与 h 形状无关
```
- [ ] **Step 2: 跑，验证 fail。**
- [ ] **Step 3-4: 实现 → pass.**（`h_eff = h` / `ones` / `h[shuffle]`；`delta[E] = −delta_theta*mu*h_eff[bin]`；I cell=0。）
- [ ] **Step 5: Commit** `feat(topic4 M3-final): W-coupled permissivity threshold delta (h/uniform/shuffle controls)`

---

### Task 3: Runner CLI 接线（h-耦合，复用 `V_th_per_neuron` 路径，μ=0 bit-parity）

**Files:**
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`
- Test: `tests/test_topic4_m3_localw_cli.py`

**Interfaces:**
- Consumes: `spatial_bins`, `build_w_resp`, `h_field`（Task 1）；`permissivity_vth_delta`（Task 2）。
- Runner 新增 CLI：`--mu`（默认 0）、`--delta-theta`、`--h-source {struct,resp}`（struct=用 `ee_degree`/`recruitment_operator` 先验；resp=用 measured `W_resp`）、`--h-scheme {post,out,hybrid}`、`--h-control {none,uniform,shuffle}`、`--mu-impl {threshold,inhibition}`（threshold 先做；inhibition Phase-2）、`--w-resp-cache PATH`（`W_resp` npz，避免每次重测）。
- 流程：若 `--mu>0`：载/测 `W_resp`（μ=0 baseline）→ `h_field(W_resp, scheme)` → `permissivity_vth_delta(...)` → 调 `simulate_kick` 前 `V_th_per_neuron += delta`；所有新参数 + bin 数 + `Λ₀=ρ(W_step)` 写进 `config` provenance。

- [ ] **Step 1: 失败测试** — `--mu` 在 `--help`；coexist smoke：`--mu 0.3 --h-source struct --h-scheme post --ee-std-u 0.2` 一把跑通，config 里 `mu/h_source/h_scheme/h_control/ee_std_u` 全在（M1 recovery 不被挤掉）；`--mu 0` → spike SHA == `M3_BASE_SHA`（bit-parity）。
- [ ] **Step 2-4: 接线 → pass.**（μ=0 短路、不动 `V_th_per_neuron`，保证 bit-parity。）
- [ ] **Step 5: Commit**（runner 不改引擎，无需 re-bless）`feat(topic4 M3-final): wire W-coupled permissivity into runner CLI (mu=0 bit-parity)`

---

### Task 4: 预注册落盘（M3-0，出任何 SNN 动力学结果之前）

**Files:**
- Create: `scripts/run_m3_localw_preregistration.py`
- Output: `results/topic4_sef_hfo/m3_local_w/preregistration.json`

- [ ] **Step 1: 预注册写死**（调 `subject_tolerance_band` 从 Task-0 `cohort_summary.json` 算 Layer-2 容差带；**`kick_boost`/`[Δ1,Δ2]` 取 Task 1.5 标定值**）：`preregistration.json` 含 — h 三方案比较计划 + 主口径 `h^post`、`A_p` primary（E spike count）、**标定后的 `kick_boost` + 响应窗 `[Δ1,Δ2]`**（值 + Task 1.5 理由）、Layer-2 subject-level AF/LR 容差带、**R0–R4 分类数值阈值（spec §6.3 预注册）**：return-to-baseline = `A(t)<μ_A+z·σ_A` 持续 `T_quiet`（`z`/`T_quiet` 写死）、sustained = 持续 `>T_sustain` / 无 quiet / 重复波 `<T_gap`、**R4a（W-aligned，早相轴 `θ_early≈θ(W)`）vs R4b（无传播结构 tonic runaway）分类阈值**、轴对齐角度阈值。
- [ ] **Step 2: 校验**：`preregistration.json` 可载、字段齐、Layer-2 band 来自真实 n=23 per-subject median（非 hardcode）。
- [ ] **Step 3: Commit** `feat(topic4 M3-final): freeze local-W pre-registration (h-scheme/A_p/window/Layer-2 band/R0-R4)`

---

### Task 5: M3-1 baseline 基底 pilot（复用 worktree 基底；早报 ⑤双向稳定度，不硬停）

**Files:** Output `results/topic4_sef_hfo/m3_local_w/baseline/`。

- [ ] **Step 1: 跑 baseline**（`--mu 0`、`--ee-std-u 0.2`、`--hub-gain 0 --degnorm-alpha 0 --gate-scale 0` 全关、**primary = 单个平滑 onset patch**（审阅 §3）、T=8000、`--dump-fullfield --dump-fwd-rev-reps`），扫几个 `(core_mean, drive)`。**`twoend_equal` 只作 legacy control / fallback**（单 patch 给不出双向时再用，不进主 claim）。
- [ ] **Step 2: 量 + 早报**：自发离散事件率（非 silent/非 tonic）、return-to-baseline 比例、事件时长、reach、读出方向、`DI`、跨事件路线稳定度、KMeans 模板稳定。
- [ ] **Step 3: 早报存在性（不硬停，Global Constraints）**：把"亚临界可读自停窗 + ⑤双向稳定度"写进 `baseline/STATUS.md`（中文，§8 三段式）作后续语境；**不**因不达标而停——继续 Task 6。

---

### Task 6: M3-2 估 `W` 三对象 + 预测性 gate + binning sensitivity

**Files:**
- Create: `scripts/run_m3_w_resp.py`
- Output: `results/topic4_sef_hfo/m3_local_w/w_resp/` + `figures/`（README 中文）。

- [ ] **Step 1: 测 `W` 三对象**（baseline μ=0 工作点，调 Task 1 `build_w_resp` → `make_step_operator` → `make_shape_operator`，多 seed、coarse-bin），写 `W_resp.npz` + `W_step.npz` + `W_shape.npz` + `Λ₀=ρ(W_step)` + 主轴(W_shape) + `h^post(W_resp)`。
- [ ] **Step 2: 预测性对照**（Task 1 `ordering_predictivity`，用 `W_shape`）：从 baseline 自发事件取 bin 激活先后，比 `rho_W` vs `rho_dist` vs `rho_rate`。
- [ ] **Step 3: binning / r_kick sensitivity（审阅 §10）**：扫 `n_bins_per_axis ∈ {4,5,6} × r_kick ∈ {small,medium} × edge-bin 含/不含`，验 **主轴稳定 / h-map rank 相关稳定 / `ρ(W_step)` 相对排序稳定 / `ordering_predictivity` 方向一致**（非逐元相同）。
- [ ] **Step 4: 早报 gate（不硬停）**：`rho_W` 是否 > `rho_dist` 且 > `rho_rate`（W 是有用机制对象的依据）。写 `STATUS.md`（§8 三段式）+ 图（`W_resp`/`W_step` 矩阵 / 主轴 / h map / 预测顺序 vs 实测 / sensitivity）+ README。**结果不达标 → 记录 + 审视退 hub，但继续 Task 7**（联合判断）。
- [ ] **Step 5: Commit** `analysis(topic4 M3-final): W resp/step/shape + ordering predictivity + binning sensitivity`

---

### Task 7: M3-3 静态 `Λ₀ × μ` 相图 pilot

**Files:**
- Create: `scripts/run_m3_lambda_mu_phasemap.py`
- Output: `results/topic4_sef_hfo/m3_local_w/lambda_mu/` + `figures/`（README 中文）。

- [ ] **Step 1: 相图**：扫 `J_EE_scale ∈ [0.75,0.85,0.95,1.05,1.15,1.30,1.45] × μ ∈ [0,0.1,0.2,0.35,0.5,0.65,0.8]`，`AR=2/θ=45°` 固定。**每个 `J_EE_scale` 列：μ=0 测一次 `W_resp`/`W_step`/`h`/`Λ₀=ρ(W_step)`（Global Constraint，别每个 μ 重测）**；各 μ 算 `Λ_eff=ρ[D_μ(h)·W_step]`（线代预测）+ 跑 SNN **两套（审阅 §7 + P1）**：(A) spontaneous mode、(B) **conditioned-on-ignition mode（主口径 = 相同 onset patch + 相同 finite kick 让 ignition 完全相同；secondary = 初始 10–20ms active mass 匹配的自发 event）** + 真实 lag/rank/KMeans pipeline → 分类 **R0–R3 / R4a / R4b**（早相轴 `θ_early≈θ(W)`，spec §6.3）。
- [ ] **Step 2: 判据（早报）**：是否存在 R2；R2→R3→**R4a** 是否连续；**R4a** 是否沿同一 `W` 主轴（早相轴对齐，**非** whole-event）；`Λ_eff` 是否预测 phenotype；conditioned mode 下 reach/duration/escape 是否随 μ 变（证 μ 改 same-W 传播、非只提点火率）。
- [ ] **Step 3: 头号对照（C5）**：在 R2/R3 边界点跑 `--h-control uniform` + `--h-control shuffle`，看相变 / 早相轴对齐是否塌。
- [ ] **Step 4: 双向 vs μ 跟踪（审阅 §9）**：逐 μ 记 `DI(μ)`、`K_selected(μ)`、fwd/rev 模板稳定度——分 low-μ 近双向 / high-μ seizure-aligned bias；若 μ 一开就完全单向 = `h`-阈值实现过强或 `h` 定义不当（红旗，记录）。
- [ ] **Step 5: 图 + README + Commit** `analysis(topic4 M3-final): Lambda0 x mu phase map (cond-on-ignition, R4a/R4b, uniform/shuffle, DI-vs-mu)`

---

### Task 8: M3-4 `recovery × μ` 相图 pilot（③ 承重判据）

**Files:**
- Create: `scripts/run_m3_recovery_mu_phasemap.py`
- Output: `results/topic4_sef_hfo/m3_local_w/recovery_mu/` + `figures/`（README 中文）。

- [ ] **Step 1: 相图**：固定 `Λ₀` 在 Task 7 找到的 R2/R3 边界，扫 `μ ∈ [0,0.15,0.3,0.45,0.6,0.75] × {τ_rec or U} ∈ [current/2, current, current*2, current*4]`（`--ee-std-u` / `--ee-std-tau-ms`）。
- [ ] **Step 2: 判据（早报）**：强 recovery=间期自限（回静息）、弱 recovery=near-critical/持续招募；证 M1 是时间自限机制、μ 是状态门控机制。
- [ ] **Step 3: 图 + README + Commit** `analysis(topic4 M3-final): recovery x mu phase map (temporal self-limit vs sustained recruitment)`

---

### Task 8.5: M3-4.5 post-event `ΔΛ_x` 探针（证 M1 是真·快速保护项，审阅 §8）

**Files:**
- Create: `scripts/run_m3_post_event_probe.py`
- Output: `results/topic4_sef_hfo/m3_local_w/post_event/` + `figures/`（README 中文）。

- [ ] **Step 1: 测 baseline vs post-event `W`**：在 R2 工作点跑 spontaneous SNN；对自发事件**结束后 50–200ms** 做 small kick（`build_w_resp(t_kick=event_end+τ)`），得 `W_resp^post` → `W_step^post`；baseline `W_step` 取 Task 6。
- [ ] **Step 2: 算 `ΔΛ_x = ρ(W_step^post) − ρ(W_step^baseline)`**，预期 **<0**（事件后传播增益短时下降）。若 full-W 太贵，退而记事件后固定窗 local evoked response / event probability 的压制。
- [ ] **Step 3: 图（`ΔΛ_x` 分布 / post-event evoked 压制）+ README + Commit** `analysis(topic4 M3-final): post-event Delta-Lambda_x probe (M1 fast protection)`

---

### Task 9: M3-5 basin `K_min(μ)` finite-pulse escape pilot

**Files:**
- Create: `scripts/run_m3_basin_escape.py`
- Output: `results/topic4_sef_hfo/m3_local_w/basin/` + `figures/`（README 中文）。

- [ ] **Step 1: escape 扫**：对每个 μ，从 onset patch 给不同强度 finite kick `K`（复用 `simulate_kick(KICK_BOOST=K)`），测 `P_escape(K,μ)`（escape = 进入持续招募、不回静息，用 Task 6 `hub_global_recruitment`-类 reach/持续判据）。
- [ ] **Step 2: 算 `K_min(μ)`** = `min K s.t. P_escape>0.5`；量 escape 事件**早相轴** vs `W_shape` 主轴（早相对齐，spec §6.3）。
- [ ] **Step 3: 判据（早报）**：`K_min(μ)` 随 μ 是否下降（separatrix 变近）；failed recruitment vs sustained recruitment 边界。
- [ ] **Step 4: 图（`P_escape(K,μ)` 曲线 / `K_min(μ)` / 轴对齐）+ README + Commit** `analysis(topic4 M3-final): basin K_min(mu) finite-pulse escape`

---

### Task 10: 联合判断 + pilot recap（pilot 全过 → 用户讨论闸口）

**Files:** Output `docs/archive/topic4/sef_hfo/m3_localw_pilot_recap_<date>.md`（§8 三段式中文）。

- [ ] **Step 1: 联合判断**（Global Constraint：存在性早报不硬停 → 这里联合判）：汇总 Task 5–9 — 间期 R2 存在？`W_shape` 预测性 > 距离/率？R2→R3→**R4a** 连续 + 早相沿同轴？均匀/打乱 h 对照塌？conditioned-on-ignition 下 μ 改 same-W 传播？`ΔΛ_x<0`（M1 保护）？basin `K_min(μ)` 下降？
- [ ] **Step 2: 三岔口写清**：(a) 联合为正 → 进 Phase 2（动态 m + 网格 + 消融 + 真实读出验收）；(b) 给不出稳定双向 → spec §9 失败口径一；(c) 连亚临界可读自停窗都没有 → spec §9 失败口径二 + 退 hub（spec §10）。
- [ ] **Step 3: recap + memory + Commit**（**用户讨论闸口：Phase 2 不在本轮自动执行**）。

---

## Phase 2（pilot 全过 + 用户讨论后再做，本轮不执行）

- **Task 11: 动态 `m_i(t)` pilot**（真新引擎件，单独 re-bless）：`τ_m·ṁ_i = −(m_i−m_0) + η·h_i·[W_step·a(t)]_i − χ·a_i(t)`；比较 recovery-dominant / permissivity-dominant / balanced；输出 `ΔΛ_post = Λ(t+Δ)−Λ(t⁻)`，与真实数据 post-event rate/extent/direction bias 做对应预测。
- **Task 12: 网格 + 消融**：spec §6.4 十个消融（**头号 = `D_m(h)` vs `uniform` vs `shuffle` 看轴向保持**）+ threshold vs inhibition 实现对照 + L 不变性（L20/L32 仅作描述，非 ③ 判据）。
- **Task 13: 真实 SEEG 读出验收 + verdict + archive**：复用 pipeline → lagPat/rank-displacement/KMeans/split-half(odd-even)/endpoint enrichment；Layer-2 subject-level 等价（复用 `layer2_equivalence`）；§8 白话 abstract + 相图 + basin + 消融表 → `docs/archive/topic4/sef_hfo/m3_localw_result_<date>.md`；更新 framework + memory。inhibition-restraint μ 实现（`--mu-impl inhibition`）也在此阶段。

---

## Self-Review

1. **Spec 覆盖：** U1 基底（Task 5，单 patch primary）✓；U2 W 三对象（Task 1 `W_resp`/`W_step`/`W_shape` + Task 1.5 标定 + Task 6）✓；U3 h-耦合 μ（Task 2 threshold + Task 3 runner；inhibition + 动态 m → Phase 2）✓；U4 readout 复用（Task 6/Task 13 Layer-2）✓；**W 三对象 + 一步构造 C6（审阅 §1：Task 1 `make_step_operator` 按源活动归一→`spectral_radius`、`h_field` 用未归一 `W_resp`、row-norm 反陷阱测试 ×2；Task 1.5 kick/窗标定）✓**；Λ₀ 测一次（Task 7 Step 1）✓；头号对照 C5（Task 7 Step 3 + Task 12）✓；三相图 + basin（Task 7/8/9）✓；**R4a/R4b + 早相轴对齐（Task 4 预注册 + Task 7/9，审阅 §4/§6）✓**；**condition-on-ignition（Task 7 Step 1-2，审阅 §7）✓**；**post-event `ΔΛ_x`（Task 8.5，审阅 §8）✓**；**binning sensitivity（Task 6 Step 3，审阅 §10）✓**；**双向 vs μ 跟踪（Task 7 Step 4，审阅 §9）✓**；**data-facing predictions（spec §6.5，审阅 §11）✓**；预注册（Task 4）✓；自限重定义=时间（Task 5/8 return-to-baseline，**无**沿轴短判据）✓；存在性早报不硬停（Task 5/6 + Task 10 联合）✓；hub fallback（Task 10 三岔口）✓。
2. **Placeholder：** Phase-2 Task 11–13 `<date>` 产出时填；`M3_BASE_SHA` 是 Task 0 输出锚；无 TODO/TBD。
3. **Type 一致：** `spatial_bins/build_w_resp/make_step_operator/make_shape_operator/spectral_radius/h_field/principal_axis/ordering_predictivity`（Task 1）→ `permissivity_vth_delta`（Task 2，吃 `h_field(W_resp,..)` 输出 + `bin_of_cell`）→ runner（Task 3，吃 Task 1+2，`--h-source {struct,resp}`）→ 相图/basin/post-event 脚本（Task 7/8/8.5/9）；复用件 `ee_degree/recruitment_operator/branching_ratio/subject_tolerance_band/layer2_equivalence/hub_global_recruitment` 签名不变；bit-parity 锚 `M3_BASE_SHA` 单一。
4. **bit-parity hazard：** h-耦合 μ=0 短路（Task 3 不动 `V_th_per_neuron`）；`build_w_resp` 只读引擎不改；动态 m（Phase 2）才碰引擎、单独 re-bless。
5. **PILOT-FIRST + 存在性不硬停：** Task 1.5/5/6/7/8/8.5/9 pilot 在动态 m/网格（Phase 2）前；存在性 Task 5/6 早报、Task 10 联合判断，**不**中途硬停（用户 2026-06-21 决定）；预注册（Task 4，吃 Task 1.5 标定值）在任何动力学相图结果前。
