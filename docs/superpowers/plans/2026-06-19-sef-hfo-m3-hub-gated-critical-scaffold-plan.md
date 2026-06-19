# M3 hub-gated critical branching scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`).
> **Spec:** `docs/superpowers/specs/2026-06-19-sef-hfo-m3-hub-gated-critical-scaffold-design.md` (commit ebbd27f) — read it for the *why*; this plan is the *how*.

**Goal:** 在 cm-scale 放电网络上建一个"病理走廊 + 高门槛广播枢纽"模型，测它能否自发产生间期 HFO 群体事件（自发/时间自限/可传/双向）、并通过慢变量从"事件死在枢纽前（结构封闭）"相变到"跨枢纽广播（发作样）"——先建基础设施 + 跑便宜的结构相图 + 可行性 pilot（不需决定的探索），把昂贵的网格/消融留到 pilot + 用户讨论之后。

**Architecture:** 单图度归一化 hub（D2）：(1) `corridor_twoend` 基底 = 各向异性走廊 + 两端触发盆 + 走廊外全局区；(2) hub = 走廊轴末端少数高出度 E cell，加稀疏长程 E→E broadcast 边（镜像现有 `ei_gate` 块）+ 全网度归一化阈值（θ_i=θ0+α·g_deg(i)，骑现有 `V_th_per_neuron`，零 `simulate_kick` 改动）+ 慢变量压低 hub θ（先两静态条件）；(3) 结构 σ 探针（纯线代，选工作点，替代 toy）；(4) 复用虚拟 SEEG readout + Task-0 指标做 subject-level 等价验收。

**Tech Stack:** Python, NumPy, SciPy sparse; `src/snn_engine/` LIF substrate（worktree base = M1 recovery + M2 gate）；pytest；现成 readout/geometry 脚本。

## Global Constraints

- **Worktree（P0）：所有 M3 代码在 `.worktrees/topic4-m3`（branch `topic4-snn-m3-hub`），fork 自 `topic4-snn-m1-recovery`（96a416f = M1 recovery `--ee-std-u` + M2 gate `--gate-scale/--l-gate/--c-gate`），merge `topic4-event-extent-audit`（带 Task-0 指标 `src/topic4_event_extent_audit.py` + spec/plan）。** 不依赖 M2-faithful 在飞的 shunting/ahead-recruit 改动（D3 并行隔离）。只 re-bless 本 worktree 自己的 `engine_versions.json`。
- **Default-OFF bit-parity（P0，神圣）：`hub_gain=0`/`hub_mask_E=None` AND `degnorm_alpha=0`（默认）时，引擎 spike 与 base 逐比特一致。** base spike SHA 在 Task 0 锚定。每块新机制 gated；默认路径无新 rng draw / 无 float 触碰。
- **引擎编辑 → re-bless（先过 parity 测试再 bless）。** 度归一化是 runner 预变换（零 `simulate_kick` 改动）；只有长程边动 `connectivity_rot.py`。
- **PILOT-FIRST（P0 硬停）：** 不跑任何网格，直到 U1.0 feasibility（Task 9）+ op-sanity（Task 10）+ interictal pilot（Task 11）+ transition pilot（Task 12）四道 gate 全过。
- **预注册（出任何 SNN 动力学结果之前，Task 8 落盘）：** 度归一化三方案（out_strength/in_strength/hybrid）比较 + 主口径选取规则、Layer-2 subject-level 容差带、hub/global 数值阈值。结果出来不许改。
- **Claim discipline（spec §3 C1–C5 + 审阅）：** 临界 = 招募算子 σ≈1（**非**静息线性 max Re λ≈0）；事件停 = **结构封闭 structural containment**（**非** intrinsic self-limit）；ictal = **synthetic feasibility bridge**（不解释临床发作）；Layer-2 = **subject-level 等价**（**非** event-level "不被拒=PASS"）；度归一化 **不设单一 primary**。
- **真实 readout：** 所有 synthetic 结论必须经真实 masked lagPat propagation pipeline，禁止只看 raster。

---

### Task 0: M3 worktree 建立 + base 校验

**Files:** worktree `.worktrees/topic4-m3`；无代码改动。

- [ ] **Step 1: 建 worktree（fork m1-recovery，merge event-extent-audit）**
```bash
cd /home/honglab/leijiaxin/HFOsp
git worktree add -b topic4-snn-m3-hub .worktrees/topic4-m3 topic4-snn-m1-recovery
cd .worktrees/topic4-m3
git merge --no-edit topic4-event-extent-audit   # 带 Task-0 指标 + spec/plan；不同文件，预期无冲突
```
- [ ] **Step 2: 校验 base 三要素都在**（M1 recovery flag + M2 gate + Task-0 指标）
```bash
grep -q "ee_std_u" scripts/run_sef_hfo_snn_cm_spontaneous_readout.py && echo "M1 recovery OK"
grep -q "gate_scale" src/snn_engine/connectivity_rot.py && echo "M2 gate OK"
test -f src/topic4_event_extent_audit.py && echo "Task-0 metrics OK"
python3 -c "from src.topic4_event_extent_audit import event_extent, matched_null_extent; print('import OK')"
```
- [ ] **Step 3: 锚定 base spike SHA**（bit-parity 锚）
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
记录输出为 `M3_BASE_SHA`（写进所有 parity 测试 + 本 plan 此处）。
- [ ] **Step 4: 跑现有引擎 smoke**：`python3 -m pytest tests/ -k "snn or gate or step0 or recovery" -q` → PASS（base 无回归）。

---

### Task 1: `corridor_twoend` 基底 + deterministic hub 选择 + region 索引

**Files:**
- Create: `src/topic4_corridor_substrate.py`（纯几何：region 划分 + hub 选择）
- Test: `tests/test_topic4_corridor_substrate.py`

**Interfaces:**
- Produces: `corridor_regions(posE, center, axis_unit, half, corridor_half_frac=0.6, hub_frac=0.12) -> dict(corridor_idx, global_idx, hub_idx, along)` — 把 E cell 按沿轴投影 `along=(posE−center)·axis_unit` 分：`|along| ≤ corridor_half_frac*half` = 走廊；其余 = 全局区；走廊内 `along` 最靠正轴端的 `hub_frac` 比例 E cell = hub（deterministic：按 along 排序取末端，**不引入 seed-dependent 位置**）。返回各 region 的 E-local 索引 + along。
- Produces: `hub_mask_E(NE, hub_idx) -> np.ndarray(bool)` — 长度 NE 的 bool。

- [ ] **Step 1: 失败测试**（确定性 + region 划分）
```python
import numpy as np
from src.topic4_corridor_substrate import corridor_regions, hub_mask_E
def test_regions_deterministic_and_partition():
    posE = np.c_[np.linspace(-10, 10, 50), np.zeros(50)]   # 沿 x 轴一条
    out = corridor_regions(posE, center=np.array([0.,0]), axis_unit=np.array([1.,0]),
                           half=10.0, corridor_half_frac=0.6, hub_frac=0.1)
    # 走廊 = |along|<=6mm；hub = 走廊内最靠 +x 端 10%
    assert set(out["corridor_idx"]) | set(out["global_idx"]) == set(range(50))   # partition
    assert set(out["corridor_idx"]) & set(out["global_idx"]) == set()            # disjoint
    assert set(out["hub_idx"]).issubset(set(out["corridor_idx"]))               # hub ⊂ corridor
    # determinism: 同输入两次 -> 同 hub
    out2 = corridor_regions(posE, np.array([0.,0]), np.array([1.,0]), 10.0, 0.6, 0.1)
    assert out["hub_idx"] == out2["hub_idx"]
    # hub 在 +x 末端
    assert min(out["along"][i] for i in out["hub_idx"]) > 4.0
```
- [ ] **Step 2-4: 实现 → pass.** `corridor_regions`：`along = (posE-center)@axis_unit`；`corridor = |along|<=corridor_half_frac*half`；`global=~corridor`；hub = corridor 内 along 最大的 `ceil(hub_frac*n_corridor)` 个（`np.argsort(along)` 取末端，确定性）。
- [ ] **Step 5: Commit**（不在并行窗口内提交，见执行说明）

---

### Task 2: 长程 E→E broadcast 边（`connectivity_rot.py`，默认关 bit-parity，re-bless）

**Files:**
- Modify: `src/snn_engine/connectivity_rot.py`（镜像 `ei_gate` 块，在 E→E AMPA 分支加 hub→远端）
- Test: `tests/test_snn_hub_longrange.py`
- Modify(after parity): worktree `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`

**Interfaces:**
- Produces: `build_connectivity_rot(..., hub_mask_E=None, hub_long_range_C=0, l_hub_long=None, hub_gain=0.0)`。当 `hub_gain>0 and hub_mask_E is not None`：对每个 **E 目标** `i`，额外从 **hub 源子集**（`posE[hub_mask_E]`）按 `_sample_partners(pt, hub_pos, hub_long_range_C, l_hub_long, 0.0, rng)` 采远端 hub 源，权重 `hub_gain*w_EE*jump_ampa[i]`，写进 `ampa_by_delay`。默认 `hub_gain=0` → 无新 rng draw → bit-parity。Guard：`hub_gain>0` 要求 `hub_mask_E/hub_long_range_C/l_hub_long`（ValueError）。

- [ ] **Step 1: 失败测试**（镜像 `tests/test_snn_ie_gate.py`）
```python
# (a) hub_gain=0 默认 -> spike SHA == M3_BASE_SHA
# (b) hub_gain>0 -> ampa(E->E) 边数增加，且新增边的源都在 hub_mask_E 内
# (c) 更大 l_hub_long -> 新增边的平均距离更大
# (d) hub_gain>0 缺 hub_mask_E/C/l -> ValueError
```
- [ ] **Step 2: 跑，验证 fail**（`unexpected keyword 'hub_gain'`）
- [ ] **Step 3: 实现** — 在 `a_is_E` 的 E→E append 之后加（gated）：
```python
        if a_is_E and hub_gain > 0.0 and hub_mask_E is not None and hub_mask_E.any():
            hub_idx_global = np.flatnonzero(hub_mask_E)        # E-local hub indices
            hub_pos = posE[hub_idx_global]
            ch = _sample_partners(pt, hub_pos, hub_long_range_C, l_hub_long, 0.0, rng, self_local=None)
            if ch.size:
                src = hub_idx_global[ch]                        # map back to E indices
                dh = np.linalg.norm(posE[src] - pt, axis=1)
                a_rows.append(np.full(src.size, i)); a_cols.append(src)
                a_w.append(np.full(src.size, (w_EE * hub_gain) * jump_ampa[i]))
                a_dly.append(p.tau0 + dh * inv_vdt)
```
+ 顶部 guard：`if hub_gain>0.0 and (hub_mask_E is None or hub_long_range_C<=0 or l_hub_long is None): raise ValueError(...)`。
- [ ] **Step 4: 跑，验证 pass**（4 tests；(a) SHA 不变）。
- [ ] **Step 5: re-bless connectivity_rot.py**（仅 worktree 的 engine_versions.json）+ smoke + commit。

---

### Task 3: 度归一化阈值变换（纯 helper，零引擎改动）

**Files:**
- Create: `src/topic4_degnorm.py`
- Test: `tests/test_topic4_degnorm.py`

**Interfaces:**
- Produces: `ee_degree(net, NE, scheme) -> np.ndarray(NE)` — 从 `net["ampa_by_delay"]` 取 E→E 子矩阵（行=目标、列=源，见 build），按 `scheme ∈ {out_strength, in_strength, hybrid}` 算每个 E cell 的度量：`out_strength` = 列和（作为源发出的权重，含长程边）；`in_strength` = 行和（作为目标收到的权重）；`hybrid` = `max(out_norm, in_norm)`（各自除以中位数后取 max）。
- Produces: `degnorm_vth_delta(net, NE, NI, alpha, scheme) -> np.ndarray(NE+NI)` — 长度 NE+NI 的阈值增量向量（I cell = 0），E cell = `alpha * ee_degree`。调 `simulate_kick` 前 `V_th_per_neuron += delta`。

- [ ] **Step 1: 失败测试** — 手搓小 net（已知 E→E 边）验证 out/in/hybrid 与列和/行和一致；`alpha=0 → delta 全 0`；高出度 cell 的 out_strength delta 最大。
- [ ] **Step 2-4: 实现 → pass.**（从 `ampa_by_delay` 累加 E→E 子块；注意只取 E 源 E 目标 `row<NE & col<NE`。）
- [ ] **Step 5: Commit**

---

### Task 4: 结构临界探针（`src/topic4_hub_criticality.py`，纯线代）

**Files:**
- Create: `src/topic4_hub_criticality.py`
- Test: `tests/test_topic4_hub_criticality.py`

**Interfaces:**
- Produces: `recruitment_operator(net, V_th, NE, drive_rest, link='linear') -> scipy.sparse` — `M[j,i] = w_EE_edge(j<-i) * gap_factor(V_th[j], drive_rest)`，`gap_factor = clip(1/(V_th[j]-drive_rest), 0, cap)` 单调 link（screening，非动力学拟合）。只用 E→E 子块。
- Produces: `branching_ratio(M, idx) -> float` — `idx` 子集上 `M` 的最大特征值实部（`scipy.sparse.linalg.eigs(k=1)`；小矩阵退化为 dense `eigvals`）。
- Produces: `crossing_branching(M, corridor_idx, hub_idx, global_idx) -> float` — 走廊→hub→全局这条路径的有效 σ（restrict 到 `corridor∪hub∪global`，但只保留"经 hub 出"的列；实现：把 corridor→global 的直接边清零，强制经 hub）。
- Produces: `sigma_phase_map(build_fn, alpha_grid, gain_grid, scheme, regions, V_th0, drive_rest) -> dict` — 对每个 (alpha, gain) 重建 net + degnorm V_th，算 `σ_corridor`/`σ_crossing`，返回 2D 数组 + 相变边界 σ_crossing=1 的等高线。

- [ ] **Step 1: 失败测试**（解析校验）— 链状网 σ = 行和；断开 hub（hub_gain=0）→ `crossing_branching=0`；σ 随 alpha↑（阈值↑）单调↓、随 gain↑单调↑。
- [ ] **Step 2-4: 实现 → pass.**
- [ ] **Step 5: Commit**

> **此 task 可由并行 subagent 建（纯函数，独立文件，单元测试用手搓矩阵，不依赖引擎）。**

---

### Task 5: Layer-2 subject-level 等价 helper（`src/topic4_m3_acceptance.py`，纯）

**Files:**
- Create: `src/topic4_m3_acceptance.py`
- Test: `tests/test_topic4_m3_acceptance.py`

**Interfaces:**
- Produces: `subject_tolerance_band(ref_per_subject_af, ref_per_subject_lr, q=(10,90)) -> dict` — 从 Task-0 真实 subject-level median AF/LR 列表算预注册容差带（分位区间）。
- Produces: `layer2_equivalence(model_subject_af, model_subject_lr, band, *, min_af=0.75) -> dict(pass, af_in_band, lr_in_band, af_median, lr_median, note)` — PASS = 模型 subject-level median AF/LR **都落在容差带内** AND `af_median≥min_af`（短一段 reject）。**非** p>α。返回描述性 overlap 作辅助。

- [ ] **Step 1: 失败测试** — 容差带 = ref 的 10–90 分位；模型 median 落带内 → pass；模型 AF≪band（短一段）→ fail；KS 只作辅助字段不决定 pass。
- [ ] **Step 2-4: 实现 → pass.** Task-0 真实 per-subject median 从 `results/topic4_sef_hfo/event_extent_audit/cohort_summary.json`（reference_distribution / per_subject）读，**预注册时落盘容差带**。
- [ ] **Step 5: Commit**

> **可由并行 subagent 建。**

---

### Task 6: hub/global 招募 + relay-timing 诊断（`src/topic4_hub_diag.py`，纯）

**Files:**
- Create: `src/topic4_hub_diag.py`
- Test: `tests/test_topic4_hub_diag.py`

**Interfaces:**
- Produces: `hub_global_recruitment(E_spk_bool, hub_idx, global_idx, corridor_idx, dt) -> dict(hub_recruited_fraction, global_E_spike_fraction, global_first_spike_after_hub_ms, corridor_onset_ms, hub_onset_ms)` — 各 region 放电比例 + 首发时间；`global_first_spike_after_hub_ms = t_global_first − t_hub_first`（>0 = 全局区在 hub 之后才燃 = relay）。
- 间期 gate 判据：`hub_recruited_fraction` 与 `global_E_spike_fraction` 都 < 预注册阈值。发作 bridge：hub 先燃、`global_first_spike_after_hub_ms>0`、`global_E_spike_fraction` 显著抬升。

- [ ] **Step 1: 失败测试** — 合成 spike：走廊先燃、hub 中间、全局区最后 → `global_first_spike_after_hub_ms>0`；全局区全静默 → `global_E_spike_fraction==0`。
- [ ] **Step 2-4: 实现 → pass.**
- [ ] **Step 5: Commit**

> **可由并行 subagent 建。**

---

### Task 7: Runner CLI 接线（hub + degnorm + hub θ 静态条件）+ coexist + re-bless

**Files:**
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`
- Test: `tests/test_topic4_m3_cli.py`

**Interfaces:**
- Runner 新增 CLI：`--hub-frac`/`--corridor-half-frac`/`--hub-long-range-c`/`--l-hub-long`/`--hub-gain`/`--degnorm-alpha`/`--degnorm-scheme`/`--hub-theta-delta`（hub cell 阈值在 degnorm 基线上再叠的量；间期=0、发作=负）。
- 流程：build net 时传 `hub_mask_E`（从 Task 1 region）+ 长程边参数；调 `simulate_kick` 前 `V_th_per_neuron += degnorm_vth_delta(...)`，再对 hub cell `V_th_per_neuron[hub_global] += hub_theta_delta`；region 索引 + 所有新参数写进 `config` provenance。

- [ ] **Step 1: 失败测试** — `--hub-gain` 在 `--help`；coexist smoke：`--hub-gain .. --degnorm-alpha .. --ee-std-u 0.2 --gate-scale 0` 一把跑通，config 里 `hub_gain/degnorm_alpha/degnorm_scheme/ee_std_u` 全在（M1 recovery 不被挤掉）。
- [ ] **Step 2-4: 接线 → pass.**
- [ ] **Step 5: re-bless（若 kick_probe.py 未动则只 connectivity_rot 已在 Task 2 bless；本 task 只动 runner，不需 re-bless）+ coexist smoke + commit.**

---

### Task 8: 预注册 + σ 相图探索（无动力学，M3.0 gate）

**Files:**
- Output: `results/topic4_sef_hfo/m3_hub_scaffold/preregistration.json`、`sigma_phase_map/{out_strength,in_strength,hybrid}.npz` + `figures/sigma_phase_map.png` + `figures/README.md`（中文）。
- Create: `scripts/run_m3_sigma_phase_map.py`

- [ ] **Step 1: 预注册落盘**（出任何 SNN 结果之前）：`preregistration.json` 写死 度归一化三方案比较计划 + 主口径选取规则、Layer-2 容差带（调 Task 5 `subject_tolerance_band` 从 Task-0 cohort_summary 算）、hub/global 数值阈值。
- [ ] **Step 2: σ 相图**（三方案各一张）：建 `corridor_twoend` net，扫 `(degnorm_alpha × hub_gain)` 网格，调 Task 4 `sigma_phase_map` → σ_corridor / σ_crossing 2D + 相变等高线。
- [ ] **Step 3: Gate**：每个 scheme 图里**是否存在**"σ_corridor≳1 且 σ_crossing<1"的间期区，且其邻域有 σ_crossing>1 的发作区。**无此区 → 记录"拓扑不支持门控"，停在 Task 8 报告**（这本身是关键早期发现）。
- [ ] **Step 4: 图 + README**（中文逐图说明 + 关注点）+ commit。

---

### Task 9: U1.0 substrate feasibility pilot（bare corridor SNN，M3.1b gate）

**Files:** Output `results/topic4_sef_hfo/m3_hub_scaffold/u1_feasibility/`。

- [ ] **Step 1: 跑纯走廊**（hub 全关 `--hub-gain 0`、degnorm 关 `--degnorm-alpha 0`、保留 M1 recovery `--ee-std-u 0.2`），`corridor_twoend`，T=8000，`--dump-fullfield --dump-fwd-rev-reps`，扫几个 `(core_mean, drive, sep_frac)`。
- [ ] **Step 2: Gate（§5.1 U1.0）**：自发离散事件率 > 阈值（非 silent/非 tonic）、fwd 与 rev 各 ≥ N_min 干净可读事件、跨事件路线稳定度 > 阈值。**通过才进 Task 10；失败 = "substrate fail" 停报告，不算到 hub。**（⚠️ ⑤双向/刻板在旧 SNN partial/unstable，这是要重新挣的 gate。）

---

### Task 10: 工作点 sanity（degnorm + hub 开，M3.2 gate）

**Files:** Output `results/topic4_sef_hfo/m3_hub_scaffold/op_sanity/`。

- [ ] **Step 1: 跑 degnorm + hub 开**（间期态 `--hub-theta-delta 0`，扫 `--degnorm-alpha`），看是否仍自发点火 + 安静 rest。
- [ ] **Step 2: Gate**：degnorm 抬高均值阈值后，若压死点火 → 在该层 re-tune `--drive`（不在 hub pilot 调）；选有离散事件的 `(alpha, drive)` 为工作点。无 → 停报告。

---

### Task 11: 间期 pilot（hub 关，两层 + 诊断，M3.3 gate）

**Files:** Output `results/topic4_sef_hfo/m3_hub_scaffold/interictal_pilot/`。

- [ ] **Step 1: 跑间期工作点**（degnorm 开 + hub θ 高即 `--hub-theta-delta 0`，长程边开 `--hub-gain>0`），二端触发盆给双向。
- [ ] **Step 2: Layer 1**（调 Task 6 `hub_global_recruitment` + fullfield geometry）：三层都报（全部/可读/不可读）；事件 FINITE（有界、L20≈L32、edge_margin>0、非 tonic、死在枢纽前 = hub/global fraction < 阈值）。
- [ ] **Step 3: Layer 2**（调 Task 5 `layer2_equivalence`）：虚拟 SEEG → Task-0 指标 → subject-level median AF/LR 落容差带内 + AF≥0.75。
- [ ] **Step 4: 诊断**（Task 6 relay timing）：hub 是中继（hub 晚于走廊起点、早于全局区）。
- [ ] **Step 5: Gate**：Layer 1 有界 + 不招募全局区 + 双向保留 → 进 Task 12。

---

### Task 12: 相变 pilot（压低 hub θ → 广播，M3.4 gate）

**Files:** Output `results/topic4_sef_hfo/m3_hub_scaffold/transition_pilot/`。

- [ ] **Step 1: 同工作点压低 hub θ**（`--hub-theta-delta` 负，模拟发作 permissivity）。
- [ ] **Step 2: 判据**：从"事件死在枢纽前"切到"跨 hub + 长程边广播到全局区"（`global_E_spike_fraction` 显著抬升、`global_first_spike_after_hub_ms>0`）；σ_crossing 越过 1 由结构探针预测、SNN 确认。
- [ ] **Step 3: Gate**：相变可复现（间期/发作判别清晰）→ 进 Phase 2。口径锁：synthetic feasibility bridge，不主张临床发作。

---

## Phase 2（pilot 全过 + 用户讨论后再做，本轮不执行）

- **Task 13: 网格 + 消融**：`(θ_hub × hub_gain × degnorm_alpha)` 网格 + spec §6.4 六个消融（去 hub 门槛/去 hub 输出/去长程边/去度归一化/固定慢变量/打乱 hub 位置）+ L 不变性（L20/L32）+ 三度归一化方案对比（主口径按 Task 8 预注册定）。
- **Task 14: verdict + archive**：§8 白话 abstract + 两层结果 + 相变 + 消融表 → `docs/archive/topic4/sef_hfo/m3_hub_scaffold_result_<date>.md`；更新 recap + memory。慢变量 v2（接 `slow_vars` φ/z）也在此阶段。

---

## Self-Review

1. **Spec 覆盖：** U1 基底（Task 1）+ U1.0 gate（Task 9）✓；U2 三手术 = 长程边（Task 2）+ degnorm（Task 3）+ hub θ 静态条件（Task 7，slow_vars v2→Phase 2）✓；U3 σ 探针（Task 4）+ 相图（Task 8）✓；U4 readout 复用（Task 11 Layer 2）✓；两层验收 = Layer 1（Task 11 Step 2，三层报告）+ Layer 2 subject-level 等价（Task 5 + Task 11 Step 3）✓；hub/global 数值判据（Task 6）✓；预注册（Task 8）✓；消融（Task 13）✓；相变 synthetic bridge（Task 12）✓。
2. **Placeholder：** Phase-2 Task 13/14 date `<date>` 是产出时填；`M3_BASE_SHA` 是 Task 0 输出锚；无 TODO/TBD。
3. **Type 一致：** `corridor_regions/hub_mask_E`（Task 1）→ `build_connectivity_rot(hub_mask_E=)`（Task 2）→ `ee_degree/degnorm_vth_delta`（Task 3）→ runner（Task 7）；`recruitment_operator/branching_ratio/crossing_branching/sigma_phase_map`（Task 4）；`subject_tolerance_band/layer2_equivalence`（Task 5）；`hub_global_recruitment`（Task 6）——跨 task 一致；bit-parity 锚 `M3_BASE_SHA` 单一。
4. **bit-parity hazard：** 长程边（`hub_gain=0` 默认）+ degnorm（`degnorm_alpha=0` 默认）+ 静态 hub θ（`hub_theta_delta=0` 默认）三 toggle 分开测 parity（Task 2 + Task 7）；degnorm 零引擎改动（骑 V_th）。
5. **PILOT-FIRST：** Task 8/9/10/11/12 五道 gate 在网格（Task 13）前；预注册（Task 8）在任何动力学结果前。
