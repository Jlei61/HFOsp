# M4-MZ 设计文件：per-neuron adaptation `m_i` + inhibitory efficacy `z_i`

- **状态**：设计锁定（2026-07-18），实现走 red-TDD。本轮 = **discovery（cheap-first → 多 seed 复核）**，不进 40s acceptance。
- **分支 / worktree**：`codex/topic4-mz-slowvars` @ `.worktrees/topic4-mz-slowvars`（从 main HEAD `2d01634` 拉出）。
- **口径级别**：**mechanism screen（探索性）**。所有 phenotype 标签是**检出标签**，不是发作主张。
- **上游同行方法稿**：`/home/honglab/.codex/attachments/.../pasted-text.txt`（§神经元与突触动力学 / §脉冲发放适应 / §活动依赖性抑制耗竭）。
- **公式合同**：`docs/snn_core_model_equations.md` §A（自发衬底）+ §B（慢变量层）。

---

## 0. 朴素话摘要（测什么 / 怎么测 / 什么算成功）

**测什么。** 同行提了一个"最小两慢变量"假说，想解释"间期短暂 HFO 事件如何被推大、变长，最后又能恢复"。两个慢变量都只挂在兴奋性（E）神经元上：

- `z_i`（抑制效能）：一个 E 细胞被持续强抑制轰炸时，它"感到"的抑制会**变弱**（现象学地代表 Cl⁻ 累积 / GABA_A 反转漂移 / 中间神经元疲劳）。抑制变弱 → 事件更容易沿病人特异传播轴摊开、拖长。
- `m_i`（放电适应）：一个 E 细胞每放一次电就累积一点"适应电流"，让它随后更难放电。放得越多、越久，这个刹车越强 → 限制高招募态、并可能让它自己恢复。

**怎么测。** 在**完全相同的病人 E1146 衬底**（narrow montage、template_source 双低阈值核、L=20mm、密度 100/mm²、AR=2、真实电极注册）上，跑四个对照臂：

1. **slow-off**（两个都关）= 唯一基线；
2. **z-only**（只开去抑制）；
3. **m-only**（只开适应）；
4. **z+m**（都开）。

本分支里所有其它慢变量（`q_I`、`g_K`、`S_G` 池、a-shunt、STD）**全部关闭**——否则分不清"同行版机制本身成不成立"。

**什么算成功 / 失败（都要如实报告）。** 用 slow-off 自发间期事件的分布做基准：

- 事件**更长 + 招募更广 + 群体活动更高**、但**没到失控（runaway）** → `expanded_bounded`；
- 上面基础上**事件后能回到基线活动带、且恢复窗内不立刻反弹失控** → `expanded_returned`（最想要的"有界招募 + 恢复"）；
- 也可能得到：全 `fragment`（碎裂）、全 `suppress`（压死）、全 `runaway`（失控）、或 `interictal_like`（跟基线差不多）。

**不预设 z+m 一定成功。** 本轮最重要的不是"找最好参数"，而是看**机制分解**：z-only 把网络推向扩大还是失控？m-only 是缩短/压制事件吗？z+m 有没有出现 z-only 没有的"有界/恢复"招募？

**红线。** 不称任何结果为"发作 / seizure"；不把 runaway 截断片段当 seizure；不从 field 分数反挑参数；本分支**允许轴向保持的扩大招募**（不要求破轴），axis 指标只作描述。

---

## 1. 科学问题与机制链

检验最小 push–pull：

```
z_i ↓  → E 有效抑制减弱 → 间期轴向事件扩大、延长           (push)
m_i ↑（随 E spike 累积）→ 适应电流增强 → 限制高招募态、可能恢复   (pull)
```

四臂对照：**slow-off / z-only / m-only / z+m**。预期检验四个臂的真实 phenotype 差异，**不预设 z+m 必须成功**。

---

## 2. 方程（逐字实现，只对 E 神经元引入 `m_i`、`z_i`）

膜电位（当前 current-based Brunel 引擎，`V_L=0`）：

- **I 神经元（保持原模型，不被 z/m 调制）**：
  $$\tau_m^I \dot V_i^I = -V_i^I + I_i^{I,E} - I_i^{I,I},\qquad i\in I$$
- **E 神经元（新增 `z_i`、`m_i`）**：
  $$\tau_m^E \dot V_i^E = -V_i^E + I_i^{E,E} - z_i(t)\,I_i^{E,I} - \eta_m\,m_i(t),\qquad i\in E$$

符号约定：`I^{X,E}` = AMPA 兴奋电流（正贡献），`I^{X,I}` = GABA_A 抑制电流（负贡献），二者都是**非负幅值**，符号在膜方程里。

**放电适应 `m_i`（仅 E）**：
$$\dot m_i = -\frac{m_i}{\tau_{\mathrm{adp}}} + \sum_k \delta(t-t_i^k);\qquad \text{每个 E spike: } m_i \leftarrow m_i + 1$$
适应电流 $I_i^{\mathrm{adp}} = \eta_m m_i$；$\eta_m$ 控制单 spike 适应电流增量。

**抑制效能 `z_i`（仅 E，$z_i\in[0,1]$，初值 1）**：
$$\tau_z \dot z_i = z_{\infty,i} - z_i,\qquad z_{\infty,i} = H\!\big(I_{\mathrm{th}}^{E,I} - I_i^{E,I}(t)\big)$$
即 $I_i^{E,I} < I_{\mathrm{th}}^{E,I} \Rightarrow z_\infty=1$（恢复）；$I_i^{E,I} \ge I_{\mathrm{th}}^{E,I} \Rightarrow z_\infty=0$（耗竭）。有效抑制 $I_{i,\mathrm{eff}}^{E,I} = z_i I_i^{E,I}$。

**数值合同（承重）**：
1. `z_i` 初值 1，每步 clip 到 `[0,1]`。
2. `m_i` 初值 0，数值不得 `< 0`（clip `max(m,0)`）。
3. z/m **只作用 E**；I 细胞不受 z/m 调制（I 细胞 `z≡1, m≡0`）。
4. `threshold()` **原样返回** `V_th_per_neuron`（双低阈值核心保留）。适应是**电流** `-η_m m`，**不是**阈值移动（区别于旧 `phi`）。
5. 第一版严格 Heaviside（`z_inf = 1.0*(I_I < I_th)`，等号处 = 0，符合"`≥`→0"）。平滑阈值只能作独立 sensitivity，不混入主实现。
6. 不加 `m_max` / `z_min` / ictal sensor / 手工状态开关 / seizure gate；不根据结果在线开关慢变量。
7. Off-by-default 字节奇偶：`use_z=False and use_m=False` ⇒ `apply_currents` 返回 `I_E - I_I`（与 `slow=None` 逐字节一致，见 §4）。

---

## 3. 状态变量作用对象（谁被改、谁不被改）

| 变量 | 作用群体 | 进入膜方程处 | 初值 | 边界 | 驱动 |
|---|---|---|---|---|---|
| `z_i` | **仅 E** | `-z_i I_i^{E,I}`（缩放抑制） | 1 | `[0,1]` | `z_inf=H(I_th_EI - I_i^{E,I})`，本步 I_I |
| `m_i` | **仅 E** | `-η_m m_i`（减性适应电流） | 0 | `≥0` | 每 E spike +1，`τ_adp` 衰减 |
| I 细胞 | — | `I_E - I_I`（不变） | — | — | — |
| `V_th_per_neuron` | 全体 | `threshold()` 原样返回 | 衬底 `vth`（双核） | — | — |

E 细胞在 `[:NE]`、I 在 `[NE:]`（`labels==0` 为 E）。core/surround 掩码用几何法从 `posE` 派生：`core = (‖posE−src_xy‖≤1.5) | (‖posE−snk_xy‖≤1.5)`（双核并集），`surround = ~core`（只在 E 内）。

---

## 4. 引擎集成（钩子 / 字节奇偶 / 不 re-bless / 种子）

**驱动器 = `simulate_kick`（`src/snn_engine/kick_probe.py:91`）**，非 `model.simulate`。慢变量协议 = 3 方法，钩子处：

- `apply_currents(self, I_E, I_I, labels=None, I_E_rec=None)` → `I_net`（`kick_probe.py:295`；本模块不用 `I_E_rec`，但签名要收）。
- `threshold(self, V_th_base)` → `V_th_eff`（`:302`；`base_vth = V_th_per_neuron`，原样返回骑双核）。
- `step(self, spk, labels, dt)` → None（`:343`；本步 spikes 推进内部态）。
- 本模块**不实现** `uses_shunt`/`shunt_g_at_E`/`nE` → 引擎自动走字面膜路径 `Vtmp = I_net + (V - I_net)*decay_V`（`:334`）。机制全走 `I_net` + `threshold`。

**字节奇偶（test 1 承重，已由算术保证）**：`slow=None` 走 `membrane_step(shunt_gaba=False)`（`kick_probe.py:83-85`）= `I_net = I_E - I_I; return I_net + (V-I_net)*decay_V`。MZ 两关时 `apply_currents` 返回 `I_E - I_I`、`threshold` 返回同一 `V_th_per_neuron`、`step` 无操作且**不消耗 RNG** → 每步 `I_net`/`V_th`/RNG 序完全相同 → 整条轨迹逐字节一致。仍写 full-`simulate_kick` parity 测试实测验证。

**不 re-bless**：`engine_versions.json` 只 guard 6 个核心文件（`kick_probe/params/model/connectivity/connectivity_rot/lfp`）。新模块放 `src/snn_engine/mz_slow_vars.py`（**不在 guard 列表**），**不改**这 6 个文件、**不调** `record_versions`。runner 里 `assert_versions`（只读）会通过。provenance 里记录这 6 文件的 sha256 作证据，但**不写** `engine_versions.json`。

**种子（每臂只差 slow-config）**：`build_substrate(seed)` 定网络（`place_neurons`+`build_connectivity_rot`+core 阈值 `seed+7/+8`）。噪声种子在每次 `simulate_kick` 前 `net["rng"] = np.random.default_rng(S["seed"])` 重置。一个 seed 建一次 `S`，四臂共享（fork-COW），每臂重置 `net["rng"]` 到 `S["seed"]` → 衬底 + 噪声实现完全相同。

**衬底调用**：`S = build_substrate(seed)`（E1146/narrow/template_source/twoend_equal/L20/dens100/AR2 全部 baked-in），`S["p"].T = T`。自发运行：`simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=S["vth"], slow=MZSlowVars(...), early_stop_runaway=True)`。

---

## 5. 模块设计 `src/snn_engine/mz_slow_vars.py`

```
@dataclass
class MZSlowVarsConfig:
    use_z: bool = False           # off-by-default → byte parity
    use_m: bool = False
    tau_z: float = 5000.0         # ms   (calibration; placeholder)
    I_th_EI: float = 0.0          # E-cell GABA current depletion threshold (calibration)
    tau_adp: float = 2000.0       # ms
    eta_m: float = 0.0            # adaptation current per unit m (calibration)
    record_calib: bool = False    # slow-off 观察模式：额外记录 I_I[E]/I_E[E] 直方图
    calib_hist_edges: np.ndarray | None = None   # 直方图边（探针预定）

class MZSlowVars:
    def __init__(self, N, V_th0, cfg=None, *, NE, core_mask_E, surround_mask_E): ...
    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None) -> I_net: ...
    def threshold(self, V_th_base): return V_th_base
    def step(self, spk, labels, dt): ...
```

**apply_currents**：
- 两关：`return I_E - I_I`（精确字节奇偶）。
- 否则：`inh = self.z * I_I if use_z else I_I`（`z[I]≡1`）；`I_net = I_E - inh`；`if use_m: I_net = I_net - eta_m * self.m`（`m[I]≡0`）。
- `record_calib` 时把 `I_I[E]`、`I_E[E]` bin 进本步直方图（承重：side-effect，不改返回值 → 不破坏字节奇偶）。

**step**：
- `use_z`：`z_inf = (self._I_I_last < I_th_EI).astype(float)`（`_I_I_last` = 本步 apply_currents 存的 I_I）；`z[E] += (dt/tau_z)*(z_inf[E]-z[E])`；clip `[0,1]`。
- `use_m`：`m[E] += (-m[E]/tau_adp)*dt`；`m[spk & is_E] += 1.0`；`m = max(m,0)`。
- 记录审计 traces（见下）。early-stop 截断帧处理：break-frame 值写入，post-stop 帧视为缺失（不当 0；§6 pitfall 1）。

**审计 traces（每步 append 标量，除直方图外内存 ~O(nsteps)）**：
`z_mean, z_min, z_core_mean, z_surround_mean, m_mean, m_max, m_core_mean, m_surround_mean, I_EI_E_mean/p90（E 抑制电流摘要）, adap_current_mean=η_m·mean(m[E]), rate_E_step, rate_I_step`。
`record_calib` 额外：`hist_I_EI[E]`、`hist_I_EE[E]` 每步直方图（仅 slow-off 用，逐 seed 处理后丢弃）。**边分箱=log-spaced `[0, ~5000]` mV 400 bins**（承重：E-cell `I_I`/`I_E` 事件相重尾到几百 mV——引擎每抑制 spike 注入 `~w·τ_m/τ_r≈12` mV，线性 `[0,100]` 会截断 ~25% `I_I` 使分位数偏低；log 边低端细分辨中位 ~1 mV、高端吃住尾巴，overflow≈0）。
**不保存完整 N×T 浮点电流矩阵。**

---

## 6. 参数来源与标定（`results/topic4_sef_hfo/mz_slowvars/calibration.json`）

同行稿**未给** `tau_z / I_th_EI / tau_adp / eta_m` 数值表（`\ref{tab_model_parameters}` 不在稿内；`slow_vars.py` 现值是 placeholder）。故走**baseline-only 分位数**路径。**只用 slow-off baseline，不得根据 z+m 结果反调。** 阈值离散/合成 fixture 校准（anti-circularity，仿 `sef_hfo_m4_termination`）。

1. **slow-off baselines**：seeds 1/3/4，T=8s，`slow=MZSlowVars(use_z=F,use_m=F,record_calib=T)`。得 `E_spk_bool, rate_E, af` + 观察直方图。（先跑 T=2s/seed1 探针定直方图边。）
2. **baseline-anchor gate（承重前置）**：`floor=P95(af[5,50]ms)`；`bar=floor+0.5*(af.max()-floor)`；`detect_events(MIN_DUR_MS=8, RETURN_FRAC=0.2, SETTLE_MS=50)`。要求每 seed ≥ `MIN_BASE_EVENTS=3` 个 **returning** 事件。**均质衬底 slow-off 是 0 事件（pure R0）——若 E1146 也 0 事件则无法刻画基线相，clean no-go at baseline，先报再停，不硬扫。** 某 seed 事件不足 → 该 seed baseline insufficient（**不借别 seed 阈值**）。
3. **`I_th_EI` 分位数**：把观察直方图在**事件步**（`[t_on,t_off]`）上池化（跨 clean seeds），取 `q50/q75/q90`。方向：z 在 `I_I ≥ I_th` 耗竭 → **q50=强耗竭、q75=中、q90=弱**。
4. **`tau_z`** 初检：`{2500, 5000, 10000}` ms。
5. **`tau_adp`** 初检：`{500, 2000, 5000}` ms。
6. **`eta_m` 离线 replay**：对每个 `tau_adp`，用 slow-off `E_spk_bool` replay m-ODE → 每 E 细胞 `m(t)`；`peak_m(tau_adp) = P95(事件窗内每细胞 max-m)`（跨 clean seeds 池化）。兴奋尺度 `I_EE_scale = P90(事件步 I_E[E])`。`eta_m_{frac} = frac × I_EE_scale / peak_m(tau_adp)`，`frac∈{0.05,0.10,0.20}`（低/中/高）。（→ eta_m 依赖 tau_adp。）
7. **写 calibration.json**：全部规则（逐字 recipe）+ 每 seed baseline 事件统计 + anchor 判决 + `I_th_EI{q50,q75,q90}` + tau_z/tau_adp 候选 + `peak_m(tau_adp)` + `I_EE_scale` + `eta_m` 表（tau_adp×frac）+ **arm-C 预注册选择规则**（§7）。**u_n0 教训**：set-point 从**安静间期**分布标，绝不从持续/发作态标（否则 `[I_I−I_th]_+≈0` 静默关掉机制）。

---

## 7. 实验 arms + discovery 网格（cheap-first，seed=1 起）

**Arm A — z-only**（m off）：`I_th_EI∈{q50,q75,q90} × tau_z∈{2.5,5,10}s` = **9 cells**。
**Arm B — m-only**（z off）：`tau_adp∈{0.5,2,5}s × eta_m∈{low,mid,high}` = **9 cells**（eta_m per tau_adp）。
**Arm C — z+m**（**不做全 4 维笛卡尔积**）：
- **预注册选择规则（写入 calibration.json，运行 C 前锁定）**：
  - 从 Arm A 9 cells 按**实测 z 耗竭**（run 内 `z_min` 全局最小值）排序，取三档 = 最接近目标耗竭 `{0.8, 0.5, 0.2}` 的 cell（weak/mid/strong）。
  - 从 Arm B 9 cells 按**实测适应电流峰**（run 内 `max(adap_current) / I_EE_scale`）排序，取三档 = 最接近 `{0.05,0.10,0.20}` 的 cell（weak/mid/strong）。
- Arm C = 3×3 = **9 cells**（z-config × m-config）。

**discovery T = 8–15s，seed=1**。先跑单 cell RSS 审计再定并发（≤2–3 full-density workers，OMP=1，fork-COW，RAM-gated；swap 满=OOM 无缓冲）。early_stop_runaway=True（runaway cell 截断省内存；bounded/returned 事件不触发早停跑满）。

---

## 8. Phenotype 合同（新标签集；轴向保持可接受）

**唯一基线 = slow-off 同 seed 自发间期事件分布。** 覆盖旧"必须破轴才算 ictal-like"门槛——本分支问题是"发作早期能量增强是否仍沿间期 scaffold"，故 **axis-preserving expanded recruitment 是目标之一**；axis score / off-axis fraction / globality **全部保留为描述性指标，不作分类门**。

标签（7 类）：`interictal_like / expanded_bounded / expanded_returned / fragment / suppress / runaway / insufficient`。

per-run 度量（复用 §5 primitives + `sef_hfo_events.detect_events` + `run_m4_dynamic_qi._first_sustained`）：事件时长、招募/参与（af）、群体活动峰（rate_E Hz）、runaway_ms、returned（`event_recovery`）。

判定树（阈值合成-fixture 校准、锁结构）：
1. `runaway_ms is not None` → **runaway**（注入式，形状无关；`RUNAWAY_HZ=120, DUR=100ms, 80% 滑窗, 20ms 平滑` — `run_m4_dynamic_qi.py:64,144`）。
2. slow-off 同 seed baseline 不足（`<MIN_BASE_EVENTS`）→ **insufficient**（不借别 seed）。
3. 否则取本 run 最大事件 `peak_run`，与 baseline 分布比：
   - **expanded** ⟺ `dur > f_dur·dur_hi` **且** `participation > f_part·part_hi` **且** `peak_rate > f_act·act_hi`（三者 AND；`*_hi` = baseline P90，`f_*` 合成校准 ~1.0–1.5）。
     - **returned**（`event_recovery`：peak 事件后 `[t_off, t_off+t_return]` 均值 ≤ baseline+m·σ_base，且余程无 runaway）→ **expanded_returned**；否则 → **expanded_bounded**。
   - **not expanded**：
     - peak af < baseline floor（活动被压死）或 0 事件 → **suppress**；
     - `n_events ≥ f_frag·base_n` 且 max 事件 dur < baseline 中位（碎裂多短）→ **fragment**；
     - 其余（跟基线相当）→ **interictal_like**。

**expanded 三条件（时长 ∧ 招募 ∧ 活动）AND、且 not runaway；returned 额外要求回基线带 + 恢复窗无反弹 runaway。** early-stop 前片段**不**改标为 expanded。

---

## 9. 输出目录

```
results/topic4_sef_hfo/mz_slowvars/
├── calibration.json          # §6 全部规则 + 值 + arm-C 预注册选择规则
├── discovery_summary.json    # arms A/B/C 汇总 + meta
├── per_run.jsonl / per_run.csv
├── per_seed/                 # P3 多 seed per-cell JSON/npz
├── readout_ready/            # 仅 bounded candidate：LFP/contacts+coords/envelope/onset/激活代理/z,m traces/per-neuron onset/scaffold fingerprint/params
└── figures/
    ├── README.md             # 中文，图渲染后写
    ├── mz_phenotype_map.png
    ├── mz_mechanism_traces.png
    └── mz_spatial_recruitment.png
```

---

## 10. 停止条件（stop conditions）

- **baseline-anchor 失败**：E1146 slow-off 全 seed 0 事件 / 无 returning 事件 → 报"基线相不可刻画"，**停**，不扫 arms。
- **E1146 数据不可达**：`build_substrate` 读不到 `results/interictal_propagation_masked/.../epilepsiae_1146.json` 或 geometry `_t_a/_t_b` → 停并问用户（AGENTS.md stop condition）。已确认主 checkout 存在 → 通过。
- **RSS/OOM**：单 cell RSS 审计后 workers 超安全线 → 降并发或串行；swap 满则停。
- **本轮到多 seed 复核止**：不自动进 40s acceptance；只有存在跨 seed bounded candidate 才在报告里**提议**下一阶段。

---

## 11. Claim boundary（允许 / 禁止）

**允许说**：
- "在 E1146 衬底上，slow-off 是否保留稀疏可恢复间期事件基线（是/否）"。
- "z-only / m-only / z+m 各产生什么 phenotype 分布（按 7 类如实报）"。
- "z+m 是否出现 z-only 没有的有界/恢复招募（检出标签层面）"。
- "（描述性）z+m 是否仍保留病人特异轴向传播 vs 变成同步全场"。
- "axis-preserving expanded recruitment 是本分支的检出目标"（分支内口径）。

**禁止说**：
- ❌ "模型已复现完整 seizure / 已证明发作机制"。
- ❌ 把 runaway 早停截断片段称 seizure / expanded。
- ❌ 从 field concordance 分数反挑参数；把 activation proxy 称真实 1–150 Hz broadband energy。
- ❌ 把单病例模型与 Fig3-B best-case seizure 配对后称"独立验证"。
- ❌ 把本分支"轴向保持可接受"口径当成对 topic4 主文档"必须破轴"框架的修改——这是**分支内探索口径**，主文档框架不动。
- ❌ 根据结果在线开关慢变量 / 加 seizure gate。

---

## 12. 交付清单

branch/worktree；改动文件；测试命令+结果；实际运行命令；calibration/discovery/figures/report 路径；当前科学 claim 与禁止 claim；git status。**不改论文 Methods，不 push，不删旧 M3/M4 结果。**

最终报告 `docs/archive/topic4/sef_hfo/mz_slowvars_discovery_2026-07-18.md` 回答 8 问 + 明确三选一（40s acceptance+field readout / 只留 recruited-state screen / clean no-go），建议须由 artifact 支撑。
