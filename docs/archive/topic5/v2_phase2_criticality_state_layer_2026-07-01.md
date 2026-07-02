# Topic 5 V2 Phase 2 — 发作前"变脆状态"是否投影到间期 HFO 几何（criticality/state 层）

date 2026-07-01 · 状态：**EXPLORATORY，preliminary（pending Phase-1 空间/顺序 null + Gate A）** · 分支 `topic5-v2-phase2`（隔离 worktree，off `topic5-v2-phase1`）

> **降级定位（2026-07-02 加）**：本结果是 **restricted axial preictal-only sanity check**——在"只看 −120~0s、只用两段窗、限于 HFO 几何匹配触点、无显式非轴向假设、按 relt=0 而非 eeg_onset 锚定"的受限实现下没得到 state 支持。**它不判定**模型真正的预测（发作早期 **轴向组织减弱 + 非轴向活动/流/模态放大**）。后继设计见 `docs/superpowers/specs/2026-07-02-topic5-v3a-mode-transition-design.md`（V3a data-side mode transition）。**可保留的耐用产出 = 动力学腿的方法学定律**：raw λmax≈0.95 被 phase/block surrogate 解释掉 → 今后所有 λmax/VAR/DMD/Jacobian 一律报 `λ_surplus`（观测 − surrogate 中位数），不报 raw。**不能写的 claim**：不能说"没有临界性 / 发作前没有 state projection"；不能把 λmax≈0.95 当临界；不能把 avalanche rank-coupling 当传播；不能把 −120~−30 差分当 onset-proximal transition；不能把 axis-only M_loading 当全脑模态。

> 归档定位：这是 V2 三层框架里 **state 层** 的探索性执行报告。主结论目前是**偏阴性/弱**，且按设计 Phase 2 单独不能下"致痫临界通路"结论——那需要 Phase 1 的三道门（Gate A/B/C）先过。数值表见 §4；主文档只引摘要 + 本链接。

---

## 0. 朴素话回顾（测了什么 / 怎么测的 / 揭示了什么）

**测了什么。** 病人在两次发作之间，那些短暂的高频异常放电总是**按一条固定的先后顺序**在电极间传开——像一条走熟的小路（我们把这条"顺序地图"叫 G_HFO，它是固定的、来自间期高频事件的传播排名）。这一层问的是：**真正发作之前的最后约两分钟里**，大脑几种"快要失稳的迹象"会不会也**沿着这同一条小路**排布。三种迹象：(1) 每个电极的信号起伏是不是变大、变慢、变毛躁（"变脆的地图"）；(2) 把这些电极曲线拟合成一个简单的联动模型后，最主导的那个"晃动模式"的权重是不是压在小路的某一端，同时整个系统的"最不稳定程度"是不是随着逼近发作在升高；(3) 把能量突然拔高的时刻当作"点亮"，这些点亮的连锁传递是不是**顺着小路往前走**。

**怎么测的。** 用发作前最后约 2 分钟的宽带能量起伏（1–45Hz，每个电极一条曲线，0.1 秒一个点）。对每种迹象都算一个"跟小路对齐的程度"，然后**跟'打乱后的随机版本'比**——如果这些迹象只是"平滑的空间场碰巧长得像小路"或"只是同一个电极自己一直亮着"，随机打乱后应该也差不多。动力学那一层的两种随机对照（打乱时间块、打乱相位）我们能自己做，已经做了 1000 次；另外两层随机对照（保留电极几何的空间打乱、保留每触点放电率的顺序打乱）依赖 Phase 1 还没建好的工具，**暂时挂着（`pending_phase1`），不会假造一个 null 顶上**。每个病人是一个独立单位，broad / narrow 两套电极集从不混在一起。

**揭示了什么。** 在整个队列看，三种迹象沿这条小路的**方向一致性都接近零**——而且不是"有人正有人负平均掉"，是每个病人自己也普遍很弱。最能说明问题的是连锁传递：**顺着小路往前走的净流量几乎为零**（队列中位数 ≈ 0）；但如果错用"自相关式"的指标，反而会看到 0.64–0.91 的高值——那是"同一个电极自己反复亮"造成的假象，不是真往前传。这**正好印证了当初设计坚持用'净前向流量'而不是'自相关'当主指标**（否则会假阳性地宣称"强烈沿轴传播"）。动力学模型的"最不稳定程度"确实都贴近临界（谱半径 0.90–0.95），但这是宽带能量被平滑之后本来就会有的假象，随机对照压不下去（这也是设计里早就点名的坑）。所以**在我们能看的这个尺度上：发作前的"变脆状态"看起来不像特别沿着那条间期 HFO 小路排布**——这是一个探索性的偏阴性结果。**注意**：Phase 2 单独本来就不能宣称"这条路是致痫临界通路"，那要等 Phase 1 的能量表达层三道门先过（§5 honest coupling）。

（内部归档代号：G_HFO = 间期 HFO typical_rank 顺序场；state 层 = K_t 易感场 / M_loading 主模态 loading / avalanche ATM forward-displacement；state_band = legacy_bb_1_45；null = spatial_constrained_permute + order_null_rank_pair（pending Phase-1 T8/T9）+ phase/block surrogate（已做）；evidence ladder §1.1 Gate A/B/C。）

---

## 1. 这一层在 V2 里的位置 + honest coupling

V2 三层：**trait**（固定的间期 HFO 几何 G_HFO，候选病理模态）/ **state**（随时间变的临界性/易感性场，本报告）/ **expression**（发作时哪个频带沿 G_HFO 放大 = Phase 1）。

**honest coupling（先读）**：Phase 2 是 §1.1 证据阶梯的 **state leg**，探索性。合并的"pathological critical mode"主张需要 Phase 1 的 **Gate A/B/C** *加上* 本层的 state 投影。**若 Phase 1 Gate A 不过，Phase 2 最多只能写成"发作前易感性动力学可能组织在间期 HFO 网络上"，绝不能写成"HFO 几何是致痫临界模态"。** 无 forecasting；Phase 2 单独不能下 cohort 级 critical-mode 结论。

---

## 2. 数据与方法

- **队列**：broad 9 例、narrow 7 例（Epilepsiae SEEG；`SUBJECTS_BY_SUB`）。每例 ~10–20 个几何匹配触点。
- **state 场**：`ictal_field_long_cache` 的 `bb_zt`（1–45Hz 宽带功率 baseline-robust-z，hop=0.1s），**只取发作前段（relt<0）**，span `[-120,0]s`。`available_pre_sec < 90s` → 跳过（记 `status=skipped`，非阴性）。
- **发作数上限**：per-perm 重拟合 null 计算量大，对发作极多的被试（如 916：44 次、1084：70 次）取前 12 次（`max_seizures=12`），`n_seizures_total` 透明记录（非静默截断）。
- **三条 leg**（subject 为单位，window→seizure→subject 中位数聚合）：
  - **2A 易感场 K_t**：每触点 late(`[-30,0]`)−early(`[-120,-90]`) 的 {variance, lag1_autocorr, **line_length_rate（primary）**} 变化 → `contact_alignment(field, G_HFO)`。
  - **2B 动力学**：preictal span 上滑窗（10s/5s）ridge-VAR(1) → 谱半径轨迹 + 最新窗主模态 `|leading_eigvec|` → `M_loading = contact_alignment(|loading|, G_HFO)`；`lambda_trend = spearman(λmax, 窗中心)`。VAR 意义门：`cv_one_step_r2 > 0` 才 `var_meaningful_flag=True`。**两种 surrogate null（相位 + 时间块），各 1000 次**。
  - **2C avalanche**：late-preictal 阈值化（z>2）→ 汇总 ATM → **primary = `atm_forward_displacement`（净前向流量）**；`atm_direction_index`（稳健性）；`atm_rank_coupling_spearman`（**仅描述性**，会混淆自持与流向）。无 power-law exponent。
- **null 纪律**：K_t / ATM 用 空间 + 顺序 null（Phase-1 T8/T9，**pending**）；M_loading / λtrend 用 相位 + 时间块 surrogate（**已做**）。`state_leg_supported` 需 `within_shaft_strong` 空间 null + 非 `weak_downgrade` 顺序 null + 至少一条 leg null 显著 → **null 挂着时恒为 False**（不越 Gate A 升级）。

---

## 3. 关键方法学验证：forward-displacement vs rank-coupling（真数据印证）

真数据上，avalanche 连锁传递的**净前向流量 `atm_forward_displacement` ≈ 0**（每个被试都接近零），但**自相关式 `atm_rank_coupling_spearman` = 0.64–0.91**（很高）。若误用后者当主指标，会假阳性地宣称"连锁强烈沿 G_HFO 往前传"。实测证明这个高值来自"同一批触点反复自己亮"（自持），不是前向流动。**这印证了 plan Patch-2 坚持 forward-displacement 为 primary、rank-coupling 仅描述性的决定。**

---

## 4. 结果（observed = final；per-subject null p = 部分 pending）

**队列 SIGNED 对齐中位数（broad，observed）**：K_t(line_length_rate) −0.06 · M_loading −0.06 · λtrend +0.03 · atm_forward_displacement −0.004。
**narrow**：K_t(line_length_rate) −0.06 · M_loading −0.17 · atm_forward_displacement（见 CSV）。
→ 三条 leg 的方向一致性都接近零；per-subject K 的符号在被试间不一致（lag1_autocorr K 从 −0.77 到 +0.69），非"平均掉"。

**动力学 λmax_late ≈ 0.90–0.95（所有被试）**：贴近临界，但为宽带平滑的 AR1 假象 → 相位/时间块 surrogate 是判读关键。`var_meaningful_flag=True`（cv_r2 ~0.83–0.90，几乎不筛选——宽带 envelope 本就平滑；**因此 M_loading/λmax 的点估计只当描述量，判读一律看 surrogate p，不看点估计**，见审阅 Important-2）。

> **判读天花板（审阅 Important-1，必须遵守）**：易感场 K_t 与 avalanche ATM **目前没有任何 null**（空间/顺序都 `pending_phase1`）。**任何非零的 observed K/ATM 都可能纯由空间自相关 / HFO-rate 地形造成，不能读作"沿 G_HFO 对齐"的证据。** 现在唯一能支撑的结论是"队列 signed 中位数接近零 → 弱/偏阴性"；具备真实（时间）null 的只有动力学 leg，且其 surrogate 检验（如 139 M_loading p≈0.25/0.75）返回不显著。**整层框成"暂时偏弱/偏阴性"是稳的，前提是主要靠动力学的 surrogate 检验说话、把无 null 的 K/ATM 幅度只当描述量。**

- 完整 per-subject 表：`results/topic5_ictal_recruitment/v2_criticality/{broad,narrow}/phase2_{susceptibility,dynamics,avalanche}_subject.csv` + `phase2_criticality_summary.csv` + `phase2_criticality_cohort.json`。
- **动力学 surrogate null（n_perm=1000）= 干净阴性**：要求相位 AND 时间块两种 surrogate 都 < 0.05 时，主导动力学模态对齐 G_HFO 的 M_loading **broad 0/9、narrow 0/7 显著**；λmax 随逼近发作升高的 λtrend 也 **broad 0/9、narrow 0/7 显著**（个别被试单侧 nominal 命中——如 253/1077 的相位 surrogate p=0.036/0.009——被"两种 surrogate 都要过"的规则正确滤掉，属 16 被试多重比较下的偶然）。所有被试 `var_meaningful_flag=True`。**这是唯一有真实（时间）null 的 leg，结论是：发作前主导动力学模态没有稳健地投影到 G_HFO，λmax 也没有沿几何升高。**
- **易感场 K_t 空间/顺序 null、avalanche 空间/顺序 null**：`pending_phase1`（Phase-1 band_scan T8/T9 未建）。**但队列级近零的 signed 中位数已独立指示弱 state-leg，不依赖 per-subject 幅度 null。**

`state_leg_supported`：broad 0/9、narrow 0/7（null 挂着时按定义为 False；非"证伪"，是"还没到能支持的门槛"）。

---

## 5. 局限 + 待办

- **Phase-1 依赖**：空间/顺序 null（K_t、ATM）+ 频带表达层 Gate A/B/C 都在 Phase-1；Phase-1 当前 stalled 于 Task 1。Stage C 将在 Phase-1 落地 `spatial_constrained_permute`/`order_null_rank_pair` 后接线 + TDD + 重跑（`get_null_fns()` 自动切换，`_apply_nulls` 现 raise NotImplementedError 占位）。
- **λmax→1 假象**：宽带 envelope 本身生成强 AR1，λmax 接近 1 不等于临界；只有 surrogate 压得下去才谈"接近失稳的空间模态"。
- **n 小 + preictal ≤300s 非平稳**：整层 exploratory，~10 触点 VAR 脆弱；不做 forecasting。
- **contact_alignment 经 contract-shim（plan Task-5 逐字）**：确定性相关，与真 Phase-1 输出一致（非 provisional）；Phase-1 落地后 `get_contact_alignment()` 自动切换。

---

## 6. 复现 + provenance

```bash
# 隔离 worktree /home/honglab/leijiaxin/HFOsp-t5v2p2, 分支 topic5-v2-phase2
for ax in broad narrow; do
  python scripts/run_topic5_v2_crit_susceptibility.py --substrate $ax     # observed K_t
  python scripts/run_topic5_v2_crit_dynamics.py       --substrate $ax --n-perm 1000
  python scripts/run_topic5_v2_crit_avalanche.py      --substrate $ax     # observed ATM
  python scripts/run_topic5_v2_crit_summary.py        --substrate $ax
done
pytest tests/test_topic5_v2_criticality.py tests/test_topic5_v2_crit_io.py \
       tests/test_topic5_v2_crit_dynamics.py tests/test_topic5_v2_crit_legs.py -q
```

**并行分工（2026-07-01，用户离开 8h 自主执行）**：并发 session `1af452a4` 做 Phase 1（band-scan，main tree，分支 `topic5-v2-phase1`，声明 Phase-2 非其 scope）；本 session `4d3ab1ba` 做 Phase 2（隔离 worktree，分支 `topic5-v2-phase2`）。两分支独立文件、可干净合并。见 memory `project_topic5_v2_phase2_session_split_2026-07-01`。
