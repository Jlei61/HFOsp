# SEF-HFO 虚拟 SEEG 观测层 — Increment-3a rate parity + 电极尺度合同 + SNN-cm 复盘 (2026-06-07)

并行加速段归档（用户 2026-06-07 "并行做不冲突的、加速虚拟电极构建"）。本轮在 SNN-cm
单条件结果之外，把**观测层主基底（LIF rate field, spec §13.1 锁定的 cm 尺主路）**铺到可跑
的程度，并把电极尺度从终端输出固化成可复现合同。

Plan: `docs/superpowers/plans/2026-06-07-sef-hfo-virtual-seeg-observation-increment3.md`。
前序：Increment-1 (commit 12c4b64…) + Increment-2 SNN Option-B (commit c5625cb)。

---

## 0. 朴素话三段（测了什么 / 怎么测的 / 揭示了什么）

**测了什么**：我们有一个会"自己点着、沿某个方向传一段、再熄灭"的神经组织模型（rate field）。
我们想验证：把虚拟电极插进去、用跟真实病人数据**一模一样**的读取流程，能不能把模型里"沿
45°方向传播"这件事如实读出来——而且要在两个独立模型（spiking SNN 和 rate field）里都读得
出来，才算这套观测工具可信。

**怎么测的**：在 rate field 里某一端"点一下"（确定性脉冲），让活动沿连接各向异性方向传开；
在源空间上摆一圈虚拟电极（真实 SEEG 间距 4mm，3 根杆不平行），把每个触点看到的活动包络抽出来，
走 `event_window_for_run → extract_lagpat → endpoint_centroid_axis`（与真实 pipeline 同一套），
读出"源→汇"主轴，跟真实点的连接方向 θ_EE 比。各向同性连接（无方向偏好）作为负对照——如果完全
没有方向，应该读不出稳定轴。

**揭示了什么（2026-06-07 收尾，见 §6 —— 本段下方"撞墙"叙述已被 §6 取代）**：rate field 的虚拟电极
在**全部四个对照**下都干净读回了传播方向：连接方向取 0/45/90° 都读对（误差 1.1/3.3/1.1°）、踢点
左右偏移不把方向拖走（20.5/3.3/20.5°）、电极杆整体旋转 0/30/60/90° 不把方向拖走（≤7.3°）、各向
同性连接（无方向偏好）采够触点却读不出稳定方向（n_part=12、无可读轴 = 正确的负对照）。这套观测路
在挑选过的 4mm 三杆几何下可用。**但 LFP 边界要收住**：这是 rate field 的 firing-rate-density 包络
读出，**不**等于"真实 4mm 尺度 SNN 虚拟电极可用"——SNN cm 复核仍点不着（卡在 SNN 点火，不是 LFP
forward），current-LFP 只在 2mm 小尺度真实点着的 SNN 事件上验过。

> **STATUS UPDATE 2026-06-07**：早先（§3, commit 70f0b81）四对照"撞几何墙"的结论已被 **§6** 取代。
> 三处检测-前端设计修复（事件窗用源场活跃比例、participation margin=10%、电极杆改对称 15/75/135°）
> 让四对照全过；这两个前端口径已锁进 spec §4.2 + sensitivity 证明它们移动的是覆盖、不是方向读数。
> §3 保留作历史记录（70f0b81 那一版用 +θ_EE 端踢 + 50% margin 的"撞墙"是真实的，被前端修复解掉）。

---

## 1. SNN cm-scale 单条件复盘（item #1）— 结论 = readout UNVALIDATED（非 failed）

终端 run 完成 + 图 + stdout 已审（`results/topic4_sef_hfo/observation_layer/inc2_cm_raw.npz`
+ `figures/inc2_cm_currentLFP.png`）。**θ_EE=45° 没有被 current-LFP readout 恢复**，但根因不是
读出坏了，而是**事件压根没点着**：

- `max_active_frac`(踢后 50ms) = 0.000；dense oracle 各向同性（ratio≈1.2）；n_part=1。
- 根因：`R_KICK=0.15mm × density=100 → 踢盘内只有 ~6 个 E 神经元`（可用的 2mm smoke 用 density
  1800–4000 → 100–226 个）。亚临界驱动(0.6)下 6 个种子点不着火。
- **措辞精确**：current-LFP 前向模型（engine `lfp.py` |I_E|+|I_I|）本身**没被测过**（没有事件可读）
  → readout = **UNVALIDATED**，不是 failed；不要据此判 LFP 模型坏了。
- per-contact baseline z-score readout + excess-front oracle（advisor 2026-06-07 修）工作正常——正是
  它给出"是点火失败、非读出失败"这个干净诊断。

**不重跑**：spec §13.1 锁 SNN-cm = 后调（rate@cm 是主路）；Option-B 锁 "grid 不行就停+报告、用户
gate"。item #1 要求的是"审完"（已审，结论=没点着），不是"重跑到成功"。仅种子修复(R_KICK↑)解决点火
但不解决更深的"事件跨度够不够铺满 montage"，大概率第三次模糊 block。**种子假设 + 是否重跑 = 用户
gate decision**（见 §4）。

---

## 2. Increment-3a 基础设施（item #3）— Tasks 1–3 DONE (TDD, committed)

commit `22430f9`。3 件加性、可复用、与事件尺度无关的纯代码（不碰 gitignored SNN engine）：

- **Task 1 `integrate_lif_field(..., return_frames=True)`**（`src/sef_hfo_lif.py`）：逐步 `rE` 帧栈
  `(nsteps,n,n)` 作为返回元组**最后一个元素**追加（**RETURN-TUPLE POSITION LOCK**，在所有 optional
  之后：peak_field/rE → ext_coh → axis → **rE_frames**）。default False 字节级不变（27 个 lif/events
  测试不动）。`rE_frames[-1]` == `return_field=True` 最终快照。审计了全部 9 个既有 call site，无一传
  return_frames → 全部不破。
- **Task 2 `pulse_stim_fn`**（新 `src/sef_hfo_rate_adapter.py`）：给 `integrate_lif_field` 的有限圆盘
  脉冲 `stim_fn(t)`（`[t_on,t_off)` 内 = `amp`·disk(center,radius)，外=标量 0.0）。**驱动 canonical
  `integrate_lif_field`，绝不复用 sigmoid `sef_hfo_pulse`（错基底）**。center = 混淆旋钮（中心=C/iso，
  偏心=kick-track）。签名加 `n,L`（构造 (n,n) mask 必需；plan 逻辑签名省略，已显式加非静默补）。
- **Task 3 `rate_event_envelope`**（`src/sef_hfo_rate_adapter.py`）：帧栈 `(nsteps,n,n)→(nsteps,n*n)`
  **C-order reshape，与 `grid_coords` 对齐**（field-space 网格污染坑：两者都源自 `_grid` 的 ij-indexed
  居中网格），复用 `sample_envelopes`。单测只锁**核心性质**（早活跃区触点 first-crossing 早于晚活跃区）
  ——**不用单向行波 + onset_front_axis**（WF-A 垂直前沿陷阱）。另加一条对真实积分器的 wiring 守卫测试。

测试：`tests/test_sef_hfo_lif.py`（+1 位置锁测试）+ `tests/test_sef_hfo_rate_adapter.py`（3 测试）。
全量 sef_hfo：**89 passed, 1 skipped**。

#5 事件窗合同：rate runner 用**锁定语义** `src.sef_hfo_snn_adapter.event_window_for_run`
（calibrate_detector floor-from-ref / peak-from-kick），不用临时 `0.2*peak`。

---

## 3. Increment-3a Task 4（rate parity runner）— θ=45 parity 成立，全四对照撞几何墙

commit `70f0b81`，`scripts/run_sef_hfo_obs_increment3a.py`，逐字镜像 Increment-2 SNN Option-B smoke
（同锁定阈 AXIS_ERR_MAX=25 / KDIR=3 / PART_MIN=7 / TAU_FAIL=0.3，同估计子 endpoint_centroid_axis，
同事件窗，同四对照），只换基底。基底 parity：原点居中网格 center=(0,0)、frame_dt=dt、iso=ell_par==ell_perp、
确定性脉冲、op=`mean_field(0.6)`、脉冲 (r=2,amp=8,30ms)。

**冻结几何配置**（先做了 advisor 的"先量瓣再信 montage"门 + 2 个备选探针，非 grid grind）：
L=24, n=96, 3 杆(10/70/130°), 4 触点/杆, **4mm 锁定间距**。探针：2杆L16 n_part5 / 3杆L16 n_part6 /
2杆L24 n_part5 / **3杆L24 n_part8 ✓**（大片→定位瓣 + 3 杆→角采样密，二者合力过 ≥7）。

verdict（`results/topic4_sef_hfo/observation_layer/increment3a_rate_parity/smoke_rate_parity_verdict.json`,
kicktrack_off=0.35）：

| 条件 | n_part | axis_err | 判 |
|---|---|---|---|
| C-track θ=0 | 1 | None | 触点不足 |
| **C-track θ=45** | **8** | **5.9°** | **PASS（read=0.994）** |
| C-track θ=90 | 5 | None | 触点不足 |
| kick-track perp±0.35 | 3 / 3 | None | 触点不足 |
| iso (AR=1) | 2 | None | **诚实 fizzle ✓** |

- **核心 parity 成立**：θ=45（两基底都标定过的标定条件）rate **5.9°** ≈ SNN **6.4°**（memory line 29）；
  iso 两基底都熄火。**两个独立模型同一套工具读出同一方向** = 双模态 parity 在主条件上的证据。
- **全四对照撞墙**：θ=0/90 + kick-track 在固定 4mm montage 下拿不到 ≥7 触点。这是 SNN 也撞过的
  **"四者不可兼得"**（memory line 31 "L=3 上 montage 四者不可兼得"）。**关键：这堵墙 rate 和 SNN 都撞
  → 与模型无关 → 是稀疏虚拟电极采样 2D 瓣的几何问题**。kicktrack_off 0.7→0.35（套用 SNN 既有 known-good）
  把 kick-track 从 1→3 触点，仍 <7。

**iso 基底差异（re-read plan §0 在 step 边界发现，未静默改）**：SNN iso 单端踢靠"各向同性不传播→
fizzle"；rate field 是连续可兴奋介质，各向同性踢仍会**圆形**传，单端时 endpoint_centroid_axis 会读
**踢点方向**而非连接轴。本轮实测 rate iso 仍 fizzle（n_part=2，montage 在偏心圆瓣外缘）→ 当前实现凑巧
诚实。但**严谨的 rate iso 应按 plan §0 = 中心脉冲 + 看事件拉长比<1.3**（去掉踢点方向混淆）——列为待定
设计点（§4）。

### 3.1 与 compaction 前 commit 8b54107 的对账（CONFOUNDED / SUPERSEDED）

本 session compaction 前有一个 commit `8b54107 "Increment-3a smoke PASS"`（err=18.4°,
n_part=10），**经核对该 "PASS" 不成立、已被本轮 D6-compliant 结果取代**：

- **违 D6**：其 montage 两杆**都平行 θ_EE**（`build_shaft(theta_rad,…)` ×2，±2mm 侧偏）。
  D6 锁"≥2 **非平行**杆"——杆平行 θ_EE 时 endpoint_centroid_axis 的回收轴被杆方向支配
  （杆=θ_EE→err 小是**循环**，非几何无关回收）。且转 θ_EE 必须转杆 → 毁掉 shaft-invariance →
  这套设计**结构上撑不起四对照**。
- **违锁定间距**：SPACING=3.0mm，非真实 SEEG depth 4mm / Yuquan 3.5mm。
- **违 #5 事件窗**：手搓 `ext>0.02` + margin 0.2，非锁定 `event_window_for_run`。
- **漏 commit + broken runner**：8b54107 只提交了 runner + analyzer + plan，**没提交**
  `return_frames`/`rate_adapter`/tests（本轮重建）→ 留下的 `scripts/run_sef_hfo_obs_increment3a_smoke.py`
  在本轮重建前是引用不存在函数的 broken 文件。

**有效保留**：8b54107 的 plan 修正（onset_front_axis→endpoint_centroid_axis；SNN cm 诚实状态）
+ `analyze_sef_hfo_obs_cm_offline.py`（offline oracle + per-contact baseline，advisor 修）仍有效。
**真正有用的洞见**：它正确诊断了"事件是 ~4mm 宽细条带"——这正是本轮非平行杆覆盖墙的根因。
**建议（用户 gate）**：删除或加 deprecation banner 到 `run_sef_hfo_obs_increment3a_smoke.py`
（confounded，与 D6-compliant `run_sef_hfo_obs_increment3a.py` 重名冲突）。

---

## 4. 电极尺度可复现合同（item #4）— DONE (committed)

commit `b3ce819`。把"真实 SEEG 间距=虚拟电极间距"（用户锁）从终端输出固化成可复现 audit + config，
防止滑回 0.4mm toy pitch。**双峰，非单一均值。**

- `scripts/audit_electrode_spacing.py`（可复现，确定性，用既有 `src.seeg_coord_loader` 公共 API，非手解 SQL）。
- `config/electrode_spacing.json`（锁定合同 + provenance + 分类带 + 双峰 + ambiguous flag）。
- 产物：`results/topic4_sef_hfo/electrode_spacing_audit/{electrode_spacing_per_shaft.csv (401 行), cohort_summary.json}`。

| 数据集 | depth (mm) | grid/strip (mm) | 备注 |
|---|---|---|---|
| Epilepsiae (27) | **4.58** (3.16–5.92, 46 杆/12 subj) | **11.0** (9.03–15.56, 230 杆/24 subj) | 双峰，分得很开 |
| Yuquan (11) | **3.5011** (3.4956–3.5109, 118 杆/11 subj) | — (0) | 单峰全 depth |

Yuquan 3.5011 = 重算 per-shaft median，复现 v3.1-lock 审定值 3.501mm（非硬编码字面量）。6 个边缘杆
（5 subj）落入 `other` 带，flag 不强塞。**取用规则（写进 config）**：按模拟电极类型选对应带，不许塌成单一均值。

---

## 5. 状态 + 用户 gate（待定决策，不自主 grind）

| 项 | 状态 |
|---|---|
| #1 SNN-cm 复盘 | ✅ DONE = readout UNVALIDATED（没点着，非坏） |
| #3 rate Tasks 1–3 | ✅ DONE committed (TDD, 89 passed) |
| #3 rate Task 4 四对照 | ✅ **OVERALL PASS**（前端修复解掉 §3 撞墙；见 §7）；检测前端合同 spec §4.2 + sensitivity |
| #4 电极尺度合同 | ✅ DONE committed（双峰 audit + config） |
| #5 事件窗合同 | ✅ rate runner 用锁定 event_window_for_run |
| #2 SNN 全四对照 verdict | ⏸️ gated（spec §13.1 SNN-cm 后调 + 几何墙用户 gate） |
| #6 异质核机制层接口 | ⏸️ Increment-3b，plan 锁"待 3a 双模态 parity 过 + Step-3 机制锁"才做 |

**待用户定（geometry/iso 设计 fork，按既有锁定 = 用户 gate）**：
1. **全四对照几何**：固定一套 4mm montage 拿不到全 θ ≥7（rate+SNN 同墙）。选项：(a) per-θ 旋转
   montage（=本就是 plan 的 shaft-invariance 对照）；(b) per-θ 调踢点端距/类型；(c) 接受 θ=45 单标定
   条件 + kick-track 作主 parity 证据（plan acceptance = "同 SNN 同定性结论 at 标定条件"，未必要求全 θ）。
2. **rate iso 对照口径**：单端 endpoint（当前凑巧 fizzle）vs plan §0 中心脉冲+拉长比<1.3（更严谨、去踢点
   方向混淆）。建议后者。
3. **SNN-cm 是否重跑**：种子修 R_KICK 0.15→~0.7（density100 下 ~150 神经元）能点火，但事件跨度 vs montage
   仍存疑；spec §13.1 已把 SNN-cm 降为后调。建议暂不重跑，先把 rate 主路四对照定下来。

阈值 / §3.5 估计子 / reframe claim / framework H1–H6 / v0.2 边界 **均未动**。

---

## 6. 用户复核轮（2026-06-07）— LFP forward 模型已验证 + Step-3 写盘回读闸门 + 清理

用户复核后给了 5 点最小合同 + 3 缺项（rate 四对照 / SNN 可点火复核 / artifact 回读）。用户自己
在跑 rate 四对照（重写 `run_sef_hfo_obs_increment3a.py` 成 C1–C4 完整 runner v2，含 C3 旋转电极不变
对照 + C4 中心脉冲 iso）；分配给本 agent 的是后两项 + 清理。

**① LFP forward 模型 = 已验证（@2mm 尺度，commit 含 `validate_sef_hfo_lfp_forward.py`）**：
之前 cm 没点着 → current-LFP 读出从没被测过。本轮用**可靠点火的 2mm 配置**(L=3, density=1000,
单端踢；= 验证过的 6.4° 配置)，对**同一个真实事件**同时用两种信号读方向：(a) 发放密度包络（已验证
参照）、(b) **电流型 LFP**(engine `lfp.py` |I_E|+|I_I| at sites = §13.4 正式前向模型)。两者都走
同一条 `event_window_for_run→extract_lagpat→endpoint_centroid_axis`。**结果**: 事件确实点着
(on max_spk_frac=0.028 vs off 0.0004); current-LFP 读出 **err=6.4° / 16-of-16 触点 / readability 0.77**,
**与发放包络的 6.4° 完全一致**。→ **电流型 LFP 前向模型在真实事件上能透过真实 pipeline 恢复方向 = 已验证**。
**caveat**: 这是 2mm 介观尺度 + toy 0.45mm 触点；**真实 4mm 间距的 cm 尺读出是另一个更重的 "后调" 问题**
(`recheck_sef_hfo_snn_cm_ignition.py`: L=12 density=1000 vs 失败的 density=100；后台 feasibility 复核)。

**② Step-3 写盘回读闸门 = PASS（`test_rate_field_artifact_round_trips_via_real_loader`）**：
用**真实 rate-field 读出**(锁定 4mm 间距 + 3 非平行杆 + event_window_for_run)写 legacy artifact
(`*_lagPat_withFreqCent.npz` + `_packedTimes_withFreqCent.npy` + `_montage.json`)，再用**下游 loader**
`load_subject_propagation_events`(Step 3 真正消费的那个)读回，断言：通道顺序保留、时序 ms→s 转对、
**非参与触点回读为 NaN-rank(无 phantom)**。证明 Step 3 消费的是写盘再回读的真实 artifact，不是临时
内存对象。8.55s 过。

**③ 清理**：confounded `run_sef_hfo_obs_increment3a_smoke.py` **已删除**（不是只贴 banner——删掉才不会
再算出那个平行杆假 PASS；其细条带洞见保留在 §3.1）。

**cm ignition 复核结果 = 仍 NO EVENT（density=1000 也不够）**：`recheck_sef_hfo_snn_cm_ignition.py`
L=12 / density=1000（= 2mm 点着的同一密度）跑完：`on max_spk_frac=0.0001`(=off 背景, ~11 神经元,
无传播事件)。即**只升密度 100→1000 在 cm 尺仍不点火**。推测根因（结构性、非单旋钮）: L=3 小片靠周期
边界 wrap 把局部连接"折叠加密"才点着; L=12 无 wrap → 局部连接真稀疏 → C_EE=800 in-degree 喂不满 →
recurrence 不够 → 不点火。要 cm 点火大概需 density≥4000(=smoke 默认; N≥576k, build 很重) 或更大 l_EE/
更强踢。**= 坐实 §13.1 "SNN-cm 重、是后调; rate@cm 才是主路"**。**不再自主 grind**（用户已许后调 +
advisor/memory 反复"grid 不行就停" + 会与用户并行 rate 跑抢算力）。

**仍缺/待**：~~rate 四对照（用户在跑 runner v2）~~ → **DONE，全过，见 §7**；cm 真实 4mm 尺 SNN 读出
（需 density≥4000 重 build, 后调）。**LFP 前向模型 = 用户问的, 已答 = 有且验证过（2mm, current-LFP
err 6.4°=firing, 16/16）**; 真实 cm 尺读出受阻于 SNN 点火（与前向模型是否成立无关——前向模型已验证）。

---

## 7. Increment-3a 四对照 OVERALL PASS + 检测前端合同 + sensitivity（2026-06-07 收尾）

`run_sef_hfo_obs_increment3a.py` runner v2（C1–C4 完整四对照）跑出 **OVERALL PASS**，§3 的"撞几何墙"
被三处**检测-前端**设计修复解掉（不动估计子/阈值）。

**四对照结果**（`rate_parity_four_controls.json`，L=24/n=96/pitch=4mm/3 杆 15·75·135°/6 触点）：

| 对照 | 隔离的混淆源 | 结果 | 判 |
|---|---|---|---|
| C1 连接轴跟踪 θ_EE=0/45/90° | 连接方向 | n_part 11/12/11，err **1.1/3.3/1.1°** | ✅ 全过 |
| C2 踢点位置 perp ±0.35/0 | 种子位置 | n_part 9/12/9，err **20.5/3.3/20.5°**（对称） | ✅ 全过 |
| C3 电极杆旋转 0/30/60/90° | 电极摆放 | n_part 12/10/12/10，err **3.3/0.0/7.3/0.0°** | ✅ 全过 |
| C4 各向同性 AR=1 中心踢 | 有无方向 | n_part **12**（采够），无可读轴 | ✅ 过（无伪方向） |

**三处前端修复（resolve §3 撞墙）**：
1. **事件窗源 = 源场活跃比例 `ext`**（非触点 aggregate）：踢是模型 artifact、真实数据无对应物；触点
   aggregate 在某些旋转下被"直接受刺激/平行波触点持续点亮"污染 → 检测器判 `returned=False` → 无窗。
2. **participation margin = 10%**（非 50%）：rate field 沿波 3–10× 幅度梯度，50% 把波前触点全切掉
   （θ=0° 从 1 → 11 触点）。
3. **电极杆改对称 15/75/135°**（原 10/70/130 对 45° 反射不对称 → C2 两侧 15.5° vs 25.5°；对称后 20.5/20.5°）。

**检测前端合同（spec §4.2 锁）+ sensitivity（`rate_parity_sensitivity.json`）—— 回应"看结果后换旋钮"质疑**：
前端两旋钮（事件窗源、margin）是**检测前端**；parity 主张 = 估计子+阈值（不动）。判定性问题不是"换旋钮还过不过"，
而是"换旋钮 axis_err 变不变"：
- **事件窗源**：在 contact-aggregate 也有窗的条件（C1×3/C2:perp0/C3:rot0,rot60）上，field vs contact 的
  axis_err **差 = 0.0°**。即换窗源移动的是**覆盖**（contact 在 4 个几何脆弱条件根本无窗），不是 read。
  **这直接用 0.0° 回答"事件窗会改通道顺序"的担忧。**
- **margin 10/20/30%**：**从不把 tracking 条件 (C1/C2/C3) 翻成 fail**，只 pass→insufficient（覆盖丢失、
  诚实）。axis_err 在 n_part≥7 处始终 < 25° 门。axis_err 数值**会**随 margin 收紧漂移（最大 ~10.2°，
  如 C1:0deg 1.1°→11.3°，因触点变少→centroid 变噪 = 覆盖→精度），**10% 是最大覆盖端故非 cherry-pick**。
- **诚实 caveat**：C3 rot30/rot90 在 contact-aggregate 下**无窗**（一根杆恰平行波→触点持续点亮→aggregate
  不 return），field-ext 是唯一能给它们定窗的 → 它们的 PASS 依赖"两种窗都有窗的条件上 read 一致(0.0°)"的
  **归纳论证**（明确写出，不掩盖）。
- `VERDICT_ROBUST=True`（前端旋钮移动覆盖、不移动 read/判定）。

**C4 iso 措辞（user 复核点）**：JSON 里 `readability=null` / `axis_err=null`，**报告为"采够触点(n_part=12)
但无可读方向(axis=None)"= 各向同性的正确负对照结果**（够触点 + 无稳定方向），**不是 readability=0**。
不要把 null 当数值 0 画/读。

**C3 rot30/rot90 = 0.0° 不是退化**（advisor sanity flag 已查）：rot30 把杆 A 转到 45°、rot90 把杆 C
转到 225°≡45°，**恰平行波** → 该杆触点沿传播线排开 → 最早/最晚 3 触点质心在 (−4.24,−4.24)/(+4.24,+4.24)
（distinct）→ 轴恰 45° → err 0.0°。是几何恰好对齐的**真实精确读**，端点质心 distinct，非 k_dir 触点塌缩。
（margin 收紧到 20% 时变 3.7°，= 随数据响应、进一步证明非硬编码退化。）

**写盘回读测试更新到最终口径**（§6 ② 那版用旧口径 contact-aggregate window + 50% margin + 10/70/130 杆，
与 runner 口径不同 → 没覆盖 runner 输出）：现 `test_rate_field_artifact_round_trips_via_real_loader` 直接用
runner 的 `_read`（run_four_controls 调的同一函数）在 C1 θ=45 PASS 条件、最终口径（field 窗/10% margin/
15·75·135 杆/−θ_EE 端踢）产 artifact 写盘回读，断言通道顺序/ms→s 单位/**参与触点集**/非参与 NaN mask 一致。8.6s 过。

**图**：`figures/rate_parity_overview.png`（四对照柱状总览）+ `figures/rate_readout_headline_theta45.png`
（主条件 §13 全链诊断：电极几何 overlay + per-contact firing-rate 包络[标注 NOT LFP] + lag-vs-轴链）+
`figures/README.md`（中文逐图）。

**LFP 边界（不变，重申）**：四对照 PASS 是 **rate field 的 firing-rate-density 包络**读出 = "rate 主观测路
在挑选过的 4mm 三杆几何下能读回方向"。**不**等于"真实 4mm 尺度 SNN 虚拟电极可用"——current-LFP 前向模型
只在 2mm 小尺度真实点着的 SNN 事件上验过（§6 ①），SNN cm 尺读出仍卡在点火（§6 cm recheck）。

**阈值 / 估计子 / reframe / framework H1–H6 / v0.2 边界均未动。** 双模态门（SNN+rate 同接口/估计子/阈值）：
rate 侧四对照已过；SNN 侧 cm 尺读出后调（点火受阻），2mm 尺度 LFP-forward 已过 = 同条件 6.4° parity 证据。
