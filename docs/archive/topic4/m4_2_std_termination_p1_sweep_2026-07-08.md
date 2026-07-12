# M4-2 —— STD 终止器 P1 sweep 结果（2026-07-08, LOCKED：3-seed clean no-go）

> 状态：**seed 1 / 3 / 4 全确认 → clean no-go LOCKED（3-seed）**。这是 M4-2A（"STD 能否把 M4
> pass-1 的有界持续态干净地终止成可再触发间期"）的 P1 go/no-go 结果。
> 图：`results/topic4_m4_dynamic_p1_sweep/figures/`：`m4_2_p1_sweep_map.png`（(u,τ) 分类 map,seed=1）+
> `m4_2_p1_mechanism.png`（persist/fragment/suppress 慢变量轨迹）+ 中文 `README.md`。
> spec：`docs/superpowers/specs/2026-07-07-sef-hfo-m4-2-std-termination-design.md`（§5 P1 / §7.2 go-no-go）。

---

## 0. Abstract（第一性原理朴素话）

**测了什么：** M4 pass-1 造出一个"进得去、出不来"的持续放电态（除法共享池把失控压成有界,但它不会自己熄）。
M4-2 打开**神经元放得越多、递归自持输入被削得越狠**的机制（短时程突触抑制 STD），看它能不能让这个持续态
**一次性干净地熄灭、并回到"安静但还能再点着"的间期**。

**怎么测：** 扫 STD 的两个旋钮——削多狠（`ee_std_u`）× 削完恢复多快（`ee_std_tau_ms`）——在同一个已确认有界
的工作点（`k_q=0.10, alpha_G=16`）上，每格真 E1146 布局长跑 15 秒；用一把**只看形态**的尺子把每格判成
持续不灭 / 一次干净终止 / 单调淡出 / 碎裂 / 压死 / 熄后反跳，外加一个独立的"熄了之后补一发 kick 还点不点得着"。

**揭示了什么：** 加 STD 只把持续态推向两个极端,**中间没有"干净终止"这一档**：不加 → 持续不灭；STD 弱 →
碎裂成一串断续小爆（~15–33 次 / 15 秒,没有一次完整发作）；STD 强/慢 → 直接压死。**没有任何一格出现"一次
持续发作 → 陡然熄灭 → 安静但可再点"。** 也就是：**在这个衬底 + 这个工作点,光靠削递归自持不能造出发作
终止；只能碎裂或压死。**

**注意分母（承重）：** seed 1/3 的无-STD 基线（Arm0）是 pass-1 那个 **bounded persist** 态,是同质分母,支持
"从同一有界持续态出发、STD 终止失败";**seed 4 的 Arm0 本身已偏 fragment**（aG16 仅 3/4-seed 干净有界,与 pass-1
multiseed 一致）,故 seed 4 只作 **seed-robustness**（同样无 terminate_clean）,**不作"同一 bounded attractor 出发
终止失败"的同质分母证据**。

---

## 1. 方法

- runner：`scripts/run_m4_dynamic_qi.py --p1-sweep`（Pool + fork-COW 共享 E1146 net；per-cell fail-loud）。
- 每格 = 两遍 same-seed retrigger（`src.sef_hfo_m4_termination.run_cell_with_retrigger`）:pass-1 分类（用
  runner baseline,非 naive 前 5%）；仅 `terminate_clean` 才跑 pass-2（`t_kick2 = offset + 2×max(ee_std_tau, tau_q)`
  的 post-offset kick,断言 pre-probe identity）。
- 判读器 `classify_termination`（阈值 synthetic-fixture 锁,spec §7.1）：`termination_class` +（独立）
  `retrigger_probe`。**go(cell) = terminate_clean AND retrigger pass**。
- 协议 = spontaneous/no-kick（= pass-1 dynamic 有界态,分母锁）；唯一的 kick 是 post-offset retrigger。
- 工作点 `k_q=0.10, alpha_G=16`（pass-1 confirmed-bounded strip）；`T=15000`；`--p1-workers 5`（OOM-safe,
  swap 满 + 另一 campaign 在跑）。

## 2. 结果

### 2.1 seed=1 全 map（coarse 3×3 + 低-u refinement + Arm0）—— 无 terminate_clean

| `ee_std_u` \ `tau` | 1000 | 2500 | 5000 |
| --- | --- | --- | --- |
| 0（Arm0） | **persist**（平台 1270ms,尾 78% 不熄）| — | — |
| 0.05 | fragment | fragment | fragment |
| 0.08 | fragment | fragment | fragment |
| 0.11 | fragment | fragment | suppress |
| 0.15 | fragment | fragment | suppress |
| 0.30 | fragment | suppress | suppress |
| 0.50 | suppress | suppress | suppress |

- **persist → fragment → suppress,两条转变都陡,中间跳过 terminate_clean。**
- 包络 spot-check（10ms）：persist 长平台不熄；fragment 平台仅 10–30ms、尾 3–9%、~9–33 bursts；suppress
  peak≤0.02 或 =0（killed）。**无一 cell 有 sustained plateau ≥250ms 后陡降到静息。**
- retrigger 全 `not_run`（无 terminate_clean）；无 runaway。

### 2.2 seed=3（coarse 3×3 + Arm0）—— 逐格复现 seed=1

Arm0 persist；`u0.15/τ{1000,2500}`、`u0.3/τ1000` = fragment；其余 suppress；**无 terminate_clean**
（Counter: suppress 6 / fragment 3 / persist 1）。与 seed=1 coarse map **完全一致**。

### 2.3 seed=4（coarse 3×3 + Arm0）—— 确认，无 terminate_clean

`u0.15/τ1000`、`u0.3/τ1000` = fragment；其余 6 格 suppress；**无 terminate_clean**（Counter: fragment 3 /
suppress 7）。**注意:seed=4 的 Arm0(u=0) 本身 = `fragment`（非 seed1/3 的 persist）** —— 无-STD 基线随 seed 变
（aG16 仅 3/4-seed 干净有界,与 pass-1 multiseed 一致）,**但加 STD 仍无一格 terminate_clean**,verdict 不受基线差影响。

### 2.4 三-seed tally（含低-u）

| run | persist | fragment | suppress | **terminate_clean** |
| --- | --- | --- | --- | --- |
| seed1 coarse | 1 | 3 | 6 | **0** |
| seed1 low-u | 1 | 8 | 1 | **0** |
| seed3 coarse | 1 | 3 | 6 | **0** |
| seed4 coarse | 0 | 3 | 7 | **0** |

**跨 seed 1/3/4 + 低-u,共 0 个 `terminate_clean`、0 个 go cell。**

## 3. 机制

代表轨迹（图 `m4_2_p1_mechanism.png`,活动 + sheet-mean `q_I` + STD 可用度 `x_dep`）显示,判别在 **STD 恢复时标**:
- **fragment（弱 STD / 快恢复,u0.15/τ1000）:** `x_dep` 在两次 burst 之间快速回充（τ=1000）→ recurrent 自驱恢复
  → 在事件真正终止前又点火 → burst 振荡;`q_I` 在这 15 秒里**仍在慢慢排空**（1→~0.15,**不是**一开始就钉在地板——
  这一点纠正了初稿"q_I primed"的粗糙说法）。
- **suppress（强/慢 STD,u0.5/τ5000）:** `x_dep` 被抽干且回充慢 → recurrent 自驱被杀 → 活动死掉 → `q_I` 因无活动
  可排反而**维持高位**（~0.8）。
- **persist（无 STD）:** `q_I` 排到地板、活动自持不熄、`x_dep≡1`。

两头之间没有"`x_dep` 恰好结束一次事件、又压住足够久不再点火"的窗口 → 无 terminate_clean。**这是与 M4 pass-1
"不可撤回"一致（consistent with）的轨迹级解释——3-seed、单工作点、本 STD 网格内,不是普适证明。**（`S_G` 未存,
本图用 activity + q_I + x_dep 三条,足够读出恢复时标机制。）

## 4. 验收（spec §7.2）

**go(cell) = terminate_clean AND retrigger pass。跨 seed 1/3/4 + 低-u,共 0 个 go cell / 0 个 terminate_clean。**
→ **clean no-go（3-seed LOCKED）**：STD 单独不足以把 M4 pass-1 的有界持续态干净地终止成可再触发的间期。
（§7.2 明确:干净 no-go 是合法结果,加强"下一杠杆"结论,不是把 M4-2 悄悄证伪。）

**结论口径（scoped,承重）:**
- **能支持:** 在**当前 E1146 衬底、当前 pass-1 工作点（`k_q=0.10, alpha_G=16`）、当前 STD 参数网格、3 个 seed**
  检查内,E→E presynaptic STD **单独**没有产生 clean, re-triggerable termination;它把系统推向 persist /
  fragment / suppress,而不是 re-triggerable interictal recovery。
- **不能支持:**（a）STD 作为生理机制**普遍**不能终止发作;（b）M4 **所有**工作点都 no-go;（c）已"证明必须"换
  D_EE 或 gK。以上是 no-go **指示**的方向,不是已证结论。

## 5. 下一杠杆

两条最清楚的分叉（**待用户定,非自主决策**）:
- **若目标 = "终止 + postictal brake"** → 优先 **mild/slow gK arm**（spec Arm 3,此前 deferred）:它最直接接住这次
  no-go 暴露的"事件后仍可再点火"问题;Epileptor 谱系用 slow-K / pump（`g_K`-adjacent）作**主**慢渗透终止变量,
  比快 recurrent-侧 STD（太快 → 碎裂）更合适（spec §10 nuance）。
- **若目标 = "从衬底上拆掉不可撤回性"** → 转 **D_EE / 衬底异质**:均匀率场衬底给不出事件后的非兴奋（可再触发）
  静息态,需结构连接或异质核提供 separation。
- **Framing 锁**：措辞用 "actual M4-2 **SIMULATION**",绝不 "real data";上面是 no-go **指示**的方向,不是"已证必须换"。

## 6. 复现

```
# seed s ∈ {1,3,4}; coarse grid
python scripts/run_m4_dynamic_qi.py --p1-sweep --confirm-run --seed <s> --T 15000 \
  --p1-u-grid 0.15,0.3,0.5 --p1-tau-grid 1000,2500,5000 --p1-workers 5
# seed=1 低-u refinement
python scripts/run_m4_dynamic_qi.py --p1-sweep --confirm-run --seed 1 --T 15000 \
  --p1-u-grid 0.05,0.08,0.11 --p1-tau-grid 1000,2500,5000 --p1-workers 5 \
  --out results/topic4_m4_dynamic_p1_sweep_lowu
```
结果：`results/topic4_m4_dynamic_p1_sweep{,_lowu,_seed3,_seed4}/{p1_sweep_summary.json, p1_sweep_traces.npz}`。
per-cell wall（竞争下）:Arm0（无STD）~60–73min（最慢）；Arm1 ~7–21min。
