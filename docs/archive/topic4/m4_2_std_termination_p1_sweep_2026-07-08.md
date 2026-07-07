# M4-2 —— STD 终止器 P1 sweep 结果（2026-07-08, DRAFT：seed 1+3 confirmed，seed 4 running）

> 状态：**seed 1 / 3 / 4 全确认 → clean no-go LOCKED（3-seed）**。这是 M4-2A（"STD 能否把 M4
> pass-1 的有界持续态干净地终止成可再触发间期"）的 P1 go/no-go 结果。
> 图：`results/topic4_m4_dynamic_p1_sweep/figures/m4_2_p1_sweep_map.png`（+ `figures/README.md`）。
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
终止；只能碎裂或压死。** seed 1 与 seed 3 的图**逐格一致**。

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

STD 削 recurrent 自持,能结束当前这一次放电；但事件一结束,衬底仍"上着膛"（`q_I` 在慢时标 `tau_q=5000`
上还没回充、背景驱动仍在）→ STD 一旦恢复就立刻再点火 → 变成 burst 振荡（fragment）,而不是"一次事件后静息"。
STD 太强则在事件成形前就压死（suppress）。**STD 只是调制正在进行的态,造不出"间期 ↔ 单次发作"的分离。**
这与 M4 pass-1 的"不可撤回"发现一致,并把它钉到机制层面。

## 4. 验收（spec §7.2）

**go(cell) = terminate_clean AND retrigger pass。跨 seed 1/3/4 + 低-u,共 0 个 go cell / 0 个 terminate_clean。**
→ **clean no-go（3-seed LOCKED）**：STD 单独不足以把 M4 pass-1 的有界持续态干净地终止成可再触发的间期。
（§7.2 明确:干净 no-go 是合法结果,加强"下一杠杆"结论,不是把 M4-2 悄悄证伪。）

## 5. 下一杠杆

- **D_EE / 衬底异质**：均匀率场衬底给不出事件后的非兴奋（可再触发）静息态；需结构连接或异质核提供 separation。
- **更慢的离子型终止器**：Epileptor 谱系用 slow-K / pump（`g_K`-adjacent）作**主**慢渗透变量（spec §10 nuance）;
  STD 是快 recurrent-侧,可能太快→碎裂。gK arm（spec Arm 3）此前 deferred,是自然的下一个候选。
- **Framing 锁**：措辞用 "actual M4-2 **SIMULATION**",绝不 "real data"。

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
