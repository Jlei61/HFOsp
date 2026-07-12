# M4-3A —— 连续 shunting 恢复变量 P1 discovery 结果（2026-07-10, 3-seed CLEAN NO-GO）

> 状态：**3-seed clean no-go（discovery 层）**。这是 M4-3A（"连续、活动驱动、电导型 shunting 的恢复变量 `n→a`
> 能否把 M4 pass-1 的有界持续态干净地终止成可再触发间期"）的 P1 go/no-go 结果。
> 前置 spec：`docs/superpowers/specs/2026-07-09-sef-hfo-m4-3-continuous-shunting-axis-coordinate-design.md`（rev2）；
> 实现 plan：`docs/superpowers/plans/2026-07-09-sef-hfo-m4-3a-continuous-shunting-impl.md`（10 task 全建完，commits `c8eddff`..`b01b90b`+`4f81c2a`）。
> 前作：M4-2 archive `docs/archive/topic4/m4_2_std_termination_p1_sweep_2026-07-08.md`（STD 终止器 3-seed clean no-go）。
> **Framing 锁：全文是 "actual M4-3A SIMULATION"，非 "real data"。no-go 是预注册合法结果（spec D7），非把 M4-3A 悄悄证伪。**

---

## 0. Abstract（第一性原理朴素话）

**测了什么：** M4 pass-1 造出一个"进得去、出不来"的持续放电态（除法共享池把失控压成有界，但它不自己熄）。M4-2 已证
"快·减法·单一"的疲劳（短时程突触抑制）造不出干净终止（只碎裂或压死）。M4-3A 换一个**一直都在、随活动累积、靠"开一条
漏电通道把增益按下去"（电导型 shunting）** 的慢变量 `a`，看它能不能让这个持续态**一次性干净熄灭、并回到"安静但还能
再点着"的间期**。

**怎么测：** 扫这个 shunt 的两个旋钮——按得多狠（`α_A`）× 累积负荷恢复多慢（`τ_n`）——在同一个已确认有界的工作点
（`k_q=0.10, α_G=16`）上，每格真 E1146 布局长跑 15 秒，3 个 seed；用一把**只看形态**的尺子把每格判成 持续不灭 / 一次
干净终止 / 碎裂 / 压死 / 熄后反跳 / 失控，外加"熄了之后早窗点不着（发作后不应期）+ 晚窗能再点（恢复）"两窗判据。

**揭示了什么：** **没有任何一格出现"一次持续发作 → 干净熄灭 → 安静可再点"。** 而且方向很反直觉——**不加 shunt 时是
有界持续（bounded persist），一加弱 shunt 反而失控（runaway），加强 shunt 则碎裂（fragment），中间没有干净终止这一档。**
也就是：**在这个衬底 + 这个工作点，这种电导型 shunt 不但不能造出发作终止，弱的时候还把本来有界的态推成了失控。**

---

## 1. 方法

- runner：`scripts/run_m4_dynamic_qi.py --m43a-sweep`（Pool + fork-COW 共享 E1146 net；per-cell fail-loud；OOM-safe workers=5）。
- 每格 = 两遍 same-seed retrigger（`src.sef_hfo_m4_termination.run_cell_with_retrigger`，`early_offset_ms=750`）：pass-1
  spontaneous 分类（用 runner baseline）；仅 `terminate_clean` 才跑 pass-2（early 窗 offset+750ms + late 窗
  offset+recovery_factor×max(τ_n,τ_q)），断言 pre-probe identity。
- 判读器 `classify_termination`（阈值 synthetic-fixture 锁）：`termination_class` +（独立）`retrigger_probe`/`retrigger_early`。
  **go(cell) = `terminate_clean` AND `retrigger_early=="attenuated"`（早窗不应期）AND `retrigger_probe=="reignite_bounded"`（晚窗可再点）。**
- 机制变量：**电导型 shunt（form A）** —— `g_A=α_A·a` 进膜更新 `V_inf=(I_net+g_A·E_A)/(1+g_A)`，`E_A=e_gaba`（阈下反转，
  引擎复用），`Vtmp=V_inf+(V-V_inf)·decay_V^(1+g_A)`。`a` **绝不整除 signed net current**（reversal-clamped）。减法项 `-η_A·a` 本轮 off（`η_A=0`）。
- 工作点 `k_q=0.10, α_G=16`（M4 pass-1 confirmed-bounded strip；`ee_std` off、`g_K` off）；`T=15000`；`u_n0=0`（见 §5 标定说明）。
- 主扫描平面 = `(α_A ∈ {2,4,8}) × (τ_n ∈ {5000,20000,40000})` + Arm0（`α_A=0` = M4-1 bounded-persist 基线）。
- seed 1/3/4（seed 1/3 Arm0=bounded-persist=主分母；seed 4 Arm0=fragment=stress，照 M4-2 §0/§2.3 分层，D10）。

## 2. 结果

### 2.1 三 seed 全 map（每 seed = Arm0 + 9 格）—— 无 terminate_clean

| seed | Arm0(α=0) | α=2（各 τ） | α=4（各 τ） | α=8（各 τ） | terminate_clean | go |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | persist | runaway | runaway | fragment | **0** | **0** |
| **3** | persist | runaway | runaway | fragment | **0** | **0** |
| **4** | fragment | runaway | runaway | fragment | **0** | **0** |

- **`τ_n` 对分类无影响**（每个 α 下 5000/20000/40000 三格同类）—— **α_A 是判别轴，τ_n 不是**。
- **跨 seed 1/3/4 共 27 sweep 格，0 个 `terminate_clean`、0 个 go cell。**

### 2.2 分类为真（runaway_ms + trace spot-check，seed 1）

| cell | class | runaway_ms | af peak | af tail10% | ⟨a⟩ | D_A_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Arm0 (α=0) | persist | None | 0.103 | 0.077（尾不熄） | 0.716 | 1.00 |
| α=2, τ=20000 | runaway | **2629.9ms** | 0.401 | —（早停） | 0.518 | 2.03 |
| α=4, τ=20000 | runaway | **4040.8ms** | 0.291 | —（早停） | 0.436 | 2.74 |
| α=8, τ=20000 | fragment | None | 0.231 | 0.036（碎裂） | 0.312 | 3.50 |

- **runaway 是真失控**（引擎 Hz 阈值 `runaway_ms` 触发早停，非误判）；**弱 shunt 峰值反而比 Arm0 高**（0.40 vs 0.10）。
- **shunt 越强 runaway 越晚**（α=2 于 2630ms、α=4 于 4041ms）；α=8 不 runaway 但碎裂（tail 0.036）。
- **D_A 随 α 单调升**（2.0→2.7→3.5）= shunt 确实在耦合（`a`≈0.31–0.52）。

## 3. 机制（scoped，承重措辞）

**能支持（3-seed、单工作点、本 α×τ 网格内）：** 电导型 `a`-shunt **不产生 clean, re-triggerable termination**；它把系统推向
**弱→runaway / 强→fragment** 两端，中间无干净终止窗。**反直觉点：不加 shunt 是有界持续，加弱 shunt 反把有界态推成失控。**

**机制假设（UNVERIFIED，不作结论）：** 一个自洽解释是 shunt 抑制了稳态放电 → 除法共享池 `S_G`（M4-1 把 runaway 压成
有界的那个刹车）积累不足 → 刹车被松开 → 失控；shunt 越强削得越狠，晚一点才松开（runaway 延迟）或直接把活动切碎
（fragment）。**但本轮 runner 未 dump `S_G` trace，此假设不可从现有 trace 证实**，只能作机制方向，不作已证结论。

## 4. 验收（spec §7 / D7）

**go(cell) = terminate_clean AND early attenuated AND late reignite_bounded。跨 seed 1/3/4，共 0 个 go / 0 个 terminate_clean。**
→ **clean no-go（discovery 层，3-seed LOCKED）**：本电导型 `a`-shunt 单独不足以把 M4 pass-1 有界持续态干净终止成可再触发间期。
（§7/D7 明确：干净 no-go 是**合法、预注册**结果，加强"下一杠杆"方向，不是把 M4-3A 悄悄证伪。）

**用户门控执行：** 0 candidate → **不进入 40s acceptance、不进入 Task 9 ablation**（无候选可验）。这符合 "有候选才进 40s/ablation"。

**边界细化（rigor，seed 1，α∈{5,6,7}，τ=20000）：** α=5/6/7 全 fragment、0 terminate_clean（§6.1）—— **确认 no-go 在细网格上稳健**（合计 30 格 0 go）。

## 5. 下一杠杆 & 标定说明

**下一杠杆（待用户定，非自主）——照 spec §5，no-go ≠ 已证 `D_EE`：**
- M4-3A 仍锁 `λ_K=0`（各向同性 Gaussian 慢场），只换了恢复变量**类型**（shunting vs STD 减法）。所以本 no-go **只加强**
  "在当前 Gaussian 慢场衬底上换恢复变量类型拿不到终止"的怀疑，**不证** `D_EE` 是唯一杠杆。
- 三条 live 分叉：① **先跑 M4-3B graph-kernel smoke**（`λ_K∈{0,0.5,1}`，spec §9）——病理轴对齐可能恰恰要 `K_graph` 才出现；
  ② deferred **`g_K` arm**（Epileptor 谱系 slow-K/pump 作主慢渗透终止变量，M4-2 §5 列为第一分叉）；③ `D_EE`/衬底异质。
- **本轮新增一条机制线索**：runaway-with-weak-shunt 提示 "shunt vs S_G 除法刹车"的相互作用值得单查（需 dump `S_G`），
  可能是"为什么加 anti-ictal shunt 反而失控"的关键——这不是 spec 里既有的三分叉，是本次 discovery 冒出来的新问题。

**`u_n0=0` 标定说明（承重，纠 plan P0b 的一个自相矛盾）：** plan 的 P0b runbook 曾写"`u_n0` = Arm0 的 `trace_un_mean`
长期均值"。**这对 discovery 是自相矛盾的**：Arm0 就是**有界持续（高放电）**态、不是安静间期基线；若把 `u_n0` 定在持续态
驱动上，则持续态里 `[u_n-u_n0]_+≈0` → `a` 建不起来 → shunt 永不启动 → 平凡 no-go（把机制关掉了）。所以本轮用 `u_n0=0`
（让 `a` 随总活动累积、在持续态里真启动 = discovery 的正确选择）。Arm0 实测 `⟨a⟩≈0.72`（`u_n0=0` 下负荷在持续态近饱和）证实
shunt 有充足"燃料"，failure 不是燃料不足。**真正的 sensor-free gate 闭合（`a_block` + 一个安静-间期 `u_n0`）属 M4-3B/C 的
interictal 情形，discovery 协议里没有安静间期相，故本轮未闭合该 gate**——`run_arm` 尚未 surface `trace_un_mean`（1 行可加）、
无 `--m43a-un0` flag、`a_block` 需专门 IED-kick 探针，三者 teed-up 待用户定标定协议后补。

## 6. 复现

```
# 主 discovery（每 seed s∈{1,3,4}，3 seed 并发）
python scripts/run_m4_dynamic_qi.py --m43a-sweep --confirm-run --seed <s> --T 15000 \
  --m43a-alpha-grid 2,4,8 --m43a-tau-grid 5000,20000,40000 --m43a-workers 5 \
  --out results/topic4_m43a_p1_seed<s>/
# 边界细化（seed 1）
python scripts/run_m4_dynamic_qi.py --m43a-sweep --confirm-run --seed 1 --T 15000 \
  --m43a-alpha-grid 5,6,7 --m43a-tau-grid 20000 --m43a-workers 4 \
  --out results/topic4_m43a_p1_seed1_refine/
```
结果：`results/topic4_m43a_p1_seed{1,3,4}/{m43a_sweep_summary.json, m43a_sweep_traces.npz}`。per-seed wall（3 seed 并发、80 core）≈ 2h。

### 6.1 边界细化结果（α∈{5,6,7}, τ=20000, seed 1）—— 确认 no-go

| α_A | class | runaway_ms | go | D_A_mean |
| --- | --- | --- | --- | --- |
| 5 | fragment | None | False | 3.02 |
| 6 | fragment | None | False | 3.28 |
| 7 | fragment | None | False | 3.35 |

**α=5/6/7 全 fragment，0 terminate_clean / 0 go。** → **no-go 在更细网格上稳健**。结合主网格：**runaway(α≤4) → fragment(α≥5)
直接过渡，边界（α≈4.5）无 terminate_clean 窗。** 全 discovery 合计 **30 sweep 格（27 coarse × 3 seed + 3 refine）、0 terminate_clean、
0 go**。（注：α=8 fragment 但 α=5,6,7 也 fragment 且不 runaway，说明 runaway 只在 α≤4 的窄弱-shunt 带出现，稍强即转 fragment；
干净终止窗在整条 α 轴上都不存在。）
