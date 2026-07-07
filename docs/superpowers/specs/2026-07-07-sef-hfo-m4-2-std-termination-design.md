# M4-2 —— 用 E→E 短时程抑制(STD)终止 M4 有界态、回到可再触发间期 设计 spec（2026-07-07, rev2, 中文版）

> 状态：**DRAFT / 待用户 review**。这是 M4 pass-1 的续集:pass-1 证明除法共享池 `S_G` 能把 runaway 变成
> **有界持续吸引子**,但那个态**不可撤回**(灌满 `q_I` 也回不了间期)。M4-2 加一个活动依赖的终止器,让有界态
> **自发熄灭并回到可再触发的间期态**。
> 这仍是机制筛查,不是"证明发作周期成立"。
> 前置:`docs/superpowers/specs/2026-07-05-sef-hfo-m4-divisive-shared-inhibition-design.md`(M4 pass-1)。
>
> **rev2(2026-07-07,并入用户 review)**:① 锁初始协议 = spontaneous/no-kick(§5);② `retrigger_probe`
> 升级为 state-continuous probe hook、`x_dep` trace 列必做 instrumentation(§8B);③ "零新引擎代码" →
> "零新动力学方程 + 非行为性 hook"(§1/§8);④ `classify_termination` 加 synthetic fixture 要求(§7.1);
> ⑤ 锁 P3 的 `k_K/tau_K/gK_max`(§5)。

---

## 0. 一句话结论

M4 pass-1 卡在"进得去、出不来":**自发**进入一个 ~77 Hz、`q_I` 钉在地板的**有界持续态**(pass-1 dynamic
实验是 no-kick、`q_I` 耗竭自点火),`S_G` 只 bound 不 terminate。M4-2 的承重改动是:**打开引擎里已有的 presynaptic E→E 短时程抑制 `ee_std`,让活动在发作期间耗竭
recurrent 自持输入,把持续吸引子推回间期。**

$$
I^{\text{net}}_{E,i}=I^{\text{ff}}_{E,i}+\frac{[\text{STD-depressed recurrent}]_i}{1+\alpha_G S_G}-q_I(x_i,t)I_{I,i}-\eta_K g_K(x_i,t)
$$

- **`S_G`(pool,pass-1)**:offset 前保持不 runaway —— **bounded stabilizer,不负责终止。**
- **`ee_std`(STD,M4-2 主终止器)**:活动耗竭 E→E 自持 → 有效 recurrent 增益跌破临界 → 事件自发熄灭。
- **`g_K`(fatigue,次级)**:postictal brake / 防反跳 —— **deferred 到 M4-2B,M4-2A 关掉(`eta_K=0`)。**

**为什么是 STD 而不是加更强的抑制(承重论证):** 这条正是 pass-1 spec §1 的逻辑。减法/加抑制**平移工作点、
不改增益**,对自放大 recurrent 环只会欠刹或过刹。STD **乘性削 recurrent 输入本身**,直接把有效增益
$\lambda^{\text{eff}}\sim\frac{d_{EE}\lambda_E}{1+\alpha_G S_G}$ 推到临界以下 —— 这是攻击持续吸引子的**结构**,
不是在输出端加超极化电流。pass-1 的可逆性实验已证:强灌 `q_I`(加抑制)回不了间期,会重新抽干。STD 的活是
**移除 drive**、不是加 inhibition,正好做 `q_I`-refill 做不到的事。

**最强科学目标:** 从 `interictal → bounded seizure-like attractor`(pass-1 终点)推进到
`interictal → bounded seizure-like event → interictal/postictal(可再触发)`。

**交付顺序:** 先跑 **P1 相平面(`ee_std_u × ee_std_tau_ms`)在已确认 bounded 操作点上**,回答
"STD 能否把有界态终止成一次干净事件、且终止后可被新 kick 再触发?"(go/no-go,§7)—— **在**任何 gK / 池-STD
交互扫描 **之前**,也**在** Arm 4 机制归因 **之前**。

---

## 1. 命门:M4 pass-1 为什么"出不来"

pass-1 有界态里:`q_I ≈ q_min`(地板)、`r_E ≈ 77 Hz`(持续)、放电把 `q_I` 抽得比恢复(`tau_q=5000`)快 →
**自锁**:`q_I↓ → 局部去抑制 → r_E↑ → q_I↓`。`S_G` 除法池把 runaway 变成有界,但**没打断这个自锁环**。

要终止,得有个变量在发作期间**累积**、最终打断自锁。两个候选(引擎里都有):

| 变量 | 机制 | 角色定位(M4-2) |
| --- | --- | --- |
| `ee_std`(E→E STD,presynaptic) | 反复放电 → depression → 削弱**自持 recurrent drive** | **主终止器**:攻自锁环的输入端 |
| `g_K`(SFA / sAHP) | 放电累积慢 K⁺ 电导 → 超极化 → 压放电 | **次级**:postictal brake(deferred 到 B) |

选 STD 作主力的三条依据(全部内部证据,不靠新文献):
1. **pass-1 spec §1 逻辑**:除法/增益侧 > 减法/输出侧(见 §0)。`g_K` 是输出端超极化(减法风格),要压出吸引盆
   必须够强,太强就全局静默 / 强反跳。
2. **M3A-M1(2026-06-18)已经是正向 pilot**:那次打开同一个 `ee_std`,结论是**"给时间自限、非空间自限"**——
   当时问的是空间围堵,判 Stage2 NULL。但"时间自限"正是 M4-2 要的。参考
   `docs/archive/topic4/`(M1 Stage2 NULL 归档)。
   **注意**:M3A-M1 是**没有池、在更早衬底**上测的;M4-2 的新东西是 **STD 叠加除法池、在 E1146 bounded
   attractor** 上 —— 这个组合从没跑过,是真·新实验,不是重跑。
3. **零新动力学方程**:`ee_std` 动力学原语已在 `simulate_kick`、recurrent-E-specific(§4)。但**不是纯
   runner 接线** —— 还需两处**非行为性 instrumentation hook**(`x_dep` trace + post-offset 第二 kick schedule,
   off 时 byte-parity,见 §8B)。

---

## 2. 承重设计决策（锁）

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | 主终止器 = 现有 `ee_std`(presynaptic STD) | §1;用户 2026-07-07 sign-off |
| D2 | **不新建 `d_EE(x,t)` 场** | 重复 `ee_std`;且 `d_EE` 撞 memory 的 `D_EE` 结构杠杆(见 §3 命名守卫) |
| D3 | `g_K` = 次级;**M4-2A 主信号(P1: Arm 0/1)关闭 gK**(`eta_K=0`)。gK 只在 **P3(Arm 3)作 rebound 抑制**(gated on P1 出现 rebound)、及 M4-2B cycling 才开。Arm 2 是**故意开 gK 的负对照**(证 gK-alone 不够,§6)| §1;避免混机制;postictal-refractory 只在 cycling 才需要,且可能反害 re-trigger |
| D4 | `S_G` 保持 pass-1 bounded stabilizer,**不负责终止** | pass-1 已锁;M4-2 只在其上加终止器 |
| D5 | 验收先"终止一次 + 可再触发"(M4-2A),**后**"极限环"(M4-2B) | 极限环要自发 re-nucleation,历史上是衬底墙(§7.3) |
| D6 | Arm 4(STD 机制归因)= **deferred attribution**,gated on ≥1 terminate-clean candidate | 它要在线取 `x_dep` 注入 matched current → 碰 integration loop,不阻塞 P1/P2(§6/§8) |

### 命名守卫（§6.2 分辨率层级）

"E→E" 在本仓库有**三个不同对象**,别混:
- **`ee_std` / `x_dep`(旋钮 `ee_std_u`,`ee_std_tau_ms`)** = 动态**短时程 depression**。← **M4-2 用这个**(时间终止)。
- **`D_EE`(静态)** = E→E **连接强度 / 尺度**结构杠杆。memory「下一杠杆 = `D_EE`/衬底」指的是它,针对
  **空间宽度**问题。M4-2 **不**碰它。
- 提案的 **`d_EE(x,t)`** = 拟新建的 postsynaptic depression 场。**D2 否决**(重复 + 撞名)。

写结果时禁止说"M4-2 拉了 `D_EE` 杠杆"——那是另一个对象、另一个(空间)问题。

---

## 3. 轴翻译:提案 4 相平面 → `ee_std` 真旋钮

`ee_std` = 标准 Tsodyks 突触前模型:每 spike 乘性耗竭 `x_dep *= (1-ee_std_u)`、单一 `ee_std_tau_ms` 恢复。
因此提案(按一个 postsynaptic `d_EE` 场写的)的轴有一半在真原语里不存在:

| 提案的平面 / 参数 | 映射到 `ee_std`? | 落地 |
| --- | --- | --- |
| plane 1 `(d_EE, q_I)` 轨迹 | ✅ 但是**诊断读出、非扫描轴** | 每 cell 画 `(⟨x_dep⟩, ⟨q_I⟩)` 弛豫轨道(§5 诊断) |
| plane 2 `(α_G, u_D)` | ✅ 直接 | 扫描平面 **P2** `(alpha_G × ee_std_u)` |
| plane 3 `(τ_D^rec, η_K)` | ✅ 直接 | 扫描平面 **P3** `(ee_std_tau_ms × eta_K)`,gK 臂 |
| plane 4 `(d_min, τ_D)` | ❌ `ee_std` **无 `d_min` clamp** | `d_min` 是 `(u, tau, rate)` 的**涌现量**;按 §7(多图去冗余)换成 **P1** `(ee_std_u × ee_std_tau_ms)` |
| `τ_D^deplete` vs `τ_D^rec` 分裂 | ❌ 只有单一 recovery tau + 每-spike u | 一个时标 |
| Hill `a_D50` / `K_D` | ❌ 每-spike 耗竭,非 rate 的 Hill | 无此层 |

**副产品:** 提案把耗竭强度拆成 `d_min` 和 `τ_D` 两个平面,真原语里它们**本来就是一个平面** `(u, tau)`,
即 M4-2 的头号平面 P1 —— 提案反而没把它当主平面。`d_min^{obs} = min_t ⟨x_dep⟩` 作为**读出**报告,不作扫描轴。

---

## 4. 状态变量 & 组合方程（引擎里怎么落）

保留 pass-1 的场:`q_I(x,t)`(局部抑制资源)、`g_K(x,t)`(疲劳)、`S_G(t)`(共享除法池,两级低通)。
M4-2 **不新增场**,而是激活现有 `ee_std`:`x_dep[NE]`(per-E-neuron 可用度)。

引擎里三者的合成(全部现有原语,见 §8):
1. E 神经元 j 放电 → 其外传 E→E 边权按 `x_dep[j]` 缩放(`ee_std_apply`,`kick_probe.py` L368),
   `x_dep[j] *= (1-ee_std_u)`(L371),每步 `x_dep += (1-x_dep)*x_rec_f` 恢复(L259)。
   → **recurrent 到达(`ring_sE`)已是 STD-depressed**,feedforward(external drive)不动。
2. 这批 depressed recurrent 累积进 `s_E_rec → I_E_rec`(pass-1 拆的 recurrent-only 路)。
3. 池在 `apply_currents` 里除 `I_E_rec`:`out[:nE] -= I_E_rec*frac`,`frac=αS/(1+αS)`(`slow_field.py` L273–276)。

合起来 E 输入 = §0 的 boxed 方程,`d_EE` 用 presynaptic 边缩放实现(不是 postsynaptic 场)。

**时标关系(承重,决定 offset 与 re-trigger):**
- `tau_q=5000`(慢),`ee_std_tau_ms` = 扫描量,`tau_K=5000`(gK,M4-2A 关)。
- `tau_S ~ 20–300 ms`(池,快)≪ `ee_std_tau_ms`(秒级)。**offset 时** activity 一降,池快速释放除法刹车,但
  STD 还压着 → 净 recurrent 增益仍 sub-critical → **STD 是 offset 的实际闸门**(设计自洽点)。
- **re-trigger** 要 `x_dep` 与 `q_I` 都恢复(各自 ~秒级)→ inter-event 尺度 ~ `max(ee_std_tau_ms, tau_q)`。
  这决定 §7 的 `retrigger_probe` 必须在慢变量**充分恢复后**才打(否则 fail 是平凡的、不可解读)。

---

## 5. 相平面:3 扫描平面 + 1 诊断读出（§7 去冗余后不是 4 个）

全部在 **E1146 真实布局**(twoend_equal 双轴向灶,与 pass-1 同衬底)、**长跑 `T=15000`**(只长跑是终判)。

**[LOCK] 初始触发协议 = spontaneous / no-kick**(= pass-1 dynamic 实验:`simulate_kick(..., KICK_BOOST=0.0,
t_kick=1e9)`,`q_I` 耗竭自点火;`run_m4_dynamic_qi.py` L189-191)。**这是分母锁**:M4-2 要终止的就是 pass-1 那个
**自发**产生的有界态;不能换 kick-triggered 态,否则 Arm 0 的 `persist` 分母漂移、终止的不是同一个 bounded
state。**全程唯一的 kick 是 §7.1 的 post-offset `retrigger_probe`。**
> 放弃的替代 = triggered protocol:若改 kick-triggered,Arm 0 必须先在同一 kick protocol 下复现 `persist`
> bounded state 才有效对照 —— 成本 + 风险更高,不选。

- **P1(头号)`ee_std_u × ee_std_tau_ms`** @ 已确认 bounded 操作点(`k_q=0.10`,`alpha_G` ∈ pass-1 confirmed-
  bounded 集;exact coords 从 pass-1 相图 verdict 读)。耗竭强度 × 恢复速度 → **STD 能否终止 bounded 态、终止得
  干不干净**。这是提案 plane 4 该有的样子。
- **P2 `alpha_G × ee_std_u`**。池 × STD。STD 是否拓宽 aG16 那条 marginal 有界带。(顺带避开"在 marginal 点叠
  robustness"——这平面本来就扫 `alpha_G`。)
- **P3 `ee_std_tau_ms × eta_K`**,**仅 gK 臂(Arm 3),gated on P1 出现 rebound**。恢复速度 × postictal brake →
  管 rebound。**[LOCK] P3 只扫 `eta_K` × `ee_std_tau_ms`;其余 gK 参数固定:`tau_K=5000`(postictal 慢于 STD
  恢复,§4)、`gK_max=1.0`(引擎默认);`k_K` 在 plan 阶段标定一次(取"bounded 事件内 gK build 到 ~O(1)·gK_max"
  的值)后固定,不进扫描轴。** 否则 P3 有隐藏自由度。
- **诊断读出(per-run,非扫描)`(⟨x_dep⟩, ⟨q_I⟩)`**:弛豫振荡投影。**`x_dep` 现在只活在 loop 内
  (`kick_probe.py` L186)、无输出 → 需 §8B 的 `dump_ee_std_trace` instrumentation 才能取。**
  - `⟨x_dep⟩` = `x_dep`(per-E-neuron)在 axis/active E 群上的均值;`⟨q_I⟩` = `q_I`(lattice 场)在同一 axis
    区上的均值。**两者表示不同(per-neuron vs lattice),读出时必须映射到同一 axis 区**(§6.2)。
  - 预期干净终止轨道:间期 `(1,1)` → onset `q_I↓, x_dep≈1` → bounded `q_I≈floor, x_dep↓` → termination
    `x_dep<crit → M(t)↓` → recovery `q_I↑, x_dep↑`(半闭合慢轨)。
  - 失败形态:卡在 `q_I≈floor, x_dep>crit`(STD 不够)或直冲静息且再触发不了(STD/衬底过压)。

---

## 6. 臂（arms）—— 别与 pass-1 spec 臂 / 提案臂混号

| M4-2 臂 | 配置 | 目的 | 用哪些平面 |
| --- | --- | --- | --- |
| 0 | pool only(`use_SG`,`ee_std_u=0`,`eta_K=0`)| 要终止的 bounded 态(= pass-1)| baseline |
| 1 | pool + STD(`ee_std_u>0`,`eta_K=0`)| **主信号**:STD 能否终止 | P1,P2 |
| 2 | pool + 强 gK,无 STD(`use_gK`,`k_K>0`,`eta_K>0`,`ee_std_u=0`)| 对照:"只加疲劳不够" | 少数标定 cell |
| 3 | pool + STD + mild gK | 防反跳 / postictal | P3 |
| 4 | STD 乘性 vs matched-subtractive | **机制归因**:证 STD 靠削自持、非普通负电流 | deferred(D6) |

**Arm 4 = deferred mechanism-attribution。** gated on ≥1 terminate-clean candidate。它要在线拿 `x_dep`(或等价
depression signal)算 `-η(1-x_dep)` matched negative current 注入 —— `x_dep` 活在 `simulate_kick` 的
integration loop 里,现有 `--mechanism`(池的 divisive-vs-subtractive)**不是** STD 的,所以 Arm 4 要新增一条
STD-subtractive hook / loop 分支(§8)。**不阻塞 P1/P2。**

**遍1 最小消融 = Arm 0 vs Arm 1**(Arm 2 作负对照)。Arm 3 只在 Arm 1 出 terminate-clean 但有 rebound 的区域跑;
Arm 4 最后。

---

## 7. 验收门（success gates）

### 7.1 双字段 schema（承重,防"安静尾巴"误读为"回到间期")

每个 cell(一次长跑)输出**两个独立字段**,不合并:

```
termination_class ∈ { persist, terminate_clean, fade, fragment, suppress, rebound }
retrigger_probe   ∈ { pass, fail, not_run }
```

- `termination_class` = 事件**形态**:
  - `persist` = 有界但不熄灭(pass-1 现状)。
  - `terminate_clean` = 高平台 → **相对陡的 offset** → 安静尾巴。(不是单调衰减!)
  - `fade` = 单调衰减到静息(**不算终止**,是衰减 transient)。
  - `fragment` = 碎裂成断续局部 burst。
  - `suppress` = STD/gK 过强,直接压死(无像样事件)。
  - `rebound` = 熄灭后自发再点火成 burst。
- `retrigger_probe` = **独立的再触发探针**:在慢变量充分恢复后(§4,`t_reprobe` > tail + ~few×`max(ee_std_tau_ms,
  tau_q)`)打一次**新 kick**(**nonzero** KICK_BOOST on source core —— 注意 primary 段是 spontaneous
  `KICK_BOOST=0`,只有 probe 段有 kick;需 §8B 的 state-continuous hook,不是 run_arm 补参数):
  - `pass` = 新 kick 重新点燃一次 bounded 事件(post-态是真·可再触发间期)。
  - `fail` = 新 kick 熄火(衰减 / 无事件)或直接 runaway(post-态不是可触发间期)。
  - `not_run` = `termination_class ≠ terminate_clean`(非干净尾巴不探针)。

**阈值标定 + synthetic fixture(承重,gate 编码结论、避免阈值循环):** `classify_termination` 的判据(offset
陡度 / 平台 vs 单调饱和 / fragment 断续度 / suppress 下限):
- **先在手工合成轨迹上单元测试**(合成 plateau→陡 offset、合成单调 fade、合成 fragment stutter、合成单调饱和
  runaway、合成平台 persist)—— 分类逻辑与阈值必须在**与仿真数据独立**的 synthetic fixture 上正确。否则"在同一
  批真轨迹上调阈值再分类"是循环论证。
- **再对真实实例做坏数据回归 sanity**(不作阈值来源):Arm 0(pool only)必须判 `persist`、pass-1 已知 runaway
  必须**不**判 `terminate_clean`。
- 最终数值阈值锁进本 spec 标定表 + plan。

**为什么拆两字段:** 一个 `terminate_clean` 但 `retrigger_probe=fail` 的 cell = **终止了但衬底再点不着** →
不是真间期回归,指回 `D_EE`/衬底,而不是"M4-2 成功"。合并成单 label 会把安静尾巴当成功。

### 7.2 M4-2A go/no-go（跑前锁死）

- **go(cell)** = `termination_class == terminate_clean` **且** `retrigger_probe == pass`。
- **go(plane)** = P1 上至少 `K_min` 个**连通** go(cell)(一个面积,非数值边缘),且出现在 **Arm 1(有 STD)但不在
  Arm 0(无 STD)**。
- **no-go(合法结果)** = 即便激进 STD(高 `ee_std_u` / 短 `ee_std_tau_ms`)也无 go(cell)。这**加强** memory 的
  「下一杠杆 = `D_EE`/衬底」结论(见 §10 Framing 锁),不是把 M4-2 悄悄证伪。
  - **reversibility 下界(正对照)**:复用 pass-1 `--reversibility`(`q_I`-refill 已证回不了间期)。若激进 STD 也
    终止不了 → "不可撤回"不是 depression 不够,是 attractor/衬底本身 → clean no-go。

### 7.3 M4-2B 极限环（advanced,不作 M4-2A 验收）

≥2 次自发事件、之间有清楚间期(`M(t) < baseline+cσ`,`q_I↑`,`x_dep↑`)。需自发 re-nucleation —— 均匀衬底
历史上是墙(M3A quasi-static NEGATIVE)。**M4-2A 用受控 `retrigger_probe` 绕开自发 re-nucleation 问题**;
M4-2B 才碰它。不作第一版验收。

---

## 8. 工程含义:动力学原语 / 引擎 instrumentation / runner 待接线（P1-1 精确版）

**A. 引擎动力学原语(已在,不改动力学):**
- `ee_std`(STD):`kick_probe.simulate_kick(..., ee_std_u=0.0, ee_std_tau_ms=0.0)`;`ee_std_apply` /
  `ee_std_recover_factor`;`x_dep[NE]`,深度线 L181–186 / L259 / L368–371。recurrent-E-specific(只缩 E→E 边、
  E→I 与 feedforward 不动)。
- `S_G`(divisive pool):`slow_field.SpatialSlowField.apply_currents(..., I_E_rec)` L254–277;fail-closed
  (缺 `I_E_rec` 直接 raise)。pass-1 已在 M4 worktree。
- `g_K`:`SpatialSlowFieldConfig.use_gK / k_K / eta_K / tau_K`;membrane `- eta_K*gK_E`(L266)。
- recurrent-only 电流拆路(`track_rec`/`s_E_rec`/`I_E_rec`):pass-1 已在 `simulate_kick` L170–271。

**B. 引擎 instrumentation / scheduling hook(要新增,非行为性 = off 时逐字节 parity;不改动力学方程):**
这两处**不是** `run_arm` 补参数能做完的,要动 `simulate_kick`—— 但只加 trace/schedule,不改方程。
- **`dump_ee_std_trace`(M4-2A 必做):** `x_dep` 现在只活在 loop 内(L186)、无输出。加 trace/summary 导出:至少
  active / axis / core 的 `x_dep_mean / x_dep_min / x_dep_tail`(per-step 或降采样),喂 §5 诊断
  `(⟨x_dep⟩, ⟨q_I⟩)`。不传 → 不 alloc、byte-parity。
- **state-continuous retrigger hook(M4-2A 必做):** §7.1 的 `retrigger_probe` 要"终止后充分恢复再打一发新
  kick",但 `simulate_kick`(L91)现在**只有单个** KICK_BOOST/t_kick 窗、也**不返回可续跑的膜/突触/slow 状态**。
  三条实现路(本 spec 选 (a),plan 落地):
  - **(a) 同一长仿真里第二个 kick schedule(首选):** 给 `simulate_kick` 加第二个 `(t_kick2, KICK_BOOST2)` 窗
    (或 kick-schedule 列表)。同一 seed、同一连续状态,最干净;复用现有 `kick_center`/`r_kick`,只加时间窗。
  - **(b) 导出 / 恢复连续状态:** 返回 `(V, ref, s_E, I_E, s_I, I_I, ring_*, x_dep, slow.{q_I,g_K,S_G,mu_G})`,
    第二段从该状态续跑。灵活但接口面大。
  - **(c) 近似:两遍同 seed 复现 + 固定 probe time(fallback):** 第一遍测 offset,第二遍同 seed 重跑到
    `t_reprobe` 再打 kick。**边界**:仅当仿真对 seed 完全确定、且第二遍前缀与第一遍逐位一致才成立;`t_reprobe`
    固定(非自适应到实际 offset)。不作首选。

**C. runner 待接线(`run_m4_dynamic_qi.py`,纯 config / orchestration):**
- `run_arm` 现在写死 `use_gK=False, k_K=0.0`,且**不透传** `ee_std_u`/`ee_std_tau_ms` 给 `simulate_kick`
  → STD 接线(透传);gK 接线**只在 P3 阶段**(§5 P3 LOCK)。
- 调 B 的 hook:开 `dump_ee_std_trace`、排 retrigger schedule。
- `classify_termination()` 双字段分类器(§7.1)+ synthetic fixture 单测。
- 加 P1/P2/P3 sweep grid / `--cells`(复用现成 `--sweep`/`--cells` 骨架)。
- 现有 `--mechanism` / `--reversibility` 是*池*的,**不是** STD 的。

**D. 唯一真·新动力学-adjacent 代码(碰 loop,deferred):**
- **Arm 4** STD-matched-subtractive:在 `simulate_kick` integration loop 里在线取 `x_dep`,算 matched
  `-η(1-x_dep)` 电流注入(标定到与乘性 STD 同等"被削掉的 recurrent 电流",类比池的 `trace_Irec_mean` 标定)。
  gated on ≥1 terminate-clean candidate(D6)。

**parity 红线:** `ee_std_u=0` + B 的 hook off(和 `k_K=0`、`use_SG=False`)时逐字节等旧路径;runner/hook 接好后
重跑 byte-parity smoke + 改了 `kick_probe.py` 则 re-bless `engine_versions.json`。

---

## 9. 实现顺序 & 算力

Cheap/discriminating-first(即便"全扫",内部排序;同 pass-1 spec §11「不要从大网格开始」纪律)。
**第一版 writing-plans plan = 步骤 1–4(到 P1 go/no-go 为止);步骤 5–6 是 gated 后续 plan。**

1. **引擎 instrumentation(§8B,非行为性)**:`dump_ee_std_trace` + state-continuous retrigger hook(选路 a);
   `ee_std_u=0` 且 hook off 时 byte-parity smoke;改了 `kick_probe.py` → re-bless `engine_versions.json`。
2. **runner 接线(§8C)+ 分类器**:`run_arm` 透传 `ee_std_*`(gK 先不接,P1 不用);`classify_termination()`
   双字段(§7.1)**先过 synthetic fixture 单测**,再在 pass-1 confirmed-bounded / runaway 实例上 sanity。
3. **计时一个 `T=15000` cell**(spontaneous 协议 + 一发 retrigger),把 P1 总 wall-clock 钉死写回本 spec。
4. **P1(`ee_std_u × ee_std_tau_ms`)多 seed**(Arm 0 vs Arm 1,spontaneous,gK off,带 retrigger probe)→ §7.2
   go/no-go。**第一个科学产物,第一版 plan 到此。**
   —— 以下 gated,后续 plan ——
5. 若 P1 出 go 区:跑 **P2**(池 × STD 拓宽);若 P1 出 `rebound`:接 gK 路 + **P3** + Arm 3(§5 P3 LOCK)。
6. **Arm 4** 机制归因(§8D,碰 loop)最后,gated on ≥1 terminate-clean candidate。

**算力现实:** 每 cell = 一个 `T=15000` 长跑 + 一发 retrigger 续跑。3 平面 × 各 ~6×6 × 相关臂 × 多 seed ≈
几百个长跑。第 3 步计时后给用户总预算;P2/P3/Arm4 按 §7.2/§D6 gate,不无条件全铺。

---

## 10. 本地锚 & Framing 锁

**本地锚:**
- M4 pass-1:`docs/superpowers/specs/2026-07-05-sef-hfo-m4-divisive-shared-inhibition-design.md`。
- M3A-M1 STD "time-not-space self-limiting":`docs/archive/topic4/`(M1 Stage2 NULL,2026-06-18)。
- 引擎:`src/snn_engine/kick_probe.py`(`simulate_kick`,`ee_std_*`)、`src/snn_engine/slow_field.py`
  (`SpatialSlowField`,`apply_currents`,gK)。
- runner:`scripts/run_m4_dynamic_qi.py`(`run_arm`,`--sweep`/`--cells`/`--mechanism`/`--reversibility`)。
- M3A 线长期结论:均匀衬底"压死 XOR 耗尽",下一杠杆 = `D_EE` / 换衬底。M4/M4-2 是**真新**杠杆;clean no-go
  加强那个结论。

**文献(STD 作终止器,支持性,不新增 consult):** Jacob et al.(JNeurosci 2019,activity-dependent synaptic
depression → seizure refractoriness);Kilpatrick(2009,SFA vs STD 时标:SFA/AHP ~40–120 ms、STD 恢复 ~200–800
ms → STD 更适合作持续态终止器);Chizhov Epileptor-2(PLoS CB 2018,含 STD 慢变量);Kramer/Truccolo(PNAS 2013,
human seizures self-terminate);Jirsa Epileptor(快子系统 + 慢渗透变量 = 弛豫振荡器骨架)。
**nuance:** Epileptor 谱系里 primary 慢渗透变量更接近 ionic/slow-K(`g_K`-adjacent),STD 是 complementary;
但对**本问题(打断 recurrent 自持吸引子)+ 本引擎(已有 recurrent STD)**,STD-primary 是对的。

**Framing 锁:** 结果措辞必须说 "actual M4-2 **SIMULATION** trajectory",**绝不**说 "real data"。
`terminate_clean` + `retrigger pass` = **模型仿真里** STD 能把有界态终止成可再触发事件;不等于"证明发作周期成立"。
任一结局(go / clean no-go)都如实报;clean no-go 把下一杠杆指回 `D_EE`/衬底,而不是继续调 STD。
