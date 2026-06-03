# 数据锚定 Step-0b 有限脉冲闸门探索（2026-06-03）

> 脚本 `scripts/explore_datalocked_step0b.py`；产物 `results/topic4_sef_hfo/finite_pulse/datalocked_step0b_exploration.json`。**结论：有限脉冲闸门没过——没找到 self-limited propagation。Step 1 保持锁定。** 这是一个合法发现，不是失败（计划早锁定"`fraction_with_window=0` 也是发现"）。并且定位到了一个清晰的结构性原因 + 一条原则性的修法。

## 测了什么

Step 1 的真正前提**不是**"有没有行波 Hopf"，而是**有限脉冲闸门**：在一个数据锚定的工作点族里，一个有限大小的扰动能不能**点着 → 传播一段 → 自己停下来**（self-limited propagation，且失控幅度高于点着幅度、留正安全余量），并且事件时长/空间范围贴近真实数据。这一步我没有跳过。

## 怎么测的（按 advisor 拆解）

把时间结构锚到上一步的发现（慢抑制 ~18ms）+ Topic 1/2 实测的事件级尺度（单次事件 ~100–300ms、传播展布 ~50–178ms、cm 级电极覆盖网格）。然后**先只看"能不能传播"**（恢复变量关掉）：从**偏离中心**打脉冲，沿传播轴量**波前位置**（不是用居中圆盘的质心——那对"径向往外扩"是盲的，会把会扩的波误读成不动）；用**波区连接**（兴奋范围 ≥ 抑制范围，避免 Amari 的"侧向抑制 → 原地鼓包"区）；并**审计稳态实际增益**。

## 揭示了什么

**闸门没过：没有任何工作点给出 self-limited propagation。** 两类工作点都不行：

- **饱和高基线**（静息发放率 rE0≈1.0）：基线已经顶到天花板，**没有余量**——脉冲推不出可检测的事件。
- **低静息基线**（rE0≈0.02，有余量）：脉冲只在**刺激盘内局部响应一下就熄灭**，波前只到刺激盘半径（~4mm）就不再前进——**不招募邻居、不传播**。

**结构性原因（关键，evidence-backed）**：sigmoid 率场**放不出"低静息 + 高增益"的稳定工作点**。在探索范围内，稳态的实际环路增益 `G_E·J_EE` 最高才 **1.26**，远低于行波/Hopf 需要的 ~3–4。原因有二、互相咬死：(i) 高的静态环路增益会让静息态**失稳**——它是不稳定的分界面，不是稳定不动点；(ii) sigmoid 在低发放率处很**平**（增益低）。所以"低静息可激 + 上阈高增益再生传播"这件事，在 sigmoid 传递函数里**本质上拿不到**——要么坐在饱和态（无余量），要么坐在低平态（无再生招募）。

**与 Exploration 1 的关系（不矛盾）**：Exploration 1 是把高增益(3–4)**直接注入**色散看的，绕过了自洽稳态——所以它显示"行波 Hopf 机制可达"。但 sigmoid 率场**自洽地放不出那个稳定的高增益态**。差距不在连接、不在恢复变量、也不在慢抑制——**在传递函数层（transfer function）**。

## 原则性修法（不是调参；advisor 同向）

把率场的"阈值-发放曲线"从 **sigmoid** 换成 **LIF 有色噪声传递函数**：LIF 神经元的 f-I 曲线在**低发放率、近阈值处很陡**（低静态发放率却有高动态增益）——这正是 coworker1 的脉冲网络能从低态传播的原因，也正是 Exploration 1 注入高增益所模拟的东西。两条具体路：

1. **换 transfer**：率场 `F_eff` 用 LIF colored-noise 形式（Brunel f-I + susceptibility）重做 Step 0a/0b——预期能放出"低静息 + 高增益"的可激稳态 → 有限脉冲可再生传播。
2. **直接用脉冲网络做底座**：coworker1 的 Brunel-LIF（`Jlibrary/ei_snn_scaffold/`）**已经会从低率态传播**；可以用它做 Step 0b 的有限脉冲闸门 + Step 4 底座，率场只作便宜的线性地图 companion。

## 对 Step 1 的意义

**Step 1 保持锁定。** 有限脉冲闸门的前提（self-limited propagation + 正安全余量 + 事件时长/范围贴数据）**未满足**——而且不是"参数没扫到"，是 sigmoid 率场的结构限制。要解锁，先做上面的传递函数修法（或换脉冲底座），再重跑 Step 0a/0b 闸门。

## 边界（诚实）

不是穷举证伪。是 advisor 指定的几项检查（波区连接 + 偏心波前度量 + 高/低基线两类工作点 + 增益审计）都指向"传不起来"，加上一个强结构性理由（稳定高增益态不存在）。没有 torture 参数去硬凑一个 pass（计划纪律 + advisor 都明确：honest null 是合法结果）。max 环路增益 1.26 是探索范围内的代表值，背后有"稳定静息态要求环路增益偏低"的理论原因，不是单纯欠采样。

（内部归档代号：finite-pulse gate self_limited_propagation / A_event / A_runaway / safety_margin、loop gain G_E·J_EE、saturated vs low-quiescent rest、Amari bump regime vs wave regime、front-advance metric off-center、sigmoid vs LIF colored-noise transfer、coworker1 Jlibrary/ei_snn_scaffold、Exploration 1 injected-gain Hopf、data_locked provenance、Step 1 LOCKED）
