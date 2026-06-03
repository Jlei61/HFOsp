# LIF-transfer route (Step 0a/0b 换传递函数) — 2026-06-03

> 用户决定：不进 Step 1、不再大扫 sigmoid；换传递函数（sigmoid → LIF colored-noise），**留在 Step 0a/0b**，最小手术。按用户三步顺序执行。脚本：`scripts/sef_hfo_transfer_preflight.py`、`scripts/sef_hfo_step0a_lif.py`、`scripts/sef_hfo_step0b_lif.py`。

## 一句话结论

**换 LIF 传递函数后，有限脉冲闸门的核心机制成立了——self_limited_propagation 实现了（点着→传播→自终止），这是 sigmoid 结构上做不到的。** 但事件时长（~49–88ms）在实测通道展布（50–150ms）的低端、短于事件包络（100–300ms），空间范围（~7mm）亚 cm。**所以 Step 1 仍按纪律等：先把时长/范围定量锚到数据，再重跑 0b 确认数据匹配窗。**

## 测了什么 / 怎么测的 / 揭示了什么

**Step 1 — transfer preflight（PASS）**：LIF Siegert f-I 在**低发放率**（1–10Hz）就有**高 loop gain**（2–5，落在行波/Hopf 需要的 ~3–4），因为近阈值涨落驱动的 f-I 很陡 + 复发权重大（τ_m·C·w~2520）。sigmoid 在低率处增益≈0（平尾），最高实现 loop gain 仅 1.26。→ 换传递函数有理由。

**Step 2 — Step-0a 自洽稳态 + 线性图**：用 Brunel LIF mean-field（Siegert）替掉 sigmoid。mean-field 复现 coworker1 的低率 SI（E ~1Hz）。线性图 → 低率工作点上出现**有限-k Hopf @ ~25–29Hz**（k*~2.4/mm），与 coworker1 经验行波吻合（不再注入假增益）。**修正（重要）**：committed `step0a_lif.py` 的 mean-field 用了 0.5-阻尼定点迭代，在 loop gain~3（斜率>1）时不收敛 → 给了**偏高工作点、夸大了近临界**（误判 ratio≥1.15 失稳）；改用 `fsolve`（0b 用的）后真工作点更低（~0.2Hz）、更稳健亚临界，真正的 Hopf onset 在更高 drive。**待办：用 fsolve 重跑 0a 定位真 Hopf onset / candidate band。** 0b 闸门已用正确的 fsolve。

**Step 3 — Step-0b 有限脉冲闸门（核心，部分通过）**：在 LIF 率场上重跑有限脉冲（偏心刺激 + 沿轴波前度量；recovery 关/开并列）：
- **self_limited_propagation 实现了**：A<4 熄灭；A≈4 局部鼓包；**A≥6 自限传播**——点着（率 0.2→64Hz）、波前**单调推进**（t=20ms@0mm → t=60ms@5.9mm，~7mm）、然后**自终止**（~120ms 前消失，dur~49–88ms）、保持**局部**（max_ext<0.28，从不全局/失控，A 上至 24 都不失控）。
- **全或无可激脉冲特征**：波前推进距离 **~6.9mm 与刺激幅度无关**（A=6→24 都 6.9mm）——脉冲形状由介质定、不由刺激定，这是真可激脉冲的标志。
- recovery 关/开结果几乎一样（亚临界振铃已自终止，recovery 此处非必需 = "Brunel 支" 主导，"Pinto–Ermentrout 支" 备选——与之前的拆分一致）。
- 4/4 候选工作点（ratio 0.95–1.10）都有自限窗；幅度安全余量稳健（上至 A=24 不失控）；工作点上至 ratio 1.26 仍稳健亚临界。
- **与 sigmoid 的决定性对比**：sigmoid 率场完全不传播（局部熄灭或饱和无余量）；LIF 率场点着→传播→自终止。**差距确实在传递函数层，已证。**

## 与数据的定量对照（这是 Step 1 前还差的一步）

- **时长 ~49–88ms**：在实测**通道激活展布 ~50–150ms** 的低端 ✓（量级对），但**短于事件包络 ~100–300ms**。且在 ratio 0.95–1.26 内调 drive **不变长**（fsolve 工作点稳健亚临界、~0.24Hz、|maxRe| 不趋 0）。要变长需更接近 Hopf（更高 drive，逼近真 onset）或换增益旋钮。
- **空间范围 ~7mm**：亚 cm，vs 数据 ~cm 电极覆盖。要 cm 级需放大核尺度（Brunel 是 mm 级运动皮层核；SOZ patch 可能更大）。

## 对 Step 1 的判定

**核心闸门（self_limited_propagation 存在 + 稳健远离失控）满足。** 但用户纪律明确要求"事件时长/空间范围贴近真实数据才进 Step 1"——当前时长偏短（短于包络）、范围亚 cm。**所以 Step 1 仍等**，建议下一步：(a) fsolve 重跑 0a 定位真 candidate band / Hopf onset；(b) 把时长锚到数据（逼近临界 + / 或 recovery τ_a 锚事件时长 → 更长事件）、把范围锚到 cm（放大核尺度到电极覆盖）；(c) 在数据匹配的窗上重跑 0b，确认时长/范围贴数据，才进 Step 1。或用户判断"50–150ms 通道展布低端在量级内可接受"则可进。

## cm-scale 探查（extent 与 duration 分别由什么定）

把连接核尺度 ×4（Brunel mm → ~cm patch；连接**强度** C·w 不变 → 同一临界性/工作点，只放大空间），grid 48mm：
- **波前推进 ~27.6mm（~2.8cm）→ 达到数据 ~cm 尺度 ✓**（空间范围可锚到电极覆盖）。
- **但时长仍 ~70–83ms，没变长**——空间放大后波也传得更快，渡越时间不变。**时长由时间动力学（振铃率 ~|maxRe|、时间常数）定，不由空间尺度定。** ~70–83ms 在实测**通道展布 50–150ms 范围内 ✓**，短于**包络 100–300ms**。
- 结论：**extent 与 duration 是两个独立旋钮**——空间核尺度定 extent（可锚到 cm）；时间动力学定 duration（要更长需更接近真 Hopf onset 或 recovery τ_a 调，**不是**放大空间）。当前 ~70–83ms 是量级内、偏短端。

## 边界（诚实）

LIF transfer 是白噪声 Siegert + 固定 σ（工作点值）+ 单极突触（无显式延迟）+ ρ=0.6 椭圆核高斯近似——近似版。定量数字（频率、时长、k*）会随精化（colored-noise 修正、动态 LIF 传递函数 H(ω)、显式延迟、σ 动态）而变。但**定性结论稳**：self_limited_propagation 在 LIF 传递函数下成立、在 sigmoid 下结构上不成立。

（内部归档代号：transfer preflight、Siegert nu(mu,sigma)、loop gain G_E·J_EE、finite-k Hopf、self_limited_propagation、A_event/A_self_limited/A_runaway、front-advance metric、all-or-none excitable pulse、sub-critical Brunel branch vs Pinto–Ermentrout recovery、fsolve vs damped mean-field、coworker1 Jlibrary/ei_snn_scaffold ground truth）
