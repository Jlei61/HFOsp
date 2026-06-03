# Exploration 1：率层色散能否"重现并解释"coworker1 的 Brunel-LIF 行波（2026-06-03）

> 脚本 `scripts/explore_brunel_rate_dispersion.py`；产物 `results/topic4_sef_hfo/linear_stability/exploration1_brunel_dispersion.json`。**这是廉价的率层机制确认/解释，不是独立定量预测**——真值（ground truth）是 coworker1 的脉冲仿真（在 Brunel Table-1 参数上跑出 ~29 Hz 的有限-k 行波）。

## 测了什么

把 coworker1 那套真参数（Brunel 论文 Table-1）的**连接形状 + 时间结构**搬进我的线性色散分析，看率层（不跑脉冲、只解特征方程）能不能**重现并解释**它脉冲网络里那条"会移动的 beta 波"——以此判断"会移动的自限事件"这个目标在率层框架里到底可不可达（= 能不能解锁 Step 1 的关键一刀）。

## 怎么测的

复用 SEF-HFO 率层色散结构（`src.sef_hfo_stability` 的 2×2 特征行列式），喂 Brunel 映射参数（膜 τ_E=20/τ_I=10、AMPA 衰减 3.5、各向异性 E→E、抑制核 0.25、权重比 = in-degree×单突触权重），把"阈值-发放曲线"的增益作为注入量扫描。做四件事：(1) 扫总增益找临界点的最不稳模 k\* 与频率；(2) **机制对照**——慢抑制（GABA 衰减 18ms）对比 scaffold 式快抑制（2ms）；(3) **增益比稳健性**——G_I/G_E ∈ {0.5,1,2}（堵"只扫对角线"的盲点）；(4) 看 Re λ(k) 曲线形状。

**近似（必须讲清）**：单极突触滤波（真实是双指数差）、静态注入增益（真实是 LIF 动态有色噪声传递函数）、ρ=0.6 椭圆核用高斯近似、延迟用常数/零。所以这是"机制对不对得上"的筛查，不是"频率/波长精确预测"。

## 揭示了什么

**稳的结论（机制）**：**慢抑制（GABA 衰减 ~18ms）把原本静止的 k=0 模变成振荡（Hopf）失稳**。机制对照坐实：其他都不动、只把 GABA 衰减从 18ms 换成 scaffold 的 2ms → 振荡完全消失，回到静止的 k=0（纯实特征值）。这是经典的"慢/延迟抑制 = E-I 振荡来源"，也**正好解释了我 scaffold（GABA=2ms）为什么只得到静止 Turing、拿不到行波**。增益比 0.5/1/2 上 Hopf 都在（不是对角线假象）。

**较弱的结论（有限 k）**：这个 Hopf 偏好"有限 k"，但偏好**很浅**（k=0 几乎同样不稳，Re 从 −0.001 升到 +0.011，摆幅仅 ~0.012）。所以它是**长波长的扫过式梯度**（k\*≈2/mm → 波长 ~3mm ≈ 网格尺度），不是锐选的短波长——这恰好和 coworker1 在 1–2mm 亚波长网格上看到的"波前扫过阵列"一致。k\* 与频率随增益比变（k\* 1.5→2.4、频率 12→29 Hz；29 Hz 落在 G_I/G_E=2）。

**措辞纪律**：率层"**重现并解释**"了机制（慢抑制驱动的有限-k Hopf，频率在 ~12–29 Hz 的对的量级），**不是**"独立定量预测了那条波"——近似不支持精确定量匹配，真值是脉冲仿真。

## 解锁 Step 1 的逻辑链（明确写出）

有限-k Hopf 存在 → 存在一个**振荡分岔可以"坐在它下方"** → 亚临界 + 噪声触发 → 一段段会移动又自己衰减的**瞬态波包** = 时间离散的**自终止传播事件** = 我们要的候选窗（Step 0b 的 self-limited propagation + 正余量）。

**recovery 是拆分不是替换**（防止反向过度修正）：慢抑制 → 会移动的**振荡（Hopf）**；亚临界 / recovery → **自终止 / 离散**。两个杠杆管两个不同特征，不是"慢抑制取代 recovery"。

## 对 Step 1 解锁的意义（暂定，待 Exploration 2 收口）

**强烈倾向 data_locked 路径可行**：只要把时间结构（尤其**抑制衰减时间常数**）锚到 Brunel/真实数据 + 各向异性 E→E，率层 re-run 大概率能在某个工作点拿到"会移动的自限窗"。**但完整 green light 必须等 Exploration 2 的数据**：如果实测的**有效抑制时间尺度不够慢**（或实测事件包络/lags 与"慢抑制 + ~20Hz Hopf"不自洽），"慢抑制"这条故事就要调整。在那之前，这只是"机制可达性确认"，不是"Step 1 解锁批准"。

（内部归档代号：rate dispersion `char_det`、finite-k Hopf vs static Turing、slow GABA τ_d=18ms、G_I/G_E gain-ratio、operating-point gain scan、sub-critical noise-kicked packet、Brunel Table-1、coworker1 `Jlibrary/ei_snn_scaffold/`）
