# MZ conductance 阶段反思与下一版动力学路线

日期：2026-07-20
范围：`codex/topic4-mz-conductance` 独立 worktree；不合并、不复用另一条并行探索线的结果

## 1. 一句话判断

当前更新解决了“膜方程是否合理、L=20 间期工作点是否存在、Z 是否能形成稳定 event-locked 推进”三个前置问题，也首次把终末 runaway 的空间表型画清楚；但 protected additive-global GABA 仍是一个瞬时、空间均匀的代数刹车，它只能移动 runaway/prevention 边界，不能单独创造一个有界、可恢复、具有自身空间组织的发作态。

## 2. 当前 paper-ready 视觉证据

正式输出：

- `results/paper-ready-figure/fig_mz_conductance_current_dynamics/figures/mz_conductance_current_dynamics.png`
- `results/paper-ready-figure/fig_mz_conductance_current_dynamics/figures/mz_conductance_current_dynamics.pdf`
- metadata：同目录 `mz_conductance_current_dynamics_metadata.json`
- compact source artifact：由 `results/topic4_sef_hfo/mz_conductance/latest_figure_capture.json` 指向

图使用同一条 `L=20 / seed=1 / beta=1/12 / Z on / M off` 自发连续轨迹，布局为：

`mechanism | returning event | early runaway | continuous electrode readout`

定量核对：

| 读出 | returning event | early runaway |
|---|---:|---:|
| 时间窗 | 5113–5163 ms | 7150–7280 ms |
| 窗内招募 E cells | 7963 / 32000 | 15008 / 32000 |
| onset-axis Spearman | **+0.959** | **−0.118** |
| onset-perpendicular Spearman | −0.007 | +0.043 |
| axis P95 span | 15.38 mm | 19.05 mm |
| perpendicular P95 span | 6.77 mm | 9.86 mm |
| source/sink core median onset-rank | 0.092 / 0.952 | 0.291 / 0.287 |

这说明当前模型不是“完全没有空间变化”：returning event 是清楚的 source→sink 轴向时序；terminal early-runaway 招募更广、两个核接近同时、原来的方向梯度丢失。但这个变化与 runaway 绑死，不能称为一个独立发作态，更不能称为可恢复时空相变。

连续时间轴同时复现：19 个 pre-runaway returning events、core `D=1-z` 阶梯上升、7180.1 ms runaway。capture 数值安全：`tau_eff_min=0.280 ms > 2dt`，`clip_fraction=0`，峰值 RSS 6.80 GiB。

## 3. 设计里正确的部分

### 3.1 把抑制从加性 current proxy 改成 conductance 是正确的

E-cell 膜方程现在显式包含 reversal potential 和 conductance-dependent effective time constant。它把“抑制只减一个电流”改成“抑制同时改变稳态电位和膜时间常数”，比旧 subtractive proxy 更接近我们要讨论的 shunting / restraint。跨 seed 的间期工作点和数值 gate 都通过，说明这个改动不是靠数值爆炸制造结果。

### 3.2 保留 local restraint、再加 protected global 项是正确的方向性对照

replacement-global 会先拿走 core 的强 local inhibition，实测 beta 越大越早 runaway；additive-global 保留 local 项后才形成 `runaway → near-prevention → suppress` bracket。这个结果清楚说明 local/global 竞争不是简单“全局比例越大越稳定”，而是必须区分：是否牺牲局部抑制、global 项是否随同一个 Z 一起耗竭。

### 3.3 Z 路线已经成为稳定的 onset/bridge 变量

Z 在两个 primary seed 都给出多个 returning events 后的单调 event-locked staircase，再进入 runaway。它适合作为 interictal→critical 的超慢推动变量：前序事件留下记忆，系统逐步接近转换。它当前不适合承担终止，因为 Z 耗竭和活动构成正反馈。

### 3.4 时空读出必须和状态判读同时存在

新图证明只看 rate 会漏掉一个关键事实：同一条轨迹中，间期事件仍保留有序轴向梯度，而 runaway 时方向结构塌缩、空间范围扩大。后续 gate 不能只问“高不高、振不振”，还必须问“传播是否仍有前沿、是否扩大、是否变成同步/无方向、能否恢复原模板”。

## 4. 设计里不正确或仍不够的部分

### 4.1 additive-global 不是一个新的动力学状态变量

当前 global 项是

\[
g_{I,i}=z_i g^{I}_{i,local}+\beta\langle g^{I}_{local}\rangle.
\]

它随当前 received GABA 立即变化，没有独立存储、激活阈值、延迟、饱和或恢复时间。因此它能改变 fast subsystem 的阻尼和 runaway 阈值，却没有给 state space 增加一个可形成新吸引子/极限环的慢维度。把 beta 调得更细，只会继续在 runaway 与 prevention 的陡边界上找点，不会自然得到一段有界发作。

### 4.2 当前 M 虽是慢变量，但反馈时机不对

线性 spike-count M 从第一个间期 spike 就开始累积，没有“只有高招募后才打开”的阈值，也没有明显饱和段。结果符合直觉：M 弱时压不住 runaway；稍强时先杀掉间期事件或把系统推入 prevention。它没有把“允许 onset”和“延迟 termination”分开，因而没有形成发作平台。

### 4.3 Z 与 M 被同一局部活动同时驱动，缺少反馈层级分工

理想生命周期需要至少三种不同角色：超慢变量把系统推近临界；较快正反馈维持一段高招募；延迟负反馈在高态内累积并终止。当前 Z 与 M 都直接看局部 activity，且 M 从低态就介入，两个变量没有形成清楚的先后相位关系，所以网络在“点不着 / 点着后压不住”之间跳变。

### 4.4 空间信息目前主要来自固定 scaffold，慢变量没有产生传播前沿的 refractory wake

Z/M 是 postsynaptic cell state；global 项又是空间均匀的 rank-1 mean。它们会改变哪里容易放电，但不会直接耗竭已经被传播波使用过的 E→E relay。于是 once runaway begins，固定 long-axis recurrent excitation 仍可反复驱动已经经过的区域，没有“前沿走过后该处暂时不能再接力”的空间自限结构。图中的 early-runaway 因而是方向梯度崩溃后的广招募，而不是一个有清楚前沿、尾迹和终止的 ictal wave。

### 4.5 当前 terminal state 不是 seizure attractor

runaway 判据命中的是持续高率并触发 early stop。它是计算上和科学上都要避免的饱和端点，不是 bounded high state。当前结果没有证明 bistability、Hopf/limit cycle、hysteresis，也没有证明同一参数下存在可返回的 ictal basin。

## 5. 对“继续加电流/加性设计是否能产生真正空间切换”的直接回答

可以帮助，但不够。

- 作为 baseline restraint 和工作点控制，conductance/additive-global 是有价值的；它让我们知道 local restraint 不能被 global mean 替换，也给出了可重复的 runaway/prevention bracket。
- 作为“间期态→发作态→恢复”的核心生成机制，它不够。一个没有独立时间尺度的代数项通常只移动分岔位置；一个从低活动就线性累积的 M 通常只把边界从 runaway 推到 suppression。
- 当前确实出现了空间表型变化，但它是 terminal runaway 的从属现象，而不是可分析的第二状态。真正目标应是让空间模式变化先成为 bounded state，再让它退出并恢复间期模板。

## 6. 我们这条独立路线：把恢复机制放到 recurrent E→E scaffold 的空间资源上

> **2026-07-20 设计修订**：普通 per-spike STD 已在 M1/M4-2 做过，并暴露出 onset 前耗竭与 fragment/suppress 问题；因此本节原始 `x_j` 方程只保留为思考起点。下一版不直接扫 `U_x × tau_x`，而先补齐 E-cell AMPA/GABA full conductance（I-cell 保持原 current 路径），确认 finite high branch，再使用 persistence sensor 门控的 local presynaptic relay resource。binding candidate：`docs/superpowers/specs/2026-07-20-topic4-mz-full-conductance-spatial-relay-design.md`。

为与另一条并行的 Abbott/global-inhibition 路线分开，本线下一步不继续细扫 beta，也不先复制另一套全局抑制池。主更新只加一个 **presynaptic E→E resource** `x_j(t)`：

\[
\tau_x\dot x_j=1-x_j-U_x x_j r_j(t),\qquad 0\le x_j\le1,
\]

并让来自 presynaptic E neuron `j` 的 recurrent weight 变成

\[
W^{EE}_{ij,eff}(t)=W^{EE}_{ij}x_j(t).
\]

角色分工：

- `Z_i`：保留为 2.5–5 s 的超慢 permissivity / onset bridge；多个间期事件逐步把系统推近转换。
- `x_j`：作为 0.2–2 s 的空间使用依赖资源。传播前沿经过一处就耗掉该处 outgoing E→E relay，留下暂时低可传播性的 wake；安静后恢复。
- protected additive-global conductance：只保留为已锁定的 baseline restraint，不再假装它是 ictal terminator。
- 当前线性 M：先从主模型关闭，后续只作 ablation。这样能判断 termination 是否真正来自 E→E relay depletion，而不是多个刹车叠加后无法归因。

这条更新直接针对当前图暴露的空间问题：既要允许 recruitment 从轴向间期模式扩展，又要让已经经过的区域暂时失去重复接力能力，从而给高态一个空间上的自限与退出通道。

## 7. 最小执行顺序

### Step 0：锁回归锚

不改当前已确认参数：`gaba_gain=1.125`、`beta=1/12 additive protected`、`q75`、`tau_z=2.5 s`、L=20、seeds 1/3。要求复现工作点、Z staircase、runaway 时间量级和当前 paper-ready 空间图。

### Step 1：只加 `x_j`，先回答“runaway 能否被变成 bounded recruitment”

先做 cheap grid：`U_x × tau_x`，M off，不加新的 dynamic global pool。每个点同时保存 rate、Z、mean/core `x`、source-space onset map、axis/perpendicular extent、15-contact readout。

目标不是立刻找完美 seizure，而是把 terminal runaway 至少分成三类：

1. 仍 runaway：资源太弱；
2. 单个短 blip / suppression：资源太强或太快；
3. bounded high-recruitment bout 后回落：进入下一步。

### Step 2：只有出现 bounded bout，才检验是否是真正 ictal state

- 高招募持续时间相对 interictal event 明显拉长，但不命中 runaway；
- source-space 招募超过间期 P90，同时不是瞬时全片同步；
- 有可重复的传播前沿或多波，而不是一个高率 plateau；
- bout 后连续回到同 seed baseline band；
- 2 s 尾窗内再次出现 returning interictal event，证明不是永久沉默；
- seeds 1/3 同方向。

### Step 3：若 `x_j` 只给“一下即灭”，再加 thresholded dynamic global brake

第二配料只在 Step 2 需要时加入：

\[
\tau_G\dot G=G_\infty(\bar r_E)-G,\qquad
G_\infty(r)=G_{max}\frac{r^n}{r^n+r_G^n}.
\]

`G` 只在广招募后跨阈值累积，用于延迟终止；它与当前瞬时 `beta<g_I>` 必须分开记名和做 ablation。这样才能把“baseline global restraint”和“ictal-state delayed feedback”分成两个物理角色。

### Step 4：动力学确认，不只靠时间轨迹

对候选冻结 `(Z, mean x, G)` 做 fast-subsystem response map，并做 Z 上扫/下扫与不同初值试验。只有看到 bounded limit cycle、hysteresis/bistable basin，或同一参数下稳定的 interictal→ictal→recovery closed orbit，才能把结果叫真正的状态切换。否则仍只叫 trajectory phenotype。

## 8. 与三条后续 workflow 的接口

在找到上述 bounded/recovered state 之前：电气相变、ecomode 空间相变、early-ictal bridge 三条线都不应继续做正式统计，因为它们现在只会分析“间期→runaway”。候选通过后再分别消费同一状态定义：

- 电气线：fast/slow phase portrait、hysteresis、limit-cycle/transition timing；
- 空间线：onset gradient、axis/perpendicular spread、front velocity、wake recovery、ecomode occupancy；
- early-ictal bridge：用 Z/x/G 轨迹定义 onset-near trend，而不是把 rate 单调上升当成临界转换。

## 9. 当前安全结论

当前模型已经能自发地产生：稳定间期事件、event-locked Z 消耗阶梯、从有序轴向传播向广招募/无方向 terminal runaway 的时空转变。它还不能产生：有界发作态、可恢复振荡、双稳态/极限环、发作后恢复间期模板。
