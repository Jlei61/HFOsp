# fig_mz_early_bridge_v2_zm_tau500 — 图说明（中文）

本目录是 MZ early-field bridge **V2（z+m，τ_adp=500 ms）** 的 paper-ready 图。**2026-07-22 已锁定为 Figure 5 上半部分的正式布局**：当前整张复合图作为一个不可拆换的上半部 block，后续 Figure 5 下半部分另行设计与拼接；除非用户明确解锁，不再改变 panel 数量、相对宽度、时间锚点、标题、legend、colorbar、spatial-probe glyph 或坐标标签。主体沿用连续 readout + 双空间场图语，并在右列加入 z–m 慢状态轨迹和 baseline/early-onset frozen-q mode context。

### fig_mz_early_bridge_v2_zm_tau500.png

上排左侧是一条**连续的 z+m native Virtual-SEEG 轨迹**（seed1，非拼接）：橙色是 ICL 杆的 11 个触点、青色是 SCL 杆的 4 个触点；`Virtual-SEEG (30–80 Hz)` 放在纵轴，不再另设面板标题。右上角图例只保留两个真正需要解释的标记：红色 early onset 和蓝色 TB sample event，不再重复列出 ICL/SCL。蓝窗是按固定规则选出的**一个间期样返回事件**（规则：`t_recruit` 之前、多数 slow-off 方向里最后一个合格事件，选取不看目标能量）。红虚线定义为图中统一使用的 **early onset**，即 operational `t120 − 120 ms`（显示坐标约 986 ms，对应绝对时间约 12836.2 ms）；原先的粉色 shading 已删除。operational `t120=12956.2 ms` 只作为 provenance 保留在 metadata 中，不再画在图上。上排右侧是同一 z+m 条件的一条 seed1 自然慢状态轨迹：横轴为去抑制 `D=1-z̄`，纵轴为适应 `a`；深蓝轨迹越过红色虚线后进入极淡红区。图中的 `𝒮` 是**首次 operational-runaway crossing 的示意边界**，不是解析拟合或已证明的 separatrix。

下排从左至右是**等宽排列**的四张空间图，横轴统一为 `TA shared axis (mm)`，最左图纵轴为 `y (mm)`：前两张分别为蓝窗事件的**触点发放次序**（viridis，rank 1 早→rank 11 晚）和 **early-onset energy**（Blues）；event-order 色条端点改为纯数字，并加宽两个中间色条与相邻 panel 的留白，避免端点数字侵入下一张图。能量窗仍使用原注册的 `t_recruit` 后 0–50 ms，换算到统一的 early-onset 时标后为 −55.7 至 −5.7 ms，因此位于红线前紧邻 early onset 的增强阶段。两张场沿 shared axis 同向抬升。右侧两张图是三 seed 平均的 frozen-q rate-field leading-mode loading，`Baseline mode` 近全局/各向同性（axis score 0.055，globality 0.985），`Early-onset mode −120 ms` 沿 shared axis 集中（axis score 0.860，globality 0.135），共用同一 magma 色标。Baseline panel 左上角新增一个灰度的局部 E-rate spatial probe 及输入箭头，**只用来示意如何对冻结空间系统施加小扰动**；leading mode 本身仍由 frozen Jacobian 求得，不是该特定 probe 的有限时间 response，也没有新增一次仿真。−120 ms 模式不是把 −100 ms 结果改标签，而是先在逐神经元层面对 −200 与 −100 ms 的 z 状态插值，再重新求 frozen-q operating point 和 leading mode；它仍是线性化 rate-field 的机制背景，不是从上方 z+m SNN trace 直接辨识出的 empirical full-SNN eigenmode，也不编码传播方向。灰点只表示固定 E-neuron 几何背景，**不表示局部招募**；operational runaway 是模型代号、不是临床发作，30–80 Hz virtual-SEEG readout 不是临床宽带功率。正式统计仍来自 slow-off held-out 双向模板的 maxAB（不是图里这一例的描述性 earliness）。

**关注点**：先看红线处的 Virtual-SEEG 是否已出现清楚的早期增强，再看右上角单条慢轨迹是否从低 `D/a` 状态穿过 operational boundary；最后比较下排前两张场的长轴梯度是否同向，以及 mode 是否由 baseline 的近全局形态转为 −120 ms 的轴向集中。

### fig_mz_v1_v2_paired_diagnostic.png

一张紧凑的**三 seed 配对诊断图（不是主图 Figure 5）**，把"只有去抑制"的 V1 和"去抑制＋快速适应"的 V2 在**同一个噪声 seed 上并排**比较。左图 = 每个 seed 同 seed 的间期时序轴对失控前能量的方向无关相关（maxAB），蓝 = V1、红 = V2，星号表示"把触点在各自电极杆内部随机重排一万次"这个检验的 p<0.05。右图 = 失控时刻 `t120`（秒），蓝 = V1、红 = V2。三个 seed 不是三个患者、V1+V2 六次运行也不是六个独立样本，只是同一块组织上的三个噪声实现。

**关注点**：V2（红）的 maxAB 是否三个 seed 都显著（尤其 V1 里偏弱、没过随机线的 seed3 在 V2 里是否变强并过线），以及右图里红柱是否都比蓝柱高（快速适应是否推后了失控时刻）。

### fig_mz_v2_axis_temporal_supp.png

一张**审阅补充图（非主图；2026-07-20）**，回答两个对照问题。**左**：把触点的**固定长轴坐标本身**当"模板"，做同样的 maxAB —— 灰柱=单纯长轴、红柱=间期模板；单凭长轴几何就已经 3/3 显著预测早期能量场（0.79/0.84/0.78，p<0.01），间期模板只多贡献 +0.10～0.13；控制长轴后残余关联全样本 2/3 显著，但**leave-one-contact-out 后只 1/3 稳健**（每柱标注 partial p 与 LOO 最坏 p：seed3 LOO 0.02 稳健、seed1 LOO 0.20 单触点脆弱）。**右**：把 0–25 / 25–50 / 50–100 ms 三个窗的 contact maxAB 连成折线（星号=杆内随机重排 p<0.05）—— 最初 50 ms 强，到 **50–100 ms 减弱**。

**关注点**：左图看间期模板相对"单纯长轴"多贡献多少、控制长轴+掉一个触点后是否还显著（间期时序是否**超越几何轴**——本轮 LOO 后只 1/3 seed 稳健、**未确立**）；右图看这条读出是不是**只在最早 50 ms** 成立。
