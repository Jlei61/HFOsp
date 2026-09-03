# Figure 3 panel 与完整排版输出

独立 panel 文件不写左上角 A–F；字母只出现在 `fig3-complete-layout`。所有独立 PNG 均由矢量 PDF 或原始 producer 以 600 dpi 生成。

本版按各 panel 进入拼板后的实际缩放比例校准坐标字体：常规 x/y label 目标为 9.0–10.5 pt，常规 tick 为 7.5–8.5 pt；A 的通道名与 F 的患者编号属于密集 tick，目标为 6.5–7.5 pt。C–F 的排版位置与 panel 尺寸未改动；顶行只重新分配 A/B 宽度以容纳两个代表性频谱表型。

### fig3-panela.png / .pdf

两个代表性发作模式的并列 signal context：左为 E10 | SZ8 broadband-type 的 raw SEEG 与 SCL9 baseline-normalized TFR，右为 supplementary 已接受的 E20 | SZ8 gamma-type raw SEEG 与 HRB1 TFR。两例都只显示 20 s baseline 邻域 −110 至 −90 s 和 clinical onset 邻域 −10 至 +20 s；中间 −90 至 −10 s 用成对斜线断轴明确标为未显示。

**关注点**：每个内部示例的两段式 raw SEEG 与 TFR 必须严格共轴；`BASELINE` 在 20 s baseline 段居中；病例/类型标题必须显著大于 `BASELINE` / `CLINICAL ONSET` 区间标注；E20/SZ8/HRB1 应清楚显示 gamma-dominant 快活动增强。断轴斜线表示删去显示区间，不表示数据缺失或时间连续。

### fig3-panelb.png / .pdf

两个代表性发作的 low bands、gamma、high-gamma 与 broadband 能量轨迹：E10/SZ8 的代表通道 SCL9 为 broadband-type，E20/SZ8 的代表通道 HRB1 为 gamma-type。

**关注点**：B 连续显示 −120 至 +20 s，不使用 A 的断轴；四图在 0 s 统一画黑色竖直虚线，左列 ylabel 简写为 `dB`。颜色只编码发作表型，不编码频带；legend 在 low-bands 图左上角的无曲线区纵向排列，只写 `Broadband` / `Gamma`，避免覆盖 onset 附近的核心变化。患者/SZ/通道身份由 A 标题给出。两例来自不同患者，只是代表性形态对照，不是患者内或 cohort 统计。

### fig3-panelc.png / .pdf

all-event Timing+Space 冻结间期 TA timing field 与固定 SZ3 的 early-ictal broadband power field。C 不使用总标题；右图以 `E10 | SZ3` / `Early ictal field` 两行子图标题标识病例与语义。左色条与 Fig2 统一为 `0 early / 0.5 / 1 late` normalized ranks，右色条标题简写为 `power` / `z`，空间 y label 为 `Y (mm)`。

**关注点**：右图色条中的 `z` 指 baseline-normalized robust z power，不是传播 rank；该病例经过形态选择，只作空间读出桥。

### fig3-paneld.png / .pdf

all-event Timing+Space 场下 clinical onset 后 0–10 s 的 gradient-field cohort Data–Null 比较：Pooled n=17、Broadband n=16、Gamma n=11。

**关注点**：Pooled/Broadband 显著、Gamma 为 n.s.；不得替换成旧 endpoint n=20 三组全显著版本。

### fig3-panele.png / .pdf

E10 peri-onset amplitude-aware template expression：左为 `max(|q_A|, |q_B|)`，右为 signed TA/TB projection；两图各自把两项 legend 纵向放在左上角。

**关注点**：这是单病例描述性轨迹，不支持 onset-emergent alignment 或机制结论。

### fig3-panelf.png / .pdf

17 名可评估患者在 −120 至 +20 s 的 all-event Timing+Space signed A/B contrast heatmap。

**关注点**：虚线为 clinical onset；主图使用 heatmap，paired inferential companion 留作补充材料。

### fig3-complete-layout.png / .pdf

将 A–F 六个无角标独立 panel 组装为带统一 A–F 字母的完整 Figure 3。

**关注点**：完整排版只负责版面与字母，不改变各 panel 的数据、统计或坐标合同。
