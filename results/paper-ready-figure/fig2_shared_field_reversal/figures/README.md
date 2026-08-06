# Fig2 shared-field reversal 最后一行候选

### fig2_shared_field_reversal_last_row.png / .pdf

左侧复用冻结间期场的统一画图函数，在相同 shared plane 上成对展示 TA/TB Viridis rank field；
横轴是冻结 shared TA propagation axis，显示带宽固定为 6 mm；四人统一裁到以各自触点范围为中心的
50 × 60 mm display-only 窗口，轴、触点、rank、kernel 与统计均不改变。4个显示案例锁定为
E15 (r=-0.44)、E14 (r=-0.42)、E13 (r=-0.28)、Y9 (r=-0.77)；它们均为负相关且二维几何易读。E958 因触点过密、图形瘦长而排除，E1146 因已在
Figure 2 前文出现而不重复。这些图用于直观说明场的反向形态，不是独立抽样验证。

右上纳入全部 12 名 shared-axis 且二维几何有效的患者，不按 axis cosine、
same/reversed 标签或 strict-stability 分组。当前 8/12
为负，中位 r=-0.353。右下是 TB earliness 与 support 在全部触点间联合打乱、
重建 TB field 后的层级 cohort-median-shift null；观测 Δmedian=-0.339，
lower-tail permutation P=0.01840。

**关注点**：图内不直接写精确 P；caption/metadata 中的 P 只对应全触点打乱的 cohort 中位移位检验。
逐患者 observed-vs-null-median 的配对 Wilcoxon 为
P=0.08813，
within-shaft cohort sensitivity 为 P=0.87836；
因此正文安全口径是“cohort median 比全触点随机化更负”，不能泛化成所有 null 或多数单患者均显著。
