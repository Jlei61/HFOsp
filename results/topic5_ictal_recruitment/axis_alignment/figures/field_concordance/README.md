# A 线 · field concordance（被试级,不是 cohort 平均）

A 线主统计是**两个二维场像不像**(间期 template-rank 场 vs 发作 activation 场,镜像不变、符号自由)。
**这套图是 A 线主结果的论文级 cohort 图,核心原则:队列级 ≠ 把场平均**,而是把每个病人的 paired field
同屏排出来、再用**每个病人自己的 null** 作统计对照。不画"平均脑"——每个人电极布局/轴向/病灶都不同,平均
场是个不存在的脑。

主口径(其余进 supplement):Epilepsiae 18 人、template A、broadband 0–10 s、符号自由/镜像不变场对齐、
只看粗(channel-shuffle)null。HFA、四层 null 阶梯图都退到 supplement。

## `field_concordance_atlas_broadband.png` —— 主视觉(每病人 paired field)

- **Panel A(顶部示意)**:只保留测量定义 `r_s = | corr_mirror( F_interictal , F_seizure ) |`；不再加额外解释标注。
- **Panel B(18 个 tile)**:每个病人一个 tile,tile 内**左 = 间期传播顺序场**、**右 = 发作起始激活场**。
  - **不画 cohort 平均**;每个病人保留自己的触点平面。左侧间期传播顺序用 viridis，右侧发作早期能量用 Reds（低能量浅色、高能量深红）。
  - 右图按该病人最终 `|r|` 的**最佳符号/镜像**显示。
  - 触点黑环 = 临床 SOZ(仅叠加)。
  - **tile 边框 = 是否过该病人自己的粗 null**:深黑粗框 = 过,浅灰细框 = 不过。
  - tile 上方只写:subject ID、`|r|=0.xx`、`+0.xx`(margin = |r|−null95)或 `n.s.`。
  - 病人按 `margin = |r| − Q95(null)` 从强到弱排序(强的在左上)。

**怎么看**:左侧看间期传播顺序，右侧独立看发作早期能量的低高分布；是否对齐以 tile 上的 `|r|`、margin 和边框为准。

## `field_concordance_null_forest_broadband.png` —— 统计支撑(不做四层乱图)

一行一个病人(与 atlas 同序):**灰点 = 该病人的 channel-shuffle null 分布**,**竖线 = null 95 分位**,
**黑点 = 实测 |r|**(过 null 95% 用黑、不过用红)。横轴 = 镜像不变场对齐 `|r|`。多数黑点落在灰色 null
右侧 = 这种场一致**不是 channel shuffle 能解释的**。FDR / 留一在归档文字里,不塞进图。

**关注点**:主图让人看到"每个病人自己的间期场和发作场都像",forest 只负责证明"这不是随机洗牌能解释的"。
**这是 A 线 primary(粗骨架)的论文主图;四层 null 阶梯 + HFA 是给方法审稿人的 supplement。**
定稿数值(FDR + 留一)见 `../../axis_alignment_FINAL.md`;加固/时间负对照(持续 scaffold 收窄)见
`docs/archive/topic5/axis_alignment_hardening_result_2026-06-15.md`。

## `epilepsiae_1146_raw_broadband_minmax_no_flip.png` —— 诊断图(原始能量显示)

这张图只用于检查显示层,不是 A 线正式 atlas。左边仍是间期 template A 的传播顺序场;右边是 1146 的发作
0-10 s broadband robust-z 能量,在 matched contacts 内做 min-max 归一到 0-1,**不做 rank-normalize,不做 sign flip**。
右图红色系中浅红=低能量、深红=高能量,所以它可以直接看原始能量高低的空间分布,但不能和左图的 0-1 当成同一物理量。

**关注点**:用来判断正式 atlas 的视觉一致性是否依赖 rank 化/符号翻转;正式统计仍看 `axis_alignment_FINAL.md`。
