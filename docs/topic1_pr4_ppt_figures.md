# Topic 1 PR-4 PPT 图说明

> 输出位置：`results/interictal_propagation/figures/ppt/fig{1..5}_*.png`
> 输出脚本：`scripts/plot_topic1_pr4_ppt.py`
> 共享样式：`src/plot_style.py`（Morandi 语义色 + `style_panel` /
> `violin_with_scatter` / `add_significance_bracket` / `savefig_pub`，
> `DPI_PUB=300`）。本脚本在导入时把 `FS_TICK / FS_LABEL / FS_TITLE /
> FS_SUPTITLE / FS_PANEL_LETTER` 全部上调到 PPT 可读字号
> （18 / 18 / 22 / 24 / 28 pt）。Day/Night 调色板换成接近黑白的
> `#EDEDED / #3D3D3D`，散点 / 连接线 / strip / label 全用同一对，
> 严格 label-to-bar 一致。

PR-4 的核心问题：**什么慢调控影响固定传播模板？** 五张顶刊 / PPT 级别的
可视化，每张图收敛到一个相关问题。

```bash
python scripts/plot_topic1_pr4_ppt.py --all
# 或者
python scripts/plot_topic1_pr4_ppt.py --fig 3
```

---

## fig1_pr4_framework.png — PR-4 三层调制框架（toy 概念图）

3 个 panel，每个 panel 直接表达一层。**没有标题、没有 banner、没有
表格**——所有信息都在 panel 内部。

- **Panel a (L1)**：`stacked area` 画 24 h 内 Template A / B 占比的
  缓慢漂移。问题："哪个 template 在 firing？"
- **Panel b (L2)**：左半 `stable` 组 / 右半 `scrambled` 组的 raster
  对比。每组 12 个 event，每个 event 画 6 channel 的 lag 点。Stable 组
  几乎重合在 template reference line 上（黑色短线），scrambled 组明显
  发散。每组角落标 `τ ≈ 0.95 / τ ≈ 0.30`。**这就是 L2 怎么算的**：把
  每个 event 的 channel rank 与 template rank 比对，τ 就是 Kendall。
- **Panel c (L3)**：`scatter + line` —— x 轴是 template channel lag，
  y 轴是 event channel lag。两组点云：
  - `compressed` (蓝)：slope ≈ 0.45，所有点贴在 y = 0.45 x 上 → lag span
    被压缩
  - `expanded`  (橘)：slope ≈ 1.55，所有点贴在 y = 1.55 x 上 → lag span
    被展开
  
  黑色虚线是 y = x（slope=1，"完全复制 template"）。**slope = lag span
  ratio**，**Pearson r = within-cluster timing precision**。这就是 L3
  指标的几何含义，一眼可读。

---

## fig2_pr4a_daynight.png — PR-4A 日夜 + dom × seizure proximity

4 个 panel，全部围绕 **同一个 L1 metric: dominant template fraction**。
suptitle 与图体之间留出大间距。

- **Panel a (switching subject) / Panel b (dominant-template subject)**：
  两个代表 subject 的 stacked occupancy timeline，**都限制在 k = 2**，
  让 stack 颜色简单可读。
  - 顶部 3% 高度的 strip 标 day (`#EDEDED`) / night (`#3D3D3D`)
  - **红色虚线** 标 seizure onset 时刻（`load_seizure_times` 从
    `pr1_seizure_*.json` / `epilepsiae_seizure_inventory.csv` 读）
  - **每个 panel 右上角内置 inline legend**（day / night / seizure
    onset），不再放在图顶部 fig.text，避免与 suptitle 挤占空间
  - panel 标题左对齐 (`loc='left'`) 进一步避让 suptitle
  - 选 subject 的逻辑（`_pick_two_subjects`）：
    - 硬约束：`stable_k == 2`、24 h ≤ timeline ≤ 200 h、至少 1 个
      in-window seizure
    - switching：最大化"在 ±2 h 窗内的 dominant-cluster 跨越次数"（按
      "near-seizure 切换"而不是总切换数排）
    - dominant：最大化"max template 平均占比"，且与 switching subject
      互斥
  - 当前默认挑出 `epilepsiae/548`（k=2, 31 sz, near-seizure switches=39）
    + `yuquan/sunyuanxin`（k=2, 8 sz, dom_frac=0.75）

- **Panel c**：cohort dominant fraction day vs night paired，n=30，
  Wilcoxon **p = 0.124（n.s.）**。配对线灰色 / day cream-edge / night
  black-edge，散点用同色填充。**结论**：日夜不显著调制 L1。

- **Panel d** *(本次新增的小实验)*：cohort per-subject Spearman
  ρ(per-bin dom-fraction, |Δt to nearest seizure|)。
  - n = 26 (4 个 subject 在 timeline 内没 seizure，不入分析)
  - **median ρ = +0.077**, 16/26 > 0
  - **Wilcoxon two-sided vs ρ = 0：p = 0.041 \***
    （即 cohort 26 个 ρ 是否中位数显著偏离 0）
  - 方向：ρ > 0 ⇒ dom-fraction 在 *离 seizure 远* 的 bin 更**高**，
    *近 seizure* 的 bin 更**低**

  **重要解读**（这是个新发现，补充 PR-4C）：
  这个信号不与 PR-4C 的 "dominant template **rate** ↑ post-ictal" 矛盾，
  而是正交：
  - PR-4C：dominant template 的**绝对 rate**（events/h）在 post-ictal ↑
  - 本图：dominant template 的**占比** (dom / total) 在 seizure 附近 ↓

  合起来：seizure 附近所有模板都被招募更多，而 *non-dominant* 模板被
  招募得更多 → 这正是 PR-5A novel-template gate 关心的方向，是 PR-5 主
  分析的间接 motivation。

  **诚实约束**：
  - p = 0.041 紧贴 0.05 边界
  - 单方向 sign 16/26（正二项 p ≈ 0.16 单侧）— direction 一致但量级小
  - 仅 cohort-level，subject-level 异质性大
  - **不能** 单独把它升级为 "novel template emergence" 机制结论；这是
    PR-5-A novel-template falsification gate 的工作

---

## fig3_pr4b_rate_coupling.png — PR-4B 速率态 × L1/L2/L3（合并版）

把原来 7 个 panel 收敛到 3 个：

- **Panel a** *(top-left)*：rate-binning + matched subsampling toy（高 vs
  低 rate state 的判定方式）
- **Panel c** *(top-right)*：subject-level dominant cluster median Pearson
  r 排序条形图，标 HC 阈值 (dom_r > 0.7)。**告诉读者为什么只有 8/30 进入
  HC**
- **Panel b** *(bottom, 全宽)*：5 列**垂直** violin，每列一个指标，按
  L1 → L2 → L3 排列：
  1. **L1 ρ**（occupancy ↔ rate）：median ρ = **−0.083**, 13/30 > 0 → null
  2. **L2 raw τ Δ**：Δ med = +0.003, Wilcoxon p = **0.349** → null
  3. **L2 centered τ Δ**：Δ med = +0.003, Wilcoxon p = **0.221** → null
  4. **L3 lag span Δ**：Δ med = +0.0008 s, Wilcoxon p = **0.135**
     (18/30 high>low) → trend；红色大点 = HC subset (n=8)
  5. **L3 Pearson r Δ**：full p = 0.265，**HC 子集 Δ = +0.083,
     Wilcoxon p = 0.016 \***（红点）

  每列下方注释 3 行：median Δ + Wilcoxon p + verdict tag（→ null /
  → trend / → HC sig*）。指标的取值尺度不同（ρ ∈ [−1, 1]、τ Δ ∼ 0.05、
  lag span Δ ∼ 0.005 s），所以每列独立 y 轴 + 0 参考虚线居中，但 5 列
  并排排在一行让 cohort 方向一目了然。

**诚实约束**（图里也有）：
- HC subset n = 8，Wilcoxon 最小可能 p = 2⁻⁸ = 0.004，p = 0.016 仅比
  下限大一档（W=1）→ 探索性，不是 cohort-level 强结论
- "lag 更展开 + Pearson r 更高" **不是** Kuramoto K(t)↑ 预测的 lag
  压缩，方向相反；**不要**写成 "coupling increased"

---

## fig4_pr4c_seizure_proximity.png — PR-4C：geometry null + recruitment-rate signal

3 个 panel（删除原 panel d take-home box），垂直布局给每行充足宽度。

- **Panel a**：seizure proximity 窗口合同 toy。
  - `xlim` 从 −4 h 开始（不再多余左边白）
  - main / aux **标签放进 axis 内部** 的白底圆角 box（不再悬在 y 轴外）
  - baseline 标签靠右对齐避开 recording gap 矩形
  - "recording gap" 后括号里的解释文字已删除（图例自身已经够说明）
  - seizure onset 用红色 dashed line + "seizure onset" label

- **Panel b**：5 (metric) × 3 (pair) × 2 (config) 的 cohort Wilcoxon p
  热力图。
  - 现在拉到全宽
  - **只在 p < 0.05 的 cell 显示数字（粗体）**，n.s. cell 留空 → 文字
    overlap 完全消失
  - x 轴 label "metric_name / main / aux" 三层（main 在 metric 行下，
    aux 单行）
  - colorbar：−log₁₀ p，在 p = 0.05 处画 reference line
  - **唯一两处显著**：main `dom_cluster_frac post_vs_pre` p = 0.020、
    aux `dom_cluster_frac post_vs_baseline` p = 0.002，跨配置不一致

- **Panel c**：dominant template rate 三状态柱状图。
  - **`baseline` 排在中间**（pre / baseline / post），符合"基线居中"的
    阅读直觉
  - main / aux 两个 config 用 hatch 区分，颜色与 panel a 同源 (pre /
    baseline / post)
  - 显著性 bracket **只在配置内画**：
    - main：baseline → post bracket（p = 0.0009 \*\*\*）
    - aux：baseline → post bracket（p = 0.0067 \*\*）
  - 两 bracket 在不同 y 高度（main 顶高 ≈ 129、aux 顶高 ≈ 108），不
    overlap、也不再跨 bar
  - 左上角白底 box 列出 main / aux 的 Δ_med + p
  - 柱顶数字直接标 ev/h 值

---

## fig5_pr4d_template_rate.png — PR-4D 模板分解事件率（描述层）

本次未改（用户没提）。结构：
- panel a/b：两个代表 subject (epilepsiae/548 + yuquan/chenziyang) 的
  gap-aware rate envelope (events/h，上) + 同色离散计数堆叠柱（下）。
  NaN 在 grid 中自然把 gap 留白
- panel c：cohort dominant rate fraction 分布（按数据集 dataset_color
  着色）
- panel d：dominant template 切换次数统计（25/30 任意切换，17/30
  meaningful 切换 ≥ 25% peak rate, 6/30 重复切换 ≥ 3 次）

---

## Take-home (整体)

> 之前贴在 fig 4 panel d 的内容，按用户要求挪到这里：

**PR-4C closure（2026-04-19，P0 fixed）**：

- 5 geometry metrics × 3 pairs × 2 configs → 30 cohort 比较，**0 个稳健**
  - 仅 2/30 nominal p < 0.05，跨配置不一致，过不了 Bonferroni alpha
    = 0.0017
- **Template internal geometry 在 seizure 邻近 FORMALLY NULL** —
  没有 shape deformation
- `rate_by_template post vs baseline`：main p = **0.0009**, aux p =
  **0.0067**，方向一致 → fixed templates 是被 **更多招募**，不是被变形
- 这条信号的机制结论由 PR-5 拆开：novel-template falsification gate
  (PR-5-A) + recruitment shift main analysis (PR-5-B)

**本次 fig 2 panel d 新发现（待 PR-5 验证）**：

- per-subject Spearman ρ(bin dom-fraction, |Δt to nearest seizure|)
- median ρ = +0.077, 16/26 > 0, p = **0.041 \***（cohort Wilcoxon）
- 方向：seizure 附近 dominant 模板**占比** ↓ → non-dominant 模板被招募
  得更多
- 与 PR-4C 的 dom rate ↑ 不矛盾，是正交读数；这是 PR-5-A novel
  template gate 的间接 motivation

---

## 风格合同（与 topic 2 PPT 同源，但本脚本本地放大字号 + 黑白 day/night）

| 项目              | 值 |
|-------------------|----|
| 脊线              | 仅保留 left + bottom，1.4 pt |
| 刻度字号          | **18 pt**（PPT 可读，比 topic 2 大） |
| 标题字号          | **22 pt** |
| 子图字母          | **28 pt 黑体**，左上角 |
| suptitle          | **24 pt 黑体**（fig 1 / fig 5 不画） |
| dpi               | 300 |
| 背景              | white |
| 数据集色          | Yuquan `#6F8FA8` / Epilepsiae `#B07A6E` |
| **Day/Night**     | day `#EDEDED` / night `#3D3D3D`（接近黑白） |
| pre/post/baseline | baseline 灰 / pre `#C49B92` / post `#A35E48` |
| 高 vs 低 rate     | high rust / low sage `#9DAA90` |
| 模板调色板        | blue / rust / mustard / sage / plum / dust |
| 显著性            | bracket + `***`/`**`/`*`/`n.s.` |
| seizure marker    | `#C0392B` red dashed line |
| Legend            | upper right / lower right，framealpha = 0.95 |

如有需要，新增图请直接复用 `src/plot_style.py` 的常量与 helpers，**不要**
再本地定义颜色或样式（必须 override 时如本脚本，集中在文件顶部 monkey
patch，不散落在画图函数里）。
