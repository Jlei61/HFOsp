# 患者特异 TA/TB 间期传播场画图规范

> 合同：`topic5_interictal_ab_field_figure_v1`
> 状态：锁定，适用于 Fig2-E 候选及之后所有患者级 TA/TB 间期场图。
> Canonical renderer：`scripts/plot_topic5_interictal_template_ab_fields.py`

## 1. 适用范围

本规范用于把冻结的患者特异 TA/TB propagation rank 映射到二维电极平面，生成单患者双面板图或全 cohort atlas。它只规范**间期模板场**；发作能量场、field concordance 和动态 movie 可以复用同一低层 field renderer，但不得把其颜色、标题或统计口径混入这里。

所有新图必须优先复用本规范列出的公共函数。不得复制现有函数后另改一套坐标、平滑、翻轴、contact 样式或 colorbar。

## 2. 数据与科学合同

1. 唯一输入是 `results/interictal_propagation_masked/template_gradient_fields/per_subject/<dataset>_<subject>.json`。
2. 加载时必须调用 `scorers_from_interictal_record()` 校验 `topic5_interictal_template_fields_v1` 合同、fingerprint algorithm 和完整 fingerprint；校验失败必须停止，不能跳过。
3. TA/TB 使用 artifact 内同一 `interictal_field.contact_order`。不得因某个下游数据缺触点而缩小 contact set、重新拟合轴或改变 plane/support。
4. `axis_a.u`、`axis_b.u` 已经是正式传播向量，正方向固定为 `early → late`。绘图层不得再取负，也不得把 `earliness_gradient_u` 或 `D_AB.u_AB` 当成单模板传播正方向。
5. 图只使用间期数据。发作、onset、subtype、SOZ、swap、decision-k、source/sink endpoint 均不得参与轴、平面、方向或显示范围的决定。
6. `geometry_2d_supported=false` 的单杆患者可进入 atlas/质控图，但不能作为 paper-ready 二维场代表患者，也不能解释为有效二维传播几何。

## 3. 平面与方向路由

### 3.1 shared 与 separate

- 只有当 `axis_pair.relation.collinear=true` 且 artifact 同时存在 `shared_a/shared_b` 时，TA/TB 才使用同一个 shared plane，标题写 `shared`。
- 其余患者必须分别使用 `own_a/own_b`，标题写 `separate`。不得因为两张图看起来相似而临时改成 shared，也不得对不同向患者强建共同切面。
- shared plane 的横轴以 TA 的 `early → late` 方向为正。若 TA/TB 反向，TB 的 rank 场应自然沿横轴反向变化；不得为了视觉相似翻转 TB 的横轴。

### 3.2 transverse 轴符号

transverse 轴的正负没有生物学意义，但必须固定，保证同一患者的 TA/TB 可比：

- shared：TA/TB 使用同一个 plane `w`；取 `w` 绝对值最大三维分量为正的符号，两幅图完全相同。
- separate：TA 先按其 `own_a.w` 的绝对值最大三维分量固定符号；TB 在 `y_B` 和 `-y_B` 中选择使同名触点相对 TA 的 transverse RMSE 最小者。
- 该步骤只能读取 contact-aligned 电极几何。禁止根据 rank、field 颜色、相关系数或预期结果选择符号。
- 不做额外的全局 y 轴反转；`imshow` 固定 `origin="lower"`。

### 3.3 坐标范围

- 所有坐标必须换回物理毫米。
- 同一患者 TA/TB 共用一组 `xlim/ylim`，范围由两幅图全部触点联合计算。
- x/y 均包含 0，最小显示跨度为 35 mm；不得逐 panel 自动缩放。
- `aspect="equal"`，保证毫米在横纵轴上等尺度。

## 4. 场与触点渲染

1. 背景与触点颜色都表示模板 rank：`0=early`、`1=late`，固定使用 `viridis`、`vmin=0`、`vmax=1`。
2. 必须调用公共低层函数 `draw_topic5_field_panel()`；不得在新脚本中复制 `_smooth_rank_field_mm()` + `imshow()` + `scatter()` 另写一个近似版本。
3. 连续场使用 artifact 冻结的 template values 和 support。显示层固定 `sigma_display=6 mm`，用于保持足够覆盖；它不能写回 artifact，也不能替换 field correlation/maxAB 使用的患者特异冻结 kernel。
4. 主图 contact 为带粗白色外圈的圆点：单患者 `size=92`、`linewidth=2.7`；atlas `size=46`、`linewidth=1.5`。
5. 默认不写电极名称，不画 source-A/source-B 圈、swap 圈、SOZ 圈或临床 onset 标记。确需 overlay 时必须作为单独 sensitivity/annotation 参数加入公共函数，不能复制 renderer。
6. 场只在 support 覆盖区域显示；不得为了“铺满画布”外推到没有支持的区域。

## 5. 单患者版式

- 结构固定为 `TA | TB | shared colorbar`。
- 大标题：`E/Y编号 · shared/separate`，19 pt、粗体、左对齐 TA panel，靠近 panel 标题。Yuquan 只能通过被 git 忽略的 private crosswalk 转成匿名 Y 编号，图和公开代码不得写真实 folder-name 映射。
- panel 标题：`TA` 用 `#B2182B`，`TB` 用 `#2166AC`，20 pt、粗体。
- 左侧唯一 y label：`transverse (mm)`；TB 不重复 y label/刻度文字。
- 两幅图共用一个居中的 x label：`Main Propagation Axis (mm)`，17 pt；不得分别重复长 xlabel。
- 只画一个 colorbar，标签固定为 `early 0 → late 1`。colorbar 用 TB axes 的 inset `cax=[1.045, 0, 0.055, 1]`，顶端和底端必须与 field 坐标框严格等高。
- 两幅图尽量紧凑。figure 高度固定 6.6 inch；宽度按联合 frame aspect 自适应，公式为 `clip(3 + 10*aspect, 8.6, 13.0)`，避免高窄患者出现巨大中间空白。
- PNG 使用 150 dpi；paper-ready candidate 同时输出矢量 PDF。

## 6. Atlas 版式

- 默认每行 4 名患者，每名患者相邻两格为 TA/TB。
- 患者标题只放在 TA 格，写 `E/Y编号 · shared/separate`；TA/TB 在各自 panel 左上角用相同红/蓝语义色标注。
- compact panel 不显示坐标刻度；全 atlas 只保留一个水平 `viridis` colorbar。
- atlas 用于 cohort 目视质控和代表患者筛选，不是 cohort-level 统计图。

## 7. 强制复用入口

完整单患者图或 atlas：

```python
from scripts.plot_topic5_interictal_template_ab_fields import (
    load_interictal_field_records,
    plot_interictal_ab_atlas,
    plot_interictal_ab_subject,
)
```

需要嵌入其他拼版时：

```python
from scripts.plot_topic5_interictal_template_ab_fields import (
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
)
```

低层连续场统一使用：

```python
from scripts.plot_topic5_field_vs_ictal_swap import draw_topic5_field_panel
```

禁止新脚本直接复制上述函数。若现有参数不能满足合法的新 overlay 或拼版需求，应向公共函数增加向后兼容的显式参数，并补测试。

## 8. 标准运行与验收

```bash
python scripts/plot_topic5_interictal_template_ab_fields.py --display-sigma-mm 6
python scripts/plot_topic5_interictal_template_ab_fields.py \
  --subjects epilepsiae_1146 --no-atlas --format pdf \
  --output-dir results/paper-ready-figure/fig2e_interictal_template_fields/figures
pytest -q tests/test_plot_topic5_interictal_template_ab_fields.py \
  tests/test_topic5_template_axis_field.py
```

每次改 renderer 后必须：

1. 重画全部 28 名 field-ready 患者和 atlas；
2. 至少目视一个宽面板患者、一个高窄患者、一个 shared-reversed 患者；
3. 核对 colorbar 等高、shared xlabel、不重叠、TA/TB transverse 可比；
4. 验证 28 张单患者 PNG + atlas 完整；
5. 运行 fingerprint audit 和上述测试；
6. 更新对应 `figures/README.md`。图义或科学合同未变时，不重复改写 README。

## 9. 允许与禁止的表述

允许：该患者的 TA/TB 间期模板在冻结患者特异平面上形成连续空间 rank 场；两轴宽泛共线时可在同一 shared plane 比较同向或反向结构。

禁止：把插值场写成未采样脑区的真实传播；把单杆图写成二维传播；把 TA/TB 视觉相似写成发作重放；把 representative subject 图写成 cohort 统计；把显示用 6 mm kernel 当成 field scoring kernel。
