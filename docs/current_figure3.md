# 当前 Paper Ready Figure 3

作者于2026-10-10明确接受本版并指定为新的正式Figure 3。版本为`visual_alignment_compact_rows_20261010`，状态为`AUTHOR_ACCEPTED_FINAL`；下列入口优先于所有旧工作树、旧图号和候选目录。

- [完整PDF](../results/paper-ready-figure/fig3/figures/fig3-complete-layout.pdf)、[预览PNG](../results/paper-ready-figure/fig3/figures/fig3-complete-layout-preview.png)
- [A–E单panel及说明](../results/paper-ready-figure/fig3/figures/README.md)：每张提供PNG、PDF、SVG；完整图也提供三种格式。
- [当前机器指针](../results/paper-ready-figure/fig3/current_revision.json)、[逐panel登记](../results/paper-ready-figure/fig3/figure3_panel_registry.json)、[文件校验清单](../results/paper-ready-figure/fig3/release_manifest.json)
- [正式绘图/重建入口](../scripts/paper_figures/build_fig3_current.py)、[布局参数与原生图层](../results/paper-ready-figure/fig3/source/README.md)

| Panel | 当前内容 | 保留的解释边界 |
|---|---|---|
| A | E10 / SZ3 broadband：raw + TFR、早期发作能量场、红色E10 TA rank场 | 同一发作编号；shared TA平面 |
| B | Y1 / SZ6 gamma：raw + F2–F3 TFR、固定0–10 s的30–80 Hz相对能量场、红色Y1 TA rank场 | 使用本患者自身TA平面，**不是shared axis**；0 s沿用冻结EEG标记，不能重称为已核实的临床起始 |
| C | 原D：pooled / broadband / gamma配对场一致性统计 | n=17/16/11及原统计语法保持 |
| D | 原E右半：E10 signed q轨迹 | Y=−1～1；TA/TB图例在右上纵排；只是显示放大 |
| E | 原F：17人signed A/B contrast热图 | 原数值、排序、分组保持；组间灰色间隔 |

B保留原X范围，Y裁切为−20～20 mm，B1/B2仅退出可见区域；全26触点仍用于场计算，不能把裁切解释为重新筛选通道。固定0–10 s的TA场相关为+0.933796；它是所选例子的描述性对应，不支持“TA=broadband、TB=gamma”的一一对应假设。A/B地图标题标明患者，保留既定语义色。

完整图为三行A、B、C/D/E。列边界计入色条、刻度、轴标签和标题；上下两处可见行间距均4.064 mm，列间距3.048 mm，画布308.11×254.24 mm。正式输出与作者刚确认的修订逐字节一致。

## 本机与远端使用

远端agent先更新到仓库`main`，然后读取本页和`fig3/current_revision.json`；`AGENTS.md`、`docs/paper_figure_registry.md`、`config/paper_figure_source_registry.json`与paper-ready根README均指向此版本。共享机器上的旧工作树以`/home/honglab/leijiaxin/HFOsp/docs/current_figure3.md`及同目录机器指针为准，不应使用旧工作树中缓存的A–F信息。

在仓库根目录运行（依赖版本见`fig3/source/requirements-layout.txt`，可用`python -m pip install -r results/paper-ready-figure/fig3/source/requirements-layout.txt`安装）：

```bash
python scripts/paper_figures/build_fig3_current.py
python scripts/paper_figures/build_fig3_current.py --output-dir /tmp/fig3-rebuild
```

第一条验证当前版本、五个panel、病例合同及文件摘要；第二条在**新的空目录**重建完整图和A–E单panel，并验证单panel逐字节一致、完整PDF渲染逐像素一致。`build_main_figure_3.py`兼容入口调用相同代码，不再恢复旧A–F。默认不覆盖已接受图。

远端包保存原生PDF/SVG/PNG图层和准确布局参数，足够复现本次出版图；不包含原始记录或私有患者映射。这是冻结图层的出版重建，不是重新计算原始SEEG。原科学producer为`revise_fig3_visual_alignment.py`，其布局代码快照和来源摘要见`fig3/source/`；完整分析仍需工作机上的原producer、冻结分析输入与数据盘。后续改变科学数据时须从这些来源生成新修订，再进行相应核验和作者目检。

## 历史边界

旧A–F正式图、metadata及旧组装代码位于[归档](../results/paper-ready-figure/archive/2026-10-10_pre_current_fig3_ae/README.md)。本机`fig3/revisions/visual_alignment_compact_rows_20261010/`保留发布前的原始修订证据；其他`revisions/`和`candidates/`也仅供来源追溯，不是当前入口。补充视频2保留旧文件名，其E10/SZ3场现在对应主图A的场部分。
