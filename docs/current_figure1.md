# 当前 Figure 1：作者定稿

2026-10-10，作者确认当前完整 Figure 1 为定稿。版本为 **`y1_final_20261010`**，状态为 **`AUTHOR_ACCEPTED_FINAL`**。正式图统一存放在 [`fig1/figures/`](../results/paper-ready-figure/fig1/figures/README.md)，以 [`current_revision.json`](../results/paper-ready-figure/fig1/current_revision.json) 和 [`figure1_panel_registry.json`](../results/paper-ready-figure/fig1/figure1_panel_registry.json) 路由；不再从旧 `candidates/` 或 `revisions/` 选版本。

- [完整 PDF](../results/paper-ready-figure/fig1/figures/fig1-complete-layout.pdf)与[完整 PNG](../results/paper-ready-figure/fig1/figures/fig1-complete-layout.png)。
- [PPTX](../results/paper-ready-figure/fig1/figures/fig1-final.pptx)：共7页，完整图及A–F单独面板；嵌入原PNG，矢量版本另见PDF。
- 单独面板命名为 `fig1-panela` 至 `fig1-panelf`，均提供PNG/PDF且无左上角面板字母。

## 接受的内容与布局

A为Y1真实脑投影、A杆和12条双极波形，A7紫色/A9蓝色，三段各0.16秒；Y1在波形轴左端对齐，80–250 Hz右对齐，顶部弯线避开标签。B保留178段HFO以及Y1 A3–A9的三个群体事件：FA134AX6/1559、1562和FA134AXF/1494；继续使用原完整打包事件S³质心，主图仅缩放显示至±150 ms。

C/E显示18通道和全部18,190事件，冻结TA为13,160、TB为5,030；1–18为显示用参与通道重排，未参与者空白，原26通道数组及原始lagPat保持。C分布无标题，逐行峰高缩放定义保留；Day/Night为扁长矩形。D/F继续使用20位Yuquan和20位Epilepsiae的原统计，保留配对点线、均值柱及检验。E/F整体上移6.35 mm后的紧凑行距已获接受，图大小和原统计均不变。

## 代码、输入与复现

当前统一入口为 [`build_fig1_current.py`](../scripts/paper_figures/build_fig1_current.py)。默认校验正式包；指定空目录可重新构建版式：

```bash
python scripts/paper_figures/build_fig1_current.py
python scripts/paper_figures/build_fig1_current.py --output-dir /tmp/fig1-rebuild
```

依赖为numpy、pillow、pypdf、matplotlib。工作机已验证环境为 `cuda_env`；需设置 `LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib`。重建使用冻结的原生PDF/PNG图层，完整PNG和PDF均已逐字节复现；这一步不是重新运行EDF检测或聚类。

各面板的科学producer与数据目录见 [`panel_data_contract.json`](../results/paper-ready-figure/fig1/source/panel_data_contract.json)。原绘图脚本及其本地Python依赖在 `fig1/source/code_snapshot/` 冻结备份。修改科学内容时应遵循对应producer和输入合同，不能用一套新的计算规则替换后仍沿用定稿标签。

[`fig1/data/`](../results/paper-ready-figure/fig1/data/backup_manifest.json)包含约13 MB的部分科学输入备份：A的波形/触点坐标/脑投影、B的178段HFO和群体事件谱及质心、C/E原26通道与18通道派生数组、D/F的40人统计输入。原始EDF和完整解剖重建仍在数据盘；本次没有修改原始数据，也没有将部分备份称为完整原始数据备份。

## 历史版本与所有工作树的入口

此前 `fig1/` 整个目录已迁到 [`archive/2026-10-10_fig1_finalization/fig1/`](../results/paper-ready-figure/archive/2026-10-10_fig1_finalization/README.md)，保留旧正式图、候选、每轮修订和原始说明。旧路径按此前相对层级在该归档中解析。定稿包只保存本版输出、来源、必要数据、校验和重建输入；历史版本不再混在当前 `figures/`。

本机多个工作树以 `/home/honglab/leijiaxin/HFOsp/docs/current_figure1.md` 和主目录的 `fig1/current_revision.json` 为共享入口。main工作树及远端保存相同定稿包和代码说明；旧工作树不得根据自己的历史 `fig1/figures/` 覆盖主目录定稿。后续更改应在独立候选目录完成，作者确认后再替换当前入口。
