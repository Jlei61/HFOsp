# 当前 Figure 1：作者定稿，2026-10-10

唯一正式入口为本目录的 `current_revision.json`；完整图与A–F单独面板在 `figures/`。`figures/fig1-final.pptx`含7页：完整图及六个单独面板。所有PNG/PDF与作者接受的紧凑行距版本逐字节一致。

## 代码与数据

当前入口：`scripts/paper_figures/build_fig1_current.py`。原生绘图代码及本次全部修订代码的冻结副本在 `source/code_snapshot/`，面板与代码/数据的对应关系在 `source/panel_data_contract.json`。

`data/a`包含真实波形、触点坐标及脑表面投影；`data/b`包含178段HFO及完整打包事件谱、质心和显示例；`data/ce`同时备份原26通道数组与18通道显示派生数组、冻结分类及昼夜标记；`data/df`为40位患者原统计读出。原始EDF和完整解剖重建仍在原数据盘，本包是部分备份，未改动原始数据。

## 校验与复现

运行 `python scripts/paper_figures/build_fig1_current.py` 校验当前包；加 `--output-dir /tmp/fig1-rebuild` 可在空目录复现定稿拼版，验证完整PNG及单面板一致。需要 numpy、pillow、pypdf、matplotlib；原始科学producer另依赖项目环境。冻结PDF/PNG图层负责精确版式复现；如果修改数据或科学计算，应使用对应面板的原producer及数据合同，不能将拼版复现表述为重新运行原始检测。

## 归档与协作

旧正式图、所有旧候选和修订已迁入 `../archive/2026-10-10_fig1_finalization/fig1/`。旧路径仅作历史来源，不再是绘图入口。主目录 `docs/current_figure1.md`、登记表和各工作树AGENTS入口共同指向本版；任何后续修改从本版派生候选，不直接覆盖已接受包。
