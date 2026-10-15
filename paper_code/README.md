# HFOsp 论文独立代码包

这是可单独复制的代码目录。数据处理、分析、建模和绘图均保留本目录内的实现副本；运行时不导入原 HFOsp 仓库或其他 worktree。目录内只有源码、配置、环境清单、说明与清空输出的 notebook。

## 从这里开始

打开 **[notebooks/01_reproduce_paper_figures.ipynb](notebooks/01_reproduce_paper_figures.ipynb)**，选择配置好的 Python 3.11 内核，从头运行。它依次检查真实事件的参与通道、重算空间信息增益、运行小规模 SNN 检查、从逐次评分重算 core 数曲线，并重建 Fig1–5 与恢复 Video 2。

```text
paper_code/
├── data_processing/             # 预处理、HFO检测、群事件构建
│   ├── run.py                  # 原始EDF的显式参数入口
│   └── src/                    # 复制进来的实现与依赖
├── analysis/                    # rank、模板、空间场和统计
│   ├── reproduce_statistics.py # 论文数值核对
│   └── src/                    # 复制进来的分析实现
├── modeling/                    # 空间SNN、Z/M、core搜索
│   ├── smoke.py                # 小规模CPU运行检查
│   ├── src/snn_engine/         # 神经元、连接、延迟和慢变量
│   ├── src/                    # core场、目标函数、读出等
│   └── scripts/                # core优化及其依赖脚本
├── plotting/                    # 绘图代码，与科学计算分开
│   ├── reproduce.py            # 统一绘图入口
│   ├── core_count_curves.py    # 原数值绘图函数的独立副本
│   ├── fig1/, fig3/, fig4/     # 正式图版重建/恢复
│   ├── fig5/                   # 待确认候选的拼版代码
│   └── vendor/                 # Fig2绘图链及其包内依赖
├── notebooks/01_reproduce_paper_figures.ipynb
├── config/processing.example.json
├── tools/                      # notebook执行及独立运行检查
├── paper_paths.py              # 统一输入/输出路径
├── source_manifest.json        # 复制来源、原文件与适配后的校验值
├── requirements.txt
└── LICENSE

paper_inputs/                   # 相邻的独立衍生输入包，不在代码目录中
paper_outputs/                  # 执行后的图、报告和带输出notebook
```

`source_manifest.json` 仅记录来源，不会从这些来源加载代码。所有引用的 Python 实现已复制到上述目录；第三方库通过环境安装。

## 环境与路径

```bash
cd paper_code
python3.11 -m venv ../paper_env
../paper_env/bin/python -m pip install -r requirements.txt
../paper_env/bin/python -m jupyterlab notebooks/01_reproduce_paper_figures.ipynb
```

默认读取相邻 `paper_inputs/`，写入相邻 `paper_outputs/`。也可在启动 Python/Jupyter 前设置 `HFO_PAPER_INPUTS` 和 `HFO_PAPER_OUTPUTS`；这些变量只指定输入与输出位置，不指定代码位置。衍生输入包需随交接另外复制，不从原项目路径自动回退。数据许可证与代码许可证分别适用。

命令行执行同一 notebook：

```bash
python tools/execute_notebook.py
```

输入包内保留正式图的原生图层和必要数值、Fig2/5 的指定候选来源，以及输入清单。它不包含完整原始脑电、患者数据库或训练 checkpoint。清单不通过或输入缺失时会明确报错。

本机验证环境使用已有的第三方安装包配置，依赖版本已记录在 `requirements.txt`；本次未完成联网全新安装验证。

## Notebook 实际复现什么

| 部分 | 输入与执行 | 输出与边界 |
|---|---|---|
| 数据处理示例 | 真实 Y1 18190 事件、18显示通道，执行复制来的参与通道/rank处理 | 与冻结掩码及13160/5030事件标签核对；不是原始脑电全流程重跑 |
| Fig2 数值 | 26名患者的原始配对得分表，重算增益中位数与单侧配对Wilcoxon | 与冻结结果核对；独立展示实际统计计算 |
| SNN | 复制的原生引擎，240神经元、100 ms、CPU | 验证关闭Z/M与基线相同；不能替代论文规模拟合或发作过渡实验 |
| Fig4 C 数值 | 冻结的逐次优化评分，经原 reducer 重新构建最佳训练轨迹 | 1–4 core、各4重复、共同31 epoch；均值/样本标准差与正式表一致，再由原绘图函数绘制 |
| Fig1、Fig3 | 冻结原生图层 + 本包内绘图代码 | 重建作者定稿版式，并检查一致性 |
| Fig4、Video 2 | 冻结正式导出 | 逐字节恢复；不声称重新仿真或从模型参数重画 |
| Fig2、Fig5 整图 | 本包内Fig2绘图链重画统计面板并拼版；Fig5从指定原生图层拼版 | 分别为 `wider_b_score_axes_20261012`、`reference_layout_20261011`，均保留待作者确认身份 |

## 从原始数据继续

`data_processing/run.py` 提供 EDF → 预处理 → 通道HFO → 群事件窗口入口：

```bash
python -m data_processing.run --edf /path/to/recording.edf \
  --config config/processing.example.json --output ../paper_outputs/raw-example
python -m modeling.smoke
```

配置文件是显式参数示例，须按患者补齐坏道、参考方式、采样率、事件窗口等；不能把这份通用配置称为已完成全部队列的最终复现。Epilepsiae `.data/.head` 读取实现在 `data_processing/src/preprocessing.py`，其记录和临床注释需单独提供。当前 notebook 使用已冻结的衍生输入，不跳过缺失原始数据后宣称全流程成功。

`modeling/scripts/topic4_core_count_optimization/` 保存 core 搜索流程，科学实现的依赖已复制。重新运行完整搜索还需提供对应网络、目标、设计配置、优化环境和计算预算；不由基础 notebook 自动启动。BO 使用的 Torch/BoTorch/GPyTorch 环境须按所选批次锁定，未包含在基础绘图环境的已执行范围。

后续整理以此独立副本为起点。正式图版本不因整理代码改变；待确认的 Fig2/5 不自动升级。原始数据全队列重算和论文规模模型搜索仍是另外的验证步骤。

## 已执行的检查（2026-10-15）

- 代码与衍生输入复制到项目之外的独立目录后，完整 notebook 从头执行通过；运行中禁止读取原 HFOsp 仓库，68个已加载项目模块均来自独立副本。
- Fig1/2/3/4/5 与 Video 2 的相应重建、数值或导出检查通过；Fig2/5 仍是候选。重新计算的 core 曲线及 Fig2/5 完整拼版已做 Agent 目视检查，未代替作者验收。
- HFO 人工信号检查覆盖检测、时长筛选及群事件窗口；模型动态加载与 core 搜索依赖的导入也在独立目录中通过。
- 包内170个复制来源、197个Python文件通过语法与来源检查，无符号链接或图件/数据文件；基础 notebook 的输出保持清空，执行结果另存。
