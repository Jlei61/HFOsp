# HFOsp：从原始 SEEG 与新 SNN 仿真开始的论文方法代码

主入口是下面三个 notebook。每个计算 cell 标明对应论文面板，依次展示输入、参数、计算和新生成的结果。代码均在本目录内独立保存，不导入原仓库或其他工作区。

1. [01 · 原始 SEEG → Fig1](notebooks/01_raw_seeg_to_fig1.ipynb)：参考/滤波、HFO 检测、群事件、时频质心、rank、K-means、患者内与队列统计。
2. [02 · 时序与坐标 → Fig2](notebooks/02_ranks_to_spatial_fields_fig2.ipynb)：rank-gradient、训练/测试分离的 Timing+Space、投影、Gaussian 场与共享场置换检验。
3. [03 · SNN → Fig4/5 与 Video 2](notebooks/03_snn_model_fit_fig4_fig5.ipynb)：模型设定、新建网络、新仿真、目标函数、core 优化、方向扫描、模型/患者匹配、Z/M、能量场与动态 GIF。

```text
paper_code/
├── notebooks/                  # 上述三个 notebook，发布副本无执行输出
├── data_processing/
│   ├── raw_to_events.py        # 原始采样 → 检测 → 质心/rank
│   └── src/                   # 本目录内的预处理、BQK和群事件实现
├── analysis/
│   ├── from_events.py         # 模板、统计、方向、投影和空间场
│   ├── spectral_fields.py     # 原始信号 → 能量 → 固定空间场评分
│   ├── model_transition.py    # Z/M进入、宽频起始场、原始临床比较
│   ├── shared_field_null.py   # 共享场相关及触点置换
│   └── src/                   # 本目录内的科学计算实现
├── modeling/
│   ├── computational.py       # 模型、simulate、fit_objective、optimize_cores
│   ├── experiments.py         # 有限CPU预算的独立实验调度
│   ├── panel_analysis.py      # Fig4G/H：平均rank与触点交叉匹配
│   ├── cohort.py              # 多患者重新拟合入口
│   ├── cohort_objective.py    # 原多杆、等权、无标签目标函数
│   ├── cpu_acceleration.py    # 与原生轨迹一致的有序scatter加速
│   ├── src/snn_engine/        # LIF、连接、延迟、Z/M、LFP代理
│   └── scripts/               # 原core优化等实现的独立源码副本
├── plotting/computational.py  # 只画传入的本次计算数组
├── workflows/cohort.py        # 多患者原始数据处理及患者单位统计
├── config/
│   ├── raw_example.json       # 明确的原始数据与患者处理参数
│   ├── snn_tutorial.json      # 40,000神经元，有限搜索预算
│   └── snn_paper.json         # 60秒/候选、4 restarts、31评估、13角度
├── tools/                    # 执行、科学不变量检查、独立性检查
├── METHODS_AND_PANELS.md      # 方法、面板映射与当前复现范围
├── source_manifest.json      # 复制源码来源与哈希；运行不回读来源
├── paper_paths.py
├── requirements.txt
└── LICENSE

paper_inputs/computational/   # 相邻输入：原始电压片段、坐标、患者观测/合同
paper_outputs/computational/  # 相邻输出：本次新事件、新仿真、图、GIF和报告
```

目录内只有代码、配置、说明和 notebook。原始临床数据、患者训练观测、仿真轨迹、图、GIF、带输出 notebook 均在旁边；它们不随代码提交。MIT 仅覆盖本项目代码，不赋予临床数据再分发权限。

## 运行

Python 3.11，依赖在 `requirements.txt`。本机已验证环境：`/data/hfosp/envs/paper_code_20261015/bin/python`。该环境使用已有安装包配置；联网全新安装未验证。

使用本机这个环境前设置 `LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib`，以加载其配套动态库；其他机器的独立安装不应照搬此路径。CPU并行时建议将 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMBA_NUM_THREADS` 均设为1，避免每个worker重复占用全部CPU。

```bash
cd paper_code
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/
# 或按顺序执行；输出 notebook 写到 ../paper_outputs/executed_notebooks/
python tools/execute_notebook.py --notebook 01_raw_seeg_to_fig1.ipynb
python tools/execute_notebook.py --notebook 02_ranks_to_spatial_fields_fig2.ipynb
python tools/execute_notebook.py --notebook 03_snn_model_fit_fig4_fig5.ipynb
```

默认输入/输出为本目录旁的 `paper_inputs/`、`paper_outputs/`；可在启动 Python 前设置 `HFO_PAPER_INPUTS`、`HFO_PAPER_OUTPUTS`。原始输入契约和准备方式见 [方法文档](METHODS_AND_PANELS.md)。缺失输入时明确报错，不退回原工作区、不读取现成图代替计算。

SNN并行入口使用Linux fork；其他操作系统未验证。默认保留真实的40,000个神经元，只缩小仿真长度和优化预算；CPU 运行需要数十分钟或更久。`config/snn_tutorial.json` 的 `max_workers` 控制并发，默认4；每个worker约数GB内存。设置 `HFO_SNN_CONFIG=snn_paper.json` 扩大预算前先查看该文件：仅core搜索就有496次、每次60秒的真实仿真，另有方向扫描等实验，可能需要数天CPU时间。

## 精确边界

- Notebook1 本机示例为 Y1 四个200秒原始片段，共800秒，重新处理后302个群事件。它演示处理方法；不是完整24小时、40人队列或历史图像逐像素恢复。预处理的 polyphase 重采样与早期 legacy 实现有差异。新患者还需自己的参考、坏道、临床排除区间和坐标。
- Notebook2 在这批新事件上重新训练两臂、交叉验证、1000次方向置换和50次training-space shuffle；不声称复现原26人队列数值。
- Notebook3 重新生成模型数据。临床FIT/CAL质心是拟合输入；Fourier特征、目标矩及原256组连续CAL抽样的尺度全部重算。优化器为透明的Sobol+局部精修，区别于历史delayed GP/qLogEI调度。新采样的图身份也不同于历史缓存图。
- core数曲线、方向响应和低误差位置密度来自新仿真；默认样本数少于正式论文。G/H有独立计算，不能由loss下降推断患者机制恢复。
- Video2由本次两工作点的事件与2ms活动场新生成；按既定资格及类别取最早事件。若两个工作点不能各产生两类可评分事件，报告 `NOT_ESTIMABLE`，不使用旧动画补齐。
- Fig4I 的完整患者队列、Fig5E 的真实发作比较和Fig5F 的ictal队列仍需各患者完整输入和执行预算。SNN不能生成临床队列。相关计算入口已分开，默认notebook明确标出未执行范围。

旧版图件恢复代码仍作为来源工具保留在 `plotting/fig1` 等目录及 `plotting/reproduce.py`，不再是 notebook 主流程，也不被这三个 notebook 调用。正式论文图、作者验收状态及本机终稿均不受本次计算示例替换。

## 检查

```bash
python tools/check_package.py
python tools/check_computation.py
```

科学检查覆盖已知时间差的质心恢复、缺席通道rank污染不变性、跨记录名称对齐、梯度方向符号、测试方向不进入训练、固定场对坐标原点平移的不变性、无自连接，以及Z/M开/关时CPU加速与原引擎的完整轨迹一致性。独立目录运行时可设置 `HFO_PAPER_FORBID_ROOT` 禁止访问原仓库；输入也须复制到独立目录。
