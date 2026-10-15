# 论文图件、方法与公开代码精简指南

> **独立计算入口已改为三个 notebook：** [原始 SEEG → Fig1](../paper_code/notebooks/01_raw_seeg_to_fig1.ipynb)、[rank/坐标 → Fig2](../paper_code/notebooks/02_ranks_to_spatial_fields_fig2.ipynb)、[新 SNN 仿真 → Fig4/5/Video2](../paper_code/notebooks/03_snn_model_fit_fig4_fig5.ipynb)。模型、目标函数、core优化及动态场均从本次输入重新计算。精简后的实现直接保存在 [paper_code](../paper_code/README.md) 内，默认使用相邻 `paper_inputs/E1146_demo/`；原始/训练输入和输出分别在相邻 `paper_inputs/`、`paper_outputs/`。计算面板与完整论文预算的区别见 [方法接口](../paper_code/METHODS_AND_PANELS.md)。下文保留定稿来源及整队列公开准备的审计记录。

更新：2026-10-15。用途：确定论文每张图依赖的方法、数据和代码，作为最后整理公开代码的依据。本文是**发布准备清单**；图件定稿、图件重建成功和原始分析复现完成是三个不同状态。

核对基线为本机作者终稿 `author_final_20261015`、当前 paper-ready 包及其来源登记。终稿 `.tex` 继续仅在本机保存，不纳入公开代码包。图件身份以[总登记](paper_figure_registry.md)、[来源登记](../config/paper_figure_source_registry.json)及各图版本指针为准；本文不替换这些记录。

## 1. 论文主线与图件入口

论文的方法链为：**间期事件的重复时序 → 电极几何上的空间传播场 → 间期场与发作起始场的对应 → 患者几何约束的空间 SNN → 慢变量调制下的模型活动变化**。前三步提供数据证据，后两步检验模型中的可能机制；拟合患者传播模式不等于识别临床致痫组织。

| 图件 | 当前身份与科学内容 | 最小来源包与入口 | 目前支持到哪里 |
|---|---|---|---|
| Fig1 A–F | `y1_final_20261010`；作者定稿。Y1 代表事件、模板与两队列时序统计 | [版本说明](current_figure1.md)；[逐面板输入合同](../results/paper-ready-figure/fig1/source/panel_data_contract.json)；`build_fig1_current.py` | 可校验完整包、从冻结原生图层重建；包内保留逐面板衍生数组与原绘图代码快照 |
| Fig2 | 登记入口仍为 A–F、`CANDIDATE`；B/E/F 使用全事件 Timing+Space 分析 | [面板登记](../results/paper-ready-figure/fig2/figure2_panel_registry.json)；全局来源登记中的 `contracts[active_contract].fig2` | 已登记绘图输入和 B/E/F producer；全事件分析的独立重算入口尚需整理。较新的 A–D 拼版仍待作者确认 |
| Fig3 A–E | `visual_alignment_compact_rows_20261010`；作者定稿。A：E10/SZ3 宽频；B：Y1/SZ6 gamma；C：队列；D：signed q；E：场对应热图 | [版本说明](current_figure3.md)；[面板登记](../results/paper-ready-figure/fig3/figure3_panel_registry.json)；`build_fig3_current.py` | 可校验并从原生矢量图层重建；原始信号分析另外依赖患者数据和对应分析配置 |
| Fig4 A–I | `core_search_final_20261011`；作者定稿。几何约束的 core 搜索、响应、模型读出与模板拟合 | [版本说明](current_figure4.md)；[数值来源与恢复说明](../results/paper-ready-figure/fig4/source/README.md)；`build_fig4_current.py` | 可逐字节恢复冻结导出；保留数值快照与历史绘图脚本，按新参数重画及重新仿真尚需整理依赖 |
| Fig5 A–F | `single_seed_zm_entry_and_onset_field`；本机顶层与 main 的历史登记尚未统一；新版 `reference_layout_20261011` 未替换正式入口 | [main 登记说明](current_figure5.md)；本机新版来源见下文 | 目前是多个有来源的版本，尚不能给出统一的最终公开重算命令 |
| 补图与视频 | 补图编号以[paper-ready 首页](../results/paper-ready-figure/README.md)为准；Video 2 已于 2026-10-15 定稿为模型拟合比较 | [Video 2 当前说明](current_supplementary_video2.md)；`build_supplementary_video2_current.py` | Video 2 可恢复冻结动画；四个代表事件分别对齐，不能解释为同时发生或独立验证 |

以上 `build_*_current.py` 均位于 `scripts/paper_figures/`。Fig2、Fig5 以下候选在共享机器主目录核实，**尚未作为完整候选包发布到本次 main 基线**，因此不作为公共 checkout 的可执行入口：

- Fig2：`results/paper-ready-figure/fig2/candidates/wider_b_score_axes_20261012/`，A–D；producer 为 `build_fig2_wider_scores.py`；`human_visual_acceptance=PENDING`。
- Fig5：`results/paper-ready-figure/fig5/candidates/reference_layout_20261011/`，左列 A/B/C、右列 D/E/F；producer 为 `build_fig5_reference_layout.py`；`replaces_formal_current=false`。A–E 来自 `fig5_fig4f_historical_zm_20260930/right_column_offset_20261008`，F 来自 `broadband_phase_statistics_20261009/two_model_tfr` 的候选队列统计；该 producer 只拼版，不重算模型或统计。

## 2. Methods 与代码的最小对应

下表用于确定应保留的科学实现；它不是所有模块已能一键重跑的声明。分析参数最终应从**实际出图批次的配置**读出，不能以当前模块默认值或旧 Methods 代替。

| 方法任务 | 输入、计算与读出 | 统计单位与必要对照 | 应保留的实现与图件 |
|---|---|---|---|
| 数据预处理与 HFO 群事件 | 原始 SEEG、通道表、坐标、发作注释 → 重参考、滤波、HFO 检测、事件合并与通道参与矩阵。终稿方法草稿描述 80–250 Hz 检测；两个数据集的参考方式与采样率分别配置 | 原始队列为 Yuquan 20 人、Epilepsiae 20 人；后续每项分析另列纳入人数和排除原因。时钟、触点顺序、事件边界必须一致 | [检测入口](../scripts/run_hfo_detection.py)、[检测模块](../src/hfo_detector.py)、患者配置；Fig1/S1。检测版本与最终输入清单仍需绑定 |
| 事件时序与模板 | 事件内各参与通道的时频质心 → 局部 rank → masked clustering；MI 衡量事件顺序与模板的一致性 | 事件在患者内汇总，队列以患者为单位；single-template / multi-cluster 比较及事件内时序置换。拆半或奇偶复现规则沿原合同 | [传播分析](../src/interictal_propagation.py)、[rank 掩码](../src/lagpat_rank_audit.py)、[运行入口](../scripts/run_interictal_propagation.py)；Fig1/S3。正式图输入以 Fig1 的 `source/panel_data_contract.json` 为准 |
| 空间信息增益 | rank 与真实触点坐标 → Timing-only / Timing+Space；按记录块交叉拟合，held-out 事件只按 rank 分配模板，再比较方向一致性 | 患者配对；方向置换与训练空间打乱。当前两组训练均纳入全部事件，缺方向只掩蔽空间视图；评分只用数学上可估计方向的共同 held-out 事件 | [增益方法模块](../src/interictal_spatial_information_gain.py)作为实现定位入口；Fig2 B/E/F 的确切输入和绘图脚本以全局来源登记为准。旧同名 runner 不能直接充当当前全事件分析命令 |
| 电极空间轴与场 | 对触点时序与几何拟合方向，固定投影、空间支持及平滑后生成模板场；患者坐标提供先验 | 患者内模板/事件；方向、触点与空间置换依各分析合同。共享轴须满足资格条件，单模板自身平面另作标记 | [模板轴与场](../src/topic5_template_axis_field.py)、[网格场](../src/topic5_gradient_grid_field.py)；Fig2–3/S4。不能把 timing-only 旧场当成全事件 Timing+Space 场 |
| 发作起始能量与场对应 | 原始 SEEG → 固定频段/时间窗的功率、基线归一化 → 冻结间期几何上的场；分别保留相似度、signed q、maxAB 等读出 | 发作先在患者内汇总，队列以患者为单位；选择 A/B 或最大匹配的步骤须在 null 中同样执行。代表事件与队列统计分别报告 | [时频场统计](../src/topic5_tspectral_field_concordance.py)及模板轴/网格模块；Fig3/S5。原有 `run_topic5_figure3_ictal_grid_rebuild.py` 等 runner 须与当前全事件输入逐项对齐后再列为最终重算命令 |
| 空间 SNN 与拟合 | LIF E/I 网络、空间连接与延迟、core/阈值调制、虚拟电极读出 → 传播特征 → 患者模板拟合 | 患者、拓扑/动力学种子、优化重启是不同单位。训练误差选出的最佳配置只能支持训练拟合；空间先验不算独立验证 | [已合并 SNN 源码快照](https://github.com/Jlei61/HFOsp/tree/eb930d8b63f96e1a832fe0ed4f3c6d7af367036f)：`src/snn_engine/`、core/目标函数模块及实际仿真 runner；Fig4/S7/Video 2 |
| 慢变量与活动过渡 | 在明确的原生 SNN 网络上加入 Z/M，固定网络、输入与种子，读取速率、慢变量、波形、空间场和进入时间 | 单条轨迹、参数扫描、患者队列分别验收；保留无慢变量参照和读出定义。高放电率或高持续电平本身不证明发作样振荡 | 同一 SNN 快照中的 `src/snn_engine/mz_slow_vars.py`、`src/topic4_zm_slow_vars.py`，及所选 Fig5 批次配置。最终批次与临床比较入口尚待统一 |

整理实现时必须保留以下定义，否则数值相似也不是相同分析：

1. **参与通道**：legacy rank 可能给未参与通道有限值。先按 `bools` 掩蔽，并核对跨 block 的通道名称/顺序；KMeans 的缺失填补不能使未参与通道获得观测身份。
2. **时序与空间缺失**：当前 Fig2 全事件合同保留无方向事件的时间信息；“参与训练”与“能计算方向评分”是不同资格。它不是旧 hard-QC/每人最多 5000 事件方案。
3. **间期到发作期**：方向、投影与空间支持在间期确定；发作数据不反向选择模板。Fig3 B 是 Y1/SZ6 的自身 TA 平面、gamma 30–80 Hz、EEG 标记后 0–10 s，不可统一写成 shared-axis 或独立临床起始。
4. **信号和单位**：HFO 幅值、功率 z 分数、dB、rank、场相似度不可互换；Fig4 的高斯加权发放密度代理也不等于电位前向模型，或 Fig5 某些版本的原始突触电流。
5. **模型身份**：原生空间 SNN、空间 rate、六群体 rate、Z/G/K 闭环分别保留。`Params` 中的 smoke 默认值不能作为论文参数；Z/M 是有效动力学变量，不直接等同实测离子浓度或已验证临床机制。

## 3. 公开代码只保留哪些内容

按下面的依赖范围整理新发布包，现有研究目录和在跑实验保持原位。最终对每张图形成“**输入 → 分析命令/配置 → 数值表 → 绘图命令 → 导出**”的可追溯链。

| 保留项 | 最小内容 |
|---|---|
| 入口与环境 | 本指南、图件总登记、逐图版本/输入合同、项目许可证；独立的图件环境和科学分析环境说明。当前根 `requirements.txt` 不是全部模型和排版依赖的锁定环境 |
| 公共科学实现 | 上表实际被选中 producer 调用的 `src/` 模块、相应 `scripts/` runner、配置与有科学意义的验证；连同递归导入依赖保留，不能只拷贝主脚本 |
| 绘图包 | 最终选定的 `fig1`–`fig5`、补图/视频；每包的逐面板衍生数据、原生图层、绘图代码、来源清单、数值统计与 null 分布。Fig1/3/4 已冻结的 `source/` 继续随包保留 |
| 仿真来源 | 每个进入论文的模型批次：源码 commit、完整参数、输入几何/标签、拓扑与动力学种子、目标函数/筛选规则、运行命令、必要轨迹/统计。完整开发分支不等于精简发布包 |
| 数据接口 | 明确输入数组的维度、通道与时间单位、患者/事件映射和可共享衍生数据；临床原始数据按其许可另行取得，路径通过配置提供 |
| 历史材料 | 本机终稿、旧候选、重复导出、临时环境、日志、大型训练 checkpoint 和无论文依赖的探索代码不进入默认发布包；保留历史索引。RNN 并行开发独立，按最终补充材料清单决定是否引用 |

SNN 当前在 `codex/snn-unified-20261015`，核对 commit 为 `eb930d8b63f96e1a832fe0ed4f3c6d7af367036f`，不是 main 中全部代码的替代版本。公开整理时先绑定 Fig4/5 实际批次，再导入其依赖；不要直接覆盖主目录的活动运行代码。

## 4. 现在可执行的图件检查

在含有完整正式包的仓库根目录、已配置的 Python 环境中运行：

```bash
python scripts/paper_figures/build_fig1_current.py
python scripts/paper_figures/build_fig3_current.py
python scripts/paper_figures/build_fig4_current.py
python scripts/paper_figures/build_supplementary_video2_current.py
```

这些默认命令只校验，不重新分析原始脑电或重跑 SNN。需要重建/恢复时，使用独立空目录，例如：

```bash
python scripts/paper_figures/build_fig1_current.py --output-dir build/paper-release/fig1
python scripts/paper_figures/build_fig3_current.py --output-dir build/paper-release/fig3
python scripts/paper_figures/build_fig4_current.py --output-dir build/paper-release/fig4
python scripts/paper_figures/build_supplementary_video2_current.py --output-dir build/paper-release/video2
```

Fig1/3 从冻结图层重建版式；Fig4/Video 2 恢复冻结导出。输出目录须为空，不能指向正式包。Fig3 的排版依赖版本见 [requirements-layout.txt](../results/paper-ready-figure/fig3/source/requirements-layout.txt)；Fig1 重建还用到 NumPy、Matplotlib、Pillow 和 pypdf。完整可移植环境仍需单独锁定；本机 `/tmp` 的补充依赖路径不能成为发布安装说明。

Fig2 的 B/E/F 数据以来源登记中 `required_inputs` 为准：`spatial_information_gain_all_events/` 与 `template_gradient_fields_all_events_timing_plus_space/`。目前定位到的旧 `run_interictal_spatial_information_gain.py` 默认使用 hard QC 和 5000 事件上限，与本图不符；在找回并整理匹配的生成代码前，只声明可追溯绘图输入。Fig5 则先确定最终版本及批次，再补统一命令。

## 5. 最终公开前需要对齐的内容

以下差异来自本机终稿与实际包的核对；本次只登记，没有改动终稿、重新选择数据或替换作者定稿。

| 项目 | 当前差异 | 完成标准 |
|---|---|---|
| 稿件 Methods 与附件 | 终稿中的数据方法和模型方法块均被 `%` 注释，当前不会进入编译；附图说明仍有占位文本，图件引用与参考文献资产需补齐 | 将核对后的 Methods 纳入投稿文档，补齐附件与编号，实际编译并核对引用 |
| Fig1 与 Fig3 图注 | Fig1 仍含 Y3/E7 与 6556 事件的旧描述，正式包为 Y1、18190 事件（13160/5030）；Fig3 仍按旧 A–F 叙述，正式包为 A–E | 图注、Results 和源数据逐面板一致；Fig3 队列分别为 pooled 17、broadband 16、gamma 11 人，不能由代表事件替代 |
| Fig2 的分析版本 | 旧稿方法为 hard QC、5000 上限、25 人；当前图输入是全事件、无硬筛选、26 人，主要增益中位数约 0.0416。新版 A–D 仍待确认；K 扫描范围也须统一 | 锁定图版；用对应分析合同修订方法与结果，补齐生成全事件输入的独立 runner，不能通过改路径调用旧算法 |
| Fig4 的面板与模型方法 | 旧稿 A–J 与现图 A–I 不同；C 为 1–4 core、各 4 次优化重复，D 为 13 点方向响应，E 为 83/413 个低误差配置密度。稿中 12 基底场/CMA-ES 方法尚未绑定这些面板来源；且 `100 × 20² = 40000` 与稿中 `N=4000` 不一致 | 以实际仿真配置统一网络规模、空间密度、拟合变量、目标函数及每面板统计单位。I 的训练误差选优不能写成 held-out 验证 |
| Fig5 与模型结论 | 顶层历史版本、新拼版与稿中参数扫描叙述未统一；旧顶层为历史手放双核，不能自动认定为 Fig4 拟合网络；候选示例场 z 与队列 dB 的定义不同 | 锁定网络/种子/时间窗和最终图版；将模型活动形态、参数扫描与患者队列分别表述，完成必要验收后再确定论文结论 |
| 公共运行环境与补充材料 | Fig2/5 候选 producer 仍有本机路径依赖；SNN 在独立分支；旧补图/视频编号存在历史含义 | 在干净 checkout 中补齐依赖并验证选定的绘图/分析命令；最终编号表包含已定稿的模型 Video 2；不把旧编号或本机可运行视为公开可复现 |

## 6. 本次核对的实际范围

2026-10-15 在干净整合工作区核对 main `04aee9fd45751b17688d04dcc8fc0be2d8c8cfed` 的正式包，并参考共享主目录尚未合入的 Fig2/5 候选及本机终稿。Fig1、Fig3、Fig4 默认检查分别通过 **123、49、124 个文件**，其中 Fig4 包含 31 个正式导出；Video 2 通过 9 个发布文件检查与全部 125 帧解码。这些是冻结包一致性检查，不代表重新完成原始数据分析或模型实验。

本指南交付后，整理顺序为：**先统一稿件与图版 → 补齐逐图分析来源和环境 → 按依赖抽取公开代码 → 在干净 checkout 验证所声明的复现层级**。新图候选仍按原要求由作者目视确认；现有正式图维持原身份。
