# Figure 4：被试特异性 SNN 的双向传播与电极读出

> 状态：正式图示工作第四版；Panel A/B 已按冻结 gradient shared-plane、Figure 5 virtual-SEEG 语法与 Figure 1 rank 语法重画，等待作者目视锁图。
> 当前代表病例：E1146。该图是单被试的模型—读出可行性与模板一致性示例，不是 cohort 证据，也不是对真实生物机制的因果证明。

## 核心问题

同一个带 E→E 各向异性长轴的 E/I LIF 网络，在长轴两端放置由患者两类间期模板最早触点定义的低阈值 E core 后，能否在无外部 kick 的背景噪声下产生方向相反的自限事件，并被同一套患者真实电极 montage 读成两类稳定传播模板。

## 正式 panel 分组

### A｜模型假设与患者特异性摆放

左侧是代码原生的单神经元连接示意，不再使用外部借图，也不重复右侧的空间布局。postsynaptic E neuron 放在两个 kernel 的共同空间中心，只保留一个 recurrent E input 和一个 local I input。当前 active artifact 中 `l_EE=0.380 mm`、`AR=2`，故 E→E 长轴尺度为 `0.380×√2=0.537 mm`；其余 E/I 核为 `0.250 mm`。因此正式图没有画成“兴奋核更窄、抑制核更宽”，因为那会与 Panel B/C 的实际仿真参数相反。

右侧读取 E1146 的 shared-plane 重跑 figdata，电极坐标直接来自冻结 `template_gradient_fields` artifact；显示时只把 20-mm sheet 中心移到原点，不再按 source→sink 轴二次旋转。两个 core 的成员仍沿用已接受的 `template_source_foci` 合同，因此这次只修正坐标框架，没有偷偷改模型 core 定义。两个 core 采用同一颜色且不标 A/B，因为模型假设它们同质；箭头被移除，因为各向异性轴不预设单向传播。

### B｜同一底物产生相反传播并被同一 montage 读出

左侧只保留 model forward 和 model reverse 两个代表事件，不再重复 Panel A 已展示的机制/底物图。两个事件图均使用冻结的 `TA–TB shared axis × y` 坐标和同一 `early → late` 归一化色标；色条标为 `relative firing onset`，对应每个 E 神经元在事件内的首次放电时刻。右侧复用 accepted Figure 5 的 virtual-SEEG 语法：从同一 spontaneous run 截取 1200 ms 连续窗口，显示 signed 30–80 Hz 波形，并同时标出一个完整 model forward 橙色事件和一个完整 model reverse 蓝色事件；不拼接轨迹，不加入发作期/runaway 标记，也不再用 peak 点和折线覆盖原始波形。两个空间图采用紧凑组内间距，并与右侧 readout 保留最小必要间隔；readout legend 移到波形轴上方，不覆盖任何通道。

Panel B 使用 shared-plane 上重新运行并通过双向门的 seed 5 spontaneous twoend artifact（model forward/reverse directional events = 1/4，15/15 触点有效，`theta_deg=-22.8°`）。右侧窗口选择规则要求同一连续轨迹中必须同时存在 model forward 与 model reverse clean event；当前 1200 ms 窗口各包含一个完整事件。方向判定使用 `k_dir=2`，这是患者电极稀疏条件下的载重参数，必须保留在 Methods、caption 或 metadata 中。

### C｜无监督聚类与真实模板一致性核验

Panel C 使用 21 个 paired network seeds。每个 seed 固定同一神经元位置与 network realization，分别运行 4 s source-only 和 4 s sink-only arm；总仿真时长 168 s。全部 arm 固定 `L=20`、`core_r=1.5`、`core_mean=17.5`、`core_std=1.0` 与 `k_dir=2`，共获得 222 个 clean directional events（model forward/reverse=103/119）。触点需在至少 15% clean events 中参与才进入图与 rank 统计；仅 SCL9（7.7%）被排除，保留 ICL1–ICL11。正式图改为与 Panel B 对齐的三块布局：clustered heatmap、mean-rank profile、model–data correlation matrix；删除 Rank dist. panel。相关矩阵行标 model forward/reverse，列标 data forward/reverse。灰格表示保留触点在单个事件中未被招募。必须注意：该 21-seed 池仍来自旧 per-template 坐标几何；在 shared-plane 下完成同规模重跑前，不能把 Panel C 写成与 Panel A/B 完全同几何的正式验证。

Pooled KMeans 得到 `model forward/reverse=103/119`、direction purity `1.000`、within-cluster tau `0.919`、shared-overlap correlation `−0.983`。在每个 paired seed 内置换方向标签后，cluster–direction association 为 `P=1.0×10⁻⁴`。LOSO purity 中位数为 `1.000`、范围 `0.714–1.000`；shared-overlap correlation 中位数 `−0.983`、范围 `−0.983–−0.358`。模型—真实模板矩阵为 `[[0.983, −1.000], [−0.963, 0.844]]`，四格方向性 channel-shuffle permutation `P≤0.001`。222 个事件用于事件级聚类显示，独立重复单位仍是 21 个 paired network seeds。

这里必须区分 Panel B 与 Panel C 的科学任务。Panel B 展示双核同网自发时，同一连续轨迹确实可出现相反事件；Panel C 为获得每类上百个可读事件，采用同一网络的 source-only/sink-only 控制 arm 检验两端各自能否产生与真实 data forward/reverse 一致的 rank pattern。它支持“两个端点具有产生相反 readout 的能力”，不支持“双核同网长期自发且方向平衡”。

新增工作点审计表明，更大的 core 不能稳定解决双核同网的方向不平衡。旧几何中 `core_r=1.5` 的 26-seed、8 s spontaneous 审计为 29/99；`core_r=2.5` 在 4 个 4 s seed 中为 11/14，但同样 4 个 seed 延长到 8 s 后变为 4/28。shared-plane 的同 seed 复核也显示 `core_r=2.5` 为 0/11，而 `core_r=1.5` seed 5 为 1/4。因此正式 B 仍保留 `core_r=1.5`，不把 larger core 当作伪装后的“平衡工作点”。

## 输出与复现

- 正式分组输出：`results/paper-ready-figure/fig4_subject_snn_e1146/figures/`
- 一键复现：`python scripts/paper_figures/plot_fig4_subject_snn_grouped.py`
- Panel A：`fig4_panel_a_model_setup.{png,pdf,svg}`
- Panel B：`fig4_panel_b_bidirectional_readout.{png,pdf}`
- Panel C：`fig4_panel_c_model_validation.{png,pdf}`
- Panel C 统计：`fig4_panel_c_model_validation_{statistics.json,per_seed.csv}`
- 工作点审计：`fig4_working_point_audit.{json,csv}`
- 总 metadata：`fig4_grouped_metadata.json`

## Caption 骨架

**Figure 4 | A patient-specific anisotropic E/I spiking network generates opposing interictal-event readouts.** **A,** Model hypothesis and E1146-specific placement. Only recurrent E→E connectivity was anisotropic, and equal low-threshold excitatory cores were positioned at the two template-source regions of the patient-specific propagation axis. **B,** A continuous spontaneous simulation window contained both forward and reverse events, producing opposing recruitment gradients in the same virtual SEEG montage. **C,** Across 21 paired network seeds, source-only and sink-only control arms yielded 103 and 119 clean directional readouts, respectively; masked-feature clustering separated the opposing rank patterns and reproduced the expected similarity structure relative to the patient’s empirical interictal templates.

## 当前解释边界

- 允许：shared-plane 的单个患者特异性模型底物可产生并读出两种相反的间期样传播次序；旧几何的控制 arm 模型 readout 与该患者真实模板顺序一致；聚类—方向关系在 paired-seed 分层统计下保留。
- 暂不允许：把当前 Panel C 写成 shared-plane 坐标下的高 n 验证；这需要对应重跑 21 个 paired seeds。
- 不允许：真实患者的两个模板已经被证明由两个低阈值 core 引起；双核同网可以长期自发且方向平衡；222 个事件等于 222 次独立仿真；单病例结果代表 cohort；模型解释了 HFO carrier 或 clinical seizure onset。
