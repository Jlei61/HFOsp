# Core A 阈值异质性 × 核内 EE 双参数分析

## 结论先说

按指定论文的思路，固定 Core A 的平均阈值，只缩窄其阈值分布，已完成 **704 个双向参数扫描点**，并补上静息态 fold 与 PD1/PD2/PD3 的二维延拓。周期 fold 只在实际求解和校验通过的区段绘出；这张图是主要边界的补充版，**不是所有周期微分岔已经闭合的完整分类**。

1. 缩窄 A 的阈值分布会削弱低输入下低阈值细胞对群体响应的贡献。A 主导的静息失稳边界先向更强 EE 移动，随后换成未改动的 B 先失稳，所以全网首次失稳边界不再继续明显右移。
2. 强连接下仍存在两种稳定周期状态：A burst / B 高背景，以及 A、B 都维持高背景振荡。同一参数下不同初始历史可以进入不同状态；在 h=0、g=1.38 已用精确周期轨道和横向 Floquet 谱验证。
3. 异质性也改变周期分支，但不能用“平均率上升”代表论文中的细胞同步性突变，也不能直接概括成“异质性越低越容易发作”。本次没有得到稳定 irregular / chaos 的证据；短窗未分类不能当作 irregular。
4. 周期轨道存在、静息态局部稳定、有限扰动能否跨入另一吸引域，是三个不同问题。**不能把低率 fold 和 cycle fold 之间整片区域直接涂成稳定双稳态**。本次额外将原 A-burst 轨道作为初始历史，在若干较低 h、g=1.14 条件下仍回到静息态，说明只看轨道存在边界会漏掉稳定性或吸引域限制。

## 与参考论文的联系

参考 Rich、Moradi Chameh、Lefebvre 和 Valiante，*Cell Reports* 39, 110863 (2022)，[原文 DOI](https://doi.org/10.1016/j.celrep.2022.110863)。原文测量的是人皮层 L5 锥体细胞的 **distance to threshold（DTT，发放阈值减静息膜电位）**，不是直接测一个网络 rate 模型的阈值参数。平均 DTT 相近的比较中，致痫组织的细胞间离散程度较低；实验样本单位为细胞，且样本数量、脑区和患者来源不完全匹配。

论文用 rheobase 异质性改变群体输入–输出函数，讨论在外源驱动变化时，多稳态和同步性突然跳变如何出现。它同时考察 E、I 异质性；E 的影响取决于 I 的设置。本文采用它的**固定平均值、改变分布宽度**这一机制操作，保留本任务原来的六群体模型和 EE 扫描轴；不是重跑论文的模型。

固定静息参考时，阈值标准差可以作为 DTT 离散程度的模型代理。但本模型没有逐细胞静息电位分布，也没有细胞同步性读出，因此没有把 h=1 标成正常组织、h<1 标成患者，更没有把某个 h 解释成病理组织分级。病理联系目前是可检验的机制假设，不是患者参数拟合结果。

## 方程中的实际改动

\[
\theta_{A,i}(h)=\bar\theta_A+h(\theta_{A,i}^{(0)}-\bar\theta_A),\qquad
\sigma_A(h)=h\sigma_{A,0},\qquad \operatorname{Var}(\theta_A)=h^2\sigma_{A,0}^2.
\]

纵轴 h 是**标准差的相对大小**；若要画方差比例，纵轴改成 h² 即可，不是另一套动力学。横轴 g 同时缩放 AA、BB 的 EE 一阶权重和平方权重，分别乘 g、g²。Core B、surround、所有 I、外源输入、神经元数、连接矩阵、突触滤波和全部物理延迟均冻结。

这仍是 `[A_E, B_E, S_E, A_I, B_I, S_I]` 六群体延迟率系统，包含突触滤波历史；不是把六个 rate 当成六维无延迟 ODE。求积采用原始阈值经验分布的 32 节点 Gaussian quadrature；对节点作同一仿射收缩，权重不变。A 的 720 个 E 细胞、均值 17.263954 mV、参考标准差 0.739744 mV 保持可追溯。h∈[0,1] 不引入高于背景的 A 阈值，也不靠裁剪补偿均值。

## 图该怎么读

- `figures/heterogeneity_two_parameter.pdf`：关键范围放大。左图为严格求出的临界曲线，中/右图为 EE 增大/减小时的有限时间吸引态候选。纵轴不再是放电率，而是被控制的异质性参数；放电率用于分类和画代表轨道。
- `figures/heterogeneity_two_parameter_full.pdf`：同一结果的完整 EE 扫描域。
- `figures/heterogeneity_onset_detail.pdf`：低率平衡态的首次 fold，以及 reference 附近 cycle fold 的已求解部分。
- `figures/heterogeneity_transfer.pdf`：阈值分布及固定外源波动方差下的群体响应；低输入局部用对数 rate 轴放大。
- `figures/heterogeneity_coexisting_states.pdf`：相同参数的两个稳定周期状态，曲线单位为 Hz / cell；I 的高 rate 不等于“抑制失效”。

实线/虚线区分 fold 和 PD 类型，不是在此图中表示整条周期分支稳定或不稳定。空心端点是本次延拓的实际边界，不是已经确认的 cusp、同宿或另一种分岔。色块只代表已采样初始历史的最终行为；相邻点之间不构成精确状态切换线。

状态命名使用 core E 的周期内范围：最大率低于 5 Hz 为 quiet，最低率低于且峰值超过 5 Hz 为 burst，最低率不低于 5 Hz 为 high background；静息候选另要求全六群体时间波动小于 0.005 Hz。这些是读出分类阈值，不能产生或定义 LP/PD 曲线。

## 已完成的数值证据

| 曲线 | 已解点数 | h 覆盖 | 状态 |
|---|---:|---|---|
| SN | 62 | 0.000000–1.000000 | 覆盖设定 h 域 |
| LPC_onset | 12 | 0.980525–1.000000 | 局部已解，端点未闭合 |
| LP1 | 5 | 0.963992–1.000000 | 局部已解，端点未闭合 |
| PD1 | 23 | 0.000000–1.000000 | 覆盖设定 h 域 |
| PD2 | 22 | 0.000000–1.000000 | 覆盖设定 h 域 |
| PD3 | 21 | 0.000000–1.000000 | 覆盖设定 h 域 |

扫描使用 11 个 h、32 个 g、上/下两种参数延续历史；每个参数点完整携带滤波和延迟历史。远低于失稳阈值的低率状态直接解平衡点，其余点积分 3–9 s；不足三个周期的两个点另延长 30 s。全六群体轨迹的跨周期 RMS 差小于 0.1 Hz 才作为周期候选，不把峰间距近似相等独立当作周期证明。

验证文件：

- `model_validation.json`：h=1 复现原模型；样本及求积后的阈值均值/方差约束。
- `equilibrium_validation.json`：低率分支的完整 DDE 特征根、40/64 阶历史离散比较、精确延迟特征方程，以及 32/64 节点阈值求积比较。没有在已检查的首次 fold 之前检测到 Hopf；这不排除其他分支上的 Hopf。
- `critical_grid_validation.json`、`critical_validation/`：代表 PD/LP 的周期网格加密。PD1–3 的横向乘子还通过 `critical_floquet_validation.json` 中两个时间步直接趋近 −1，其余领先横向乘子在单位圆内。
- `cycle_fold_offgrid_validation.json`：onset cycle fold 从 4096 到 8192 网格的轨道缺陷与临界零模检查；长周期端的零模精度比轨道率残差更早受限。
- `validated_attractors/`：4 个代表周期轨道的网格、时间步及横向 Floquet 检验；其中 4 个通过稳定周期吸引子检验。它们是数值实例，不是全网格每点都做过谱分析。
- `long_transients/`：两个短窗无法分类的慢周期条件的独立延长轨迹；不修改其他条件原来携带的状态历史。

## 尚未闭合的部分与拒绝记录

LP1 的低 h 延拓端点尚未分类。Onset cycle fold 的周期随 h 降低变长，必须同时满足周期方程残差和真实的分支切向量符号改变；仅有很小的 d g / d s 会把平坦长周期尾段误认成 fold。本次已排除这种候选，采用更严格的符号改变判据与更高周期网格重算。

原 h=1 主图中极窄的 LP0a/b/c 与 PD0 系列没有获得可靠的 h 延拓，保留在原参考结果中；本次不外推成新的二维曲线。`boundaries/` 的失败与拒绝点保留供数值审计，但汇总表和图只使用接受的区段。早期 `state_scan/` 存在文件名碰撞，整个目录被 `REJECTED.json` 明确弃用；有效扫描是 `state_scan_v2/`。

目前可接受为**六群体降阶模型的双参数主要边界与吸引态候选图**。不能据此声称原生空间 SNN 已复现这些阈值、病理组织已被分类，或同步性韧性机制已在患者层面验证。图已作 Agent 数值与目视检查，仍待用户人工看图。

## 关键代码与复现

源码目录：`scripts/topic4_core_heterogeneity_v11/`。`common.py` 定义均值不变的异质性参数；`equilibria.py` 解低率 fold；`periodic_boundaries.py`、`pd_joint.py` 和 `fold_joint.py` 解周期临界条件；`trace.py` 是二维延拓入口；`integrate.py`、`state_scan.py` 负责历史连续的双向扫描；`figures.py` 生成全部图。

在此 worktree 根目录执行：

```bash
python scripts/topic4_core_heterogeneity_v11/equilibria.py
python scripts/topic4_core_heterogeneity_v11/state_scan.py --workers 8
python scripts/topic4_core_heterogeneity_v11/trace.py PD2 --N 1024 --step .05 --suffix _rerun
python scripts/topic4_core_heterogeneity_v11/trace.py LPC_onset --N 4096 --step .002 --suffix _rerun
python scripts/topic4_core_heterogeneity_v11/validate.py equilibria
python scripts/topic4_core_heterogeneity_v11/figures.py
python scripts/topic4_core_heterogeneity_v11/report.py
```

`state_map_summary.csv` 保存有效扫描读出，`bifurcation_curves.csv` 保存实际临界点；两者的 source 字段分别回到轨迹或周期轨道。新实现依赖此前已发布的模型版本 `5a5516a8`，未改动其他 Fig5、空间 rate 或 Z/M 任务。

`_rerun` 为独立复跑目录，避免覆盖本次接受结果；绘图入口默认读取 `figures.py` 中明确列出的接受目录，不会自动将新复跑结果升级为正式输入。
