# 六群体 rate 降阶模型：方程、分岔代码与复现入口

本包发布此前讨论的确定性六群体延迟率模型、分岔分析代码、必要的冻结输入与代表轨道，以及分岔 PDF 和结构图。它属于历史模型分支；原生空间 SNN 与这些降阶分支的定量对应尚未通过验证。

## 先看哪些代码

| 任务 | 代码 |
|---|---|
| 模型定义、输入均值/方差、阈值积分、参数缩放 | [`model.py`](../scripts/topic4_core_bifurcation_v2/model.py) |
| 原生图到六群体的连接矩投影，原环境可选步骤 | [`prepare.py`](../scripts/topic4_core_bifurcation_v2/prepare.py) |
| 平衡点 fold 与伪弧长延拓 | [`branches.py`](../scripts/topic4_core_bifurcation_v2/branches.py) |
| 六群体延迟系统时间积分，仅用于生成周期求解初值 | [`dynamics.py`](../scripts/topic4_core_bifurcation_v2/dynamics.py) |
| 自由周期 Fourier 配点求解 | [`periodic.py`](../scripts/topic4_core_bifurcation_v2/periodic.py) |
| 平衡态延迟谱与周期轨道 Floquet 稳定性 | [`eigen.py`](../scripts/topic4_core_bifurcation_v2/eigen.py)、[`floquet.py`](../scripts/topic4_core_bifurcation_v2/floquet.py) |
| 周期 LP、PD 及真实 2T 子支 | [`folds.py`](../scripts/topic4_core_branch_connections_v5/folds.py)、[`flip.py`](../scripts/topic4_core_branch_connections_v5/flip.py)、[`antiperiodic.py`](../scripts/topic4_core_branch_connections_v5/antiperiodic.py) |
| 后续解析增益、倍周期和全局分支分析 | [`v7 代码说明`](../scripts/topic4_core_network_bifurcation_v7/README.md) |
| 刚查看的两页分岔 PDF | [`v9/figures.py`](../scripts/topic4_core_bifurcation_layout_v9/figures.py) |
| 分岔类型、临界模态与 T/2T 对照图 | [`v8/figures.py`](../scripts/topic4_core_bifurcation_v8/figures.py) |
| 六群体红蓝结构图，E 圆圈、I 三角形 | [`plot_topic4_six_rate_schematic.py`](../scripts/paper_figures/plot_topic4_six_rate_schematic.py) |
| 发布包的有界数值核验 | [`verify_package.py`](../scripts/topic4_core_bifurcation_v2/verify_package.py) |

## 简化方程与实际设置

群体顺序为 $r=(r_E^A,r_E^B,r_E^S,r_I^A,r_I^B,r_I^S)$。S 表示 surrounding；六个 rate 变量之外，还包含突触滤波状态与延迟历史，因此不是严格的六维常微分方程。

$$
\tau_i\dot r_i=-r_i+\Phi_i(\mu_i,V_{E,i},V_{I,i}),\qquad
\tau_{r,j}\dot h_j=r_j-h_j,\qquad
\tau_{d,j}\dot c_j=h_j-c_j.
$$

把膜时间常数、突触面积等系数吸收到有效连接 $J,K$ 中，可以写成：

$$
\mu_i(t)=\mu_{\mathrm{ext},i}
+\sum_{j\in E,d}J_{ij}^{(d)}c_j(t-d)
-\sum_{j\in I,d}J_{ij}^{(d)}c_j(t-d),
$$

$$
V_{E,i}=V_{\mathrm{ext},i}+\sum_{j\in E}K_{ij}r_j(t),\qquad
V_{I,i}=\sum_{j\in I}K_{ij}r_j(t).
$$

精确源码记法为 $J_{ij}^{(d)}=\tau_{m,i}A_jW_{d,ij}$、$K_{ij}=\tau_{m,i}\sum_dQ_{d,ij}$；$A_j$ 保留原生离散更新的突触 DC 面积。$W$ 为每个接收细胞的一阶连接权重矩，$Q$ 为平方权重矩。递归方差使用当前 rate 的独立发放近似，均值则保留突触滤波和所有原延迟箱。

- 两核 E 的阈值分布较低，surround E 和 I 阈值保持参考值；两核的细胞数、阈值分布和投影连接并不完全对称。
- 扫描参数 $g=J_{EE,\mathrm{core}}$ 为无量纲倍率，同时仅作用于 A 核内部和 B 核内部 E→E：对应 $W$ 乘 $g$、$Q$ 乘 $g^2$。其他连接固定。
- 两核之间没有直接跨核 E/I 连接，分别通过 surrounding E/I 间接耦合。
- 只有 core E 私有 Poisson 输入保留在外源均值、方差中；共享 OU 已移除，其余外源输入为确定性均值。本模型不含逐时随机驱动或适应变量 a、Z/M。
- $\Phi_i$ 对实际经验阈值分布求积，包含复位、不应期及 colored-Siegert 修正。率响应时间常数属于尚未通过原生网络动态标定的闭合假设。

## 已保存的结果与结论边界

- [两页分岔 PDF](../results/topic4_sef_hfo/core_bifurcation_composite_v9_20260916/figures/core_bifurcation_composite_comparison.pdf)：A/B 联合分支及 a–e 代表波形。
- [类型和倍周期七页图册](../results/topic4_sef_hfo/core_bifurcation_types_v8_20260916/figures/core_bifurcation_types.pdf)：区分平衡点 fold、周期 LP、PD 和同宿型极限证据。
- [结构图 PDF](../results/topic4_sef_hfo/six_rate_model_structure_20260917/figures/six_rate_model_structure_colored.pdf) · [SVG](../results/topic4_sef_hfo/six_rate_model_structure_20260917/figures/six_rate_model_structure_colored.svg)。
- [分岔解释报告](../results/topic4_sef_hfo/core_bifurcation_types_v8_20260916/scientific_report.md) · [原生对应检查](../results/topic4_sef_hfo/core_spatial_readout_v10_20260916/scientific_report.md)。

低率态有 fold；更早存在 resting 与周期 burst 共存窗口。后续有周期 LP、PD 及高背景周期态，但有限 T/2T/4T 模式不等于非周期 irregular burst，也没有建立 A-E 与 B-I 周期相等导致分岔的因果结论。双 core 高背景不等于全网全部高背景，更不直接证明抑制失效。

## 独立检出后的快速核验

依赖为 Python 3.11、NumPy、SciPy、Numba、Matplotlib、Pillow；PDF 页面核验另需 `pdftotext`。此次使用 NumPy 1.26.4、SciPy 1.13.1、Numba 0.60.0、Matplotlib 3.9.2。

从仓库根目录执行：

```bash
python scripts/topic4_core_bifurcation_v2/verify_package.py
python scripts/paper_figures/plot_topic4_six_rate_schematic.py
python scripts/topic4_core_bifurcation_layout_v9/figures.py
python scripts/topic4_core_bifurcation_layout_v9/validate.py
python scripts/topic4_core_bifurcation_v8/figures.py
python scripts/topic4_core_bifurcation_v8/validate.py
```

第一项仅重新求低率 fold，并把四条已保存代表周期轨道代回实际延迟率方程，检查参数作用位置、临界零模和轨道残差；它不重新启动完整参数扫描或原生 SNN。其余命令重画已保存的结果，结构图和 v8/v9 图件仍待人工目视验收。

`model.py` 从当前检出目录读取冻结的 `projected_graph.npz`。v8/v9 图入口用 `rate_paths.py` 将历史结果表中的原绝对路径解析到当前检出目录；原始 JSON/CSV 的数值及来源路径保留，没有回退读取原机器目录。代码公式、权重及已存轨道未因路径修改而改变。

## 本次发布范围

发布包含 v2–v10 的分析源码、六群体投影、主要平衡支/周期曲线、v9 条件表引用的周期轨道、关键 LP/PD 模态与三条 2T 子轨道，以及报告、图件和结构图。完整逐步延拓数组、原生 SNN 的大型图缓存与逐条件轨迹、原始临床数据和下载文献不在本包中。历史报告中指向这些未随包发布文件的链接仍保留来源含义，不能据此宣称所有历史脚本均可脱离原环境运行。

`prepare.py`、v7/v10 的 `native.py`、原生空间与患者对照入口依赖原基底工作树和 `/data/hfosp`，不属于上述独立核验命令。已打包的冻结投影让六群体方程及主要图的复验无需这些依赖；v2–v7 的早期代码还可追溯到 `codex/recent-results-20260916-root` 的快照 `9a0d0e8a64d1013a5fd9ae04a051a4f74e6ea6a4`。

[发布文件来源清单](../results/topic4_sef_hfo/six_rate_model_publication_20260917/source_manifest.json) · [本次方程核验](../results/topic4_sef_hfo/six_rate_model_publication_20260917/equation_validation.json)。
