# Fig. 5：模型谱系、runaway 跳变与转变前易感性审阅

## 1. 一句话判断

当前 two-core Z/M 模型能展示返回型间期事件到持续高放电 runaway 的转变；但不能把现有证据写成“低活动时所有扰动都消散，转变前沿连接长轴统一放大”。Figure 4 当前展示资产、rev22 工作点辨识和 Figure 5 冻结仿真属于不同版本，正文需要先接好这条链。

本轮按作者要求，只做进入 runaway 的过程；退出、恢复和持续深调制振荡不作为交付门槛。参考示意图仅提供构图和科学问题，不能提供轨迹、分叉、显著性或易感性结论。

## 2. 完成程度与版本关系

**本轮目标的科学完成度：65/100。** 原始轨迹、空间输出、精确 checkpoint 和配对扰动均有实际资产；局部分叉类型、对各向异性的因果归属、多 topology 稳健性及 rev22 基底接入仍未完成。这个分数是审阅判断，不是统计量或工程测试通过率。

| 层次 | 当前可核实版本 | 正确解释 |
|---|---|---|
| Fig. 1–3 | root 下 `results/paper-ready-figure/fig1/`–`fig3/` | 刻板事件、空间传播、与早期发作场的经验联系。本轮检查了渲染和登记，并未重新审计整个临床数据层。 |
| 当前 Fig. 4 图片 | `figure4_panel_registry.json` 指向 rev11 NLC frozen-substrate confirmation，B 预留 | 仍展示连续 Node field；不能因为最新模型回到 two-core，就把现图当成已更新的 two-core 结果。 |
| two-core 机制工作点 | rev20 `dualcore_s39 + Joint=1.25` | 两核约 1.8 mm、总计 1499 个 E 神经元。Joint 实际为 `g_EE=0.625, g_EtoI=1.25`。保留患者事件分布距离、双模板和 OOD 三个分开的读出。 |
| 正在辨识的模型 | rev22 DCI；主要工作区 `.worktrees/topic4-dual-core-mechanism-scan`，后处理在 `.worktrees/topic4-rev22-postfit` | 固定 core 数量、位置、预算与 Vth 映射；调 E→E、E→I redistribution、椭圆方向偏移与长短轴比。没有独立的曲率参数，也没有重新自由优化 X/Y。 |
| 本轮 Fig. 5 | rev21 Z/M，沿用 rev20 Joint=1.25 | 40,000 cells、topology 2542、dynamics 2642、OU 连续开启。不是 rev22 新的最优连接工作点。 |

two-core 中心为 `(1.5377343, 1.2264180)` 与 `(18.6066121, 2.4177134)` mm；其冻结 field 类型是 `manual_dual_core_budget_matched`。data-driven 修饰被数据选择的模型几何/连接表达，不能暗示这些位置是患者真实病理 core 的测量值。

rev22 的全四维训练选择点为 `dci_p030`：`g_LEE=0.0454, g_LEI=1.4076, theta_FT=-8.57° (offset), AR_FT=1.6284`。另有不同子模型的候选；它不是已经通过全部盲验证的唯一正式工作点。本轮读取时，`full_postfit_chain.status` 为 `OPENING_SELECTION_BLIND_VALIDATION`，对应聚合进程仍在运行；未干预该任务。

## 3. P0 / P1 关键问题

### P0（若据此作空间机制结论）：旧 C 图把空间轴转置混用

`src/topic4_rev21_zm_transition.py` 的空间计数按 `flat=iy*grid_n+ix` 存成 `[time,y,x]`。旧 `_spatial_rate_maps` 直接除以 `histogram2d` 返回的 `[x,y]` occupancy；onset map 也被当成 `[x,y]` 拟合方向，core contour 又与 `imshow` 使用了不同的转置。结果会错置图上热点、传播方向和 core 位置。

本轮统一在进入计算时转成 `[x,y]`，只在渲染时转回 image 的 `[y,x]`。加入不对称 occupancy 的回归例，防止对称网格掩盖错误。修正后的三窗方向余弦为 0.995890、0.992735、0.947163；保留它们在 metadata，不把高余弦画成各向异性因果验证。

### P1：旧 B 图的 adaptation current 多乘了一倍

旧主图从 round reference 读 `eta_m=0.0074515944`；实际候选 `rev21_ts_tz3000_ta500` 的 gain 是 `0.0037257972`。因此旧图的 `eta_M*M` 数值大了一倍。本轮从唯一匹配的 candidate manifest 读取，缺失或重复就报错；已添加回归检查。

### P1：旧空间扰动取在 onset 后，不能回答转变前易感性

旧两状态为 1000 ms 和 2615.4 ms，后者晚于 scientific onset 2515.4 ms。新 checkpoint 取既有 `w_pre_ms` 起点 2015.4 ms；从既有 baseline checkpoint 连续复现，population-rate 每个时间步完全一致。新 200 ms 观察窗结束在 2215.4 ms，严格早于 onset。

### P1：equilibrium fold 不是已验证的 runaway 分叉

既有 `s=0.355450824` fold 的邻近平衡态在原生 delay 系统下不稳定。此前有界/tonic 跳边 `0.428<s_c<=0.429` 是特定准备状态下的 operational escape bracket，不能称为已经证明的 limit-cycle fold、crisis 或热力学相变。

## 4. 科学性结果：扰动没有支持原先的简单故事

新试验沿用既有 16 个分层随机位置、16-cell 注入与 200 ms 窗，每个 probe 都从同一个状态恢复，sham 使用相同后续随机驱动。注入那一帧的强制放电从响应中排除；所有位置保留，16/16 在两状态都通过原 E1 可估计门。

| 读出 | Low activity | Pre-onset |
|---|---:|---:|
| 0–50 ms 额外后继放电，跨位置均值 | 1770.4375 | 754.25 |
| 0–50 ms 额外后继放电，跨位置中位数 | 9.5 | 175.5 |
| 0–200 ms 额外后继放电，跨位置均值 | −4488.4375 | 569.9375 |
| 0–200 ms 额外后继放电，跨位置中位数 | 163.5 | 193.5 |

短窗 10/16 个位置增加，长窗为 7/16。低活动期几个大响应点使均值和中位数方向不同；这也显示结果依赖响应窗口。图展示 signed extra spikes 和全部位置配对，不仅显示正值，也不删掉强响应位置。

当前安全结论是**单条轨迹上的空间响应随状态变化**。尚不能说“低活动时扰动必定被消散”，也不能说“转变前全局易感性普遍上升”。更不能把本次两个状态之间的差异单独归因于 Z，因为快变量、M 与 Z 同时不同；图中没有旋转/各向同性干预来单独识别连接长轴的作用。

这里的 common future noise 指每个状态内部的 probe/sham 配对；两个时间点各用原轨迹自己的后续随机驱动，并未跨状态复用多套相同 future-noise realization。因此这些空间点也不能替代独立噪声重复，当前只是两个具体 checkpoint 的条件响应。

## 5. 工程性与本轮产物

新入口：

- `scripts/prepare_fig5_preonset_checkpoint.py`：从旧 low checkpoint 精确延续到 pre-onset。
- `scripts/run_fig5_preonset_perturbation.py`：相同 16 个位置的 pre-onset probe-minus-sham。
- `scripts/run_fig5_upper_runaway_map.py`：原生 0.1 ms、10 s、两种准备状态的上侧 Z×M 扫描。
- `scripts/paper_figures/build_fig5_transition_susceptibility.py`：从真实数组重画六联图、诊断补图、PNG/PDF/SVG、metadata 和中文 README。

扫描配置在 `config/fig5_upper_runaway_map_20260905.json`。保留 rate、regional rate、M、Z、频率、调制深度、末段漂移和未定类别。频率/深度只作诊断，不加回 tonic runaway 门槛。首次扫描的 JSON 序列化遗漏了 modulation profile 的 NumPy→list 转换，已修复后重跑；首次配置留在 `interrupted_serialization_contract.json`，不作为结果。

全部更改限定在 Figure 5 工作区；没有改变 Figure 4 参数辨识、根工作区已有冲突或其他 Topic 5 任务，也没有 commit/merge。

## 6. 最小修改路线与下一步

1. 用本次修正的坐标、实际 gain、真正 pre-onset probe 和上侧扫描完成可审阅候选图；保留原被拒绝版本。
2. 等 rev22 盲验证正式闭合，再冻结一个带完整几何/连接 hash 的 Figure 4→5 输入包。换基底后必须重做转变与扰动，不能仅换标签。
3. 在这个基底上以多个 topology×dynamics seed 比较等强度、沿轴/离轴匹配距离的扰动；用旋转/各向同性连接对照识别方向性。用快状态匹配或 Z-clamp 识别慢抑制的作用。
4. 若正文需要具体分叉名称，再对相关有界吸引子做延续和稳定性分析；若只讲进入 runaway，保留 operational escape boundary 足够，不扩展到发作恢复。

核心目标始终是：同一个能够产生患者样间期传播的冻结网络，是否以及怎样进入 runaway；空间结构是否改变转变前对局部输入的响应。

## 7. 本轮扫描闭合与最终图

80/80 条轨迹完成，40 个参数格各含两种准备状态；运行步长 0.1 ms、驻留 10 s、末 1 s 判读，保留原生 realized delays。pre-fold 准备得到 21 个 bounded、19 个 tonic runaway；recruited 准备在本次上侧范围内 40/40 都保持 tonic。本次没有 unresolved，也没有两种准备都 bounded 的格；不向未扫描区外推。

| M gain 相对值 | 从 pre-fold 准备出发的 sampled escape bracket |
|---|---|
| 0× | 0.420 < s_c ≤ 0.428 |
| 1× | 0.428 < s_c ≤ 0.429 |
| 2× | 0.429 < s_c ≤ 0.440 |
| 4× | 0.429 < s_c ≤ 0.440 |

在 s=0.429，1× M 的终末群体率为 316.161 Hz；2× 与 4× M 仍为 221.973 和 220.633 Hz。该比较支持 adaptation 改变进入 runaway 的操作边界；2×/4× 的精确阈值尚未在这个括号内细化。两种准备不同的结果表明初值依赖，但有限时间扫描不构成完整吸引子计数或局部分叉定型。

主图为 `results/paper-ready-figure/fig5_transition_susceptibility_v3/figures/fig5-transition-susceptibility-v3.{png,pdf,svg}`，附相同 stem 的 metadata。另有 carried-state / pre-onset 响应时间图，以及两状态 0–50 / 0–200 ms 响应场补图。后者区分“从哪里刺激”与“响应在哪里出现”，同一行使用共享 signed 线性色标。

验证：41 项相关测试通过；80/80 扫描数组及其输入 SHA-256 完整核对；新 probe 输入 hash、原轨迹 hash、checkpoint 和 sham 精确复现均通过。三张 PNG 与主图同状态 PDF 已目检，PNG/PDF/SVG 来自同一次 producer 执行。候选状态是 `CANDIDATE_PENDING_AUTHOR_REVIEW`，没有作者视觉接受，也没有据此升级生物学结论。
