# figures — MZ 病理轴走廊刺激位点比较

模型侧机制探索图。**NOT 临床 / NOT 真实发作 / NOT DBS 疗效**；上限="在这个由病人间期梯度轴映射出的 Z+M SNN 里，中段虚拟抑制改变了模型 runaway 时间和模型传播范围"。

### mz_gradient_corridor_stimulation_cohort.png

队列统计主图，**两联面板**，结果直接从点的分布读出（无面板标题、无轴上统计文字块、图例里不塞统计数——数值在 `cohort_statistics.json` 与本 README）。n=4：E1146/E590/E958/yuquan_zhaochenxi。**A 位点效应**：逐病人 middle−endpoint 受限无失控时间（C_run 实心 / C_best 空心）+ 零线 + **两条队列中位数线（all-available 实线、complete-case 虚线，两套分析集并列、不钦点）**。点在 0 上=中段更好、0 下=端点更好。**B seed 稳定性**：逐病人 C_run 在各 seed 的值（一 seed 一色）——同侧紧簇=位点偏好 seed 稳定，跨 0 散开=seed 依赖。

**关注点**：**混合/阴性 + 个体异质**。A 面板两条中位数线都在 0 稍下、彼此接近（all-available C_run 中位数 −293、complete-case −613；C_best −949/−1176；都 +2/4、精确符号翻转**对均值** p=1.000、Wilcoxon 1.000）→ 结论对分析集稳健。B 面板：E958 三 seed 紧簇正（稳健中段胜）、E1146 两有效 seed 负（稳健端点，seed3 有 headroom 仍偏端点=非删失伪迹；seed4 baseline 20s 不失控无臂）、E590 全正、zhaochenxi 跨 0（seed 依赖）。**真正稳定=个体异质，非统一中段优势**。跨走廊传播（far-reach）**不作面板**：共同 1s 刺激后窗下逐病人 far_delta ~±0.01、贴近 0、只有 E958 窗口稳健 → 无窗口稳健位点差异（旧"每臂各自 [stim_off,t_run)"会因中段推迟 runaway 拿到更长窗而虚高，已修）。**selective corridor disruption 未被支持**；"短走廊→中段/长走廊→端点"只是待检验假设（n=4，走廊长度与 core 距离/覆盖/删失混杂）。

### baseline_dynamics_seed1.png

基线动力学诊断（合格性判读）。每行一个病人：群体 E-率（黑，红点线=120Hz runaway 判据）+ active-fraction（蓝，虚线=冻结事件条）+ 平均 z（青），红竖线=runaway 时间、橙带=[45%,75%] 刺激窗。

**关注点**：确认冻结候选在每个病人几何里产生的是**间期离散可恢复事件（黑率迹清晰见 40–70Hz 爆发后回 0）从 t=0 起贯穿全程 → 缓慢 z-耗竭爬坡（z 只从 1.0 掉到 ~0.905）→ late 段越过 120Hz runaway**。事件幅度随 z 耗竭单调增大，所以事件条必须用早期（z≈未耗竭）间期尺度校准（~0.022–0.033），否则被 late 爬坡抬高会漏掉早期小事件（这正是早先误判"0 个 pre-stim 事件"的原因，已修）；修后 4 例逐病人 pre-stim 可恢复事件 13–18 个 → 全部合格。⚠️ E1146 走廊最长→runaway 最晚（≈T_max）→RRT headroom 极小。
