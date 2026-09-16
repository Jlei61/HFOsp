# Fig5 自主探索审阅 · 2026-09-13 09:45

已结束01:40–09:40的限定探索，以下是实际交付及仍按原合同继续的计算。

## 目前能回答什么

弱而快M（ηM=.005、τM=1秒）已经产生完整五状态：10.59秒首次进入，12.8–13.8秒只补Z，15.2秒确认恢复，26.89秒再次进入，完整观察至37.5秒。M全程开启且未清零。这排除了“补Z后模型结构上必然不能再runaway”，但返回是外部干预造成的。

第二个预定噪声种子也已再次进入：首次13.74秒，补Z15.95–16.95秒，恢复确认18.35秒，第二次28.47秒。其尾部随访已完成至39.0秒。

原ηM=.02、τM=2秒工作点的Z-only随访已到1000秒；释放后923.5秒未再次进入。另一个共享状态的完整90秒配对中，额外清M和额外清所有细胞快状态仍未再次进入，后期平均Z、M及E率很接近。既有M的直接衰减残留不能独自解释长期现象；清除后的活动及噪声实现仍可影响状态。90秒阴性不等于永久稳定。

外部将同一高态的M增益调至2可压制高放电；调至.2仍保持高率。前者趋于近沉默，既不是固定参数下的自主终止，也不是恢复有限事件的证据。

为什么仅打开M不一定足够：代码中每个E细胞每次发放给M加1，之后按τM衰减，反馈电流是ηM×M。在近似恒定放电率下，时间平均M约为τM(s)×r(Hz)，因此ηM=.005、τM=1秒、约500Hz时适应电流只有约2.5mV等效值。这个量级估计解释了“放得很快却没有被弱M压下”的可能性；它不是稳定性或分岔证明。

这不只是量级猜测：原弱M长随访的第一个种子在59–60秒，所有E细胞的原始GABA均在耗竭门槛以上，平均Z约4.45×10⁻⁵，E平均率500Hz，M反馈2.500mV；兴奋输入约1579mV，而乘Z后的抑制仅0.079mV。第二种子49–50秒及τM=2秒两种子29–30秒也持续越过门槛。此时现有方程令Z继续趋向零，M已接近发放与衰减的平衡，并非还会无限积累。这不支持“只要继续等待M累积就会自然返回”的解释；仍不能据此排除噪声作用、其他参数或其他状态中的返回。

原ηM=.02、τM=2秒的75.5秒实测高态，M约457，当前反馈约9.15mV，而平均净驱动约514mV等效值。保持同一状态代数地改ηM到.2或2，净驱动分别约432和−392；随后真正SNN的5秒续跑也分别维持高率和被压制。两层证据共同支持反馈相对回返兴奋不足这一解释，但尚未证明固定参数能兼顾进入与自主返回。

## 两个尚未补上的科学缺口

高态目前更像快速持续放电：完整例子的第二高态采样E神经元约2.1ms一次发放，目标1–150Hz的群体burst包络很弱。虚拟电极30–80Hz滤波后的形状不能证明原网络具有所需振荡。

早期能量经真实0.1ms重放核查，只有2/15模型电极和113/400原生网格增强。与固定Fig3C的空间ρ=0.668不能替代能量增强。1ms采样的混叠确实抬高部分主招募带低频功率，但校正后仍未复现目标。

第二种子的模型电极增强0/15，ρ=-0.339，也未复现临床早期增强。其原生空间图记录分辨率为0.1ms；两种子使用同一Fig3C，不按相关值挑选参照。

## 完整新版图候选

| 实际模式 | 参数与观察 | 图 |
|---|---|---|
| 补Z后返回并再次高率 | ηM=.005 / τM=1s，37.5s完整；E2用0.1ms原生计数 | [完整Fig5](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/early_energy_high_resolution_replay/candidate/figures/fig5.png) · [转变放大](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/early_energy_high_resolution_replay/candidate/figures/fig5_transition_zoom.png) |
| 补Z返回，长程未再进入 | ηM=.02 / τM=2s，1000s | [完整Fig5](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/completed_Z_only_1000s_fig5/figures/fig5.png) |
| 高率未自行恢复 | ηM=.02 / τM=2s，原有90s数据 | [完整Fig5](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/existing_mode_figures/weak_fast/figures/fig5.png) |
| 观察窗内未进入高率 | ηM=.02 / τM=20s，原有90s数据 | [完整Fig5](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/existing_mode_figures/weak_20s/figures/fig5.png) |

[第二噪声种子的当前Fig5](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/early_energy_high_resolution_replay_seed9108402/candidate/figures/fig5.png)，实际图中覆盖39.0秒；仿真完整随访是否结束以上述结果文件为准。

同一条轨迹的旧图、完整图、能量修正版和数值重放只算一条证据；原有90秒数据不会计入本轮40个完整随访样本。所有候选仍待用户目视及科学验收。

原弱M网格中较晚补Z的同种子对照已保存至80.0秒：先保留首次高态后60秒原生演化，再于70.80–71.80秒只补Z，已观察恢复。第二次高态尚未在该保存前缀中观察到。这是原M网格的进行中前缀，完整随访仍未结束，不能记为最终阴性或新增独立种子。[对应整图](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/late_Z_refill_fig5/through_80.0s/figures/fig5.png)。

## 扫描与剩余计算

M网格完整随访 0/40；首次终点已确定 8/40；两种子齐备的F格 3/20。
固定M=.02/τM=2秒的Z扫描 128/147 完成。它和弱M示例是不同工作点，不能把这张Z图直接当作弱M的同条件F。
首次进入时间从保存的全E发放计数核验，以≥200Hz持续200ms的确认时间计；未进入且未跑满180秒保留待定，不能提前填删失。完整返回/再进入随访继续独立进行。

[运行和checkpoint清单](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/comprehensive_inventory/latest.md) · [M参数图](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/m_parameter_modes_fig5_20260913/figures/M_entry_time_and_fraction.png) · [Z参数图](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/m_on_z_kinetics_20260912/figures/entry_time_and_fraction.png)

## 继续与停止标准

继续既定40条M轨迹（20个参数组合×2种子）、147条Z轨迹（49个组合×3种子）及已开始的有限对照，完成实际首次进入、原生恢复与再进入分类。高率计算明显更慢，GPU替换仅改变已验证逐位一致的加和执行，不缩短观察窗。
不增加90秒电压/突触拆分，因为先决条件“全快状态清除可再次进入”没有出现。09:40以后不增加探索条件；已有队列保留原参数、种子和完整终点。
下一里程碑先收齐原M网格并逐条区分：未进入、进入后自行返回、人工补Z后返回、再次进入、以及未恢复高率。首次进入时间F与这些完整轨迹分类分别交付；不能用首次终点齐备替代60秒原生高态观察和返回后的随访。
对既定网格中实际出现的自主返回候选，再沿用目前0.1ms原生计数审计局部和群体burst包络，并检查返回后是否恢复有限事件。只有这两端成立，才值得用同底物的简化模型复现、延拓固定点/周期轨道并命名分岔。若没有候选通过，当前M扫描应如实收束，提出反馈机制的明确改变后进入下一版；不通过重新滤波、挑窗口或额外几何变量来替代这一缺口。
在固定参数下同时看到有限事件、自主高态结束以及再次事件之前，不把这条线当作已完成的自主发作周期模型。若全部预定M参数组合仍无法兼顾两端，应先修正高态反馈与原生节律机制，再讨论患者能量匹配和分岔命名。三维轨迹仅是观察投影，不能直接充当闭合向量场、nullcline或Hopf证据。

[90秒配对诊断](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/reset_matched_90s/scientific_review.md) · [原生节律审计](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/recurrent_high_native_rhythm/analysis.json) · [能量观测审计](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/early_energy_high_resolution_replay/scientific_review.md) · [弱M高态实测平衡](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/weak_M_high_state_balance_snapshot.json) · [M电流量级](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/M_current_headroom/scientific_review.md) · [同状态增益续跑](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/high_state_M_gain_probe/scientific_review_complete.md)


较晚补Z对照的[首次进入放大](/data/hfosp/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/late_Z_refill_fig5/through_80.0s/figures/fig5_entry_zoom.png)与[补Z返回放大](/data/hfosp/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/late_Z_refill_fig5/through_80.0s/figures/fig5_refill_zoom.png)已完成代理目视核查；放大仅限A–C，其他面板保留完整轨迹上下文。
