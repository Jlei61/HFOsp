# 2026-09-13 07:07 持续探索：完整五状态与观测分辨率

**Goal仍active，截止09:40，当前剩余约2h35。** 前一轮/本轮均有实质进展，不是blocked或完成。

## 已完整完成

- GPU1 dense PID276790 **已正常退出**。`early_Z_lookup_dense/runs/early_z_refill_s9108401/result.json` 和DISPLAY同名result实际end37.5。first10.59/confirm10.79；Zrefill12.8–13.8；recovery13.2/confirm15.2；second26.89/confirm27.09；无M清零/无第二外部干预；second确认后10秒观察已完成。
- 完整候选 `early_Z_lookup_dense/candidate/figures/fig5.png` 及zoom已生成，完整图已Agent目视并在metadata标记。full1–5=True，但**不是自主终止，也没有通过低频burst和早期能量验收**。完整case `scientific_review.md` 已写。旧1ms E2仍2/15contact↑、114/400native↑、rho.668。
- 原CPU early seed1 **20秒完整块已写出**：与dense0–20的12组观察全部位级一致，其中10–20是独立执行，覆盖进入/补Z/返回；证据 `early_Z_lookup_dense/full_observation_parity_through20s.json`。没有声称20秒完整膜/RNG状态逐数组比较，那个证据仅在200ms全引擎QA中。
- 原Mgrid e0_t0seed1 GPU到20秒后，从0–12.5全部观察与早Z分支干预前完全一致。证据 `cuda_ordered_scatter_qa/M_grid_pre_intervention_observation_parity.json`。`M_first_endpoint_snapshot.json` 当时首次终点2/40（两不同参数各seed1），完整配对格0。

## 新关键科学发现

`analyze_topic4_recurrent_high_native_rhythm.py` 固定4个1秒窗口：1–2有限事件，11.8–12.8firsthigh，19.2–20.2afterrefill，34.89–35.89secondhigh。无需新仿真，读取完整0.1ms80-neuron raster和1ms全网计数。图 `recurrent_high_native_rhythm/figures/native_rhythm_and_sampling.png` 已生成/目视；analysis/review已写。

第二高态全E/I均率476.606/600.432 Hz；分层采样E的1–150Hz相对RMS仅0.0632%，三组E的ISI均2.1ms、CV近0。有限事件与恢复后明显是间歇包络。**不能笼统说完全没有任何振荡**：细胞是~476Hz快速周期放电；但目标的低频群体burst尚不支持。原始1ms全E相对RMS.0681%，400格median1.396%、q901.950%，这些还含下述分辨率影响。

**实际发现混叠**：同一采样E在secondhigh从0.1ms改1ms计数，1–150Hz相对RMS放大16.95倍（firsthigh4.08倍），有限事件接近1。不是不同采样N的比较。高阶快速脉冲可混叠到低频；1ms native字段不能当未经混叠的神经元1–150Hz谱。

## 正在跑的唯一新增测量重放

因上述实际观测问题，新增**一个12秒同条件测量重放**（不是另一个生物学条件/完整长程副本/额外F样本），明确记录于window和plan；原M40终点完全不变。

- `scripts/replay_topic4_early_energy_high_resolution.py`，PID**417978**，GPU1，OUT=`early_energy_high_resolution_replay`。实际M=.005/τ1、seed9108401、从原始0秒到12秒，额外存400格 `field_0p1ms.npy` 和全E/I `population_0p1ms.npy`，observer只读spikes，不用RNG。原9个物理源仍未改变。最新真实3秒RUNNING。
- 必须与既有dense0–12的12组旧观察完全相等；0.1ms聚合至1ms必须和原field/counts逐项相等，才写`qa.json PASS`。源worker输出COMPLETE12秒只是测量重放终点，不能当原M40完整试验。
- 自动分析 waiter `scripts/analyze_topic4_early_energy_high_resolution.py --wait`，当前PID**434716**（旧431350在修复完成瞬间QA/PID竞争后有意重启，仅分析器）。等待qa，随后重算同基线1–8.59/同target10.59–11.59的native能量；电极/患者不变，生成 `energy_comparison.json`、`figures/native_energy_resolution.png` 及 `candidate/figures/fig5.png`。等待器会核对实际运行PID，失败不伪造完成。
- 共享plotter增加**可选**`a['early_field_0p1ms']`和来源metadata；E2标题明确Native spikes·1或0.1ms；D仍用原1ms场，全时间线37.5秒，0.1ms高分辨率只覆盖前12秒。默认1ms三个旧输出数组已与旧metadata完全一致，证据 `default_1ms_compatibility.json`。完成后必须视检新E2，不能提前说修好了能量匹配。

## 原Mgrid第二种子加速

GPU1密集分支结束后，原e0_t0seed2旧CPU4084843从10秒checkpoint接GPU1，**PID421172**。新Mcontroller **421173**（旧337612有意停止协调）；其余16个worker逐个PID/create_time不变，包括GPU0 native seed1 **337611**。记录 `cuda_ordered_scatter_qa/M_grid_seed2_adoption.json`，归档 `M_grid_e0_t0_s9108402_resume_parent.pkl`。controller已正确接管无失败；新seed2实际推进12秒PRE_ENTRY，旧CPU16秒progress留在audit可参考。原60秒自然返回/180秒恢复后观察/430秒最大不变。

GPU1现在同时为原Mseed2和12秒测量重放服务；每个约2.3–2.4GiB CPU RSS，机器约115GiB可用，GPU远低于24GiB上限。没有OOM，其他任务未停。GPU0原Mseed1已29秒HIGH。

## 其余与下一步

06:59库存：Z122/147 complete、25running；M0/40完整、18running；fullfast90 pilot4073743绝对159秒、无再进入、终点166.5；原ZM-only3202691仍未完成1000秒；CPU early30171/31978保留。

优先等0.1ms replay及其自动Fig5，审查实际能量变化是否仍失败；fullfast90完成后做完整90秒配对判读，若无进入不得启动预案的电压/突触拆分90秒试验。Mseed2到20秒将给第一个完整两种子F格，届时可运行plotter --update刷新首次终点图；最终交付前还要刷新已完成case的F，不能永久保留灰图。09:40按照完整requirements audit提交实际结果/未完进程清单，停止新增探索，但不要缩短既定随访。

`snapshot_topic4_fig5_exploration.py` 已加入highres重放/QA/analysis来源、CPU20秒全观察parity和recurrence counts核查；**五状态按唯一轨迹去重**（原前缀/完成图/修正能量图不能算多条）。当前9个完整布局artifact但仅1条五状态轨迹。
