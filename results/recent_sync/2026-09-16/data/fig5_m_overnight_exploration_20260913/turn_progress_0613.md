# 2026-09-13 06:13 自动探索进展

Goal 仍 active；09:40 停止新增探索。此轮没有改变任何九个已固定的物理源文件；再次逐个哈希核对通过。

## 本轮完整结果及图

- 同一 75.5 秒高状态的 ηM=.2 对照已完成到 80.5 秒，与 ηM=.02、2 的完整 5 秒三列比较生成并目视审查。末段 E 为 476.23 / 474.98 / 0.0011 Hz；η=2 的恢复确认 77.5 秒。外部参数突变只能证明干预能压下放电，η=2 目前接近沉默，不能当作恢复到间期的固定参数自主周期。`high_state_M_gain_probe/analysis.json` 已记 Agent 审查；见 `scientific_review_complete.md`。
- 早补 Z 的 .005/1 秒轨迹 Fig5 已实际保存并目视检查至 17.5 秒，①–④存在，⑤未出现，完整后续仍跑。最新数值已到 24 秒。A 参数标题与 Refill/Release 已分离；正式 C 保持 Z 与 M 双轴。E2 2/15 电极、114/400 网格增强，rho .668，科学验收不通过。
- 原始 .01/2 秒网格另生成并目视检查 `through_20.0s/figures/fig5.png`：16.43 秒进入，仍高率；④⑤空缺。E2 2/15 电极、108/400 网格增强，rho .789，同样不能以相关替代能量增强。
- 两个新图的 qualification、metadata、scientific_review 已更新 Agent 审查；未直接修改运行中 watcher 的 index。

## 数值加速：同一已存在任务，非新条件

`src/topic4_cuda_ordered_scatter.py` 完整 SNN 200 ms 高态对所有观测、完整引擎状态及 RNG 已位级通过。GPU backend sha 630260733fb93783f010a23c5e2ed5087b593552b76c7338d5c4572ee3250ec8。

- GPU1 密集观察同条件副本 PID276790 从 11.5 秒完整 checkpoint 继续，保留每 .5 秒观测；仍是原 early_Z seed1 的同一个条件，非额外 F 样本。原 CPU 两分支 PID30171/31978 继续，用于独立核查。12 秒 mean Z/M 和 tracker 已与 CPU 一致。
- η=.2 诊断旧 PID4123227 在78秒 checkpoint 有意停止，GPU0 PID304004 接续并已完整结束。旧数值记录副本 PID211021 也在验证后由276790接续；不再轮询旧 PID。
- 新 wrapper `scripts/resume_topic4_m_modes_cuda.py` 让**原始 M40 的 e0_t0_s9108401**从10秒完整checkpoint续算GPU0，PID337611。当前真实11秒进展已重新得到first onset10.59/confirm10.79，未更改60秒自然高态观察、180秒恢复后窗口或最大430秒。
- 为避免原controller误判失败，先暂停仅controller134670，再停止单个目标4084842，归档完整10秒checkpoint，启动同任务GPU worker，再启动同lookupcontroller PID337612。其余14个worker的PID和启动时间逐个未变，controller已正确接管。完整记录 `cuda_ordered_scatter_qa/M_grid_adoption.json`，归档 `M_grid_e0_t0_s9108401_resume_parent.pkl`。

## 运行状态与后续

最新库存 06:09：Z120/147完成、27运行；M0/40完整、16运行；Z+M reset 长程至约356秒；全快状态90秒pilot绝对时间147秒（终点166.5）；补Z dense 24秒仍RECOVERED，原两CPU branch继续。无新失败，机器约120GiB可用。

`scientific_review_midwindow.md` 已整合科学判断；`snapshot_topic4_fig5_exploration.py` 新增读取完整高M对照及图的E2增强计数，避免仅引用旧77秒匹配前缀。F代码实际已经支持**已保存原始spikes确定首次进入终点**，无需等完整返回随访；未完成首次180秒观察仍pending，两种子均确定才填格。不要把F灰色误解为所有统计必须等430秒。

下一步：等 e0_t0 原始GPU在20秒写出完整块，可核对其10–12.5秒与早补Z dense在干预前的全观测位级一致（不必再运行仿真）。继续看dense是否产生⑤；更新Z扫描完整三种子单元；全快状态90秒pilot到166.5须分析是否进入，阴性不得触发原计划的细分状态试验。剩余预算约3.5小时，勿追加盲目参数网格。Goal 不应此时完成或阻塞。
