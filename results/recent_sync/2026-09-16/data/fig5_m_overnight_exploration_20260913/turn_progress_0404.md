# 03:57阶段记录

注：文件名保留原引用；实际系统时间核对为03:57，原题头时间为手工记录误差。

8h goal仍active，截止09:40。当前不是blocked，不得宣称整轮或完整Fig.5科学验收完成。

## 新科学结果
- 实际75.5s高态代数M电流图已生成并目视：η=.02/.2/2时净驱动超过阈值的E细胞87.20%/87.02%/2.125%。仅瞬时诊断，完整续跑仍判定返回。
- η=.2前500ms仍高率391Hz；η=2已完成5s并外部降至低态。前者仍运行，不能把未完成结果当5s阴性。
- 新reset_spatial_balance图已生成并目视。固定10–70s vs970–1000s：meanZ .84315/.83819，E13.463/13.797Hz，M电流.5386/.5536；Z空间RMS差.00753、rate空间RMS差.947Hz。首次进入前也已长期处于大致Z平衡，平台本身不是不再进入的解释。单轨迹描述性，无永久保护或等状态主张。

## 数值加速
scatter_lookup_qa/full_network_qa.json PASS：真实高态200ms全部引擎状态与观测逐位一致。profiler：499.5s仿真中scatter464.9s；另setup145s。合成微基准中lookup1.44x，非完整仿真速度保证。
新src/topic4_serial_spike_scatter_lookup.py保持edge/浮点加法顺序，整数取模提前。旧9个物理源码hash未变。
仅后续未启动M40通过新wrapper run_topic4_m_modes_scatter_lookup.py调用原worker，并每run记录computational_variant.json。
Mcontroller134670 (supervise_topic4_m_modes_lookup.py)替换4084835，仅换controller；12个既存worker PID/starttime完全保留。证据scatter_lookup_qa/adoption.json。新controller仍原40条件、同种子、同科学endpoint、同资源上限、同结果绘图；discover能识别带原runner参数的新wrapper。
正式在跑的M、Z、reset、两early分支均未切换核或重启。

## 新图自动接续
plot_topic4_ongoing_fig5_prefixes.py --watch PID81847。读取exact已关闭checkpoint与prefix，不伪造result；load(end_step)前10s一致性QA已过。监视阶段变化自动全布局，但目前0prefix图，因为最新科学高态仍在未落盘10s块内。完整长随访继续；prefix不添加F样本。

## 当前进度
Z147 112complete/32running/3pending，M40 0complete/12running/28pending，均无失败。M四个已运行条件progress显示首次高态，但完整counts/checkpoint未落盘不能提前声称整条完成。
Earlyseed1首onset10.59/confirm10.79，seed2 13.74/13.94，分别与原亲本一致；各已到11/14s，2s观察后自动1sZ补充。PID30171/31978，supervisor28233。仍未完成release/复发。
Z-only1000s完成且923.5spostrelease未再入；Mclear320s未再入；fullfastpilot112s绝对时间未再入。

## 后续要看
- high eta.2完成后重跑analyze_topic4_high_state_M_gain_probe.py。
- early完成/落盘前缀后由watcher生成整图，Agent必须目视并检查状态⑤是否真实存在。
- 第一个lookup新Mworker启动时检查computational_variant、guard、source与PID，不能误把wrapper视作不同生物条件。
- 不再因为计时慢重启或缩短原科学观察窗。60s高态原M协议会远长于8h，保存运行并如实交代。
- goal原supervisor4073736只记录 inherited+fastpilot；09:40收尾需另外检查window.json列出的highgain、early、prefixwatch和lookupcontroller。
