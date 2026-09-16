# 02:53 自主阶段记录

Goal继续到09:40，本轮有实质进展，不应标记完成。

## 新证据

同一75.5秒高态、Z/M及快状态与OU不清零，外部ηM从0.02变为2.0的5秒干预已完成。77.5秒满足低活动返回标准；末1秒E均率0.00106Hz，末Z约0.832。未改参数的同源对照末1秒476.23Hz，Z约0.206。说明原有M电流足够增强可以退出该高态，但这仍是外部参数步骤，不能证明恒参数自主终止。ηM=0.2还在执行。200ms原参数续跑的counts/raster/Z/M逐项QA已PASS。

新脚本scripts/analyze_topic4_high_state_M_gain_probe.py已生成并目视检查high_state_M_gain_probe/figures/same_high_state_M_gain.png/pdf：每列实际75–75.5秒共同前缀，灰点线是外部步骤、绿虚线是返回确认，raster与Z/M电流对应，后者右轴为log。第三条件完成后再运行脚本更新三列并再次目视。

## 早期Z恢复支路

scripts/run_topic4_early_z_refill_branches.py监督器4186102已部署，尚无分支仿真：等待e0_t0两个种子进入高态至少2秒后的第一个已保存完整checkpoint。共享前缀与全engine逐字节校验复制；只做1秒Z恢复，M与噪声/快状态不断，放开后最多60秒或第二次高态后10秒。源40条仍按60秒原生高态观察继续。分支后自动完整Fig5，不进入F增加样本。脚本对输出目的地做branch-dir约束；prepare之后9个物理源文件hash不变QA通过。约02:50两源分别10秒、8秒尚未进入。

## 本轮实现错误与修复（需保留透明记录）

新分支prepare()第一次执行时source-hash循环重用了path变量，把plan误写到stage_config.json（02:42）。并非其他任务改动。新分支自检立即拦住后续仿真；原Z监督器进入sticky DRAINING，已有32workers不受影响。已用git HEAD/index均匹配冻结协议470de457...dd9fae的2922字节原文件恢复，重命名plan_path/source_path并增加只能写自身输出目录的保护，复核M与Z所有源hash一致。

原Z控制器3191746只停止控制器，32worker的PID/starttime完全保留；新scripts/supervise_topic4_z_kinetics_resume.py控制器4186101接管恢复原147任务、原排序与QA，110完成/32运行/5待派发。M控制器4084835仍0/40完成、12运行，无失败。高M探针eta2已完成，eta0.2已通过恢复后源检查。详见source_repair/incident.json、controller_recovery.json，错误原内容留档。不要再次误称其他任务修改配置。

## 后续优先

1. eta0.2完成后重画同高态对照并审查。现有探针监督4123227执行，不能重复派发。
2. 早期恢复监督4186102出分支后检查engine/prefix audit和连续Z/M，完成后目视完整Fig5。新branch没有第⑤时必须留空。
3. M40目前完整轨迹0，F只读完整result，尚未使用已确定的首次进入端点。可在原始闭合counts验证后允许F纳入已确定首次端点，分开first-endpoint complete与full-trajectory complete，未到180且无entry不能填删失。此显示改进尚未实施。
4. fast-state pilot4073743到95秒已恢复无再入；原M-clear1000秒任务仍跑。不要把短窗口无entry当永久不再入。
5. Z扫描接近完成后读完整响应，M各模式完整Fig5继续按原验收。截止09:40停止新增探索，已有固定任务/有限支路记录实际状态继续，不伪造完成。

原goal监督4073736的latest/deadline_snapshot只含旧三批+fastpilot，尚不含后来high_M_probe与early_z_refill；最终审查需要另外读取这两个status。window.json已记录全部路径与正确PID。
