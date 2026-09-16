# 04:21阶段记录

上一轮progress，本轮progress：完成真实Z-only1000秒整Fig.5及转变放大图；goal仍active，09:40前继续按已有边界推进。

## 新交付
completed_Z_only_1000s_fig5/figures/fig5.png/.pdf和fig5_transition_zoom.png/.pdf均已生成、目视。source_qualification.json/figure_qa.json保留来源和验收。
- 前238秒来自旧240s原始包；后续读取Z-only已完成chunks238–1000，238–240不重复。A原接触readout2kHz、B80cells0.1ms、C Z/M5ms连续到1000秒。
- 前238s原生空间场1ms，后段5ms。没有把5ms计数伪造为1ms；全文count conservation已过。当前4个D快照及全部E2均在1ms前缀内，E2逐数组与旧240s结果exact/allclose1e-12一致，Fig3C图片SHA相同。
- 按本轮Fig.5明确观测规则重新从counts计算return：75.85–77.85s两秒低活动窗口，77.85s确认external Z return。原干预仍75.5–76.5，不改变任何仿真或参数；旧layout硬写78.1仅早期展示，不能混淆为新干预差异。
- finite events before268/after3415只为单条轨迹的描述，非独立样本；state⑤实际缺失，full_1_to_5 false。F仍当前M40灰色待定，不加本对照为重复。
- E2全局进入73.48–74.48s的15contacts均未能量增强，rho.814是相对空间次序，不能当成功。另有early_energy_timing已有第一局部72.26和持续局部72.65的窗对照，不重复算或事后改主终点。

## 代码
plot_topic4_m_parameter_modes.py仅为混合原生空间采样增加严格全局计数验证/元数据；50ms D直接积分实际1ms或5ms完整bin，E2若没有实际1ms目标场则明确不可估计1–150Hz。所有物理hash/worker不变。
plot_topic4_completed_Z_only_1000s.py是本次producer，observation-only tracker去除模拟停止字段；源任务1000s已完成。
只读prefixwatcher现PID178478，加入plotter源码变更检测/reload/按版本重新渲染，后续无需每次改图就重启。window.json已更新。替换的是绘图watcher，不是仿真。

## 下一步
等待early_Z_refill两条continuous siblings进入restore/release/再进入并落盘，Agent看真实新prefix和完整结果。观察fullfast90s实际结束及η.2五秒probe；不要因持续计算而重启。Mclear1000尚远未完成；已有Z-only1000阴性不是永久保护证明。
当前状态见progress_0421.json。09:40完成窗口报告时必须区分8h探索交付、已经完成的继承对照、未完成的原40/147固定批次，不能把本新布局当新的M组合样本。
