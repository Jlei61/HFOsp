### yuquan_y1_hfo_group_event_demo.png

这张图作为主文 `Fig1-A`，按老论文 D panel 的 `split_contiHigh + normedSpecs_cat + spec center` 逻辑重做 Yuquan Y1 的间期 HFO 群体事件示例。当前版本固定展示 3 个较干净的群体事件（packed event indices: 22, 237, 1458），左侧显示同一批事件窗口内的 80-250 Hz bandpassed SEEG stacked traces；右侧重新计算 concatenated event signal 的 normalized spectrogram，每个通道保留 50-300 Hz 频率厚度，并用红色质心点/线标出每个群体事件内部的先后关系。

旧代码来源主要是 `ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/for523_p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py` 和 `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`。通道从 E/K 候选通道中筛选，并排除了当前视觉不够干净的 E9/K10 display rows；右侧不重复显示 y-label，通道名只保留在左侧 traces。当前 layout 为窄版，两个 x 轴都从 0 开始，右侧 spectrogram 图像边界固定到完整 concatenated event duration，避免前后白边。完整选择信息见 `yuquan_y1_hfo_group_event_demo_metadata.json`。

正式复现入口是 `scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py`。同目录里的 `candidate_*` 图只用于人工筛选，不是当前验收版本。

**关注点**：原始高频波形中可以看到跨通道同步出现的 HFO 群体事件；右侧红色质心轨迹显示同一群体事件内不同通道的早晚关系，是后续传播 rank/template 主图的直观入口。
