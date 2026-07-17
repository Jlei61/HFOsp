### legacy_hfo_n178_schematic.png

这张图是主文 `Fig1-a1` 的单 HFO 形态入口，直接读取 legacy 人工标注资产 `zhangkexuan_pickSigs.npz` 与 `zhangkexuan_annot_v4.pik`；其中 label 1 恰好包含 178 段 HFO，与参考图的 `HFO n = 178` 完全一致。黑线、黄色均值、raw Spec 和 normalized Spec 均按 `p16_mechan_events_specComp.py` 的原计算重现，不再用 Y3 检测片段近似替代。

三行的 x 轴固定为完整的 0–0.6 s；频谱首末 cell 只在绘图坐标上延伸到片段边界，因此消除了 x 轴两端白边，但没有修改任何谱值。完整合同见 `legacy_hfo_n178_schematic_metadata.json`。

**关注点**：检查标题是否为红色 `HFO n = 178`、黑色叠加波形与黄色均值是否清楚、raw/normalized 两张谱是否从 x=0 连续铺到片段末端。

### yuquan_y3_hfo_group_event_demo.png

这张图作为主文 `Fig1-a2`，按老论文 D panel 的 `split_contiHigh + normedSpecs_cat + spec center` 逻辑重做 Yuquan Y3 的间期 HFO 群体事件示例。当前版本固定展示 3 个较干净的群体事件（packed event indices: 22, 237, 1458），左侧显示同一批事件窗口内的 80–250 Hz bandpassed SEEG stacked traces；右侧恢复原 50–300 Hz 显示框架与 per-channel/per-event max scaling，并用红色质心点/线标出每个群体事件内部的先后关系。

旧代码来源主要是 `ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/for523_p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py` 和 `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`。A1/A2 现在共享同一个谱量函数：magnitude 后做 Gaussian smoothing（σ=1.5）；A2 保留其原 50 ms Hamming 窗与 40 ms overlap，不再把 magnitude³ 当作显示谱。红点取同一 magnitude 图上主峰所属、≥峰值 70% 连通增强区的加权质心，并排除事件两端各一个 50 ms STFT 窗以避免拼接边界伪峰；真实 STFT cell 坐标和无白边修复仍保留。完整选择信息见 `yuquan_y3_hfo_group_event_demo_metadata.json`。

正式复现入口是 `scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py`。图内匿名标签已按 private crosswalk 锁定为 `Yuquan Y3`。旧 `yuquan_y1_*` 文件只作历史 artifact 保留，同目录里的 `candidate_*` 图只用于人工筛选，不是当前验收版本。

**关注点**：原始高频波形中可以看到跨通道同步出现的 HFO 群体事件；右侧红色质心轨迹显示同一群体事件内不同通道的早晚关系，是后续传播 rank/template 主图的直观入口。
