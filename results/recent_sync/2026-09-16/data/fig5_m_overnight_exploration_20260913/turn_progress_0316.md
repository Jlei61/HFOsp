# 约03:16继续记录

本轮为实质进展：首次终点汇总修复、事件对齐早期Z恢复2条实际启动、完成配对20秒M等增益分析和图件目视。Goal仍到09:40，不标完成。

## 当前早期恢复分支已实际启动

监督28233、worker30171(seed9108401)/31978(seed9108402)，均已实查活跃CPU和进程身份。原先4186102监督已在没有任何支路运行时停止；新版本使用从原两条e0_t0缓存的精确10秒checkpoint(无high)，engine保存后重载摘要完全相同，前缀原始观测链接保留。

重要修订：不再等high后下个10秒存盘点恢复Z。worker包装原tracker_step，调用原事件逻辑但禁用其60秒恢复计划；未建立返回时改为首次high确认2秒后的下个10ms边界补Z1秒，再释放。若原生已返回不补Z。M与参数/所有快状态/随机历史不动。再次进入后10秒或释放/原生返回后60秒，first-entry horizon180，绝对上限243.5秒。母队列原60秒native观察仍在跑。该观察协议不同，原始物理worker文件未变，来源hash仍通过。

实际branch plan/current_design/trigger_qa均更新，初版plan存档。监督会自动完整Fig5；图未出现⑤就留空。支路不加入原M网格F的样本数。next_early_refill_design.md现在是实际执行版，不再是未执行旧方案。

## F可用端点与完整轨迹分开

scripts/plot_topic4_m_parameter_modes.py新增first_entry_from_counts与committed_first_endpoint：只读原40条已闭合1ms E/I及分区计数，检查连续块和计数守恒，按原10ms/200Hz/200ms定义确认首次进入，或真的满180秒才删失。后续恢复仍跑不阻塞已发生的首次终点进入F；不足180且无entry继续PENDING。grid.rows仍仅完整result，新增endpoint_rows和established_first_endpoints，避免误把部分轨迹当完整图条件。

first_endpoint_qa.json PASS覆盖19/20高箱、跨块高段、179/180秒删失、不计horizon后事件；测试数据不是科研数据。当前首次已确定仍0、完整轨迹0，F灰色无虚构着色。已运行--update、目视F轴和灰格正确。旧协议/种子/物理源不变，execution_plan.md已追加观测解释。

## 已完成配对等增益20秒真实前缀分析

scripts/analyze_topic4_M_matched_gain_prefix.py，输出matched_gain_prefix/analysis.json/scientific_review.md/figures/equal_gain_different_memory.png/pdf，已目视。

四组K=eta*tau=0.04：(.04,1),(.02,2),(.01,4),(.005,8)，两种子，固定0–20秒闭合实际数据，无额外仿真。保存的全局OU和E/I输入率均值跨条件逐点相同。较慢M初期积累晚，累积放电更多且Z更早下降：tau8组后10秒均Z为0.8273/0.8359，对应tau1为0.8414/0.8448。但只观察前20秒，不推断最终进入/返回，时间箱不当实验重复。

## 仍需等的实际任务

high-state eta0.2 probe仍活跃(4123227)，开始时进度标BUILDING、尚无第一完整checkpoint；不是terminal，不可按等待超时重启。eta2已完整返回，未改参数对照末1秒476Hz；等eta0.2完成后重新运行scripts/analyze_topic4_high_state_M_gain_probe.py并目视三列。py-spy只读采样因权限不可用，未尝试提权或修改运行任务。

原Z控制4186101正常、M控制4084835正常；source_repair原配置错误已完全恢复，本轮没再触碰物理源。fastpilot4073743仍跑，后续按实际结果区分M/快状态/噪声，而不是把短窗未进入当稳定。

09:40最终快照仍需额外读取high_state_M_gain_probe和early_z_refill_branches两处status，因为旧goal监督4073736只自动汇总旧三批+fastpilot。所有新job有限；到时不加新探索，保存实际在跑/完成及科学未通过项。
