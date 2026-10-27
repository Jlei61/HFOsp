# Fig5 完整搜索后的 TA 偏好选点管线

等待原 1600 条训练及 100 条原提名点复测完成，再对固定 17 位 shared axis 患者在完整 32 次候选中执行：单 core 有符号 ρTA>ρTB 约束下最低 J，双 core 原最低 J。沿用真实 TA/TB、两项 Holm，不交换模板身份；输出独立候选 ta_preferred_search_complete_vertical_legend_20261027，不自动切换当前入口。

新提名单 core 的独立噪声复测未由本管线执行；cohort_comparison_complete=false，完成状态为 TA_CONSTRAINED_CANDIDATE_READY_REPLAY_PENDING。冻结模板中的旧图层仅用于 A–E。入口是主仓库 scripts/paper_figures/finalize_fig5_core_count.py --watch。

F 横轴以 TA–MTA、TB–MTB 居中标注两组点，右上方竖排显示空心点 1 core、实心点 2 cores 的图例，星号和括号下移以避免重叠，无额外轴标题；完整搜索后的显著性符号从届时两项 Holm 校正 P 重新计算，阈值为 0.05/0.01/0.001，精确 P 保留于统计文件。
