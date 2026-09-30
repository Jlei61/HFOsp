# Figure 5 自主回返候选

用户2026-09-24要求用已见到的自主进入—退出—恢复—间期返回更新A，并推进Z–K状态与分岔分析。这套候选使用seed9108405首个完整循环0–60秒，固定网络与输入，自主动力学已在既有240秒轨迹中完成；本次重画没有新增或干预该轨迹。四次完整返回是同一条轨迹内的事件，不能当四个独立种子。

- A：完整原生raster及原间期、退出、首次返回三个300毫秒放大窗。
- B：与A同步的Z、K、实际施加G及有效M。
- C：同一轨迹原生50毫秒空间帧，保留物理core与原区域率观察半径的区别。
- 状态图：保留原Z–局部有效抑制–E率坐标，再给Z–K–E率投影。局部抑制从明确记录的Z×localII按细胞数加权，未混入新增全局G。

约16.70秒是已有双核/全局退出检测窗口的回溯起点，不等于所有细胞此刻都已经安静；16.83秒开始共同低活动，16.85秒空间帧显示退出后余波。第一次返回事件49.31秒，下一次高态进入59.74秒，右边缘保留这一重新进入。

所有图是候选，Agent已检查PNG与同版PDF，人工目视待验收。旧正式E（ηM×τM旧模型扫描）与F（旧进入轨迹患者比较）未随新模型重算，不能拼接后宣称全图已统一为新增反馈模型。状态投影的回返不证明极限环或双稳态。

复现：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/paper_figures/build_fig5_autonomous_loop.py`。仿真与新分析合同位于`/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/`。

条件响应的配套空间图为`figures/fig5-zk-spatial-fields.png`（另有PDF/SVG）：每条件保留原生400格点末10秒平均率，以及全E和双核均率。它用于检查全网均值是否掩盖不同活动区域，不能代替传播逐帧图；复现脚本为`scripts/paper_figures/build_fig5_conditional_spatial_fields.py`，18/18最终主矩阵图已自查；科学审阅见仿真根目录primary_scientific_review.md。

本轮全部图、科学审阅、结构对照和复现入口见[交付索引](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/delivery_index.md)。本轮18条主条件、16条结构条件与2条自主120秒对照均已完成，最终科学结论及技术核对见交付索引；作者目视验收仍待完成。
