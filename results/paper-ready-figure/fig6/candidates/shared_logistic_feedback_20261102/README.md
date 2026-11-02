# Figure 6：共享 logistic 连续仿真候选

> **作者展示决定（2026-11-02）**：本候选改变了原有展示语义，不用于当前周报。周报已恢复此前 Figure 6 及图注；本目录保留共享 logistic 的仿真和绘图结果。

全局抑制 G 与适应增量 M 共用参数化 logistic，替换此前两处 clip。采用同一网络、同一噪声种子 9108404 的 120 s 原生轨迹；R0=350 Hz、k=40 Hz，其余参数保持。

[完整 PNG](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [详细图注](caption_zh.md) · [显示配置](display_configuration.json)

轨迹在 22.04 s 进入持续高活动，27.09 s 退出，34.40 s 两处 core 的 Z 恢复参考值；之后出现 48 次短事件，覆盖 14.54 s，并于 71.81 s 再次进入。所有宽度对照、判据与参数见[原生验证](../../../../topic4_sef_hfo/logistic_feedback_20261102/README.md)。

A–D 使用同一条新仿真，重新计算波形、完整 0–120 s 状态轨迹、空间活动和患者最佳场匹配。D 采用原有方法，对每个模型进入窗口从 E10 的 25 次合格发作中选择最高有符号场相关者，本次分别为 SZ6 和 SZ17（r=0.824、0.750）；该示例属于描述性最佳匹配。E 仍为原历史模型的 13 患者、140 次发作队列结果，没有换成 logistic 模型重算。

A 的 13 条显示波形来自全部 15 个分析触点的固定顺序，80–1000 Hz 全信号滤波，两段保持共同增益及时间尺度。C 的 20 ms 原生放电率与 A 的八个时刻严格对应，每行按连续进入窗口的时空最高值归一化。B 使用 5 ms 全体 E 计数率及仅用于显示的 5 ms 高斯平滑，不作为分岔或预测证据。

本次 PNG 由同一 PDF 渲染；所有面板文字位于画布内，A/C 原生计数守恒，E PDF 与历史文件逐字节相同。当前状态为 **PENDING_AUTHOR_VISUAL_REVIEW**，正式 `fig6/current_version.json` 未修改。

重绘入口为 `scripts/paper_figures/build_fig6_logistic_feedback.py`，配置见本目录 `display_configuration.json`。原生数据保存在 `/data/hfosp/topic4_sef_hfo/logistic_feedback_20261102/runs/logistic_r350_k40_s9108404`，临床比较继续依赖此前同一组冻结输入。实际导入的项目源码冻结在 `source/python/`，路径及哈希见 `source/render_dependencies.json`；源码快照用于保留依赖，不包含原始网络、临床数据或 CUDA 环境。
