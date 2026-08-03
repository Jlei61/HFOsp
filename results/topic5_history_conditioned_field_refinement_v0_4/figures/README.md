# 图件说明

### history_conditioned_field_refinement_six_panel.png

六联图依次展示集合值任务、四个嵌套模型、M3 相对静态 M0 的患者级变化、M3 相对冻结状态 M1 与非递归 M2 的增量、真实历史相对完整顺序打乱及同患者 history-swap 的变化，以及 M0/M3 超出 matched channel null 的绝对信息。主 endpoint 固定为 clinical onset 后 0–10 s、1–45 Hz contact-energy field；1–150 Hz 仅为 no-retrain sensitivity。静态 A/B 未读取发作早期 target，但来自全记录间期事件，因此整体分析是回顾性的，不是完全前瞻预测器。

**关注点**：先看 C 的 M3−M0，再用 D/E 区分增量来自 recurrent dynamics、简单历史汇总、真实顺序还是发作匹配历史；F 检查模型是否只是在弱静态锚点附近做相对改善。

### representative_history_refinement.png

按预先固定的中位效应规则选择患者和发作，分别展示 A、B 两个候选场在静态 M0 与 history-refined M3 下的 contact rank，并与真实 early-ictal target 对照。由于正式 endpoint 使用绝对 Spearman，曲线仅为显示而逐候选做了符号对齐；它不表示模型预测了唯一传播方向。

**关注点**：观察残差修正是局部调整静态 A/B，还是把候选场整体改写；该病例只作模型行为展示，不替代 15 人统计。
