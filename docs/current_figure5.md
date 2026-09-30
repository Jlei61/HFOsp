# 当前 Figure 5

当前整合入口为 [`results/paper-ready-figure/fig5`](../results/paper-ready-figure/fig5/)，身份 `single_seed_zm_entry_and_onset_field`，A–F，producer 为 [`build_fig5_single_seed_panels.py`](../scripts/paper_figures/build_fig5_single_seed_panels.py)。用户2026-09-30将既有Figure 5补实验纳入验收整合范围；本次不重新设计图、不改变既有数值结论。

E使用固定seed9108401的70个唯一参数点。19条长随访由8条续跑和11个新参数组成；顶端颜色截断于1000秒。70点中33点在1000秒内进入，1点在1535.03秒进入，12点观察至3000秒未进入，24点仅有1000秒未进入证据。因此不能将整张图解释为统一3000秒随访、进入概率或正式分岔图。

- [完整图PDF](../results/paper-ready-figure/fig5/figures/fig5-complete-layout.pdf)
- [详细图注与来源](../results/paper-ready-figure/fig5/README.md)
- [70点表](../results/paper-ready-figure/fig5/entry_points.csv)与[原始计数审计](../results/paper-ready-figure/fig5/long_followup_audit.json)
- [模型版本区别](topic4_model_versions.md)：空间间期rate、六群体rate、空间Z/M及原生Z/G/K闭环分别读取。

历史元数据中的人工待审标签原样保留，用于记录产物生成时的状态；此次用户授权的既有结果整合不修改旧时间点的记录，也不把候选动力学升级为已经建立的正式分岔。
