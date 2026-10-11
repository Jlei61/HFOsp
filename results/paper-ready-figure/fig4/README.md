# Figure 4 正式定稿

2026-10-11作者确认的Paper Ready Figure 4，状态为AUTHOR_ACCEPTED_FINAL，版本为core_search_final_20261011。三行A–I，画布300×232 mm；B/C/D/E/H/I数据轴为50×50 mm。A采用v15：三个淡紫色问号区域、两个与E图密度峰对齐的core、上方两条OPTM箭头，下方不加箭头或文字；浅绿底纹仅在中央取样框，右侧保留局部高斯采样权重。

[完整预览](figures/fig4-complete-layout-preview.png) · [PDF](figures/fig4-complete-layout.pdf) · [A–I单图](figures/README.md) · [历史版本登记](historical_versions.json)

正式入口：`python scripts/paper_figures/build_fig4_current.py`。添加`--output-dir /tmp/fig4-export`可从随包冻结导出恢复31个PNG/PDF/SVG文件，逐字节一致；这是图文件恢复，不是参数重绘或重跑SNN。原绘图代码、数值输入及原生导出保存在source，来源解释见source/README.md。

旧正式v10、58个候选包、旧动画及重复图均已迁入`../archive/2026-10-11_fig4_finalization/`；此前更早归档仍原地保留。后续修改先输出独立修订，不能用旧工作树或旧producer覆盖正式图。
