# 当前 Paper Ready Figure 3

版本：`visual_alignment_compact_rows_20261010`；状态：`AUTHOR_ACCEPTED_FINAL`，作者于2026-10-10确认。正式输出为[完整图及A–E单panel](figures/README.md)，机器入口为[current_revision.json](current_revision.json)，说明为[docs/current_figure3.md](../../../docs/current_figure3.md)。

A为E10/SZ3 broadband，B为Y1/SZ6固定0–10 s、30–80 Hz gamma；两行均为波形/TFR、发作能量场、间期TA场三列。C/D/E保留原统计，完整可见列宽包含色条，行距4.064 mm。单panel不含字母，完整图含A–E。

校验：`python scripts/paper_figures/build_fig3_current.py`。重建到空目录：`python scripts/paper_figures/build_fig3_current.py --output-dir /tmp/fig3-rebuild`。旧入口`build_main_figure_3.py`转到同一安全入口，不再生成旧A–F。重建由冻结原生图层完成，科学重算仍使用工作机原始数据。

旧正式图及旧入口归档在`archive/2026-10-10_pre_current_fig3_ae/`；`candidates/`和`revisions/`只作历史来源，不能据其日期或目录名覆盖当前指针。
