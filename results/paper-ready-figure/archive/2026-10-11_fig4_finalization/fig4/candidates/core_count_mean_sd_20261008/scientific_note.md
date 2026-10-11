# E1146 core 数量实验：均值与标准差预览

2026-10-08 已核对实时进程、status.json 与 stage_completion.json。3／4 core 的两轮阶段于2026-10-04完成248／248次评估，当前无该批运行进程；1／2 core 保留用户认可的阶段，原96点计划未全部执行，独立噪声复测暂缓。
本次重画使用全部现存评分；逐文件核对与2026-10-04完成版输入一致，没有新增仿真结果。其他三位患者仍暂停，随附图只更新显示方法。

每个 epoch 是一次参数配置评估；每条重复先取截至该点的最佳训练 loss，再对同一患者的优化重复取算术均值及样本标准差（ddof=1）。E1146 每组4次重复；其他患者每组2次重复。阴影不是标准误或置信区间。
四组与三组图比较共同前31次评估；1／2 core 独立图保留共同前39次。原始完整轨迹保存在 source 中；不加入组别专属归一化，不截断标准差带。3／4 core 每次重复含15个保留初始配置及两轮各8个反馈优化配置。

| Core数 | 第31个epoch的最佳loss均值±SD |
|---|---:|
| 1 | 3.3102 ± 0.6120 |
| 2 | 1.7035 ± 0.2056 |
| 3 | 1.2649 ± 0.1829 |
| 4 | 1.4365 ± 0.1585 |

2 core 的平均训练loss比1 core低48.5%；3 core 当前最低，4 core未进一步降低。该比较反映当前预算内的训练拟合，不等于独立噪声确认或全局最优。

当前预览已完成逐点原始评分复算、曲线与阴影数值核对，以及六组PNG/PDF目视检查；保留 Epochs／Loss 标签，无标题或图注。正式Figure 4文件未改动，候选待作者检查。

复现：
```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/paper_figures/plot_fig4_core_count_preview.py --snapshot results/paper-ready-figure/fig4/candidates/core_count_mean_sd_20261008/source/core_count_input_snapshot.json --out <新候选目录>
```
