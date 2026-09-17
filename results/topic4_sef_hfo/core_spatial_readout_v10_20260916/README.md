# Core 分岔与空间读出：2026-09-16 修订候选

本版完成分岔读出的修正、四条降阶轨道的网络率图，以及相同核内连接参数的原生 SNN 空间对照和三项传播指标。**四条降阶分支与四种原生空间状态的对应尚未建立**；当前原生冷启动结果不能填成 b–d 分支的空间结果。数值及来源核验见 `validation.json`；图片等待用户目视检查。

- [修订分岔主图](figures/00_corrected_bifurcation_network.png) · [可编辑 SVG](figures/00_corrected_bifurcation_network.svg)
- [同参数 SNN：波形、固定触点与三项指标](figures/03_native_network_contacts_metrics.png)
- [包含两种传播标签的相邻事件快照示例](figures/04_native_a_snapshots_03.png)
- [a 对应参数的完整二维动画](figures/08_native_a_4to6s.gif) · [b 对应参数](figures/08_native_b_4to6s.gif) · [c/d 共用参数](figures/08_native_cd_4to6s.gif)
- [全部图片说明](figures/README.md) · [科学结果与边界](scientific_report.md)
- [三项数值 CSV](native_summary.csv) · [四条分支的对应状态表](state_correspondence.csv) · [全部检测事件及排除原因](event_inventory.csv)

复现入口依次为 `scripts/topic4_core_spatial_readout_v10/{native.py --batch,reduced.py,analyze_native.py,figures.py,spatial_figures.py,validate.py}`。使用 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`，各脚本限定 BLAS 单线程。`native.py --batch` 只运行本次三个参数的 12 秒固定拓扑/噪声对照；已完成结果不会再次仿真。旧 v2–v9 文件及其他工作树未改写。
