# Figure 2B：留出三维方向最近模板误差

2026-10-23按作者确认，方向任务的分配与评分均改为三维方向：每个测试事件取到TA/TB两条冻结训练方向的最小有向夹角，单位为度。不再使用rank MAE或旧Matching Index选择模板。每折对事件取均值，再对互补两折等权；统计单位为患者。

沿用两种方法原有训练结果：仅rank聚类与rank+空间方向聚类，每种各两个模板；本次未重训、未依据测试方向转动模板。两折共26人，1159183个方向可估计事件，完整复用六个训练/测试划分，全部测试序列在旧缓存中均有覆盖。参考事件方向仍由完整rank与触点坐标拟合，目标是方向覆盖/重建，不是早期预测或独立测量的传播真值。

## 新null

固定训练模板及患者电极坐标，在每个模板的有限rank与已定位触点支持内置换rank值，重拟合3D梯度轴；相同随机键跨四个模板共享，相同支持时使用完全相同的通道置换。保持模板rank值集合、参与支持、真实测试方向和电极几何；破坏rank与空间位置的对应。每次置换后重新按最小三维夹角选择模板，200次置换只用于两个主分析折。该null是模板空间对应对照，不是重新聚类的空间信息消融，也不保持两个模板的夹角。原事件—模板配对置换对于最小夹角任务不成立，未沿用。

## 结果

患者中位误差：Rank SEQ 51.2128°，Spatial TRAJ. 45.0385°；24/26人空间模板误差较低。双侧患者配对Wilcoxon原始P=2.270937e-05，两张图六比较Holm P=6.8128109e-05。

Rank版的MAE和rank-channel shuffle结果原样复用；主图六项比较重新一起Holm校正。五种误差×七种读出完整保存在metric_holdout_comparisons.csv，35项统一Holm；不同读出不作为独立重复。旧按rank选模板后算方向的结果保留在`../rank_direction_holdout_20261022/`。

## 图与复现

完整 Figure 2 和 A–E 单图位于 `../../paper-ready-figure/fig2/candidates/projection_typography_20261023/figures/`；独立绘图复现步骤和数据范围见该候选根目录的 `README.md`。B 是三维最近方向误差统计；C 保留第一折真实方向的二维投影、圆内图例与两条空间模板实线。

从原始数据重新执行实验使用 `python scripts/run_fig2b_nearest_direction_holdout.py --workers 2 --n-null 200`，需要运行清单所列完整上游数据与缓存。本提交保存逐患者 JSON 和统计汇总，完整事件级 NPZ 保留在分析机器。

## 留出敏感性与验证

最近三维角误差在35项统一Holm校正后，合并两折、两个单折、random_0、random_2显著；时间前后留出（校正P=0.5431）和random_1（校正P=0.1905）未显著。最近方向cosine loss同样为5/7通过，不能称所有划分均稳定。

测试：`python -m pytest tests/test_interictal_nearest_direction_holdout.py tests/test_interictal_two_target_holdout.py tests/test_interictal_rank_holdout.py -q`，15项通过。既有事件级数值复核记录见本目录 `source_snapshot/nearest_direction_validation.json`；当前排版检查见 `../../paper-ready-figure/fig2/candidates/projection_typography_20261023/validation.json`。

当前完整Figure 2已按作者要求拆分为A–E：B为方向误差统计，C为两个rose，原C/D顺延D/E。见`../../paper-ready-figure/fig2/candidates/projection_typography_20261023/figures/fig2-complete-layout.pdf`；本次数值不变。
