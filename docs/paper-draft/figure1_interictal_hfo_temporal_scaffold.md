# Figure 1：间期 HFO 群体事件的患者特异性时序组织

> 状态：temporal-scaffold 主图合同 v2（2026-07-12，拆成独立 panel）  
> 定稿候选：`results/paper-ready-figure/fig1_interictal_hfo_temporal_scaffold/figures/fig1-panel{a1,a2,b1,b2,c,d1,d2}.{png,pdf}`
> 单一复现入口：`scripts/paper_figures/plot_fig1_interictal_hfo_temporal_scaffold.py`

## 1. 当前图能与不能支持什么

| 结论 | 当前是否直接展示 | 裁决 |
|---|---|---|
| 跨通道 HFO 群体事件 | 是 | raw traces、spectrogram 和质心轨迹支持 |
| HFO-rich channel set 与 clinical SOZ 相关 | 是 | supporting clinical anchor，不等于传播被限制在 SOZ 内 |
| 患者内时序组织高于 null | 是 | masked（shared-participant）MI data-vs-null |
| 分模板后刻板性提高 | 是 | cluster-aware uplift |
| 模板跨记录时间复现 | 否（已移出 Fig 1）| 属 cohort 结果，写入正文 / Table S3；split-half 面板不进主图 |
| 同一患者内可分时序模板 | 例患层是 | Panel c1/c2 均用 E7：上排为时间原序，下排将同一批全量有效事件按 TA/TB 重排；本 panel 不单独声称 opposing-template pair |
| 同一三维空间轴的相反读取 | 否 | contact map、held-out axis 和 paired-axis cosine 进入下一张 spatial 主图 |

## 2. Panel 对照与真实 producer

Panel a/b/c/d 分别落成 `fig1-panel<id>.{png,pdf}`；c1/c2 是 `fig1-panelc` 内上下对齐的两个 subpanel，不再输出独立文件。单一入口 `scripts/paper_figures/plot_fig1_interictal_hfo_temporal_scaffold.py`。

| 文件 | 图中内容 | 直接画图代码 | 关键输入 artifact | 当前裁决 |
|---|---|---|---|---|
| `fig1-panela1` | 178 段人工标注 HFO 的叠加波形、黄色均值、raw/normalized 平均谱 | `scripts/paper_figures/plot_fig1_single_hfo_schematic.py` | legacy `zhangkexuan_pickSigs.npz` + `zhangkexuan_annot_v4.pik`（label=1 恰为 n=178）| 严格回到参考图的数据与谱计算合同；三行 x 轴铺满 0–0.6 s，无白边 |
| `fig1-panela2` | 80–250 Hz 群体事件波形、normalized spectrogram、主高频增强区质心轨迹 | `scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py` | Y3 `FC10477Q.edf` + `_gpu.npz` + `_packedTimes.npy` | A1/A2 统一为 magnitude + Gaussian σ=1.5；A2 恢复原 50 ms Hamming 时间分辨率与 per-event max scaling，不再显示 magnitude³；红点取同图主峰 ≥70% 连通区质心 |
| `fig1-panelb1` | Yuquan refined HFO count 对 clinical SOZ 的 ROC | 复用 `scripts/plot_refine_soz_validation.py` 的 loader/ROC 定义 | `results/hfo_detection/<subject>/_refineGpu.npz` + Yuquan SOZ JSON + `config/subject_params.json` | 现场重算 n=20、mean AUC=0.873；阴影为插值后 subject ROC 的 SEM |
| `fig1-panelb2` | Epilepsiae 同上 ROC | 同上 | 同上 + Epilepsiae SOZ JSON | 现场重算 n=15、mean AUC=0.955（20 例中 1125/384/620/818/916 无临床 SOZ 标注，无法进 ROC）|
| `fig1-panelc` | 同一例患 E7：上排 c1 为时间顺序 heatmap + rank distribution + day/night strip；下排 c2 为 TA/TB 聚类顺序 + mean rank | `scripts/plot_interictal_propagation.py --masked-features --pr3` 的原始 heatmap/day-night/rank helpers | 同一份 masked JSON + raw `*_lagPat*.npz` | c2 绘制全量 6,556 个有效事件，TA/TB 两个 n 之和严格等于 c1 的 n；TA/TB 间为白底灰色斜线断带并截断 x 轴；右下质心 marker 缩小；上下三列对齐、间距收紧、c2 无指标标题 |
| `fig1-paneld1` | MI data vs permutation null | 直接复用 `scripts/plot_interictal_propagation.py` 的 `violin_with_scatter` + `add_significance_bracket` | 40 个 masked per-subject JSON | `legacy_mi.masked=true` 硬检查；violin + box/IQR + whiskers + subject points + 显著性括号；画布不写 cohort 计数；y 从 0 起；数据集标签不加粗并贴近 x 轴 |
| `fig1-paneld2` | overall vs within-template MI | 同上 | 同上 | 底层为 masked rank-concordance `overall_tau` / `within_cluster_tau_mean`；图面统一用 MI 简写；画布只保留 median ΔMI，不写 cohort 计数；两轴从 0 起；恢复右下灰区和数据集图例 |

## 3. 定稿布局

独立 panel 输出，最终拼版在外部完成，不出 composite：

| 文件 | 叙事 |
|---|---|
| `fig1-panela1` | 单通道 HFO 形态示意 |
| `fig1-panela2` | 群体事件现象 |
| `fig1-panelb1` / `fig1-panelb2` | 两队列 SOZ supporting clinical anchor |
| `fig1-panelc` | 同一位 E7 的时间原序与全量 TA/TB 聚类重排 |
| `fig1-paneld1` / `fig1-paneld2` | 40 例 MI/null + within-template uplift |

图题限定为：

> Interictal HFO population events exhibit recurrent patient-specific temporal organization.

## 4. 剩余锁图项

- [x] paneld1（MI）已用 masked（shared-participant）统计重画：40/40 significant，masked median 0.228；
- [ ] 作者目视确认最终版式后锁图，不再更换 exemplar 或统计版本。
