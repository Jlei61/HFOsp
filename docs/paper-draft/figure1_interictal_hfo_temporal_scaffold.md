# Figure 1：间期 HFO 群体事件的患者特异性时序组织

> 状态：temporal-scaffold 主图合同 v2（2026-07-12，拆成独立 panel）  
> 定稿候选：`results/paper-ready-figure/fig1_interictal_hfo_temporal_scaffold/figures/fig1-panel{a,b1,b2,c1,c2,d1,d2}.{png,pdf}`  
> 单一复现入口：`scripts/paper_figures/plot_fig1_interictal_hfo_temporal_scaffold.py`

## 1. 当前图能与不能支持什么

| 结论 | 当前是否直接展示 | 裁决 |
|---|---|---|
| 跨通道 HFO 群体事件 | 是 | raw traces、spectrogram 和质心轨迹支持 |
| HFO-rich channel set 与 clinical SOZ 相关 | 是 | supporting clinical anchor，不等于传播被限制在 SOZ 内 |
| 患者内时序组织高于 null | 是 | masked（shared-participant）MI data-vs-null |
| 分模板后刻板性提高 | 是 | cluster-aware uplift |
| 模板跨记录时间复现 | 否（已移出 Fig 1）| 属 cohort 结果，写入正文 / Table S3；split-half 面板不进主图 |
| 可复现 opposing template pair | 例患层是 | Panel c2 用 `epilepsiae:958` 展示 TA/TB 相反模板；cohort funnel（candidate 16 / reproduced 15）写入正文，不单独出 panel |
| 同一三维空间轴的相反读取 | 否 | contact map、held-out axis 和 paired-axis cosine 进入下一张 spatial 主图 |

## 2. Panel 对照与真实 producer

每个科学信息独立成图，落成 `fig1-panel<id>.{png,pdf}`，每张左上角打 panel id，不拼 composite。单一入口 `scripts/paper_figures/plot_fig1_interictal_hfo_temporal_scaffold.py`（Panel a 直接复制 group-event demo 成品）。

| 文件 | 图中内容 | 直接画图代码 | 关键输入 artifact | 当前裁决 |
|---|---|---|---|---|
| `fig1-panela` | 80–250 Hz 群体事件波形、normalized spectrogram、时频质心轨迹 | `scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py` | private artifact + `_gpu.npz` + `_packedTimes.npy` | 复用 Y3 demo 成品直接复制；按 private crosswalk 显示为 `Yuquan Y3` |
| `fig1-panelb1` | Yuquan refined HFO count 对 clinical SOZ 的 ROC | 复用 `scripts/plot_refine_soz_validation.py` 的 loader/ROC 定义 | `results/hfo_detection/<subject>/_refineGpu.npz` + Yuquan SOZ JSON + `config/subject_params.json` | 现场重算 n=20、mean AUC=0.873；阴影为插值后 subject ROC 的 SEM |
| `fig1-panelb2` | Epilepsiae 同上 ROC | 同上 | 同上 + Epilepsiae SOZ JSON | 现场重算 n=15、mean AUC=0.955 |
| `fig1-panelc1` | 例患 958 时间顺序 heatmap + rank distribution | `scripts/plot_interictal_propagation.py --masked-features --pr3` 的 helper | masked JSON + raw `*_lagPat*.npz` | 非参与 cell 灰显；inter-template r=-0.81；n 报全量，均匀抽样展示 1200 events |
| `fig1-panelc2` | 例患 958 TA/TB 聚类顺序 + mean rank | 同上 | 同上 | KMeans k=2；within-template τ 高于 overall τ |
| `fig1-paneld1` | MI data vs permutation null（40 例）| `scripts/plot_interictal_propagation.py --masked-features` 统计合同 | 40 个 masked per-subject JSON | masked（shared-participant）MI：40/40 significant，masked median 0.228 |
| `fig1-paneld2` | overall τ vs within-template τ（40 例）| 同上 | 同上 | 40/40 above diagonal；median Δτ=+0.198 |

## 3. 定稿布局

独立 panel 输出，最终拼版在外部完成，不出 composite：

| 文件 | 叙事 |
|---|---|
| `fig1-panela` | 群体事件现象 |
| `fig1-panelb1` / `fig1-panelb2` | 两队列 SOZ supporting clinical anchor |
| `fig1-panelc1` / `fig1-panelc2` | 例患 958 时间顺序 + TA/TB 模板 |
| `fig1-paneld1` / `fig1-paneld2` | 40 例 MI/null + within-template uplift |

图题限定为：

> Interictal HFO population events exhibit recurrent patient-specific temporal organization.

## 4. 剩余锁图项

- [x] paneld1（MI）已用 masked（shared-participant）统计重画：40/40 significant，masked median 0.228；
- [ ] 作者目视确认最终版式后锁图，不再更换 exemplar 或统计版本。
