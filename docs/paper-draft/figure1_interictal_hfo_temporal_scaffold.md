# Figure 1：间期 HFO 群体事件的患者特异性时序组织

> 状态：panel + complete-layout 主图合同 v6（2026-09-04）
>
> 定稿候选：`results/paper-ready-figure/fig1/figures/fig1-panel{a,b1,b2,c,d,e,f}.{png,pdf}`
>
> 完整排版：`results/paper-ready-figure/fig1/figures/fig1-complete-layout.{png,pdf}`
>
> Figure 1A：固定裁剪自旧 Supplementary Figure S6 TIFF，作为无患者身份语义的 representative SEEG implantation context
>
> 单一复现入口：`scripts/paper_figures/build_main_figures_1_2.py --figure 1`

## 1. 当前图能与不能支持什么

| 结论 | 当前是否直接展示 | 裁决 |
|---|---|---|
| 跨通道 HFO 群体事件 | 是 | B1/B2 的 waveform、spectrogram 与质心轨迹支持 |
| 患者内时序组织高于 null | 是 | D 的 masked shared-participant MI data-vs-null |
| 同一患者内可分两类时序模板 | 例患层是 | C/E 使用 E7 同一批 6,556 个有效事件；E 按 TA/TB 重排 |
| 分模板后刻板性提高 | 是 | F 的 template-aware MI uplift |
| HFO-rich channel 与 clinical SOZ 相关 | 不在本图 | 已移到 supplementary SOZ 包，不再误占 Figure 1B |
| 同一三维空间轴的相反读取 | 否 | 进入 Figure 2 |
| 机制因果 | 否 | 本图只建立真实数据时序 scaffold |

## 2. Panel 对照与 producer

| 文件 | 图中内容 | 直接画图代码 / 输入 | 当前裁决 |
|---|---|---|---|
| `fig1-panela` | 代表性脑表面与 SEEG 植入空间背景 | `build_main_figures_1_2.py`；`ReplayIED/tiffs/fig_s6_画板 1.tif` 固定裁剪 | 不绑定患者编号；不把触点颜色作为新统计 |
| `fig1-panelb1` | 178 段人工标注 HFO 的叠加波形、黄色均值、raw/normalized 平均谱 | `plot_fig1_single_hfo_schematic.py`；legacy `zhangkexuan_pickSigs.npz` + `zhangkexuan_annot_v4.pik` | 0–0.6 s 无白边；n=178 |
| `fig1-panelb2` | Yuquan Y3 群体事件波形、normalized spectrogram 与质心轨迹 | `plot_fig1_hfo_group_event_legacy_style.py`；`FC10477Q.edf` + `_gpu.npz` + `_packedTimes.npy` | magnitude + Gaussian σ=1.5；主峰 ≥70% 连通区质心 |
| `fig1-panelc` | E7 时间顺序 masked rank heatmap、day/night strip、rank distribution | `plot_interictal_propagation.py --masked-features --pr3` helpers | 非参与触点为空白；n=6,556 |
| `fig1-paneld` | MI data vs permutation null | 同一 masked 40 人 JSON；原 violin/box/points/bracket painter | `legacy_mi.masked=true` 硬检查；y 从 0 起 |
| `fig1-panele` | E7 全量事件按 TA/TB 重排及 mean-rank profiles | 与 C 同一 artifact 与 channel order | TA n=4,621、TB n=1,935，总数守恒为 6,556 |
| `fig1-panelf` | overall vs within-template MI；右下为 single-template vs multi-cluster 配对分布 | masked `overall_tau` / `within_cluster_tau_mean` | 两轴从 0 起；40/40 提高；配对 Wilcoxon P=1.82×10^-12 |

## 3. 输出与组装纪律

- 每个 panel 独立输出 PNG/PDF，文件内不画左上角字母；统一 builder 另生成带 B–F 字母的完整排版。
- B 在主图中由 B1/B2 两个素材块共同组成，但两者仍分别输出，便于版面微调。
- C 与 E 不得重新抽样或更换患者；两者必须共享事件全集、channel order 与 mask 合同。
- Figure 1A 必须直接来自登记的旧 Supplementary Figure S6 TIFF 固定裁剪；不得用新画的近似脑图替代，也不得从裁剪内容反推患者身份。

图题限定为：

> Interictal HFO population events exhibit recurrent patient-specific temporal organization.

## 4. 剩余锁图项

- [x] A–F 已按当前 panel 字母独立输出；A 为 TIFF 衍生的 600 dpi raster/PDF，B–F 为 600 dpi PNG 与矢量 PDF。
- [x] 已生成带 A–F 字母的完整排版；独立 panel 内不重复角标。
- [x] 旧 `panela*`、SOZ ROC `panelb*`、C/E 联合文件与 `d1/d2` 命名已归档。
- [ ] 作者目视确认最终版式后锁图，不再更换 exemplar 或统计版本。
