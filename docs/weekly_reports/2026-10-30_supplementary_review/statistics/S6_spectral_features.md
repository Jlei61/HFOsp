# S6：频谱分类、示例与早期能量特征

## 科学问题

发作起始阶段的频谱增强具有不同类型。S6 展示队列分布与 E20 的患者内分布，并说明为什么宽带型采用 1–150 Hz、非宽带 gamma 优势型采用 30–80 Hz 来构建发作早期能量场，对应主图 3A–C 的频带选择。两例时间零点均为已核对的临床起始。

## 频谱分类

图内及图注的 Gamma 对应原分类中的 gamma 优势型（`gamma_nonbroadband`）；原始分类规则、字段和逐发作标签保持不变。

统计单位为发作，共 23 名患者、324 次发作；比例按发作数计算，未按患者等权。复用 `topic5_early_spectral_overlap_v3` 的逐发作标签；本次核对类别计数，没有以后续质量审阅标签替换原分类。

六个检测频带为 1–4、4–8、8–13、13–30、30–80、80–150 Hz。各频带触点轨迹减去远端基线中位数后，取跨触点第 75 百分位并作 2 s 平滑。候选起点后 5 s 内超过基线第 99 百分位且持续至少 2 s，记为该频带增强。分类优先采用 −15 至 20 s 早期范围内已接受的频谱起点，否则执行原有早期范围搜索。

六频带中至少五个增强、且三个低频带（1–13 Hz）和三个快速频带（13–150 Hz）各至少两个增强，归为宽带型。未达到该标准但 gamma 增强的归为 gamma 优势型；其余若至少两个低频带增强，归为图中“仅低频型”，否则为其他。此处“仅低频型”沿用原分类名，表示剩余病例中的低频主导类，并非额外要求所有快速频带均无增强。

A 在 Broadband 与 Gamma 标签旁分别列出 1–150 Hz 和 30–80 Hz，说明后续能量读出的频带；该标注不改变上文的六频带分类标准。

A 的 Low bands 以第二行 1–13 Hz 标示原分类使用的 delta、theta、alpha 检测频带；这与 E 中 1–30 Hz 的功率读出范围分别定义。

## B：同一患者的类型分布

B 选用与 D 相同的患者 E20，以原有分类表中该患者全部 17 次发作为分母，每次发作计一次，未依据类型、空间分析资格或示例触点功率删选。宽带型 6 次（35.3%）、gamma 优势型 6 次（35.3%）、其他类型 5 次（29.4%），仅低频型 0 次；零计数类别不绘制扇区，完整四类计数保留在表中。该患者的宽带型和 gamma 优势型并列最多，合计 12/17（70.6%）。

选例理由是 E20 已作为 D 的 gamma 示例，且同一患者也有宽带型发作；Y15 的 7 次已分类发作均为宽带型，不能承担这一患者内对照。B 描述 E20 内两类发作的共存与占比，不估计这一现象在全部患者中的普遍程度，也不推断单次发作内部的类型转换。逐次发作标签及完整类别分母分别保存在 `data/panel_b_patient_seizures.csv` 和 `data/panel_b_patient_distribution.csv`。

## 图中读出与能量场

C/D 的 raw EEG 采用 CAR，分别保留 Y15 的 2000 Hz 与 E20 的 1024 Hz 原始采样。两例分别从 15 和 10 个来源触点中等间距显示六个，包含固定时频示例触点；每例内保持共同波形显示尺度。时频谱使用 1 s 窗、约 0.1 s 步长，逐频率以 −120 至 −90 s 基线中位功率归一化为 dB。为并列展示基线和起始，脑电及时频图显示 −110 至 −90 s 与 −10 至 20 s 两段；E 保留连续 −120 至 20 s。

E 为 D3 与 HRB1 的频带总功率相对基线的 dB 增幅，曲线作 2 s 平滑。各频带子图的左上角均以 12 号字竖排图例，蓝色为 Y15/SZ3 Broadband，绿色为 E20/SZ8 Gamma；白底不遮挡曲线或 0–10 s 早期窗口。后续空间分析中，宽带型采用 1–150 Hz，gamma 优势型采用 30–80 Hz：对每个触点的 log 频带功率，以基线中位数及 1.4826×MAD 稳健标准化，再对 0–10 s 取均值。图中 dB 曲线用于展示频谱变化，与空间场中的稳健 z 分数分别定义。

两例时间零点均为临床起始。分类按既有跨触点规则决定，不以这里两个示例触点的功率曲线重新分类。四类的全部发作数也不等同于主图 3 空间分析的资格筛选后样本数。

## 病例核对

Y15/SZ3 为 `yuquan_litengsheng` 的零基索引 2，原清单记录 `litengsheng_sz_003`，对应独立事件 `litengsheng_episode_003`；代表触点为 D3。原 EDF `FC1047M4.edf` 的 `clinical onset` 标注为 2343.9645 s。旧清单使用更早的 `SZ2` 标记（2338.9965 s）并把它存于 EEG 起点字段；本图按已核对的独立事件清单使用真实临床起始，晚于原标记 4.968 s。图内 SZ3 沿用患者既有编号，不能与 EDF 原始标记名称 SZ2 混用。

Y15/SZ3 的原有跨触点分类为六个频带均增强。D3 按既有 broadband 选触点规则，在 0–10 s 内比较各触点 alpha、beta、gamma、高 gamma 及 broadband 增幅的第 95 百分位，优先选跨频带增强最充分的触点；未使用空间相关选择。

E20/SZ8 为 `epilepsiae_635` 的零基索引 7，记录 `63500800102`，HRB1。其事件、时频谱、频带轨迹和临床对齐均保持上一版。

Y15/SZ3 的 0–10 s 平均增幅：1–30 Hz 为 5.57 dB、gamma 为 5.48 dB、高 gamma 为 8.36 dB、broadband 为 5.62 dB。E20/SZ8 的 gamma 为 12.55 dB、broadband 为 −1.14 dB。原始数值见 `data/example_early_band_power.csv`。

## 来源

- 分类：`results/topic5_ictal_recruitment/peri_onset_energy_timing/early_spectral_phenotype/per_seizure_spectral_overlap_state.csv`。
- 分类 producer：`scripts/plot_topic5_early_spectral_phenotypes.py`。
- 原始示例读出：`scripts/paper_figures/plot_fig3_raw_spectral_context.py`。
- 本图 producer：`scripts/paper_figures/build_supplementary_onset_spectral_features.py`。
- 临床起始核对：`results/topic5_extended_type_association/yuquan_episode_review_20261017/episode_map.csv` 与原 EDF 标注；保留 `yuquan_candidate/edf_annotation_check.json`。
- 本次病例缓存、来源和参数：`data/`、`metadata.json`。

[图件](../figures/current/S6.png) · [示例频带数值](S6_example_early_band_power.csv) · [队列类型计数](S6_cohort_spectral_summary.csv) · [E20 类型计数](S6_patient_distribution.csv) · [E20 逐次发作标签](S6_patient_seizures.csv)
