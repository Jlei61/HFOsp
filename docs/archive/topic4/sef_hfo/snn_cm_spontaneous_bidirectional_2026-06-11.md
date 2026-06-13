# cm-SNN 自发异质性放电 → 两端病灶 → 双向传播模式 → 真实流水线读出（2026-06-11）

> **状态**：验收通过（仪器层面）。承接虚拟 SEEG 观测层 spec `2026-06-06-sef-hfo-virtual-seeg-observation-layer-design.md`（§4.3 cm-SNN substrate）与 cm 行波归档 `snn_cm_traveling_wave_2026-06-08.md`、异质核扫描 `snn_cm_heterogeneity_scan_2026-06-10.md`。本轮把"虚拟 SEEG 读出层"从"人为戳一下读单事件方向"推进到"**病灶自己点火、读出一串自发事件、并被与真实病人完全相同的聚类流水线认出两套相反方向模板**"。

---

## 0. 朴素话三段

**测了什么**：在一块能插真实 4mm 电极的厘米尺寸放电网络里，造一小块"更容易点着"的异常组织（虚拟病灶），**不人为戳它**，让它在背景噪声下自己点火出一串事件；把病灶放在传播轴的两个端点，看会不会出现两种相反的传播方向；再像真实病人一样隔着虚拟电极把这些事件读成"通道先后排名"，走真实数据那套聚类，看能不能自动认出两套相反模板。

**怎么测的**：先确认"病灶越异常、自发事件越多"是渐变的、并找到"事件够多、够干净（满片行波、背景安静）能读出"的档位；再在这个档位上、病灶分别放 −端 / +端各跑一次，每端先过一道门（自终止 ∧ 方向对 ∧ 触点够 ∧ 背景安静、且至少 10 个干净事件）才允许把两端事件池化进真实流水线；最后看真实聚类给出几簇、簇间相不相反。如果完全没有方向结构，聚类不会分出两套、簇间相关也不会是负的。

**揭示了什么**：在干净档位上，**−端病灶自发产生的事件全是一个方向（正向）、+端病灶全是相反方向（反向）**；池化 30 个事件喂进真实流水线，它**自动分成两簇（各 15）、两簇模板方向相反、跨时间折半可复现**。所以"模型自发产生 → 虚拟电极读出 → 真实流水线认出两套相反模板"这条链在仪器层面闭合了。**定位是"仪器对齐"、不是"机制重现"**：连接轴只有一条，模板空间近一维，分成两簇本来就半被迫——这张图证明的是"读出+流水线能如实认出双向"，不是"模型发现了双向机制"。

（内部归档代号：cm-SNN substrate spec §4.3，self-ignite / spontaneous train，stage2 gate，stable_k=2，inter-cluster corr，rank-displacement swap=strict，masked PR-2/PR-2.5。）

---

## 1. Phase-1：自发档位确认（先确认档位再读出）

**动机**：早先的扫描只量"到底点不点火"（峰值活跃比例），范围里多数已饱和自发；要先量"每秒几个**能分开、会自终止、满片**的事件"，并找渐变、可读的档位。

**做法**：−端病灶，平均门槛 18→16 × 离散 {宽 1.5 / 窄 0.5} × 2 种子、各 1.5s，用锁定的 `detect_events` 数事件。**大方向波门**：`returned ∧ 参与神经元≥5000 ∧ |方向相关|≥0.5 ∧ peak≥0.02`（排除几十神经元、方向乱的小波动）。背景安静度 = **事件窗之外**活跃比例 p95（非检测器头 5–50ms 校准 floor）。

**结果（`ratemap_summary.json` + `figures/spontaneous_regime.png`）**：
- 自发率随异常（平均下移）渐增，但**起点非零**（连平均=18、只核内参差都有稀有大成核——尾驱动）。
- **干净读出带分参差**：宽参差平均 17.5/17 干净满片行波，≤16.5 核变"准连续"、事件碎片化（mean16 真基线跳到 0.027、大方向波率掉到 0）；窄参差 ≥17.5 几乎只有小波动，≤17 才点、一路到 16 都干净。
- **修正**：早先"纯参差（mean=18）不自点火"是 cm runner ~200ms 短窗漏掉稀有大成核；1.5s 窗里它确实点（稀少、巨大、干净）。

---

## 2. Stage-2：带门的双向读出扫描

**门（每端都要过才池化，`run_sef_hfo_snn_cm_spontaneous_stage2.py`）**：期望方向干净事件 ≥10（−端期望正向、+端期望反向，方向符号错就不算）∧ 真事件间基线 <0.01。**两端都过才池化**进真实 masked 流水线；过不了的只报告、不池化。montage = ∥(θ_EE)+⊥(θ_EE+90) 对齐两杆、4mm。

**结果（`stage2_summary.json` + `figures/stage2_summary.png`，T=3500）**：

| 档位 | −端干净/基线 | +端干净/基线 | 门 | 池化结果 |
|---|---|---|---|---|
| **主 17 宽** | 15 / 0.0007 | 15 / 0.0005 | PASS | stable_k=2、15/15、inter-corr −0.95、forward/reverse 1 pair、swap=**strict**、split-half reproduced |
| **低异常 17.5 宽** | 13 / 0.0006 | 12 / 0.0004 | PASS | stable_k=2、12/13、swap=**strict**、reproduced |
| **过热 16.5 宽** | 0 / **0.027**（准连续） | 13 / 0.002 | **FAIL**（−端） | 不池化（门按设计拦下过热） |
| **参差 17 窄** | 8<10 / 0.0002 | 8<10 / 0.0002 | **FAIL** | 不池化（**干净但事件不够 = underpowered，非机制失败**；加长 T 应可过） |

门的两颗牙都验证了：拦"过热不干净"（16.5 宽）+ 拦"干净但事件太少"（17 窄）。

---

## 3. 真实流水线闭合 + 一个关键发现（剔死通道）

**池化记录走真实流水线**（`pool_and_cluster_spontaneous.py` → `compute_adaptive_cluster_stereotypy` / `compute_time_split_reproducibility` / `compute_swap_score_sweep`，全 masked）。

**关键发现——剔除无活动电极后正反检测变干净**：用全 12 通道时簇间相关被 masked-插补的死通道（B0/B1/B4/B5，从不参与）拉成 **+0.44、forward/reverse:none**；**剔除死通道后真实管线在活动电极（6∥+2⊥）直接给 inter-corr −0.95、forward/reverse 1 pair (r=−0.95)** —— 模板层也干净认出两套相反模板，与通道级 rank-displacement **swap=strict**、共享触点 spearman −0.945 一致。**死通道无信息、其插补值是污染源，剔除非 cherry-pick。** 这把早先"+0.44 是插补伪影、真反相只在共享触点 / rank-disp"的 caveat 升级为"活动电极上、全套真实管线就给 −0.95"。

---

## 4. 图

- **`figures/core_model_<config>.png`（主图，3 行）**：A 正向模式（异质性/病灶地图 + 事件传播地图 + 融合电极读出），B 反向模式（同基底、传播反向），C 真实 per-subject 流水线（lagPat + KMeans k=2 两套相反模板，活动电极）。**只融合 c/d（电极读出），保留 a/b 两重模式的空间地图**。
- **`figures/train_<tag>.png`**：∥+⊥ 电极整合进一个面板、只画活动电极、长时多事件；**一条 peak locus 跨两电极**（两电极都当观测，按沿传播轴位置排，∥ 橙 / ⊥ 青）。
- **`figures/model_propagation/model_<config>_bidir_propagation.png`**：模型走真实病人同款 per-subject 图（活动电极）。
- **`figures/stage2_summary.png`**：四档门控 + 池化结果总览矩阵。
- **`figures/spontaneous_regime.png`**：Phase-1 档位三联（大方向波率 / 真基线 / 事件大小 vs 平均）。

---

## 5. 诚实口径（锁）

- **仪器对齐、非机制重现**：单一连接轴 → 模板空间近 1 维 → stable_k=2 半被迫。图标 `model:`，不混进病人队列。
- **双向性是"强制读出 / 两端各设源"的结果**：两端病灶是人为放的（承 spec §7 口径，两端各设源 = 读回两种输入方向，不是"模型自发交替"产生正反）。两焦点放在**同一**网络里同时活会互相搅浑（早先 dephase=0.3 失败）；干净路径是两端分开跑再池化（= 真实数据里正/反是不同事件）。
- **stage2 科学目标**：验证哪些自发档位能稳定产生正反两端模板并被真实流水线识别，不是"扫异常程度"。

---

## 6. 脚本 + 结果

- runner：`run_sef_hfo_snn_cm_spontaneous_readout.py`（自发无 kick 多事件读出 + 写 legacy lagPat/packedTimes/montage + true_inter_event_floor）；`run_sef_hfo_snn_cm_spontaneous_ratemap.py`（Phase-1，`--reaggregate` 不重跑）；`run_sef_hfo_snn_cm_spontaneous_stage2.py`（带门双向扫描）；`diag_sef_hfo_snn_cm_spontaneous_train.py`（可行性诊断）。
- 池化 + 流水线：`pool_and_cluster_spontaneous.py`。
- 画图：`plot_sef_hfo_snn_cm_spontaneous_{train,mechanism,regime,stage2}.py`、`plot_model_propagation.py`、`plot_model_core_figure.py`。
- 结果：`results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/`（`stage2_summary.json` / `ratemap_summary.json` / `pooled_bidir/<config>/masked_pipeline_summary.json` / `figures/`）。engine guard：复用 `snn_heterogeneity/engine_versions.json`。

---

## 7. 待办（未做、不影响本轮验收）

- `--reaggregate` 加 expected-vs-found 缺格检查。
- 参差对照 17 窄加长 T（T=5000）补上门控那一格（它干净、只是稀）。
- 同一网络内两焦点错相位分时点火（更大 dephase / 不对称尺寸 / 恢复时间）——更"自然"但更糊，scoped。
