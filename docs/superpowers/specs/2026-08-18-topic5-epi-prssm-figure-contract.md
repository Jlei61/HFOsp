# Epi-PRSSM v0.1 预计产出图形合同

**状态：** exploratory figure specification；不是现有论文 Fig1–Fig4 的替换许可

**日期：** 2026-08-18；根据科学审阅修订

**科学 spec：** [`2026-08-18-topic5-epi-prssm-v0_1.md`](2026-08-18-topic5-epi-prssm-v0_1.md)

**实施 plan：** [`2026-08-18-topic5-epi-prssm-v0_1.md`](../plans/2026-08-18-topic5-epi-prssm-v0_1.md)

## 0. 图的叙事顺序

图按四个独立科学问题组织，不让 H3 成为整套项目的主裁决：

1. **Figure A：数据对象与统一模型阶梯。** 三种状态是否分开，G0–G3 和 R0–R3 分别回答什么。
2. **Figure B：H1 generator。** 慢状态是否存在，数据支持到 leaky、linear graph recurrent、nonlinear graph recurrent 还是 resource-anchored 哪一层。
3. **Figure C：H2a event distribution。** 慢状态是否改变完整事件和相似前缀的 suffix。
4. **Figure D：H2b seizure link。** 冻结间期状态是否在发作前移动并预测 early-ictal recruitment。
5. **Figure E：H3 exposure extension。** IED exposure 是否更新功能状态；它是独立机制扩展，阳性或阴性都不改变 A–D 的科学资格。

这些是内部稳定 `asset_id` 对应的候选包，不提前占用论文 Figure 编号：

| asset_id | provisional role |
| --- | --- |
| `epi_prssm_architecture_ladder` | Figure A |
| `epi_prssm_generator_evidence` | Figure B |
| `epi_prssm_event_distribution` | Figure C |
| `epi_prssm_seizure_link` | Figure D |
| `epi_prssm_exposure_mechanism` | Figure E |

只有作者确定 paper slot 并更新 `docs/paper_figure_registry.md` 后，才能生成 canonical `fig5/` 或 `fig6/`。不得写入当前 `results/paper-ready-figure/fig1`–`fig4`。

## 1. 全局 paper-ready 风格

### 1.1 输出与画布

- 单栏候选宽度 89 mm；双栏主图 180 mm；优先双栏。
- 独立 panel 不画左上角字母；complete layout 才统一写 A、B、C……。
- 每个 asset 同一次运行输出 PNG、矢量 PDF 和 metadata JSON。
- PNG 用 600 dpi；PDF 单页、字体嵌入、透明对象和 raster layer 经过检查。
- 白底、无外框、坐标轴朝外；不使用阴影、渐变背景或 3D 透视。
- 主图字号目标：panel letter 10–11 pt，标题 8–9 pt，轴/图例 7–8 pt，tick 6.5–7 pt。
- 主数据线 1.2–1.5 pt；个体线 0.5–0.8 pt；参考线 0.6–0.8 pt。

### 1.2 信息密度

- 每个 panel 只回答一个问题。
- 标题用名词短语，不在标题里写结论句、完整样本量或工程字段。
- exact P、完整分母、drop reasons、seed 和内部超参数放 metadata/README/caption。
- 主图优先 patient-level dots、paired effects、median 和 interval；不以 bar+SEM 代替患者分布。
- 个案 trajectory 只作解释，必须紧邻 cohort panel，不能单独承载结论。
- exploratory 结果不用星号制造 pass/fail 感；必要时写 exact P 和 interval。

### 1.3 语义配色

| 对象 | 颜色 | 用法 |
| --- | --- | --- |
| fixed patient scaffold/baseline | `#3F3F3F` | 图、静态 repertoire、reference |
| observer correction | `#9A9A9A` | 细虚线和 correction-on 控制 |
| G0 leaky | `#8C8C8C` | 中性基线 |
| G1 graph-CLDS | `#4C78A8` | 稳定线性 recurrent |
| G2 graph-GRU-ODE | `#238A8D` | 非线性 graph recurrent |
| G3/resource-anchored | `#6A51A3` | resource-anchored generator |
| IED exposure R2/R3 | `#A35E48` | 独立 H3 路径，不在其它图抢主位 |
| clinical onset | `#B33A3A` | 竖线或浅红窗，不作为状态色 |

仓库既有物理量继续服从全局规则：

- contact rank/order 使用 `viridis`，标 `early → late`；
- 带符号的 effect/Δstate 使用 0 居中的 diverging 红蓝；
- TA/TB 如出现，沿用冻结红/蓝语义，只作 downstream overlay；
- SOZ 只能是黑环 overlay，并标 `SOZ overlay only, not metric input`；
- 禁止 jet/rainbow。

### 1.4 统计语法

- 主统计单位为 patient；seizure/event/window 不画成独立 cohort 点。
- patient effect 用小实心点；同一患者多模型用细灰配对线。
- cohort summary 用粗中位线和 bootstrap/CI；不遮住原始点。
- model ladder 用 aligned paired forest 或 slopegraph，避免大面积柱状图。
- horizon/timescale curve 显示 patient bootstrap band，并在下方标 eligible denominator。
- 阴性结果保留 effect、interval 和 denominator，不留空白 panel。

## 2. Figure A：数据对象与统一模型阶梯

**asset_id：** `epi_prssm_architecture_ladder`

建议为 2 行布局：上行数据/状态/生成器，下行 observer/readout/独立假设。

### A. Three state objects

同一时间轴展示：

- fast event state \(\mathbf s_{e,k}\)；
- slow generative state \(\mathbf z_e=[\mathbf H_e,r_e]\)；
- observer state \(\mathbf c_e\)。

三者用不同形状和背景带；不得只靠颜色区分。图中直接写 `event-internal`、`physical/generative`、`inference memory`。

### B. Patient scaffold and baseline

画 node-level patient graph、固定 \(\boldsymbol\mu_p\) 和 dynamic residual。固定 baseline 用 charcoal；dynamic graph field 用 teal。明确：patient baseline 不等于 slow state。

### C. G0–G3 generator ladder

四列小示意共享输入输出：

- G0：独立 leakage；
- G1：稳定线性 graph messages；
- G2：门控非线性 graph messages；
- G3：resource 调制 damping/gain。

只有 G1–G3 画 node-to-node recurrent arrows。G0 不得画成 graph RNN。

### D. Observer versus physical transition

physical transition 用实线；observer correction 用细虚线。primary observer 箭头只进入 graph-state estimate，不直接写入每事件 resource。

### E. State-conditioned event readout

显示 `prefix + pre-event state → suffix/STOP`；完整 event 完成后才进入 observer。用竖直时间界线表达 no-self-event leakage。

### F. Independent hypothesis map

H1、H2a、H2b、H3 四个盒子独立出结果卡。H3 只以侧支箭头连接，不画成 H1/H2 的通行门。

**Figure A 禁止：**

- 把 G0 写成 autonomous graph RNN；
- 把 resource 画成已测量代谢变量；
- 把 TA/TB 画进 observer；
- 把 H3 阳性写成整个模型成立的必要条件。

## 3. Figure B：H1 generator evidence

**asset_id：** `epi_prssm_generator_evidence`

### A. Data and support landscape

患者 event/source count、clock-time coverage、H5/H10/H20/H40 anchor support。使用小型 dot/rug，不画工程 dashboard。

### B. Static versus dynamic variance

每患者显示 fixed \(\boldsymbol\mu_p\) 与 within-patient dynamic residual 方差分解。patient points 必须全部可见。

### C. Open-loop horizon

横轴 H5/H10/H20/H40，纵轴相对 static/G0 的 held-out loss difference。G0–G3 同轴、同 denominator；0 线明确。

### D. Model ladder patient effects

按 `G0 → G1 → G2 → G3` 展示逐患者 paired increments，并分别标 G1−G0、G2−G1、G3−G2。不要只画“best model”。

### E. State reset and correction budget

上半显示 reset curve；下半显示 correction energy 与 open-loop performance。用于识别 observer 是否吞掉 generator，不用双 y 轴。

### F. Graph state exemplar

一个预先按 coverage 选择的患者展示 node-level state snapshots，G1/G2/G3 使用相同坐标、相同归一化和同一时间点。若为 spectral sensitivity，必须显式标 `spectral sensitivity`。

**允许结论按 ladder 分支写：** leaky state、structured graph recurrence、nonlinear increment、resource-anchor increment。Figure B 不需要 H2/H3 结果才能成立。

## 4. Figure C：H2a 慢状态是否改变事件分布

**asset_id：** `epi_prssm_event_distribution`

### A. Adapter ladder

no state、initial-state、Node FiLM、low-rank edge gate 的容量递增对比。每种 adapter 配 state/no-state matched control，避免把容量增益写成状态增益。

### B. Full-event distribution

逐患者显示 masked order/rank、STOP 和 participation-residualized repertoire 的 state increment。participation/extent 作为次级小 panel，不占主位。

### C. State-swap counterfactual

correct state 与 patient-internal matched state 的 paired NLL/effect。点和配对线为主；不画单个平均柱。

### D. Ambiguous-prefix support map

横轴 prefix families，纵轴患者或 support strata；颜色只表示 train support，不表示 outcome。明确哪些患者 `targeted eligible`。

### E. Same prefix, different suffix

支持充分患者示意：相同/相似 prefix 在低/高或不同 slow-state region 下的 suffix distribution。显示真实事件数和 uncertainty，不挑单个最漂亮 trial。

### F. Frozen TA/TB projection

只在模型冻结后，将 predicted event distribution 投影到 TA/TB 解释面。TA/TB 不得与 slow state 共用颜色，也不得写成模型输入。

**Figure C 判读：** full-event 和 state-swap 是全队列主体；ambiguous-prefix 是高特异性加固。支持不足不是 H2a 失败。

## 5. Figure D：H2b 间期状态是否连接发作

**asset_id：** `epi_prssm_seizure_link`

Figure D 只有 `INTERICTAL_MODEL_FREEZE.json` 存在后才能生成正式版。

### A. Frozen validation design

只画一次：interictal training → freeze → seizure label release → pseudo-onset/LOSO。冻结点用锁形图标；不堆 pipeline 细节。

### B. Last-observation open-loop

时间轴显示最后允许 observer update 的 IED、correction-off 区间和 clinical onset。明确 onset state 是自主积分所得。

### C. Seizure-aligned trajectory

单 seizure 浅线、患者内 summary、患者级 cohort effect 分层展示。resource 与 graph-state summary 分轴；clinical onset 为红线/浅红窗。

### D. Matched pseudo-onset

画每患者真实 onset effect 与 matched pseudo-onset distribution 的差，不只画两条平均曲线。

### E. Increment beyond nuisance

state-only、rate/IEI/source/time-of-day、nuisance+state 的 patient paired effects。若 sleep/vigilance 分母不同，单独标 denominator。

### F. Early-ictal transfer

onset 前冻结 state 对 early-ictal masked order/field/extent 的患者级效应。个案 field 必须与 cohort effect 同屏，并沿用 `viridis early → late` 或冻结 field 配色。

**禁止升级：** seizures are caused by resource depletion；IED drives onset；state is a seizure clock。

## 6. Figure E：H3 exposure mechanism extension

**asset_id：** `epi_prssm_exposure_mechanism`

Figure E 不再是整个项目的主裁决图，而是 H3a/H3b 的独立证据卡。

### A. R0–R3 nested ladder

画 no-resource、autonomous resource、single-event depletion、integrated exposure。唯一 rust 箭头是 exposure forcing；其它模块保持相同颜色和尺寸。

### B. Resource and exposure impulse response

并排画 R1 recovery、R2 impulse 和 R3 integrated response，明确 \(\tau_r\) 先冻结、\(\tau_x\) 后比较。不得把模型轨迹标成真实代谢测量。

### C. Timescale curve

主图显示 fast/medium/slow；完整 5/15/30/60/120 min 用浅点或 Extended Data。clock-time 与 event-count control 同轴或上下对齐，不能只展示最佳 \(\tau\)。

### D. T1/R1 versus R2/R3 open-loop

逐患者显示至少一个 non-load endpoint：masked order/rank、suffix 或 participation-residualized repertoire。participation/extent 放次级 inset。

### E. Innovation and directionality

同一 panel 展示 frozen-T1 innovation、state-matched shuffle、time reversal 和 event-count control。颜色突出 real direction，null 用灰；不以一条漂亮 trajectory 代替 controls。

### F. H3a and H3b evidence cards

左右分开：

- H3a：interictal predictive + innovation；
- H3b：只读冻结的 preictal/early-ictal direction consistency。

不要用一个联合 pass 灯覆盖所有患者。H3a 阳性/H3b 阴性、H2b 阳性/H3a 阴性等分支必须能在图上单独显示。

## 7. Extended Data / Supplementary

建议至少保留：

1. just-in-time synthetic recovery，按 Goal 分图，不做巨型 truth×model 总矩阵；
2. node-level versus spectral sensitivity 和 eigen-alignment audit；
3. observer-resource flexible correction control；
4. numerical stability、time constants 和 boundary occupancy；
5. dataset/support strata；
6. full clock/event-count kernel grid；
7. seed stability 和 negative patient examples；
8. learned encoder sensitivity。

## 8. README、metadata 与输出路径

每个 asset 输出到：

```text
results/epi_prssm/v0_1/figures/<asset_id>/
├── figures/
│   ├── README.md
│   ├── <asset_id>.png
│   ├── <asset_id>.pdf
│   └── <asset_id>-complete-layout.{png,pdf}  # 需要多 panel 完整拼版时
└── <asset_id>_metadata.json
```

`figures/README.md` 每张图使用：

```markdown
### filename

2–4 句说明这张图展示什么、分母是什么、允许和不允许得出什么。

**关注点**：一句话指出目视检查重点。
```

metadata 至少包含：

- `asset_id`、`paper_slot=TBD`、`status=EXPLORATORY`；
- input hashes、code revision、config、run IDs；
- patient/event/seizure denominators 和 exclusions；
- split、seeds、model family、state dimension；
- frozen versus development status；
- endpoint/null/statistic；
- color mapping、normalization、axis limits；
- claim boundary；
- PNG/PDF same-state fingerprint。

## 9. 视觉与科学验收

每个 figure package 必须同时通过：

1. **数据合同：** patient/channel/event 对齐、mask、denominator、split 和 freeze 状态正确；
2. **科学合同：** panel 真正回答对应 evidence card，不把例图当 cohort evidence；
3. **统计合同：** patient-first、null 对应 estimand、阴性结果不消失；
4. **视觉合同：** 字体、线宽、遮挡、白边、色条、panel 对齐和匿名化目视通过；
5. **同状态合同：** PNG、PDF、metadata 来自同一次运行；PDF 单页且可打开；
6. **包装合同：** README 在实际图生成后写完，文件名和 `asset_id` 一致。

通过脚本或文件存在不等于视觉验收。H3 Figure E 即使完全阴性也可以作为高质量机制排除图；它不再决定 Figure B–D 是否可报告。
