# 间期传播模式外推 vs 电极自身顺序 —— 谁更能预测发作能量（1-3 vs 2-3，cohort 报告，2026-07-01）

## 摘要（朴素话）

**测了什么**：癫痫间期（发作之间的安静期）会反复出现小的群体放电事件，这些事件在电极间传播有个稳定的"先后顺序模式"。我们看：这个间期顺序模式，能不能预测**发作时的能量分布**——尤其是那些间期太安静、没被原分析纳入的"隐身"电极（它们发作时确实会亮）。并比两种做法：**(1) 用可信核心的顺序模式，外推到这些隐身电极**；**(2) 直接用隐身电极自己那点间期顺序**。

**怎么测的**：裁判 = 发作能量（独立的 ground truth）。对每次发作，算"预测的顺序 vs 这次发作的能量"的相关，对该被试多次发作取中位数；看做法 (1) 外推 能不能比 (2) 自身顺序 更准。全队列 16 个几何可用被试 × 两个频段（broadband 1–45Hz、HFA 60–100Hz），逐个统计谁赢。

**揭示了什么**：**间期顺序模式在队列层面能预测隐身电极的发作能量，但强度中等、被试差异很大**——相关的队列中位数只在 0.44–0.56（自身顺序 C1 0.50–0.52），且**broadband 5/16、HFA 7/16 被试低于 0.4（很弱）**，不是"都在 0.4–0.8"。"核心外推"在**队列统计上**超过随机重排（16 被试里 9/16 过 channel null，binomial q<0.05），**但能同时稳过三层随机对照的被试不多**（broadband 5/16、HFA 2/16），且"电极自身间期顺序(C1)"本身没做同样的随机对照证明。**而且"核心外推"并不系统性地比"电极自身的间期顺序"更好**：16 个被试两种做法**大致打平**（broadband 7:5:4、HFA 5:8:3，1-3:2-3:平）。→ "间期刻画网络"（弱-中等、队列层面）成立；"外推比用电极自己的间期信号更强"**没立住**。

（内部代号：1-3 = `F_core_only`（narrow-core 场外推预测隐身电极），2-3 = `C1`（隐身电极自身 broad `typical_rank`）；裁判 = seizure energy `bb_auc`/`hfa_auc`；per-seizure |Spearman| median；margin δ=0.03。上游 cohort = `energy_field_extrapolation_FINAL`。）

## 1. 目的与范围

- **目的（用户定）**：说明"我们从间期事件传播找到的模式，能很好地刻画癫痫网络"。
- **本报告回答的具体问题**：把间期顺序模式**外推**到间期隐身电极（broad∖narrow），预测它们的**发作能量**，并比"核心外推(1-3)"能否胜过"电极自身间期顺序(2-3)"。
- **范围**：**16-subject broad-geometry cohort**（E590/E1084/E1146 于 2026-07-01 补跑上游 broad propagation 后纳入，口径同既有 13：top_n=20 broad lagPat + masked，见 `scripts/build_broad_lagpat_patch_epilepsiae.py`）。**⚠️ 其中 E590/E1146 隐身电极仅 4 个（narrow 池本就大、broad 扩得少）= 低功率**，两频段基本 tie。

## 2. 方法（精确）

- **1-3（core-extrapolation, `F_core_only`）**：间期顺序场只用 narrow 核心通道建（A/B 两模板取 max、LOO），外推评估到每个隐身电极位置作预测。
- **2-3（own order, `C1`）**：每个隐身电极自身的 broad `typical_rank`。
- **裁判**：发作后 [0,10]s 能量（`bb_auc` primary / `hfa_auc` sensitivity），rank01。
- **统计量**：每发作在隐身电极上算 |Spearman(prediction, energy)|，对发作取中位数 = per-subject 值；这与上游 cohort（`energy_field_extrapolation_FINAL`）逐位一致。
- **胜负**：δ=0.03；F_core > C1+δ = 1-3 赢，C1 > F_core+δ = 2-3 赢，否则平。
- **图**：单被试单行 5 格（发作能量场 | 1-3 外推场 | 2-3 自身场 | colorbar | 逐发作箱线），16×2；cohort 汇总配对图 + 计数。数据源 `results/topic5_ictal_recruitment/field_extrapolation/cohort_per_subject/*.json`。

## 3. 结果

**cohort 计数（16-subject，δ=0.03）**：

| 频段 | 1-3 赢 | 2-3 赢 | 平 | 1-3 延伸（过三层 null）|
|---|---|---|---|---|
| broadband | 7 | 5 | 4 | 5/16 |
| HFA | 5 | 8 | 3 | 2/16 |

- **1-3 与 2-3 大致打平**，两频段都没有系统性方向 → 核心外推不比电极自身间期顺序更优。补 3 个后结论不变（新增的 590/1146 因 nhid=4 基本 tie，1084 bb 略 1-3 / hfa 2-3）。
- **sensitivity — 剔除 nhid<6（E590/E1146）**：broadband 7:5:2、HFA 5:7:2（n=14）→ **结论仍不变**（低功率被试不改变"打平"）。故 590/1146 纳入全表但另标低功率。
- **相关强度（复核，不要写满）**：F_core_only broadband 0.20–0.80（中位 0.56）、HFA 0.27–0.84（中位 0.44）；C1 broadband 0.27–0.80（中位 0.50）、HFA 0.39–0.80（中位 0.52）。**多数落在中等区间，但 broadband 5/16、HFA 7/16 被试 F_core < 0.4（很弱），subject 异质性大**。
- **延伸分层写（review P1，按 FDR 逐 null 精确）**：**channel null 两频段过**（bb/hfa 均 q=0.0）；**anchor null 两频段过**（bb q=0.0、hfa q=0.0002）；**within-shaft null 只 broadband 过**（bb q=0.0），**HFA within-shaft 是趋势但 FDR 未过**（p=0.043、q=0.055）。所以延伸主要靠 channel+anchor null 支撑；不可写"三层 null 两频段都 q<0.05"。**per-subject 同时稳过三层 null 者不多**（broadband 5/16、HFA 2/16）。C1 本身**没做**同样的随机对照，故不能对称地说"C1 也证明了延伸"——只知 C1 与 F_core 打平。
- per-subject 差异大：E916/HFA 1-3 明显赢（0.84 vs 0.61）、E583/broadband 2-3 赢（0.54 vs 0.61）；逐个见 `cohort_1v3_vs_2v3_tally.md`。
- **均值场会 oversell**：单被试图上排"均值能量场"看着 1-3 常常更像发作能量，但一旦要求"每次发作都成立"（(D) 箱线），优势多数消失——这正是用逐发作统计的理由。

## 4. 对"刻画癫痫网络"目的的判读（措辞天花板）

- ✅ 可写：**"间期传播顺序模式在队列层面能预测间期隐身电极的发作早期能量（核心外推 9/16 过 channel null，binomial q<0.05），说明间期模式刻画了这片网络的空间组织——但强度中等、被试异质（队列中位约 0.44–0.56，broadband 5/16、HFA 7/16 被试很弱、<0.4），且 per-subject 稳过三层 null 者不多（bb 5/16、HFA 2/16）。"**
- ❌ 不可写：①**"核心外推比电极自身间期顺序更能刻画网络"**（1-3 vs 2-3 打平，未立）；②把 590/1146（nhid=4 低功率 tie）当有效证据；③把单被试均值场的"看着 1-3 赢"当结论（逐发作会翻）；④"发作招募顺序延伸 / 发作早期特异"。
- **一句话**：间期模式刻画网络 = 支持；外推优于电极自身间期信号 = 不支持（大致打平）。

## 5. 图 / 工件清单

- 单被试图（16×2=32 张）：`results/topic5_ictal_recruitment/field_extrapolation/figures/triptych_preview/{sid}_extrapVown_{broadband,hfa}.png` + `README.md`（旧迭代图已移 `_superseded/`）。
- cohort 汇总：`figures/cohort_1v3_vs_2v3_summary.png`（16，dashed*=nhid<6 低功率）；计数表：`cohort_1v3_vs_2v3_tally.md`。
- 上游 cohort（权威 FDR）：`energy_field_extrapolation_FINAL.{json,md}`（16）、`cohort_per_subject/*.json`（32）。
- 代码：`scripts/plot_topic5_field_extrapolation_triptych.py`、`src/topic5_field_extrapolation.py`（`predicted_interictal_order` core_names 外推）、**`scripts/build_broad_lagpat_patch_epilepsiae.py`**（补 E590/E1084/E1146 上游 broad propagation 驱动）。
- 关联：spec `docs/superpowers/specs/2026-07-01-topic5-energy-field-extrapolation-design.md`；能量 cohort archive 段 `docs/archive/topic5/field_extrapolation_pilot_2026-06-30.md`。

## 6. 状态
- **Task 5 DONE**（2026-07-01）：E590/E1084/E1146 补跑上游 broad propagation→rank_displacement→contact_plane geometry→能量 cohort，纳入后 **16-subject**，结论不变（1-3 vs 2-3 打平）。590/1146 nhid=4 低功率已标注。
- 全部 untracked、未提交（仍在 topic4 分支 `codex/topic4-m3a-v2-2`）。commit 计划：field-extrapolation 单独一批（含 `build_broad_lagpat_patch_epilepsiae.py`），V2-criticality / Topic4-m3a 各自分开；`FIGURE_INDEX.md`+`topic5_seizure_subtyping.md` 是跨线共享文件、需拆分处理。待用户点头 + 定干净 Topic5 base。
