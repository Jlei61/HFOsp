# Topic 5 间期传播场外推到发作隐身 territory — Pilot 结果（2026-06-30，exploratory）

## ✅ Phase-2 cohort 正式版（2026-07-01，review 4 项修复后；subject=单位 / 主张=F_core_only / 14-hypothesis BH-FDR）

修复了 review 的 4 项后重做（A/B 两模板 max；主张量改 **F_core_only**=场只用 narrow core 建、隐身电极不进场；cohort 独立单位=**subject**；显式 14-hypothesis FDR 表）。13→16 个有 broad 几何被试（Task5 于 2026-07-01 补 E590/E1084/E1146） × 两频段，channel/within-shaft/anchor null B=2000。FINAL：`results/topic5_ictal_recruitment/field_extrapolation/energy_field_extrapolation_FINAL.{json,md}`；图 `figures/cohort_energy_F_core_vs_baselines_{bb_auc,hfa_auc}.png`。

**结果（诚实，两层分清）：**

**① 延伸（F_core_only 过随机 null）= 成立，但按 null 分层（16-subject，FDR 精确）。** **channel null 两频段过**（bb q=0.0 过的被试 9/16、hfa q=0.0 9/16）；**anchor null 两频段过**（bb q=0.0 7/16、hfa q=0.0002 6/16）；**within-shaft null 只 broadband 过**（bb q=0.0 7/16），**HFA within-shaft 是趋势但 FDR 未过**（p=0.043、q=0.055、3/16）。**不可写"三层 null 两频段都 q<0.05"。** → **间期传播场能外推预测间期"隐身"电极的发作早期能量分布**（channel+anchor 支撑），答审稿人"结构是否延伸到间期电极之外"=**是（主要靠 channel+anchor null）**。

**② 场必要性（F_core_only 赢逐通道/几何基线）= 大部分不成立，且关键基线从未被赢：**
- **vs 自身间期顺序 C1：两频段都不显著**（bb q=0.32 n_pos=8 med_diff+0.021、hfa q=0.79 n_pos=7 med_diff−0.012）。**= 隐身电极"自己那条间期顺序"预测发作能量，和核心外推场一样好** → 场并非比逐通道顺序更优。**这是关键基线，两频段都没赢。**
- vs 自身能量 fingerprint C2：bb 边界不显著(q=0.052 n_pos=10)、**hfa 显著(q=0.005, 13/16 正)**。
- vs 半径(沿轴距离)：**bb 显著(q=0.042 n_pos=11)**、**hfa 显著(q=0.031 n_pos=12)**（注：13→16 后 broadband radius 由不显著翻为显著）。
- vs pca1_geometry（主采样几何方向，**非** shaft-direction，见下）：bb 不显著(q=0.109 n_pos=8)、**hfa 显著(q=0.022 n_pos=12)**。

**分母 = 16-subject broad-geometry cohort**（E590/E1084/E1146 于 2026-07-01 补跑上游 broad propagation 后纳入，0 skipped；见 `scripts/build_broad_lagpat_patch_epilepsiae.py` + 报告 §5 复现路由）。**⚠️ E590/E1146 隐身电极仅 4 个 = 低功率**；剔除 nhid<6 后计数不变（sensitivity 见报告 §3）。所有结论限定此 broad-geometry cohort（非完整 Topic-5 narrow cohort）。

**⚠️ pca1_geometry ≠ shaft-direction（review 修正）**：该几何基线实现为"沿隐身电极坐标主 linear 轴(PCA1)投影"，**不是**"单杆方向 / ≥2 非平行杆"基线。故 "hfa F_core>pca1_geometry q=0.005" **只能**说"胜过主采样方向的位置"，**不能**说"排除了电极杆方向伪影"。

**净判读（措辞天花板）：**
- ✅ 可写："**间期传播 field 对间期隐身/broad territory 的发作早期 activation energy 有外推能力**（extension：channel+anchor null 两频段过 FDR、within-shaft 只 broadband 过，16-subject cohort）；该外推在 HFA 频段胜过更粗的能量/主采样几何基线（C2/radius/pca1_geometry），broadband 只胜过 radius。"
- ❌ 不可写：①"场的独特必要性成立"——**场从未显著赢过'隐身电极自身间期顺序'(C1)**（bb q=0.32、hfa q=0.79），隐身电极自己的间期顺序已够用；②"排除/赢过 shaft-direction artifact"（pca1_geometry ≠ 杆向基线）；③"三层 null 两频段都过"（HFA within-shaft q=0.055 未过 FDR）；④"完整 Topic-5 narrow cohort 已封板"（仅 16-subject broad-geometry cohort）；⑤"发作招募顺序延伸"/"发作早期特异"。
- **broadband 上场只赢 radius（q=0.042）、不赢 C1/C2/pca1_geometry**；HFA 上赢 C2/radius/pca1_geometry、不赢 C1。**关键基线 C1 两频段都没赢** = 部分回答审稿人（延伸真）+ 强主张（场必要）不立。
- per-subject 稳过三层 null 者不多（bb 5/16、hfa 2/16）；`screen_strict`（三层 null ∧ 赢 C1/C2/radius/pca1）确定性版 **bb 0/16、hfa 1/16（仅 E916）**；`screen_channel_c1c2_only` bb 3/16（E253/E620/E635）、hfa 5/16（E1096/E1125/E139/E620/E916）。cohort 层面 C1 未被赢。

**早期单模板 pilot 数字（F broad-LOO "6/13、screen 3/13"）已废**：那是 hidden 互借 + 未控基线的虚高；F_core_only + 全基线 + FDR 后回落到上述诚实结论。

**工程备注（review round-2）**：①**可复现性 fix**：within_shaft null 原用 `hash(shaft_name)` 分组，Python 字符串 hash 每进程随机→改组处理顺序→改 RNG 抽样序→hfa within_shaft n_pass 在 3↔4 漂；改稳定 sorted-unique 整数编码 + `PYTHONHASHSEED=0` 后确定性(916 hfa within_shaft_p 两跑=0.0020)。承重结论(C1 paired 全确定; 延伸 q≪0.05)不受此 ±1 边界影响。②**screen 两档**：`screen_channel_c1c2_only`(仅 channel null+C1/C2 margin) vs `screen_strict`(三层 null 全过 ∧ 赢 C1/C2/radius/pca1)——**确定性版（16-subject）**：bb strict 0/16、**hfa strict 1/16(仅 E916)**；`screen_channel_c1c2_only` bb 3/16、hfa 5/16。**"场优势个案"只认 screen_strict**（E139 曾因 hash 非确定性误入，修复后剔除）。③**full-suite 非全绿**：本模块 17 TDD 全过 + 纯 additive(无改共享模块、无既有 test import 本模块)；仓库 full `pytest -q` 有 ~15 pre-existing failures(Topic1/4 缺 artifact/script + 1 synchrony 合同)，**与本工作无关但不宣称 full-suite green**。

---

## ⚠️ 更正（2026-06-30 同日，用户指出用错了发作量）

**第一版用错了发作量**：我用了发作**招募顺序**（z-ER 的 r_sz），它跨发作本来就不稳（早就知道），所以阴性是被错基础拖的。**真正让 field similarity 显著的口径是"发作整体能量"** `bb_auc`（broadband 1–45Hz 功率、对发作前 baseline robust-z、[0,10]s 均值），**每次发作算一次 |场相关| → 对发作取中位数**（稳健聚合，不要求招募顺序稳定）。

**在这个正确基础上重跑（13 个有 broad 几何 + t0 cache 的被试，channel-shuffle null B=2000）**：
- **间期场显著延伸到隐身电极（F 过 null p<0.05）：6/13**（E253 F=0.745、E916 0.635/48 发作、E620 0.573、E583 0.544、E1125 0.409、E635 0.370）。
- **场明显赢过逐通道（F−C>0.03）：3/13**（E253 +0.319、E620 +0.280、E916 +0.040）。E583/E1125 F≈C（场无额外增益，逐通道自身 broad rank 也能预测）。
- **与"narrow field_concordance 是否显著"无干净对应**：最强的 E253/E620 在 narrow 上**不**显著、却在隐身电极上场远胜逐通道；narrow 显著的 E922 反而不延伸。per-subject 异质性大。

**逐通道能量基线 C2（用户点的关键："场是否赢过逐通道能量"）**：C2 = 逐通道**能量 fingerprint** 基线 = per-seizure |corr(隐身电极间期 baseline 活跃度 `bact`, 其发作能量)|→中位数。**语义精确（review P1）**：是"通道自身能量指纹预测发作能量指纹"，abs 把反相关也算强基线（保守），**不是**单纯"活跃通道恒活跃"。bb_auc 基础：**场同时赢 C1(自身顺序)+C2(自身能量)**：
- **margin 依赖（review P1，未锁会漂）**：用 δ=0.03（F−C1>0.03 且 F−C2>0.03）= **3/13**（E253/E916/E620）；放宽到任意正 margin = **4/13**（E583 F0.544 vs C1 0.519 差 0.025 卡边）。**Phase-2 前必须锁 margin**（已在新 spec §2 锁 δ_FC=0.03）。
- E253 最干净（F0.745 vs C2 0.234）。数据 `f_c1_c2_bb_auc.json`。

**结论措辞天花板（review P1）**：最多写 **"间期传播 field 对部分 hidden/broad territory 的发作早期 activation energy 有外推能力，在 E253/E620 等被试上明显超过逐通道顺序+能量基线"**。**不可**写"证明发作招募**顺序**延伸"（顺序问题已 negative）、不可写"发作早期特异"。且这是 **pilot 正信号**——仅 channel-shuffle null，未补 within-shaft/anchor/radius/shaft 基线 + paired+FDR（见新 plan Phase 2），cohort 级"场独特优势"尚未确证。

**HFA 平行版（`hfa_auc` 60–100Hz 能量，sensitivity）**：F 显著 **8/13**、场赢自身顺序 C1 **7/13**（E1077/E1150/E583/E620/E916/E139/E922）；E583 在 HFA 上场优势明显（F0.497 vs C1 0.231，+0.266），但 E253（bb 最强）在 HFA 仅边缘（p=0.055）→ **band 依赖、被试异质**。数据 `per_subject_hfa_auc/`+`pilot_summary_hfa_auc.json`。(HFA 的 C2 尚未算，留 cohort 版。)

**口径修正后的判读（描述性 pilot，非 cohort）**：在"发作整体能量"这个稳定基础上，**间期场确实能预测一部分间期隐身电极的发作能量分布，且在 E253/E620 等被试上明显比逐通道强** → 初步支持"场的必要性"。但 ①约半数被试不显著；②显著者里多数 F≈C（场不比逐通道强）；③仅 channel null + abs-corr，未上 within_shaft/anchor 阶梯 null、半径/杆向基线、cohort 二项+FDR。**结论：正信号、值得做 cohort 版（Phase 2 改在 bb_auc 基础上），但还不是确证。**

图：`figures/cohort_activation_F_vs_C.png`（13 被试 F vs C 散点）。数据：`per_subject_activation/*.json` + `pilot_summary_activation.json`。

下面是**第一版（r_sz 招募顺序基础，错基础）**的记录，保留作为"为什么不能用招募顺序"的对照：

---

# 【第一版，错基础】Pilot 结果（z-ER 招募顺序，被数据现实阻塞）

设计 spec：`docs/superpowers/specs/2026-06-30-topic5-interictal-field-broad-extrapolation-design.md`
实现 plan：`docs/superpowers/plans/2026-06-30-topic5-interictal-field-broad-extrapolation.md`
代码（未提交）：`src/topic5_field_extrapolation.py`(+9 TDD)、`scripts/run_topic5_field_extrapolation_pilot.py`、`scripts/plot_topic5_field_extrapolation.py`
产物：`results/topic5_ictal_recruitment/field_extrapolation/{per_subject/*.json, pilot_summary.json, figures/*}`

## 测什么 / 怎么测（朴素话）

审稿人质疑现有"间期方向↔发作早期一致"只覆盖间期有明显 HFO 群体事件的电极、信息增益有限。我们想看：发作会点燃间期"隐身"（放电太少、被 narrow 速率阈值挡在外的）电极，**用间期可信核心搭的传播顺序场，能不能预测这些隐身电极在发作时按什么先后被招募**，并比"空间场(F)"是否赢过"隐身电极自己那条噪的间期排名(C)"和"离源远近(radial)"。

- 训练=间期 broad 顺序场（`propagation_geometry_broad` 的 `typical_rank`，support 加权 kernel 回归，留一评估到每个隐身电极位置）。
- 测试=发作 z-ER 招募序（Layer A `broad_ER.r_sz`，低=早）。
- 同向 → 正相关；F 赢 C 且过随机/半径 = 场把噪电极补活了。

## 结果：两层阻塞

**第一层（队列级，致命）——发作招募排序本身在队列里大面积不稳。** 有 broad 几何记录的 9 个被试里只有 **583 一个**发作排序可信（producer_health=stable、s_sz=0.55、22 个可用发作）；**916** stable 但只 3 个发作（隐身电极几乎都拿不到稳定 r_sz，只 1/14 可评）；其余 7 个全 unstable（s_sz 0.01–0.30，跨发作招募顺序几乎不一致）。**发作招募顺序若本身跨发作不稳，就没有一个稳定的"发作方向"可供间期场去预测**——本检验的前提对 ~15/16 被试不成立。另有 7 个（含 442/548/590/958 几何干净的 ECoG）连 broad 几何记录都没有。

**第二层（唯一干净被试 583）——场相对逐通道无增益，且"隐身=噪"前提不成立。**
- F=0.187、C=0.220、radial=0.220、F_p=0.283（n=13 隐身电极）。三者都弱、且 **F≈C≈radial**：场、逐通道、纯"离源远近"给的是同一个弱信号，方向场和通道特异间期排名都没带来额外信息。
- 隐身电极 support 中位 **0.647**，并不比全 broad（0.616）低 → 对 583 这批 broad∖narrow **不是"放电少的噪电极"**，"场救活噪电极"的机制在这没有抓手。
- 几何不退化（隐身电极沿轴铺 −4..55mm，轴长 49.3）→ 弱是真弱，不是采样问题。

## 与已有 Topic 5 结论的关系

这个阴性**印证**主线：间期↔发作只共享**粗骨架**，精细发作招募是**逐发作变化**的（加固 2026-06-15 + hfa_joint 复验都指向"粗稳、细不稳"）。本 pilot 在更大电极范围上又撞到同一堵墙——隐身电极的精细发作招募顺序，既不稳定（队列）、对唯一稳定被试也不被间期方向场超越逐通道地预测。**"场必要性"未被证明，但不是 bug，是数据里精细发作结构不够支撑这个外推主张。**

## 措辞纪律

- 不写"证明了场的必要性"，也不写"场无用"——只说：**在"精确预测隐身电极发作招募顺序"这个尺度上，场没看出比逐通道更强；且队列里多数被试发作排序本身不稳到没有可预测的方向**。
- 583 是 n=13 的单被试描述，非队列结论。

## 下一步岔路（待用户定，本轮未开始）

1. **降尺度到粗骨架**：不预测精确顺序，只问隐身电极落在间期轴的源侧/汇侧能否预测发作早/晚（二值）——但这≈radial 已经给的 0.22，且就是已有粗骨架结论。
2. **换发作量**：发作激活强度（bb_auc）是否比招募顺序跨发作更稳（但用户本轮明确要 z-ER 顺序；且 bb_auc 是旧 axis_alignment 用的、也只到粗对齐）。
3. **补 broad 几何**：给缺失的 442/548/590/958 等产 broad 几何记录——但它们发作排序也 unstable，补了大概率仍卡第一层。
4. **接受阴性收口**：把"间期方向延伸到隐身 territory"判为"数据精细结构不支撑"，写进 main doc 作 §3.6 同类的 exploratory-negative。

## 工程备注

- 发作可信度门 `ictal_reliability`（producer_health∈{stable,moderate} ∧ s_sz≥0.3 ∧ n_ok≥5）是关键 gate，最初 plan 漏了，pilot 暴露后补入（agent C 早提示 producer_health 多 fail）。
- 符号已验证：r_sz 低=早（vs 从 channel_onsets 重算的 earliest=low spearman +0.94~0.97）。
- 代码全 untracked、未提交（当前在 topic4 分支，待用户定干净 Topic 5 base）。
