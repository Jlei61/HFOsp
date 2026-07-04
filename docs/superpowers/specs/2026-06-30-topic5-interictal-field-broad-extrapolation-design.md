# Topic 5 — 间期传播场外推到发作隐身 territory：方向延伸检验（design spec）

- 日期：2026-06-30
- 状态：**CLOSED-NEGATIVE（2026-07-01 restructure）。** 本 spec 只锁**一个**问题 = 间期顺序场预测发作**招募顺序**(z-ER `r_sz`)。结果阴性——发作招募顺序跨发作本就不稳，无可预测的稳定方向（见 `docs/archive/topic5/field_extrapolation_pilot_2026-06-30.md` 第一版段）。
- **能量问题（间期顺序场 → 发作早期能量空间分布）是另一个问题，另开 spec**：`docs/superpowers/specs/2026-07-01-topic5-energy-field-extrapolation-design.md`。本 spec 不再扩展，保留作 order-问题的设计记录；正文下述 z-ER/带符号口径仅对 order 问题有效。
- 触发：合作者/审稿人质疑现有"间期↔发作方向一致"主张只覆盖间期参与电极、信息增益有限
- 分支：当前在 `codex/topic4-m3a-v2-2` —— ⚠️ 实现前须切干净 Topic 5 base（见 §10）

---

## 0. 摘要（朴素话）

审稿人质疑：你们说"用间期事件两类模板的 source 搭出来的病理传播方向，和发作早期一致"，可如果这条方向只覆盖间期那一小撮有明显 HFO 群体事件的电极，那它跟"直接看电极空间分布"差不多，有信息增益但不多。

改法：发作会点燃比间期那一小撮大得多的 territory，很多电极间期因为放电太少被挡在分析之外（"隐身"），但发作时确实被招募。我们用间期可信核心搭出的传播场，去**预测这些间期隐身电极在发作时该按什么先后被招募**，再拿它们发作时真实的招募先后来验证。

如果"用空间场（带坐标）"在这些噪电极上比"逐通道直接相关"预测得更好，就证明了"场"不是装饰——是它把单看信不过的噪电极，靠核心信息补成了可用的方向。这正是审稿人要的"场的必要性"。

（精度代号见正文。本摘要适用 CLAUDE.md §8 朴素话纪律；正文为设计合同，用代号求精度。）

---

## 1. 背景与已有结论的边界

现有 A 线 `axis_alignment` / `field_concordance`：

- 实现：`scripts/run_topic5_axis_alignment.py` + `src/topic5_axis_alignment.py` + 场引擎 `src/propagation_contact_plane_readout.py`。
- 统计量：把每触点的间期招募顺序 `typical_rank` 和发作 [0,10]s 激活 `bb_auc`，按真实电极坐标投到归一化平面、各自核回归平滑成场，比两场的 **绝对值、镜像不变** Pearson |corr|（`corr_pair_mirror_invariant`）。即 **无符号、只测同轴（允许正反翻转）**，不是方向梯度拟合。
- 操作集：间期 axis record 与发作 cache **按名取交**、≥6 触点 = **narrow 参与触点**。
- method-of-record：`results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.{json,md}`。
- 已有结论（加固 2026-06-15，`docs/archive/topic5/axis_alignment_hardening_result_2026-06-15.md`）：粗骨架稳健（broadband 过 `channel` null）；**且时间负对照显示这是"持续存在的患者骨架"，非发作早期特异招募**（发作前远端窗对齐不弱于发作窗）。

**关键缺口（审稿人指向）**：以上全在 narrow 参与触点上，从未碰间期隐身电极。

---

## 2. 测什么 / 怎么测

### 2.1 训练 / 测试 = 按条件分，不按电极分

- **训练（间期）**：在 **broad 电极池**上搭"间期传播方向场"。
- **测试（发作）**：在同一 broad 池上取每电极**发作招募 onset 时序**，验证间期场方向能否预测它。
- 间期、发作是两批独立信号，"间期场方向能否预测发作 onset 方向"是干净的跨条件问题；单个电极同时进两个场**不构成信号泄漏**（泄漏只在"用电极自己的间期顺序预测它自己的发作顺序"那种留一电极设计里才成立，本设计不是）。

### 2.2 新 territory = broad ∖ narrow

- narrow 池 = 速率阈值 `pick_k`(mean+k·std on HFO counts) 卡出的高发放核心（per-subject，~4–52 ch）。
- broad 池 = `pick_k=-2.0` + `pack_top_n∈{20,40}`（`scripts/pilot_broad_lagpat_repack.py`），纳入"有间期事件但放电少"的电极。
- **broad ∖ narrow** = 这些"间期隐身（数据量少、单通道 rank 噪）但发作被招募"的电极 = 本检验的靶。
  - Yuquan top_n=20 已 materialized：`results/lagpat_broad/qc_table.csv::added_channels`。
  - Epilepsiae 推导 = `interictal_propagation_masked_broad/per_subject/` 与 `interictal_propagation_masked/per_subject/` 两个 json 的 `channel_names` **精确字符串差集**（非 base-stripped）。

### 2.3 被测量与符号性（对现有方法的实质改动，CLAUDE.md §6.1 教训）

现有方法比"间期顺序 ↔ 发作激活强度"且无符号；本检验改成：

- **间期侧**：间期招募顺序场（`typical_rank`，低=早=源），建在 broad 池上。
- **发作侧**：发作招募 **z-ER 顺序场** = 用**与之前对比同款的 z-ER recruitment ordering**（Layer A ictal ER：baseline-robust-z-ER → Page-Hinkley CUSUM onset `t_onset_sec` → 每发作按 onset 早晚排序，早=源；per-subject 聚合）。**沿用既有"早场"构造，不另发明 onset 提取**；**不是** `bb_auc` 激活强度。（用户 2026-06-30 锁：发作侧用之前用的那套 z-ER 顺序。）
- **统计量**：**带符号**的场方向对齐（用户已确认 signed；不取镜像不变绝对值）——要分"同向延伸"vs"反向"，不只是"同轴"。同向 = 间期早的触点发作也早，带符号相关应为正。

理由：用户主张是"传播方向 + 招募序列 + 发作早期被招募"，对应 z-ER 招募顺序 + 有方向，不是激活强度 + 无符号。复用引擎可以，被测量与符号性必须换（reuse 要对问题，不能光对签名）。

### 2.4 F vs C（承重验收门 = 回应审稿人的核心）

- **F** = broad∖narrow 上"带坐标的空间场方向对齐"（间期顺序场 → 发作 onset 场）。
- **C** = broad∖narrow 上"逐通道直接相关"（间期顺序 ↔ 发作 onset rank 的带符号 Spearman，无坐标；口径≈现有 `src/topic5_axis_alignment.py::along_axis_sign` 1D side-channel）。
- 主张兑现 = **F 赢过 C**（过自己 null 的余量更大 / 对电极二次抽样更稳）。机制 = narrow 核心通过空间场把单看信不过的噪电极补成可用方向，这是逐通道做不到的。

---

## 3. 预注册结果表（跑前锁死；CLAUDE.md §5 + acceptance-gate 纪律）

| F（场方向对齐） | C（逐通道相关） | 含义 | 行动 |
|---|---|---|---|
| 好 | 差 | 场把噪电极救活了 = 核心信息补充 | **主结果成立** → 进全队列检验 |
| 好 | 好 | 逐通道已够、场没赢 | broad∖narrow 不是合适的隐身度量 → **pivot 换 territory 定义** |
| 差 | 差 | 间期路没延伸到隐身 territory、发作另走他路 | **科学阴性**（发作超出间期骨架），非度量问题，不可与上一格混 |
| 差 | 好 | 逐通道行、场不行 | **场的数学没搭对**（如过度平滑），修方法非改结论 |

"假设不过"必须按这四格区分诊断，跑完不得事后揉成单一"失败"。

---

## 4. Null 与基线

复用 `src/topic5_axis_alignment.py` 4 层 channel-permutation null（`channel` / `within_shaft` / `anchor_matched` / `joint`），打乱发作 onset 向量；**外加**：

- **随机轴 null**：间期方向随机化，F 须过。
- **几何半径基线**：只用"离间期源远近"预测 onset —— F 必须赢过它，证明是**方向**对了不是"近的先亮"。
- **杆向基线**：要求 ≥2 根非平行杆参与，F 必须赢过"沿单根杆方向"的平凡对齐（Topic 4 D6 教训，否则对齐只是杆恰好平行）。
- **坏数据负对照**：打乱真实 onset 再打分应当塌掉。
- **时间对照（持续 vs 发作特异）**：发作前远端窗重测 —— 与加固一致，**不要求**发作特异；持续骨架延伸进隐身 territory 也算回答审稿人，但措辞如实（见 §8）。

---

## 5. 验收阈值（pilot 与全队列分层；跑前锁、跑后不调）

**Pilot（1–2 被试，描述性）**：

- territory 非平凡：broad∖narrow 中**有效靶电极 ≥6**（非 None onset + 有 mapped 坐标 + status 可用）。
- 单被试干净 F-good/C-bad：在 broad∖narrow 上，F 过自己 `channel` null 的 p95 且赢半径 + 杆向基线，而 C 不过。
- 过 pilot 才进全队列。

**全队列（pilot 后，16 Epilepsiae + 适用 Yuquan）**：

- 沿用现有 cohort stats：二项 + Wilcoxon + LOSO-worst-p + BH-FDR。
- F vs C 在 broad∖narrow 上的对照走全队列（如"F 过 null 的被试数"显著多于"C 过 null 的被试数"）。

---

## 6. 数据源与复用图

**复用（不重造）**：

- 平面 / 平滑 / 场相关引擎：`src/propagation_contact_plane_readout.py`（`make_plane_grid` / `smooth_field` / `build_readout_record` / 场相关）——本检验用**带符号**版，不取镜像绝对值。
- 坐标：`src/seeg_coord_loader.py::load_subject_coords`（Yuquan `fs_native_ras_mm` / Epilepsiae `mni152_1mm`；用前 `assert_coord_result_is_mm_for_main_analysis`）。命名：Layer A = monopolar CAR（`AMR1`），t0/lagPat = bipolar-alias-left；喂坐标 loader 前对齐命名。
- null 结构：`src/topic5_axis_alignment.py`。
- phantom mask：broad lagPat rank **必须** mask（`src.lagpat_rank_audit.build_masked_kmeans_features` / `use_masked_features` / `mask_phantom`；broad 跑本就全程 masked）。

**新建 / 改**：

- 间期 broad 方向场：在 broad 池上跑 contact-plane readout（`results/spatial_modulation/propagation_geometry_broad/...` 若已存在则复用，否则按 narrow 同法在 broad 池产 `_t_a.json` 类记录）。
- 发作 onset 场：从 `results/data_driven_soz/layer_a_ictal_er_rank/per_subject/<ds>_<id>.json` 取 `per_er.broad_ER.seizure_records[k].channel_onsets[ch].t_onset_sec`（**用 raw 每发作 onset，不用 producer_health-fail 的 `r_sz` 聚合**）；早=源。
- 带符号场方向对齐统计量 + F vs C 对照 + 半径 / 杆向基线 + §3 结果表判读。

**artifact 关键事实**：

- Layer A `channel_onsets` **覆盖全植入 montage**（1077 sz0 = 121 ch 全植入，102 达阈、19 None）→ 间期隐身、发作招募的电极**有真实 onset**。
- t=0 = 临床 onset；窗 [-120,+30]s；onset = baseline-robust-z ER 上 Page-Hinkley CUSUM 首越（`hop=0.1, win=1.0, pre=300`）。

---

## 7. Pilot 选人

资格 = 有 narrow 轴 + 有 broad（broad∖narrow 非空）+ 有 Layer A onset + 几何干净（两间期源分开 + ≥2 非平行杆）+ broad∖narrow 有效靶 ≥6。

- 候选交集（broad ∩ Layer A onset，Epilepsiae）：139, 253, 442, 548, 583, 590, 635, 916, 922, 958, 1073, 1077, 1084, 1096, 1146, 1150。
- 先挑 1–2 个最干净：建议从 583 / 590（field_concordance 强信号）+ 1077（全 montage 121 ch、territory 大）里选；最终由 broad∖narrow 有效靶数 + 几何（两源分开 + 非平行杆）定，plan 时确认。
- Yuquan 多数 onset 稀（5/9 CUSUM 不触发）+ 部分 narrow 本就 >20（n_added=0、无新 territory）→ Yuquan 作补充非主。

---

## 8. 不主张什么（措辞纪律）

- **不主张"发作早期特异"**（加固时间负对照证其为持续骨架）；只说"间期方向延伸进发作招募的隐身 territory"。
- pilot 是描述性 per-subject，不下队列结论；队列结论须 pilot 过 + 全队列检验过。
- "场赢逐通道"是承重主张，须 §3 表 F-good/C-bad 那格 + 阈值过，缺一不可；不得只凭 F 好就写"场更优"。

---

## 9. 已知风险 / caveat

- z-norm 中后期不可靠 → 只用 onset / 早期时序（本检验正是早期，吻合）。
- producer_health 队列多 fail → 只 gate rank 聚合 `r_sz`，不 gate raw `channel_onsets`；本检验用 raw。
- status 过滤：per-seizure 只用 onset 可用的发作；不足靶电极的发作 drop。
- broad 有几个 Yuquan 被试 n_added=0（narrow 本就 >20）→ 排除出本检验。

---

## 10. Open（plan 时定）

1. 间期 broad 方向场：用现成 `propagation_geometry_broad` 还是新跑（先验证它是否存在 + 是否 masked）。
2. 带符号场对齐的精确定义：场梯度夹角 vs 带符号场相关 —— 二选一并锁。
3. pilot 阈值数值（F 过 null 的具体口径、F-vs-C 余量阈值）。
4. 是否上 LOO（留一电极、用其余 broad 建场预测它）作样本外加固（从 broad 拿掉一点≈内插非外推，不踩范围-精度坑）。
5. 干净 base 分支（当前在 topic4 分支上，须切干净 Topic 5 base 再实现）。
