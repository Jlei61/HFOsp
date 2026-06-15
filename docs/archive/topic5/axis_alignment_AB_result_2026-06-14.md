# Topic 5 主线结果归档：间期传播轴 ↔ 发作早期激活的"轴对齐"（A 线 primary + B 线 secondary）

> 日期：2026-06-14 · 状态：**A+B 已执行，是 network-axis pivot 阶段唯一有队列结论的部分**
> 主队列：Epilepsiae 18 被试（T0 合格 354 发作）。Yuquan 结构性只有 1 被试有合格发作，不成队列。
> 上游计划：`docs/archive/topic5/network_axis_pivot_plan_2026-06-13.md`（A/B/C/D/E 全貌）
> 定稿数值表：`results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.md`

---

## 0. 白话摘要（§8 三段式）

**测了什么。** 把每个病人"平时"和"发作时"的两样东西摆到同一张脑内电极平面上比：
平时（无发作）那些高频小事件在各触点上的**发放先后顺序**（哪个触点平均最早响、哪个最晚），
连起来是一条"传播轴"；和真正**发作头十秒**里各触点"烧"得多旺（能量高低）。
问：这两个空间梯度（哪边早 ↔ 哪边旺）是不是落在同一条线上。

**怎么测的。** 如果两者毫无关系，平时的传播轴跟发作激活在空间上应该对不上、相关接近 0。
我们算两者"不分正负、只看是否共线"的相关，再跟四种一层比一层严的"洗牌重排"比：
(1) 把发作激活在**所有触点之间**随机洗牌——看有没有**任何**粗的共享空间结构；
(2) 只在**同一根电极杆内部**洗牌——看对齐是否比"哪根杆"还细；
(3) 在**基线活跃度相近**的触点之间洗牌——排除"本来就活跃的触点"这个混淆；
(4) **同时**按杆 + 按活跃度洗牌——最严。18 个 Epilepsiae 病人各自先跟自己的随机分布比，
再做队列层面的二项 + 符号检验 + 留一稳健 + 多重比较校正。

**揭示了什么。**
- **粗层面（有没有一根共享网络主轴）——稳，是真的。** 四种激活量都明显赢过"全触点随机洗牌"，
  多重校正和留一都扛得住。
- **细层面（比电极杆/活跃度还细的对齐）——看用哪种激活量，不普遍成立。** 只有"**快活动 60–100 Hz**"
  能稳稳赢过最严的"同杆 + 同活跃度"联合洗牌；主指标宽带功率只在粗层稳。
- **边界：这是符号自由的"轴/梯度共线"，不是"发作沿平时顺序逐个触点重放"。**

一句话：**"平时传播主轴和发作早期激活共享一根粗网络骨架"——稳；"对齐细到比解剖杆和活跃度还精确"
只在快活动量上稳，值得单独再固化验证一次。**

（内部归档代号：A 线 = mirror-invariant |axis_alignment|；4 nulls = channel / within_shaft /
anchor_matched / joint；指标 = broadband[PRIMARY] / hfa[sensitivity] / ramp[sensitivity] / ei[B 线 exploratory]。）

---

## 1. 测的统计量（A 线主检验）

对每个被试：

- **间期轴**：从已成熟的传播-几何观测层取该被试的间期模板 A 排名场（`*_t_a.json` 的
  `typical_rank` ∈ [0,1]，每触点平均发放先后），即"传播轴"的逐触点标量。
- **发作激活**：从 T0 特征缓存取该被试所有合格发作头 10 s 的逐触点激活，跨发作取均值。
- **统计量** `real = | corr_pair_mirror_invariant(interictal_rank_field, ictal_activation_field) |`
  —— 镜像不变 = 符号自由 = **只判共线/共轴，不判方向重放**。反向共线（间期早端 ↔ 发作高激活端）
  仍记为对齐。

**预注册纪律（计划 §六）：primary 只有 A 一条（broadband），其余全 exploratory / sensitivity。**

## 2. 四层 null（彼此独立，不是嵌套）

| null | 洗牌方式 | 控制什么 / 回答什么 |
|---|---|---|
| `channel`（粗） | 发作激活在全部触点间随机置换 | 有没有**任何**共享的粗空间结构 / 网络主轴 |
| `within_shaft`（精细） | 只在同一电极杆内部置换 | 对齐是否比"哪根杆"的解剖更细 |
| `anchor_matched`（活跃度） | 在基线活跃度分箱（4 bin）内置换 | 排除"高基线活跃度触点"混淆 |
| `joint`（联合，最严） | 同杆 × 同活跃度箱内同时置换 | 两个混淆一起卡 |

**每被试 null = 每个重排实现里"跨发作取中位"，得到 B 个实现的分布，再取 95 分位**
（不是先把每发作的 B 次塌成中位再 p95——那是早期版本的 bug，已修正，现版本 null 更严）。
`effective_shuffle_n` 标记某层 null 实际能洗动的通道数；< 4 视为退化，不计入 `n_adequate`。

## 3. 队列统计（每层 null 各算一遍）

- **二项**：18 个被试里"真值高过自身 95 分位"的人数 vs 5% 偶然率。
- **Wilcoxon**：真值 − null 中位，整体是否偏正。
- **留一（LOSO）**：去掉任一被试后的最坏 p。
- **BH-FDR**：对整族 metric × null 的 Wilcoxon p 做 Benjamini-Hochberg 校正。

## 4. 冻结参数（T0 eligibility，计划 §四）

```
baseline      = [-90, -60] s（≥30 s 干净，guard=[-60,0]，eeg-onset-aware）
pre 窗自适应  = max(120, min(|eeg_rel|, 300) + 120) s
montage_frac  = 0.80    min_ch = 6    min_gap_prev = 300 s
broad_band    = 1–45 Hz    hfa_band = 60–100 Hz
T0 合格       = 18 epilepsiae 主队列 / 354 合格发作（yuquan 仅 1 被试合格，不成队列）
```

---

## 5. 结果（定稿表 `axis_alignment_FINAL.md`，B=1000；B=2000 同号）

`n_pass/n` = 高过该层 95 分位的被试数 / 18；`adeq` = null 实际洗 ≥4 通道的被试数。

| 指标 | 档 | null | n_pass/n | adeq | 二项 p | Wilcox p | FDR q | LOSO-worst p |
|---|---|---|---|---|---|---|---|---|
| broadband | PRIMARY | channel | 5/18 | 18 | 0.0016 | 0.0080 | **0.0196** | 0.0153 |
| broadband | PRIMARY | within_shaft | 4/18 | 18 | 0.011 | 0.049 | 0.066 | 0.087 |
| broadband | PRIMARY | anchor_matched | 5/18 | 18 | 0.0016 | 0.077 | 0.091 | 0.13 |
| broadband | PRIMARY | joint | 0/18 | 13 | 1.0 | 0.75 | 0.75 | 0.86 |
| hfa | sensitivity | channel | 8/18 | 18 | ~0 | 0.0014 | **0.0075** | 0.0028 |
| hfa | sensitivity | within_shaft | 8/18 | 18 | ~0 | 0.0028 | **0.0088** | 0.0055 |
| hfa | sensitivity | anchor_matched | 8/18 | 18 | ~0 | 0.0005 | **0.0075** | 0.0011 |
| hfa | sensitivity | joint | 3/18 | 13 | 0.058 | 0.016 | **0.029** | 0.028 |
| ramp | sensitivity | channel | 9/18 | 18 | ~0 | 0.0012 | **0.0075** | 0.0023 |
| ramp | sensitivity | within_shaft | 4/18 | 18 | 0.011 | 0.024 | **0.037** | 0.044 |
| ramp | sensitivity | anchor_matched | 5/18 | 18 | 0.0016 | 0.0033 | **0.0088** | 0.0064 |
| ramp | sensitivity | joint | 3/18 | 13 | 0.058 | 0.32 | 0.34 | 0.46 |
| ei (B 线) | exploratory | channel | 7/18 | 18 | ~0 | 0.0033 | **0.0088** | 0.0064 |
| ei (B 线) | exploratory | within_shaft | 4/18 | 18 | 0.011 | 0.024 | **0.037** | 0.044 |
| ei (B 线) | exploratory | anchor_matched | 6/18 | 18 | ~0 | 0.013 | **0.029** | 0.025 |
| ei (B 线) | exploratory | joint | 0/18 | 13 | 1.0 | 0.039 | 0.055 | 0.070 |

**读表口径（q < 0.05 记"稳赢"）：**
- **broadband（主）**：只稳赢 `channel`（粗层）；`within_shaft` / `anchor_matched` / `joint` 都过不了。
- **hfa（快活动，灵敏度）**：四层全稳赢，含最严 `joint`（q=0.029）——细对齐**唯一**全层稳的量。
- **ramp（爬升斜率）**：赢 `channel` / `within_shaft` / `anchor_matched`，不赢 `joint`。
- **ei（B 线）**：赢 `channel` / `within_shaft` / `anchor_matched`，不赢 `joint`。

**强例 |r|（broadband B1000，眼看证据）**：E583=0.90、E590=0.88（反向共线）、E916=0.70；
弱例 E384=0.30、E1096=0.36（连最粗都不赢）。

---

## 6. 允许 / 禁止的措辞（写论文 / 主文档时照搬）

**允许：**
- "间期传播主轴与发作早期激活的空间梯度**共享一根粗的网络骨架**"（channel 层稳，FDR + LOSO 扛住）。
- "**符号自由的轴/梯度对齐**"——共线即对齐，含反向共线。
- "比解剖杆/活跃度更细的对齐**只在快活动（60–100 Hz）上稳健**，宽带主指标止于粗层"。

**禁止：**
- ❌ "发作沿间期传播路线逐点重放"——本统计是符号自由共线，不支持逐点重放。
- ❌ 把 B 线（ei）或 hfa（灵敏度）写成 **primary cohort claim**——primary 只有 broadband 一条；
  B/hfa/ramp 是次级 / 灵敏度读出。若将来要把 hfa 升为 co-primary，必须提前重做 alpha 分配。
- ❌ "宽带主指标证明了细对齐"——宽带只过粗层。

---

## 7. 工件清单

**代码（纯函数 + TDD）**
- `src/topic5_t0_features.py`（6 tests）— onset 窗、激活均值、ramp 斜率
- `src/topic5_axis_alignment.py`（10 tests）— 4 类 shuffle + matched_channels + effective_shuffle_n
- `src/topic5_ei.py`（4 tests）— EI-like（延迟在分母 = 惩罚项）

**驱动 / 缓存**
- `scripts/run_topic5_t0_eligibility.py` — T0 audit（`--resummarize` / `--restart`）
- `scripts/build_topic5_t0_feature_cache.py` — per-subject npz（`--augment-baseline` / `--add-ei`）
- `scripts/run_topic5_axis_alignment.py` — A 线统计（`--activation` / `--B`）
- `scripts/aggregate_topic5_axis_alignment.py` — 分档 + FDR + n_adequate → `axis_alignment_FINAL.{json,md}`
- `scripts/run_topic5_axis_sweep_v2.sh` — 4 metric × 2 B 全扫 → 图 → 聚合

**图**
- `scripts/plot_topic5_field_concordance.py` — **field 主结果论文图（被试级，非 cohort 平均）**：
  ① `field_concordance_atlas_broadband.png`（Panel A 示意 + 18 病人 paired-field 墙，左间期 order 场/
  右发作 activation 场、最佳符号镜像、深框=过自己粗 null、按 margin 排序）；② `field_concordance_null_forest_broadband.png`
  （实测 |r| vs 每病人自己 channel-shuffle null，5/18 黑过 null95%，E590/E583 远超）。**四层 null 阶梯退 supplement。**
- `scripts/plot_topic5_axis_alignment.py` — 队列条带图（真值 vs 4 层 95 分位）
- `scripts/plot_topic5_axis_alignment_fields.py` — 每被试两面板场图（同一 mm 平面、同一套 viridis、
  符号对齐到轴）；左 = 间期传播顺序（模板 A），右 = 发作激活（标注用哪种激活量计算）
- `scripts/plot_topic5_null_ladder.py` — **四层 null 阶梯图**：行 = 4 层 null 由松到严、列 = 4 指标，
  绿格 = FDR 校正后稳赢；一张图读出"主结论 = 粗骨架（rung 1）；只有 HFA 爬到最严 rung 4"的层级。
- `scripts/plot_topic5_axis_direction_rose.py` + 纯函数 `src/topic5_axis_direction.py`（13 tests）
  — **方向玫瑰图**：发作激活方向轴向均值归一化 0°/180°（黑线、双向），两模板逐事件方向画成整圆空心
  直方图；ECoG/SEEG 由 Epilepsiae SQL `electrode_array.type` 判定。**双重身份（不冲突）：A 线的方向补充
  （A 线主体是 field 相关）+ C 线主流图（逐发作方向按子型上色）**；忠实的 A 线对齐图仍是场图（见下"方向 vs 场"注）。
- `scripts/plot_topic5_axis_direction_aline.py` — **方向版 A 线图（仅干净被试 ≥8 触点且 R_axial≥0.6）**：
  间期轴用 typical-rank 场梯度（A 线真正的间期轴）对发作轴，Δ=方向夹角，标 A 线场 \|r\|；队列散点
  `Δ vs |r|`（broadband Pearson r=−0.77, n=6）显示干净被试上方向与场一致。`figures/aline_direction/`。

> **方向 vs 场（2026-06-15 审阅澄清，重要）**：A 线统计 `|corr_pair_mirror_invariant|` 是 **y-镜像不变的
> 场值相关**（折掉横向），**不是梯度方向比较**。所以场 \|r\| 高 ≠ 梯度方向一致：干净被试（922、zhangkexuan）
> 二者一致（Δ≈1°、\|r\| 高），少触点/横向被试（583 \|r\|=0.90 但 Δ=52°；590-hfa Δ=89° 但 \|r\|=0.63）二者
> 背离。**忠实展示 A 线对齐的是场图 `fields/`**；方向玫瑰是 C 线方向诊断；方向版 A 线图只在干净被试上画并
> 显式标注镜像不变 caveat。**被检验、显著的量永远是场 \|r\|（FINAL 表），不是方向角 Δ。**

**结果根**
- `results/topic5_ictal_recruitment/axis_alignment/`
  - `axis_alignment_FINAL.{json,md}` — 定稿
  - `axis_alignment_{broadband,hfa,ramp,ei}_B{1000,2000}.json` — per-metric per-B
  - `figures/` + `figures/fields/` — 含 README.md（中文逐图）
- `results/topic5_ictal_recruitment/t0_feature_cache/` — per-subject npz/json
- `results/topic5_ictal_recruitment/t0_eligibility_{audit.csv,summary.json}`

---

## 8. Handoff（交接）

### 已完成（可引用）
- A 线（broadband，primary）+ B 线（ei，exploratory）+ 灵敏度（hfa / ramp）全部跑完、定稿入表。
- 结论：**粗网络骨架共享 = 稳**；**细对齐 = 仅快活动量稳**；**符号自由共线 ≠ 逐点重放**。
- 全部纯函数有 TDD；4 层 null 含退化标记；队列统计含 FDR + LOSO。
- 每被试场图 + 队列图就绪，README 中文齐。

### 下一步候选（按优先级）
1. ✅ **已做（2026-06-15）：hfa×joint 冻结复验 + split-half + 负对照** →
   `docs/archive/topic5/hfa_joint_confirm_2026-06-15.md`。**结果 = 不升格**：full 干净复现
   （joint Wilcox=0.022）、偶数半显著（0.035）、**奇数半不显著（0.078）→ split_half_robust=False**；
   负对照四层全部打回非显著（joint=0.95）确认非假阳性。**hfa×joint 维持灵敏度档（real-but-not-robust），
   升 primary 需独立第二队列。** 主线（粗骨架）不受影响。
2. ✅ **已做：A 线场图全量重渲 + README**——19 被试 `*_axis_vs_broadband.png`，左标"模板 A"、
   右标激活量；另加**四层 null 阶梯图** + **方向玫瑰图**（19 每被试 + 队列汇总），见 §7 图清单。
3. ✅ **已做（2026-06-15）：C 线（子型修饰）** → `docs/archive/topic5/subtype_direction_cline_result_2026-06-15.md`。
   **结果 = 队列不可行 + 无信号**：18 跑→14 status_ok→**仅 2 个**（E548/E583）几何+成簇+子型三门同过能进检验，
   且子型轴角分离 0/2 显著（broadband/hfa 一致）；C↔A 机制连接（子型轴角异质↔A 线奇偶不稳）无可见关系。
   修复了审阅 5 点（轴角/极性两层分离 + A 线连接定量三标量 + Step-0 成簇门 + SEEG/近一维几何门 + seizure_id 对齐合同）。
   与主线一致：共性是粗网络锚，不是细到子型可分的方向重放。
4. **D 线（发作前漂移）**：planned，未执行。先验低；阴性也有价值（说明刻板间期事件非短时程预警）。
5. **E 线（Yuquan 触点级结局）**：预测端就绪，**结局标签卡在医院随访数据**（Engel/ILAE/复发不在 repo、
   病例文档只到术后 24h）。关键路径，等标签。

### 未决 / 边界
- 主队列只有 Epilepsiae（18 被试）；Yuquan 结构性只 1 被试有合格发作，**不能形成第二队列做外部验证**。
- hfa 细对齐若要升 co-primary，须提前 alpha 分配，不可事后挑主终点。
- "共享网络骨架"是结构性共性，**不蕴含因果驱动**（计划 §2.1：间期既可促也可抑发作）；
  本结果不支持"间期驱动发作"的因果叙事。
