# Topic 5 C 线结果归档：发作子型 ↔ 发作激活方向（病人内 modifier，exploratory）

> 日期：2026-06-15 · 状态：**已执行（全队列 broadband + hfa）** · 层级：**exploratory 病人内 modifier，非队列主张**
> 计划：`docs/superpowers/plans/2026-06-15-topic5-C-line-subtype-direction-plan.md`（v2，审阅修复版）
> 上游：A 线主线 `axis_alignment_AB_result_2026-06-14.md` + hfa×joint 复验 `hfa_joint_confirm_2026-06-15.md`
> 代码：`src/topic5_subtype_direction.py`（纯函数，24 tests）+ `scripts/run_topic5_subtype_direction.py`
> 结果：`results/topic5_ictal_recruitment/subtype_direction/`（per-subject JSON + `cohort_summary_{broadband,hfa}.json` + figures）

---

## 0. 白话摘要（§8 三段式）

**测了什么。** 同一个病人，每次发作往脑子哪个方向"烧"起来（从触点平面上的能量梯度拟一个方向角）。问：不同种类的发作
（已有的 z-ER 子型分类），方向是不是不一样。这里"方向不同"分两种、必须分开：(1) **那根轴转没转**（能解释 A 线时强时弱）；
(2) **轴没转、只是从这头烧到那头掉了个个儿**（这种掉头对 A 线没影响，只能描述）。

**怎么测的。** 每个病人把子型标签在自己的发作之间随机打乱两千次，看实测的"子型间方向差"能不能超过随机。两层分开：
轴角层（θ 与 θ+π 算同一根轴）、极性层（保留掉头）。进检验前还设了三道门：子型必须 status 干净、丢掉只有 1–2 次的小子型后还得剩≥2 个各≥3 次、
发作方向本身得先成簇、电极几何不能是 SEEG 或近一维（否则方向角不可靠）。对齐用"发作身份号"连接（不按位置盲配）。
还把"子型轴角分离"和"A 线奇偶分半的不稳程度"放进一张图直接对照。

**揭示了什么。** **C 线作为队列主张不可行，作为病人内描述也没看到信号。** 18 个跑过的病人里，14 个有干净子型标签，
但只有 2 个（E548、E583）几何够干净、方向成簇、子型够多能真正进检验——**2 个等于 2 个个案，凑不成队列**。这 2 个里
子型的轴角差都不显著（宽带、快活动各 0/2）。更广地看，绝大多数病人出局是因为：要么只有 1 个有效子型（丢掉小子型后），
要么发作方向根本不成簇，要么是 SEEG / 近一维电极（方向角不可靠）。把"子型轴角越异质"对"A 线越不稳"画出来也**看不出关系**
（轴角分离最大的那个病人 A 线不稳只算中等；轴角分离≈0 的那个反而最不稳）。
**这不是"证明子型不影响方向"，是"在我们够得着的精度内没看清、且根本凑不出队列"。** 与 A 线、Stage 1/2/2b 一致：
共性是粗的解剖/网络锚，不是细到子型可分的方向重放。

（内部归档代号：C 线=subtype×activation-direction；Q_axis=轴向 axial separation perm / Q_pol=极性 circular separation perm；
对齐=seizure_id via `t0_eligibility_audit.csv`；几何门 `coord_aspect<0.15` 或 SEEG；clustering 门 `R_axial≥0.5`；
z-ER `per_band[gamma_ER].subtype_label` status=ok；A 线不稳=hfa×joint `|corr_even−corr_odd|`，`split_half_robust=False`。）

---

## 1. 设计要点（审阅 5 点修复全部落地）

| 审阅点 | 修复 | 落点 |
|---|---|---|
| 1. 轴向统计抹掉 180° 反向 | 拆成 **Q_axis（轴角，连 A 线）+ Q_pol（极性，仅描述）** 两层，各自独立置换 | `subtype_separation_stat(mode=axis/pol)` + 回归测试 `test_axial_collapses_180_but_polarity_does_not` |
| 2. A 线连接不是检验 | 每被试 **三标量同表**：子型轴角分离 T_axis/p × 奇偶子型不平衡 TV × A 线奇偶不稳 \|corr_even−corr_odd\| | `aline_connection()` + figure C；**A 线奇偶分半口径复用 `seizure_parity_subsets(cache eligible_idxs)`**（与 hfa×joint 复验同一切分） |
| 3. 玫瑰图先验被过读 | 加 **Step-0 成簇门**（逐发作方向先得成簇才检验）；图 A 画**逐发作 tick**不是间期模板瓣 | `direction_clustering()`；实测证实方向普遍不成簇 |
| 4. SEEG/近一维几何脆 | 加 **几何门**：电极类型 + 触点云二维纵横比 `coord_aspect`；SEEG 或 aspect<0.15 → case-series | `coord_aspect_ratio()` + `_electrode_kind` |
| 5. 对齐别重造轮子 | 复用 `load_topic5_subtype_labels`（硬 `status==ok`），按 **seizure_id** 经 audit CSV 连接，零重叠 fail-loud | `align_subtype_to_direction()`（namespace-mismatch raise，禁止位置盲配） |

- **只有 Q_axis 有资格解释 A 线**（A 线符号自由，纯极性掉头不影响），Q_pol 仅描述。
- C 线 = epilepsiae-only（z-ER 无 yuquan extractor）。

## 2. 队列结果（band=gamma_ER，B=2000，seed=20260615）

n_run=18 → status_ok=14（1077 `insufficient_n`；1125/384/620 无 z-ER JSON）→ **cohort-eligible=2**。

| id | 档位 | 几何(类型/aspect) | 成簇 | perm 子型 | Q_axis(broadband) | 出局原因 |
|---|---|---|---|---|---|---|
| 548 | **cohort-test** | ECoG/0.51 | 成簇 | k2(丢[3,4]) | p=0.71 | — |
| 583 | **cohort-test** | ECoG/0.90 | 成簇 | k2(丢[2]) | p=0.49 | — |
| 916 | case-series | SEEG/0.48 | 成簇 | k2 | p=0.90 | SEEG |
| 922 | case-series | ECoG/0.11 | 成簇* | k2 | p=0.91 | 近一维(aspect=0.11) |
| 958 | case-series | ECoG/0.58 | 不成簇 | k2(丢[2]) | p=0.68 | 方向不成簇 |
| 1084,442 | case-series | ECoG | — | 1 子型 | — | 仅 1 有效子型 |
| 1096,1146,253,1150,139,590,635 | case-series | SEEG | 多不成簇 | 1 子型 | — | SEEG + 子型不足/不成簇 |

\* 922 "成簇"是近一维电极伪影（tick 全压在电极线方向），几何门正确拦下。

**verdict（两套激活一致）：cohort-eligible 2/14；其中 Q_axis p<0.05 = 0/2（binom-vs-5% 参考 p=1）。** hfa 版同样 0/2
（548 T=0.43 p=0.22、583 T=0.13 p=0.54）。

## 3. A 线连接（机制链：子型轴角异质 → A 线不稳？）

5 个能算 T_axis 的被试（2 实心合格 + 3 空心 caveat）画在 figure C：**T_axis 与 A 线不稳无可见关系**
（E958 T_axis≈1.38 最大但不稳中等 0.065；E922 T_axis≈0.02 却不稳最高 0.077；2 个合格点 E548/E583 都在低 T_axis 区）。
→ "子型轴角异质解释 A 线 split-half 不稳" 这条候选机制**不被支持**（且合格点太少，本就不足以做队列连接）。

## 4. 允许 / 禁止措辞

- **允许**："C 线在本队列无法成立——只有 2 个被试几何+成簇+子型都够进检验，且都没看到子型轴角分离（exploratory、描述性）"；
  "子型轴角异质未见与 A 线奇偶不稳相关，A 线 split-half 不稳更可能是样本量/噪声而非子型方向异质"。
- **禁止**：把 Q_axis 与 Q_pol 混成"方向随子型变"（必须指明轴角 / 极性）；用 Q_pol 解释 A 线；把 n=2 写成队列结论；
  "证明子型不影响方向"（实际是没看清 + 凑不出队列）；把 C 线升 primary。

## 5. 工件

- 纯函数：`src/topic5_subtype_direction.py`（`axial_distance` / `circular_distance` / `subtype_separation_stat` /
  `within_subject_perm_p` / `direction_clustering` / `coord_aspect_ratio` / `align_subtype_to_direction` /
  `oddeven_subtype_imbalance`）— `tests/test_topic5_subtype_direction.py` 24 tests 全绿。
- 驱动：`scripts/run_topic5_subtype_direction.py`（`--pilot` / `--subjects` / `--activations`）。
- 结果：`results/topic5_ictal_recruitment/subtype_direction/`：per-subject `*__subtype_direction_{broadband,hfa}.json`、
  `cohort_summary_{broadband,hfa}.json`、`figures/`（30 张被试玫瑰 + 2 张 C↔A 连接 + 中文 README）。

## 6. Handoff

- C 线**到此为止**：队列不可行（合格 2 个）、无信号、机制连接不成立。不建议继续 squeeze（换方向定义/换激活都不会把
  "只有 2 个被试够格"这个结构性瓶颈解决——瓶颈是子型够多 + 方向成簇 + 非 SEEG/近一维 三者同时满足的被试太少）。
- 与 network-axis pivot 其它线一致的总图景：**共性是粗网络/解剖锚（A 线 channel 层稳），不是细到子型/路径可分的方向重放**。
- 关键路径仍是 **E 线（Yuquan 触点级结局）**，卡在医院随访标签。
