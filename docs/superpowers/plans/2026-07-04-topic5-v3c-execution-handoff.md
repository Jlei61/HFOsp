# Topic 5 V3c — Execution Handoff (for an autonomous executing agent)

> **Read order**: this doc (WHY + per-step purpose + claim gates) → then the plan
> `docs/superpowers/plans/2026-07-04-topic5-v3c-soz-axis-coverage.md` (HOW: exact code + TDD)
> → the spec `docs/superpowers/specs/2026-07-04-topic5-v3c-soz-axis-coverage-design.md` (contract of record).
> Execute with `superpowers:executing-plans`, INLINE in the current worktree, milestone A→G, small commits.
> Spec+plan committed at `ca866d8` on branch `topic5-v2-phase1`. Nothing implemented yet.

---

## 0. 核心科学目标（朴素话：测什么 / 怎么测 / 想揭示什么）

**测什么。** 把每个电极触点上三张**互相独立**的标签叠起来：① 这个触点在不在病人**平时那条高频小放电的传播路线**上（间期传播轴 `A`）；② 临床医生有没有把它圈成**发作起始区**（`S`）；③ 发作时它**多早被点亮**（招募 latency）。

**怎么测。** 先数**覆盖**——这条轴盖没盖住临床 `S`、往外多盖了多少（`A∖S`），并且盖得是不是**超出"同样植入几何下随机摆一条同样大小的轴"**。再在**同一次发作里**比"多盖出来的触点 `A∖S`"和"轴上的临床 SOZ 核 `A∩S`"谁先被点亮。都跟随机重排比，以病人为单位。

**想揭示什么。** 间期高频网络是不是把临床发作起始区**整个包住、还往外扩了一圈结构化的组织**；那多扩的一圈到底是**和 SOZ 同时亮**（间期网络可能标出临床 SOZ 之外、也参与早期发作的组织 → 接术后 outcome capstone），还是**SOZ 之后才亮**（下游扩散，医生正确地没圈）。**两个方向都是有价值的结果，没有"阴性=失败"。**

**为什么是这个形状（关键背景）**：探针实测发现原设想的问法（"非轴向更多是 SOZ 外"）**按定义就是同义反复**——非轴向的定义是"从不放间期高频"，SOZ 恰是"高频最旺"，两者天然互斥；而且 `A ⊇ S`（9/9 被试轴盖住了全部 SOZ）。所以真正有信息的方向翻成了**覆盖 + 轴外扩**。coverage 承重、latency 门控。

（内部代号：`A`=`is_axis`（间期传播模板成员）；`S`=SOZ JSON∩all-clean 池；`A∩S`/`A∖S`/`S∖A`=覆盖/外扩/漏掉；latency=`bb_zt` 首次超阈。）

---

## 1. 三个不可动摇的执行纪律（违反=科学污染）

1. **coverage 是 primary，永不依赖 latency**。coverage 只用空间标签 + 空间 null。latency 塌了不影响 coverage。
2. **latency 是 label-blind-assay-QC 门控的 secondary**。QC **只看 latency 这个量本身可不可信（不看 SOZ/surplus 组差）**；QC 不过 → latency 降 descriptive-only，不进 cohort 推断。这防 data-dependent endpoint selection。
3. **broad primary / narrow sensitivity，永不 pool**。分目录、分 verdict。narrow(n=3) 只作方向/effect-size 一致性，不作独立推断。

外加：**集合语言 A/S 全程**，不用 "off-axis" 指 `¬A` 或 `S∖A`；**subject-first**（禁 pooled channel-level p）；**tier=exploratory，无 forecasting**；**outcome=future/blocked**（Yuquan 标签未到）。

---

## 2. 每一步的目的（milestone → task；HOW 在 plan，这里只讲 FOR-WHAT）

### Milestone A — 覆盖的地基（config + 集合代数 + 空间 null）
- **Task 1 config `v3c:` 块** — 所有阈值/门的单一真相源；每个下游 gate 从这里读，不散落魔数。
- **Task 2 `coverage_metrics`** — 集合代数核心（`A∩S`/`A∖S`/`S∖A` + coverage/surplus/jaccard）。primary 的全部原料。**R1 提醒**：broad 轴近固定大小≈20，`surplus_fraction` 半机械、只作描述符，不承独立权重。
- **Task 3 `coverage_null_distribution`** — same-shaft 保每杆轴数的 null。这是把"轴碰到 SOZ"变成"**轴覆盖 SOZ 超出植入几何**"的科学牙齿。复用 `label_permute`（不重造）。

### Milestone B — V3c-1 覆盖结果（primary 交付）
- **Task 4 SOZ loader + `axis_soz_join`** — 把三张独立标签接起来（轴/SOZ/池）。join 就是这条线活着的地方；SOZ 名字与 cache 同名空间（monopolar），直接交。
- **Task 5 覆盖 runner** — 产出 primary：per-subject coverage + 自己 null 分位 + cohort-median null + LOSO(n≥2 守卫)。这是 V3c-1 的交付物。

### Milestone C — latency 测量的可信度门（label-blind）
- **Task 6 `first_crossing_latency`** — latency 测量原语，带 finite/t0/censored 三分。**t0/censored 的比例本身就是结果**（告诉你 latency 到底能不能用；1077 有 56% t0 就是被这个抓出来的）。
- **Task 7 QC 纯函数** — 判 latency 是不是有效 assay 的 label-blind 门。**签名不接受 `S`**（编译期防 SOZ 泄漏）。
- **Task 8 `extract_latency_matrix`** — 从 cache 抽 per-contact latency，**缺触点 raise**（P1-4：防把一个触点的 latency 静默安到另一个触点上）。
- **Task 9 assay-QC runner** — 在轴触点上跑 label-blind QC。它的输出**决定 V3c-2 是 mechanistic 还是 descriptive-only**。这是门，不是结论。跑完对照 spec §附录 A.2（1077 t0≈56% 该 fail、1150 cens≈44% flag）。

### Milestone D — V3c-2 latency（门控 secondary）
- **Task 10 `auc_late` + `delta_t` + null** — 时序统计量：秩 AUC（对 censoring/cell-size 稳）+ 秒级 effect size + within-shaft null。
- **Task 11 latency runner** — 门控 secondary 结果：`AUC_late`(A∩S vs A∖S 主 / S vs A∖S 临床敏感) + 带符号 `Δt` + **drop_censored/exclude_t0 sensitivity + `sensitivity_concordant`**。只在 assay-QC 过的被试上有意义。`latency_cohort.json` 的字段是 Task 14 的合同（缺 → Task 14 fail-closed）。

### Milestone E — V3c-3 外扩空间组织（覆盖双条件的第二腿）
- **Task 12 `surplus_spatial_metrics` + distance null** — `A∖S` 是不是空间有结构（贴 SOZ）vs 弥散过泛化。
- **Task 13 surplus-spatial runner** — 产 `surplus_spatial_cohort.json`（cohort-median 距离 null + `n_spatial_eligible` + LOSO n≥2）。**喂 coverage 双条件**。

### Milestone F — 判决 + 图
- **Task 14 summary** — 判决引擎，**"能跑"变"能 claim"的唯一闸口**：coverage 双条件（`coverage_primary_pass=coverage_sig ∧ spatial_sig`，spatial 还要 `n_spatial_eligible≥3`）、latency 四分类、措辞按 null 授权分级。**这一步的正确性 > 前面所有 runner**。
- **Task 15 图** — paper-grade forest（覆盖/AUC，带 null 区间）+ QC 三图。**先 render→用户目视→再 commit**（memory: figure self-contained）。

### Milestone G — 全队列跑 + 文档
- **Task 16** — broad+narrow 全跑 + archive doc（§8 三段式 abstract，**先 invoke `hfosp-plain-language-recap`**）+ 主文档 §3.9 指针 + FIGURE_INDEX + **回填 spec §附录 A.1 的 1146 PENDING 行**（用真实 `broad/coverage_subject.csv`）。

---

## 3. 每个输出授权什么话（claim gate；越界=禁止 claim）

| 输出条件 | 能写 | **不能写** |
|---|---|---|
| `coverage_null_p < 0.05`（单独） | "轴覆盖 SOZ **超出植入几何**" | "超出 HFO 富集"（要 HFO-rate null，是 follow-up、未实现） |
| `coverage_primary_pass=True`（coverage_sig ∧ spatial_sig, n_spatial≥3） | "**specific axis-SOZ spatial organization**" | 若 spatial 不过：不得写"结构性/特异" |
| latency `H-B_supported` | "surplus 在 SOZ 核之后招募 = 下游 scaffold" | — |
| latency `H-A_compatible` | "**compatible with** onset-synchronous（描述性）" | "证明了同步/等效"、"医生漏掉真 SOZ"（n=6 无功率做等效检验，R3） |
| latency `surplus_earlier_unverified` | "低延迟尾，**待 t0 伪影核查**，unverified" | 任何机制解释 |
| latency `indeterminate` | "latency 未分辨清" | 把 nonsig 当 H-A |
| narrow 任何结果 | 方向/effect-size 一致性 sanity | narrow 作独立 inferential cohort |

`S∖A`（如 635 的 3 个）单独画、单独解释（轴漏掉的临床 SOZ），不是噪音。措辞用中性 `axis-surplus / A∖S`，**证前不用"那圈/peri-SOZ"**。

---

## 4. 不要重新引入的 5 个已修 bug（审阅 P1-1..P1-5）

执行时如果你"简化"掉下面任何一条，就是把已修的科学污染放回去：

1. coverage 主张**不能只靠 p**——必须双条件（Task 14 `coverage_primary_pass`）。
2. `delta_t_med` 缺失**不能默认 0**——Task 14 `_require()` fail-closed。
3. H-B **用带符号** `delta_t_med >= +thr`，**不是 `abs`**（surplus 更早是另一类 `surplus_earlier_unverified`）。
4. `extract_latency_matrix` 缺触点**必须 raise**，不能 filter rows 后按全 names 索引（静默错配）。
5. censor/t0 sensitivity（`drop_censored`/`exclude_t0`）**必须算并进 `sensitivity_concordant`**；H-B 要求它 True。

（HFO-rate-matched null 是**明确 scope cut**、非首轮——在它建成前 coverage 措辞封在"beyond implantation geometry"。）

---

## 5. 执行机制（autonomous）

```bash
# 每 milestone 内逐 task TDD：写失败测试 → 跑红 → 最小实现 → 跑绿 → commit（小提交已授权）
# Milestone A→B（无外部 mount 依赖的纯函数/单被试）先跑通；integration 测试需要 cache mount。

# 全 cohort 跑（Task 16）：
for c in broad narrow; do
  python scripts/run_topic5_v3c_coverage.py --cohort $c
  python scripts/run_topic5_v3c_latency_qc.py --cohort $c
  python scripts/run_topic5_v3c_latency.py --cohort $c        # gated on assay-QC csv
  python scripts/run_topic5_v3c_surplus_spatial.py --cohort $c
  python scripts/run_topic5_v3c_summary.py --cohort $c
  python scripts/plot_topic5_v3c.py --cohort $c
done
pytest tests/test_topic5_v3c_coverage.py tests/test_topic5_v3c_latency.py tests/test_topic5_v3c_io.py -v
```

- **数据底座**：轴/池=`classify_subject_contacts(ds,"broad"/"narrow",cfg)`；latency=`results/topic5_ictal_recruitment/ictal_field_long_cache/<ds>.npz`（`bb_zt__{si}`/`bb_relt__{si}` + meta `eeg_onset_rel`）；SOZ=`results/{epilepsiae,yuquan}_soz_core_channels.json`；坐标=`seeg_coord_loader.load_subject_coords`（缺 MRI→`{}`，V3c-3 退 shaft-only）。
- **cohort**：broad=139/253/635/1077/1096/1150/1146（442/958 无 broad cache→narrow only）；narrow=1096/1146/253/442/958。broad/narrow 成员**部分不同、非严格子集**——README 分开标。
- **产物根**：`results/topic5_ictal_recruitment/v3c_soz_axis_coverage/{broad,narrow}/`，图目录必带中文 `figures/README.md`。
- **收尾**：实现完 → `superpowers:requesting-code-review` → 再决定 merge/PR。**不要**自行 push 或 merge，除非用户点头。

---

## 6. 停下来问用户的情形

- SOZ 名单 / cache / 坐标 mount 缺失且非预期（预期缺失=graceful skip，记 reason，别静默丢被试）。
- 出现 spec §9 禁止 claim 边界的判断分歧（措辞该封在哪一档）。
- 需要动 spec 合同（改阈值/改 primary 定义）——spec 是合同，改前问。
- Task 16 的 1146 回填数值与 spec 附录估计冲突时——以执行值为准，但记录差异。
```
