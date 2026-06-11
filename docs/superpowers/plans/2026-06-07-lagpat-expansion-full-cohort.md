# 全 cohort lagPat 扩展计划（2026-06-07）

> 状态：plan draft，pilot-validated，执行前需 advisor review（§11 gate）。
> 上游：`docs/superpowers/plans/2026-06-06-mechanism-paper-clinical-outcome-strategy.md`（§5.4 覆盖率 size-confound + "跳出 rate/SOZ 需更广搜索"）。
> 动机：现有全部模板都建立在 lagPat refine（按 HFO 率 `mean+k·std` 选通道）选出的 ~4–16 通道核心上 → 模板结构上无法证明携带 rate/SOZ 之外信息（核心 ⊆ focus）。本计划用**放宽通道选取 + 逐通道可复现 + 解剖对照**，系统性检验"传播结构是否延伸到 rate 核心之外、捞回紧模板漏掉的致痫网络"。

---

## 0. 一句话（main question — user reframe 2026-06-07）

原 lagPat 通道池太窄(只看 SOZ/高率核心 ~4–16),**无法问"传播模板在 SOZ 内外怎么铺开、是否覆盖临床网络、能否支撑更大的 KMeans 模板寻找"**。把通道池放宽到 ~20(top-N by event count)重算模板,主问题 = **扩大后稳定传播模板能否覆盖 SOZ 内外临床网络 + 是否出现 SOZ 内外分工 + 更大 KMeans 是否仍稳定**。扩大通道池本身就是有效问题,**不是**为"证明跳出 rate"(后者降为 sensitivity,§5.R)。**Pilot(2 subjects)= broad lagPat 非 phantom 噪声、捞回临床网络外圈通道(见 §1),现推全 cohort。**

## 1. Pilot 结果（zhaojinrui + chenziyang，2026-06-06，`results/lagpat_broad_pilot/`）

| subject | core→broad | added 上门槛 | 可复现 | 落在报告网络 | 读法 |
|---|---|---|---|---|---|
| zhaojinrui | 4→20 | 14 | 14（bar 0.347 偏松*） | **14/14** | 捞回 C/D/E 扩散网络(紧 4-ch 模板漏掉、正是它复发的扩散区) |
| chenziyang | 10→20 | 10 | 7（bar 0.075 严） | 5/7 | 捞回 E/D 临床起源(template-source G 漏掉的真起源) |

*caveat（写入 §5）：core-relative bar 对 core 大小敏感 —— core 小(n=4)→bar 估计差且偏松，"14/14 可复现"部分靠松 bar；**解剖对照(bar-independent)是稳信号**。

**Pilot 判读**：放宽通道池能捞回可复现、且落在报告记录致痫网络上的传播结构 = 传播结构延伸到 rate 核心之外 + 直接修 §5.4 覆盖率低估。**方法成立,推全 cohort。**

## 2. 目标 + 优先级（user reframe 2026-06-07）

**主问题（推全 cohort 的依据）**：扩大 lagPat 通道池后,稳定传播模板能否覆盖 SOZ 内外临床网络、是否出现 SOZ 内外分工、能否支撑更大的 KMeans。这本身就有意义,**扩大通道池 = 有效问题**。

**rate-independence 降为 sensitivity（§5.R）,不作全 cohort 推进的 gate。** (advisor 2026-06-07 cat 提醒:"可复现 + 落在网络"两道闸是 rate-confounded,不能当"跳出 rate"证据 —— 该提醒成立,但目标本就不是跳出 rate,所以把 rate 检查诚实地放在解释层即可。)

**优先级(执行序)**：① pilot 补正式 QC 表(确认非 phantom) → ② Yuquan 全 cohort broad re-pack(分批,先不上 Epilepsiae) → ③ 每批 broad-event QC 再决定进 KMeans → ④ broad lagPat 上重跑 KMeans/stable template(masked) → ⑤ broad 模板上做 SOZ 内外 → ⑥ rate 相关只作解释层 → 然后 Epilepsiae。

## 3. Cohort 范围 + 数据门

- **可 re-pack（有 `_refineGpu.npz` 全植入计数 + 逐记录 `_gpu.npz`）**：~15 Yuquan + 19 Epilepsiae。
- **排除（明示,不静默）**：7 个 Path-D Yuquan（无完整 gpu/refineGpu 检测,见 `run_interictal_propagation.py` L113-118）;pengzihang（坏数据 total_hours=2,见 MEMORY）;gaolan/sunyuanxin（有模板无病例 doc,Yuquan 解剖对照参考缺）。
- 每 subject 落 `status`（packed / excluded+reason）,不让排除项静默消失。

## 4. 参数合同

- **broad pick**：`pick_k=-2.0`（放行 ~全部 alive 通道）+ `pack_top_n=20`（按 event count 取 top-20）。pilot 用此值,n_pick=20 稳定,中位参与 ~11 通道/事件（非全 phantom）。
- **masked features 全程**（`mask_phantom=True` / `use_masked_features=True`）—— 越广越必须,守幽灵排名。
- **平行结果树** `results/interictal_propagation_broad/`（**不覆盖** `interictal_propagation_masked/`）;re-pack 中间产物 `results/lagpat_broad/<subject>/`（dry-run 模式,**不碰** `/mnt/yuquan_data` 原始树）。
- 复用 `scripts/pilot_broad_lagpat_repack.py` 的 monkeypatch 注入(不改 shared config);Epilepsiae 走对应 lagPat 路径(epilepsiae lagpat backfill 链)。

## 5. 主判据（broad-event QC → KMeans 稳定 → SOZ 覆盖）

**5.1 broad-event QC（步骤③,进 KMeans 前必看,`scripts/qc_broad_lagpat.py`）** —— 三道闸:
- n_broad 接近 20(≥15);
- 每事件参与通道中位数合理(≥5;20 通道下中位 3–5 = phantom 噪声,不进模板);
- 广版事件数未崩(≥100,否则无法稳定聚类 → 标 `broad-ineligible`,不静默)。
- `eligible_for_kmeans` = 三闸全过。pilot 验证:chengshuai/chenziyang/zhaojinrui broad=20、中位参与 11、事件充足 → 全 eligible;新增通道 8–14 落在临床网络(非噪声)。

**5.2 broad KMeans/stable template（步骤④,`compute_adaptive_cluster_stereotypy(..., use_masked_features=True)`,全程 masked)**：
- 窄版答"高率核心内有无稳定传播";广版答"更大临床网络里传播是否仍稳定、是否 SOZ 内外分工"。
- 报 broad stable_k、簇内 τ、identity-bias;窄 vs 广**只 subject-level 对照**(事件集已变,不做事件一一对应)。

**5.3 SOZ 内外（步骤⑤,§6）** —— 主读数(见 §6)。

**§5.R rate 检查（SENSITIVITY / 解释层,不 gate）**：报新增通道发放率分布 + Spearman(发放率, 传播早晚位置)。**相关高** → 诚实写"广版主要扩展了高发放临床网络"(仍有价值);**相关低** → 额外加分,非主目标。不卡全 cohort。

## 6. SOZ 内外分析（步骤⑤,主读数,`scripts/broad_template_soz_analysis.py`）

broad 模板出来后优先做(SOZ 标签 = `results/yuquan_soz_core_channels.json`;临床网络 = §1 persisted `yuquan_clinical_networks.json`):
1. **source/sink endpoint 有多少在 SOZ 内 / 外**。
2. **新增通道是否主要补到临床报告的扩散区**(added ∩ clinical network 比例)。
3. **broad 模板是否把原来"全在 SOZ 内"的 endpoint 扩展成 SOZ-core + peri-SOZ / extra-SOZ 结构**(窄 vs 广 endpoint 的 SOZ-coverage,subject-level)。
- Epilepsiae 侧 SOZ 参考 = `epilepsiae_electrode_focus_rel.json` 的 `i`(+`l`)触点(触点级,无扩散标注,判读弱于 Yuquan)。
- 窄 vs 广只 subject-level 对照(事件集已变,不事件一一对应)。
- 喂 mechanism paper §5.4 覆盖率重定义 + 结局分析。

## 7. 下游重跑链（广 lagPat 一旦产出）

1. 模板/聚类/stable_k/identity-bias（`compute_adaptive_cluster_stereotypy` 等,`use_masked_features=True`）。
2. endpoint（source∪sink）+ 覆盖率**重定义**（§5.4：用报告/focus 网络而非 refined-pool 核心）。
3. PR-6 endpoint→SOZ、Topic 4 H1/H2 在广 cohort 上的 sensitivity（**平行报告,不替换** masked 主结论;广版是 robustness 层）。
4. 与窄版对照**不作 standalone load-bearing**（放宽阈值同时改了事件集,双变量混淆);结论压在 §5–§6 逐通道判据上。

## 8. 批次 + 算力

- ~1.5–2.2 min/记录 × 记录数（Yuquan ~12–16 records/subject;Epilepsiae 更多 blocks）。Yuquan ~15 subj ≈ 数小时;Epilepsiae 视 block 数。
- 分批跑（per dataset / per 5 subjects）+ 进度落盘 + harness-tracked waiter;每批后 spot-check n_pick / 参与度,异常即停。

## 9. 全 cohort 验收（main question = broad 覆盖/稳定/分工）

- **broad 模板有效 + 扩展覆盖临床网络**：多数 eligible subject broad stable template 稳定(silhouette/AMI 过门槛) **且** 新增通道显著落在临床记录网络 **且** endpoint 从"全在 SOZ 内"扩展出 peri/extra-SOZ 结构 → "扩大通道池后稳定传播模板覆盖 SOZ 内外临床网络、出现内外分工"可声明。
- **broad 不稳定 / 退化**：多数 subject broad-ineligible(QC 三闸不过)或 broad 模板聚不稳 → "传播结构集中在高率核心、放宽后即散",诚实边界。
- **rate 解释层(§5.R)并列报**:ρ(rate,pos) 高 → "广版主要扩展高发放临床网络"(仍有价值);低 → 加分。**不作 PASS/NULL gate**。
- **MIXED 分层**：按 core 大小 / etiology / dataset(pilot 提示 core 小者扩展收益大);不强行单一判读。
- 每 subject 输出:narrow_n / broad_n / broad stable_k / endpoint_in/out_SOZ / added_in_net / ρ(rate,pos) / eligible / 分层。

## 10. 风险 / 开放问题

- **低事件记录**（pilot 见 n_ev=2/3 的记录）贡献噪声 → 聚合时按事件数加权或设 per-record min_events。
- **置换 null 算力**：逐通道 × 全 cohort,需 vectorize / 限 n_perm。
- **Epilepsiae 解剖参考较粗**（focus_rel 触点级但无"扩散"标注）→ Epilepsiae 解剖对照判读弱于 Yuquan,标注清楚。
- broad pool 定 20 是 pilot 值;cohort 前可对 1–2 个 subject 扫 top-N∈{15,20,30} 看稳健性(可选)。
- 与临床结局分析的接口:广版 endpoint/网络是覆盖率重定义(§5.4)的输入,等 Yuquan 随访回来 merge。

## 11. 执行步骤 + 进度（autonomous run 2026-06-07,user away ~8h）

1. ✅ Pilot（2 subjects）= re-pack 机制验证 + broad 非 phantom（中位参与 11，新增落临床网络）。
2. ✅ user reframe 并入（main question = broad 覆盖/KMeans;rate→sensitivity §5.R,不 gate）。
3. ✅ 基础设施就位:re-pack 驱动 `pilot_broad_lagpat_repack.py`(--out-dir);临床网络 persist `parse_yuquan_clinical_networks.py`→`yuquan_clinical_networks.json`(18 subj);QC `qc_broad_lagpat.py`(已验证);broad-template+SOZ `broad_template_soz_analysis.py`(测试中)。
4. ✅ **Yuquan broad re-pack 完成**(20 subj,4 批,QC 每批):18 eligible,2 broad-ineligible(huanghanwen 49ev / zhourongxuan 28ev,低事件)。
5. ✅ broad-event QC（§5.1）done → `results/lagpat_broad/qc_table.csv`。
6. ✅ broad KMeans（§5.2,masked）+ SOZ 内外（§6）+ rate sensitivity done → `broad_template_soz.csv` + `COHORT_SUMMARY.md`。**结果(§9 验收)**:**17/17 broad stable_k=2**(模板稳定);扩展 subject(narrow<20)endpoint 伸出 SOZ(多 1–2/6 in-SOZ)+ 新增落临床网络(zhaojinrui 14/zhangjiaqi 13/…)→ **SOZ 内外分工出现**;rate–pos 相关弱(−0.64…+0.41,sensitivity)。**caveat**:top_n=20 只扩 narrow<20;7 个 large-narrow 被收窄,需 top_n=40。
7. ▶ **Epilepsiae broad re-pack** launched(detached PID,`broad_lagpat_repack_epilepsiae.py`,smoke n_ch=20 验证;patch target = `load_refine_chns_for_subject`)→ `results/lagpat_broad_epilepsiae/`。多日(~40–50h serial,1073=231/916=435 records),未完;建议 parallelize。
8. ⏳ Yuquan 验收 PASS → 写入 mechanism paper plan（§5.4 覆盖率重定义 + SOZ 内外分工 + rate 解释层）。**待 user 决策**:7 个 large-narrow 是否跑 top_n=40。

**autonomous 决策权(user away)**:QC 三闸判 eligible 我按 §5.1 阈值自动判;broad 模板/SOZ 按脚本跑;rate 只报不 gate;遇系统性失败(整批 QC 崩)即停并记录,不盲目续跑。

## 12. 扩展 phase（user 2026-06-07：动态 top_n + 双 geometry 验证）

**12.1 动态 top_n**（修 §5.2 的 large-narrow 收窄问题）：per-subject `top_n = max(20, narrow_n + 15)`,保证每个 subject 都比窄池宽 ≥15。驱动 `pilot_broad_lagpat_repack.py --dynamic`(已验证 hanyuxuan 22→37,n_pick=37 中位参与 23)。输出平行 dir `results/lagpat_broad_dyn/`(不覆盖 top_n=20 的 `lagpat_broad/`)。批跑 + QC 同 §5.1/§8。

**12.2 双 geometry 验证**(`broad_geometry_validation.py`)：模板端点 geometry 不应是单一定义的 artifact → 用两种有效定义交叉验证:
- **def-a**:propagation rank top-3 source ∪ top-3 sink(dominant broad cluster)。
- **def-b**:PR-6 **rank-displacement swap-k 节点**——forward/reverse broad 模板对上 `compute_swap_score_sweep`→`decision_k`→`derive_swap_endpoint`(top-k∪bottom-k by rank_a)。带 swap_class/p_fw。
- 判据:per-subject **Jaccard(def-a, def-b)** 高 = geometry 对定义稳健;两定义下 endpoint 都伸出 SOZ = 结论不依赖定义。
- 先在 top_n=20 cohort 跑(即时),再在 dynamic cohort 跑。输出 `geometry_validation.csv`。

**12.3 进度 + top_n=20 geometry 结果**(`results/lagpat_broad/geometry_validation.csv`):
- **非饱和 dk 处两定义强一致**:liyouran dk=3 Jaccard=**1.0**、zhaojinrui dk=3 Jaccard=**1.0**、chengshuai/xuxinyi/zhaochenxi dk=2 Jaccard=0.67 → endpoint geometry 对定义稳健。
- **但多数 subject `decision_k=10` 饱和在 n/2**(20 通道池太小,已知 decision_k≈n/2 饱和)→ def-b = top-10∪bottom-10 = 全 20 通道,退化,Jaccard 塌成 6/20=0.3。**def-b 在 top_n=20 不可用。**
- 脚本已加 `dk_saturated` flag。

**12.4 dynamic geometry 结果（key,`results/lagpat_broad_dyn/geometry_validation.csv`）—— def-b 退化,bigger pool 未修复**:
- dynamic 大池(36–66 ch)上 **decision_k 仍饱和在 n/2**(hanyuxuan dk=18/n=36、zhangbichen dk=33/n=66、songzishuo dk=26/n=52)→ def-b = top-(n/2)∪bottom-(n/2) ≈ 全通道,退化,Jaccard 塌成 ~0.1–0.18。
- **这是已知的 `decision_k≈n/2` perm-null 饱和**(见 H2b,`project_topic4_h2b_direction_axis_outcome`),**非池子大小问题,大池修不了**。
- **结论(诚实)**:① def-b 解出小 decision_k 时(top_n=20 liyouran/zhaojinrui dk=2–3)**与 def-a 一致(Jaccard 1.0)→ endpoint geometry 稳健**;② 但多数 broad 模板 forward/reverse swap 读作**全局**(dk=n/2),def-b 退化、不能作独立交叉验证。**⇒ def-a(rank top-n endpoints)是可靠 geometry 定义;def-b 只在 swap 局部化时确认它。**
- **待 user 决策**:def-b 既然 decision_k 饱和,要么 (a) 接受 def-a 为主定义 + 用 decision_k 作"swap 局部 vs 全局"描述符,要么 (b) 换第三种独立 geometry 定义(如 spatial compactness / PCA 轴)。
- 进度:dyn batch1 done;dyn batch2(13 small)RUNNING;Epilepsiae 多日。dyn batch2 完 → QC + dynamic SOZ(def-a,大池)。
