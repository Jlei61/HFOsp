# SEF-HFO SOZ 定位:发放率 vs 传播几何 头对头 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **v2 2026-06-06**:按 user review 收紧——去过度承诺(改"检验是否")、主对照改 event-count-matched 公平比较 + 采样时长可靠性曲线为主分析、加 count-matched 抽样噪声 null、几何主分析改 both-ends endpoint、缺测通道不填 0(定义通道宇宙 + 报 SOZ 覆盖率)、去掉自动 commit(改 checkpoint/review)。
>
> **v3 2026-06-06**(Task 1 前数据契约审计 + user 拍板,见 `docs/archive/topic4/sef_hfo/channel_universe_montage_diagnostic_2026-06-06.md`):
> 1. **核心主张收窄**:几何**寄生在高率区**(传播顺序要求通道参与足够多群体事件=高发放率)。诚实主张 = "**放电活跃区内,传播顺序比发放计数更耐采样时间漂移的 SOZ 指纹**";**不**主张"几何定位率够不着的 SOZ"(那部分是低率区,率与几何**共同盲区**,补检测也救不回——实测:几何漏掉的 SOZ 通道中位率第 32 百分位 vs 覆盖的第 76)。
> 2. **队列 = 28**(13 yuquan + 15 epilepsiae);原始交集 29,`total_hours≥12` 剔 pengzihang(2.0h 垃圾率)。comparison-A 合格 26(gaolan covU=0 / huangwanling |U|=4 判 insufficient 仍报覆盖率)。
> 3. **率↔几何蒙太奇桥接(yuquan)**:率命名逐被试混双极/单触点。single→精确同名;bipolar→几何触点 `X`→唯一双极对 `X-next`(已验证 lagPat first-contact 约定:geom⊆首触点∧⊄次触点),无唯一对→missing 不填 0。
> 4. **数据集分工**:epilepsiae 为主队列(长程多日 117–435h=采样时长脊梁;覆盖率≈1.00);yuquan 为补充层(覆盖率中位 0.52,诚实暴露低率共同盲区)。
> 5. **几何敏感性**:主=连续两端端点(§3.1);+ 离散前 n(n=1,2,3) + rank-displacement 互换节点版;PR-2 反向节点仅作溯源不当分数。
> 6. **§0 已确立的 "12 yuquan p=0.11" 含 pengzihang 污染**,结果归档须标注待 total_hours≥12 重算。

> **v4 2026-06-07**(comparison-A 主队列 null 后,user review 重定核心问题):
> **核心问题改写**——不再问"几何 vs 率定位 SOZ"(焦点型主队列活跃区≈SOZ,无非-SOZ 对照,AUC≈0.5 是结构性的,非发现)。新核心:**在已知 SOZ / HFO 活跃区内部,当采样时间短、发放次数估计很吵时,传播 source 顺序能不能比发放计数更稳定地指出"哪几个通道是源头/端点"。**
> 1. **comparison-A 降级为数据契约诊断**(只说明"主队列不适合粗粒度 SOZ 二分类定位");不再是主线。
> 2. **新主分析 = SOZ 内部 source 稳定性**:在每被试 `SOZ∩U` 内,全程数据给两个参考靶(率靶=全程计数 top-k;source 靶=全程平均传播排名最靠前 top-k);用短窗/固定事件数窗重算,比较短窗 top-k vs 全程 top-k 的 Jaccard / 排名相关,随事件数/时长/昼夜变化;加 count-matched null 分清"事件少→不稳"还是"真实时间漂移"。
> 3. **主几何指标改 source-only**(传播最靠前=源头);sink-only / both-end endpoint 作敏感性。
> 4. **"低 rate"必须按窗口低事件数 / SOZ 内 rate 分位分层**,不能把低事件通道踢掉后再谈低率;主张范围 = "活跃 SOZ 内部精细稳定识别",不覆盖低率 SOZ。
> 5. **主句**:HFO 发放率能粗标病理活跃区,但在 SOZ **内部**做细粒度靶点选择时随采样窗漂移;传播 source 顺序更稳定,可能提供比单纯计数更可靠的 SOZ 内部结构指纹。
> 6. **null 方向**:时间连续窗 top-k vs 全程(observed)对比 随机抽 M 个事件 vs 全程(count-matched null);observed<null = 超出抽样噪声的真实时间漂移。率应 observed<null;source 若 observed≈null = 稳定无漂移。

**Goal:** 在同一批 SEEG 被试上,**检验**间期传播几何端点作为 SOZ 定位器,是否(a)定位不输 HFO 发放率,且(b)用更短/更少事件就达到更高的重测一致性——若两者成立,几何端点是比单纯发放率更稳定的 SOZ 网络指纹。

**Architecture:** 纯只读分析,消费三类已存在的 per-subject artifact(SOZ 标签 / masked 传播几何 / 每通道率 + 每事件时序)。新增分析模块 `src/sef_hfo_soz_localization.py` + TDD,runner(主分析:可靠性曲线 + 静态定位;补充:B1–B4),论文级图,archive doc。**不重跑检测、不重训聚类**;几何时间分辨率用"每窗每通道平均传播排名"获得。

**Tech Stack:** Python 3, numpy/scipy/pandas, sklearn(`roc_auc_score`), matplotlib;pytest TDD。复用 `src.rank_displacement`、`src.lagpat_rank_audit`(masked features)、`src.interictal_propagation`(事件加载)。

---

## 0. 背景与核心主张(spec,收拢此前讨论)

### 已确立(数据侧,审计加固过,不重做)
- 稳定传播模板(正/反两套),`stable_k=2`,34 被试。
- 源/汇端点在两套反向模板间互换(H2,n=23)。
- 通道身份偏置高(~71%,簇内 92%)。
- 率-几何时间解耦(H4,23/23,p=1.4e-6)。
- 发放率**方向上**指向 SOZ(Topic 3:重建后 12 yuquan 中 9 个 SOZ 率>非SOZ,中位差 +75/h,Wilcoxon p=0.11 → 方向对、弱而吵)。**[v3 警示]** 此 12/9/p=0.11 含坏数据 pengzihang(2.0h),待 `total_hours≥12` 重算;结果归档须标注。

### 核心主张(条件式,不预设结论)
> **[v3 收窄]** **在 HFO 放电活跃区内**(几何端点能定义的高率通道),HFO 发放计数是高方差、状态依赖、采样时长敏感的读数。本计划检验:在**同一片活跃区、同一批通道、同一批事件**上,传播顺序的起止端是否(a)定位不输发放计数、(b)用更短/更少事件达到更高重测一致性,则**传播顺序是比发放计数更耐采样时间漂移的 SOZ 指纹**。

**[v3 边界]** 几何**寄生在**高率区(排顺序需通道参与足够多群体事件)。本计划**不**主张几何能定位发放率够不着的(低率)SOZ:实测几何漏掉的 SOZ 通道是低率(中位第 32 百分位 vs 覆盖的第 76),那是率与几何的**共同盲区**,补做非 lagPat 区域检测也救不回(通道太安静排不出稳定顺序)。这比"率不反映 SOZ"强且更不易被领域反驳(率确实指向 SOZ,见 Topic 3 + 文献)。本计划是**检验**收窄后的主张,不是预先宣告它成立。详见 `docs/archive/topic4/sef_hfo/channel_universe_montage_diagnostic_2026-06-06.md`。

### 红线
- 不解释 clinical seizure onset、不解释 HFO 载波频率。
- SOZ 只作 held-out 真值,不作训练标签(不反向拟合)。
- 一篇文章,数据为主,模型只占一节(0d 方向组织机制);不分论文。
- **昼夜只能作脑状态的时间段代理**;无真实 vigilance/sleep stage 时,不得声称"证明觉醒/睡眠驱动"。

---

## 1. 决策状态

### 已锁
- **D-geom-polarity(v2 改)**:主分析 = **起止两端 endpoint geometry**;敏感性 = source-only / sink-only / 模板方向。不把 SOZ 强等同"最先点火端"。
- **D-cohort(v3 改)**:n=**28**(13 yuquan + 15 epilepsiae);原始三源交集 29,`total_hours≥12` 剔 pengzihang。不重跑检测。comparison-A 合格 26(gaolan/huangwanling insufficient 仍报覆盖率)。
- **D-masked-only**:几何与每事件排名只走 phantom 修复后的 masked 路径。
- **D-no-split-paper**:一篇文章。
- **D-B-design(v2 改)**:**主分析 = 采样时长可靠性曲线 + event-count-matched 率vs几何 + count-matched null**;B1–B4 作补充(B2 漂移随间隔 / B3 昼夜代理 / B4 多日 / B1 逐窗离散度)。
- **D-daynight**:08:00–20:00 作时间段代理(Epilepsiae 按 Europe/Berlin);**语言降级**(代理状态,不证驱动);yuquan 时区 Task 1 冻结。
- **D-topic3-doc**:本次**不改** Topic 3 主文档;补 yuquan 导致的数变化只在新 archive 记 provenance;待 SOZ-localization 结果稳定再统一更新。
- **D-no-autocommit**:任务步骤用 **checkpoint/review**,不自动 commit;提交由 user 显式要求。

### 待确认(开跑前)
- **D-cohort-35**:6 个缺 `_gpu.npz` 的 yuquan 默认不补;仅当结果边缘/审稿需要满队列再重跑检测。

---

## 2. 队列 + 数据契约(同 v1,schema 已核实)

队列 = `SOZ 标签 ∩ masked 几何 ∩ 每通道率` 交集 = **29**(14 yq + 15 epi)。Task 1 冻结成 `results/topic4_sef_hfo/soz_localization/cohort.json`(单一真值)。

**(a) SOZ 真值** `results/{yuquan,epilepsiae}_soz_core_channels.json` = `{subject:[ch_name,...]}`。
**(b) 每通道率** `results/spatial_modulation/per_channel_metrics/{ds}/{subject}_perchannel.json` → `channel_metrics[]{ch_name,n_events,event_rate,region_label}`。
**(c) masked 几何** `results/interictal_propagation_masked/rank_displacement/per_subject/{ds}_{subject}.json`:顶层 `channel_names/soz_channels/pairs`;`pairs[0]{channel_names,joint_valid,rank_a_dense_full,rank_b_dense_full,swap_sweep{decision_k,swap_class,p_fw},soz_mask,clinical_soz_set_relation}`;dense 排名 `0`=源…`n_valid-1`=汇。
**(d) 每事件时序** yuquan `{record}_packedTimes_withFreqCent.npy`+`{record}_lagPat_withFreqCent.npz`;epilepsiae `.../all_recs/{block}_lagPat_withFreqCent.npz`(块号=1h)。加载复用 `src.interictal_propagation` + **masked**(`src.lagpat_rank_audit`)。

---

## 3. 通道宇宙 + 分数定义(v2 重写:缺测不填 0)

### 3.0 分析通道宇宙(开工先定,缺测不当 0)
- **宇宙 U** = 该被试 **HFO-active ∩ 几何可定义**通道:几何 `joint_valid=True` **且** 经 §3.2 蒙太奇桥接后映射到的率通道 `n_events ≥ MIN_CH_EVENTS`(默认 30)。
- **[v3] 蒙太奇桥接(yuquan 率命名逐被试混双极/单触点)**:single→几何触点 `X` 取精确同名率通道;bipolar→`X` 取唯一双极对 `X-next` 的率(已验证 lagPat first-contact 约定);无唯一对/未检测→该触点 **missing**(出 U,不填 0)。epilepsiae 全 single。
- `U` 之外的通道标 **missing/excluded**,不进 AUC/Jaccard,**不填 0**。
- **必报 SOZ 覆盖率**:`soz_coverage = |SOZ_core ∩ U| / |SOZ_core|`。若 SOZ 大量落在 U 外(几何测不到),这是几何的**真实局限**,明写不藏。
- 率与几何**在同一 U 上**比较(公平);率在 U 内每通道 `event_rate`(U 内通道按定义有事件,非 0 伪填),几何在 U 内有排名。

### 3.1 几何分数(主:both-ends endpoint)
设某对 joint 通道 `c` 的 dense 排名 `r_a(c)`/`r_b(c)`(0..n-1),`n=n_valid`:
- **主 `geom_endpoint(c) = mean( 2|r_a/(n-1)-0.5| , 2|r_b/(n-1)-0.5| )`** — 起或止端→1,中间→0;对 A/B 平均(端点集在反转下稳定)。
- 敏感性:`geom_source(c)=1-r_a/(n-1)`、`geom_sink(c)=r_a/(n-1)`、模板方向版(若 PR-2.5 `fwd_rev_source` 指定 forward 则按它定向)。
- **[v3] 端点定义分辨率敏感性(user 点名)**:除连续两端端点外,再测 (i) **离散前 n 端点**(模板最源/最汇各 top-n,n∈{1,2,3} 标 0/1),(ii) **rank-displacement 互换节点版**(`swap_sweep.decision_k` 端点,经 `src.rank_displacement.derive_swap_endpoint`)。三者同在高率域、覆盖率限制相同——验证活跃区内排名稳健性,不改覆盖故事。**PR-2 反向节点仅作溯源/漏斗,不当定位分数**(整模板配对层,非通道级;见 AGENTS.md Topic 4 H2 input source order)。

### 3.2 率分数
- `rate_score(c)=event_rate(c)`(全程,对照 A);时序分析用每窗事件计数(§5)。
- **[v3] 蒙太奇映射**:`event_rate(c)` 经 §3.0 桥接取得——single 取同名率通道;bipolar 取唯一 `c-next` 双极对(first-contact 约定);无唯一映射→`c` 为 missing。**桥接是精确一一对应(非平均/平滑)**:lagPat 把双极对按第一触点命名,已逐被试验证(geom⊆首触点∧⊄次触点)。litengsheng 3 触点(A9/A10/B9)率未检测→missing。

> **反循环**:率读"响多少次"(数量),几何读"排第几"(顺序),机制正交。几何分数绝不用 participation rate 派生。

---

## 4. 对照 A — 静态 SOZ 定位(必要前置)

每被试在宇宙 U 上:真值 `y(c)=[c∈SOZ_core]`;`AUC_geom_endpoint = roc_auc_score(y, geom_endpoint)`,同算 `AUC_rate`、敏感性几何分数;`topk_overlap`(k=|SOZ_core∩U|)。退化:`|SOZ_core∩U|<2` 或 `|U|<5` → `insufficient`,排除报分母。
队列:配对 Wilcoxon `AUC_geom_endpoint` vs `AUC_rate`(单边 geom≥rate?)+ n + 中位差 + geom≥rate 被试数 + 队列 `soz_coverage` 分布。
**判读语言锁**:只允许"几何定位不输/略优于率(中位 AUC 差 X,n=Y,SOZ 覆盖率中位 Z)";**不允许**"率不反映 SOZ"。

---

## 5. 对照 B — 采样可靠性(主分析 = 可靠性曲线 + 公平 + null)

### 5.0 每窗几何分数(不逐窗重聚类)
某时间窗 w,通道 `c` 几何分数 = w 内该通道参与事件的 **masked 平均传播排名**(再按 §3.1 归一)。无需重训模板;与每窗率用**同一批事件**。地板:窗内事件 < `MIN_WIN_EVENTS`(30)或通道参与 < `MIN_CH_WIN_EVENTS`(5)→该窗该通道 missing(非 0)。

### 5.1 主分析① — event-count-matched 率vs几何(公平核心,回应 review#2)
- **预算 M ∈ {50,100,200,500, full}** 个事件。对每个 M:取窗(连续 M 事件,沿记录滑动多个起点)→ 在 **同一 M 个事件**上同时算 `rate_target_M`、`geom_target_M`(各取 top-k,k=|SOZ∩U|)。
- 指标:`J_to_full = Jaccard(target_M, target_full)`(target_full = 全程靶)。
- **公平**:每个 M 上率和几何用**同一批事件**,消除"窗大=更稳"混淆。报 `J_to_full(M)` 曲线,率 vs 几何。
- 附:把 M 映射到**等效小时数**(用该被试事件率)供临床读;但**主轴是事件数预算**。

### 5.2 主分析② — 采样时长可靠性曲线(临床可读,回应 review#3)
- 时长 `d ∈ {1,2,4,8,24h}`(Epilepsiae 加 48/96/全程)。对每个 d:多个起点取长度 d 的窗 → `J_to_full(d)`,率 vs 几何。
- 汇总指标:**`hours_to_J08`** = 达到 `J_to_full ≥ 0.8` 所需最短时长(率 vs 几何;达不到则报 censored)。
- 这是最临床可读的脊梁图:"录多久,靶才稳定到 0.8"。

### 5.3 主分析③ — count-matched 抽样噪声 null(决定能否说"漂移",回应 review#4)
- **问题**:短窗靶变,是脑状态变,还是短窗事件太少的抽样噪声?
- **null**:对每个 M(或 d),从**全程事件**随机抽 M 个(打散时间→去掉时间结构,只留有限采样)→ 算 `J_to_full_null`,重复 `N_NULL`(默认 200)→ 得抽样噪声下的 `J_to_full` 分布。
- **判**:观测窗(按时间连续)的 `J_to_full` 是否**显著低于** count-matched null(即时间连续窗比"同样多但打散的事件"更不一致 → 存在超出抽样噪声的**真实时间漂移**)。率和几何各做。
- **claim 闸门**:
  - 率观测 < null(漂移 > 抽样噪声)**且** 几何观测 ≈ null → 最强:"率有真实时间漂移、几何没有"。
  - 两者都 ≈ null → 只能说"短采样两者都受抽样限,几何因更省事件而更早稳",**不能说**"状态驱动"。
- **昼夜归因**(§5.5 B3)只有在率漂移 > null 时才允许往"状态代理"说,且仍只是代理。

### 5.4–5.7 补充分析(B1–B4,非主)
- **B1 逐窗离散度**:固定窗(事件数预算或 daypart)逐窗 `AUC(w)` 的 CV/IQR,率 vs 几何。
- **B2 一致性 vs Δt**:窗对 `Jaccard(target)` 对时间间隔 Δt 分箱,率衰减 vs 几何平。
- **B3 昼夜代理**:day(08–20)vs night 靶 Jaccard,率 vs 几何。**语言:时间段代理状态,不证驱动**;且依赖 §5.3 null 通过才允许状态措辞。
- **B4 跨天**:Epilepsiae 多日,day-1 vs day-N 靶 Jaccard;yuquan 单日跳过报分母。

---

## 6. 护栏(硬守)
1. **通道对齐**:几何/SOZ 用单触点名(对齐各自 `channel_names`);`pairs[i].channel_names` 与顶层可能异序,先按名重对齐。**[v3] 率可能用双极命名**:不能假设率名集合==几何名集合;须经 §3.2 桥接(single 精确同名 / bipolar `X-next` 唯一对)把率映射到几何触点,映射后才在 U 上索引;断言桥接后向量长度==|U|,无 0 伪填。
2. **缺测=missing 非 0**(§3.0);必报 `soz_coverage`。
3. **公平**:主对照率 vs 几何用**同一批事件 / 同事件数预算**;窗宽不对称(1h率 vs 6h几何)**禁作主对照**(1h率仅作附加展示)。
4. **抽样噪声分离**:任何"漂移/状态"措辞必须先过 §5.3 count-matched null;否则只写"短采样不可靠"。
5. **masked-only**;**SOZ 不入模**;**如实报分母 + 覆盖率 + censored**。
6. **反循环**:几何不用率派生;每窗率/几何同批事件不同读数。

---

## 7. 交付物
- `src/sef_hfo_soz_localization.py`(纯函数:通道宇宙 / 分数 / AUC / 窗口化 / 可靠性曲线 / count-matched null / 稳定性指标)。
- `tests/test_sef_hfo_soz_localization.py`(TDD,每分数/指标/护栏一条)。
- `scripts/run_sef_hfo_soz_localization.py` → `results/topic4_sef_hfo/soz_localization/{cohort.json, comparison_a.json, reliability.json, supplements_b.json, per_subject/}`。
- `scripts/plot_sef_hfo_soz_localization.py` + `figures/README.md`(中文)。
- `docs/archive/topic4/sef_hfo/soz_localization_2026-06-06.md`(全量 + provenance,含 Topic 3 重建说明);主文档摘要+回链留待结果稳定。

---

## 8. 图(论文级,§7 多面板纪律,自包含无代号)
- **面板 1(对照 A)**:每被试 `AUC_geom_endpoint` vs `AUC_rate` 配对散点 + y=x + 队列中位 + SOZ 覆盖率标注。问题:几何定位是否不输率。
- **面板 2(主③可靠性曲线)**:`J_to_full` vs 事件数预算 M(率 vs 几何 + count-matched null 带)。问题:谁更省事件达到稳定 + 漂移是否超抽样噪声。
- **面板 3(主②时长曲线)**:`J_to_full` vs 时长 d + `hours_to_J08` 配对。问题:录多久靶才稳。
- **(补充图)** B2 Δt + B3 昼夜。
- README 逐图中文 + `**关注点**:`。

---

## 9. TDD 任务分解(每步末 = checkpoint/review,**不自动 commit**)

### Task 1: 冻结队列 + 通道宇宙 + 时区
**Files:** Create `scripts/build_soz_localization_cohort.py`; Create `tests/test_soz_localization_cohort.py`; Output `cohort.json`
- [ ] **Step 1: 失败测试** — 三源 mock,`build_cohort` 只留三源齐全且 SOZ 非空;每条带 `channel_universe`(HFO-active∩几何可定义)+ `soz_coverage`。
- [ ] **Step 2: 跑确认 FAIL** — `pytest tests/test_soz_localization_cohort.py -v`
- [ ] **Step 3: 实现** — 三源求交;按 §3.0 算 U + soz_coverage;返回。
- [ ] **Step 4: PASS**
- [ ] **Step 5: 真实数据** — 生成 `cohort.json`(**报实际 n**:原始交集应 29,`total_hours≥12` 后应 28;断言 raw==29 且 kept==28,不 hard-fail 单一数字),打印每被试 `montage` + `|U|` + `soz_coverage`;打印一 yuquan `packedTimes` 首末时刻,冻结 `meta.daynight_tz`。参考 `results/topic4_sef_hfo/soz_localization/cohort_preview.json`(已生成的预览,Task 1 须复现其数)。
- [ ] **Step 6: checkpoint/review**(暂停给 user 看 U + 覆盖率分布;**不 commit**)

### Task 2: 几何分数(endpoint 主 + source/sink 敏感性)
**Files:** Create `src/sef_hfo_soz_localization.py`; Test file
- [ ] **Step 1: 失败测试**
```python
def test_geom_endpoint_main_and_sensitivities():
    chans=["a","b","c","d"]
    pair={"channel_names":chans,"joint_valid":[True]*4,
          "rank_a_dense_full":[0,1,2,3],"rank_b_dense_full":[3,2,1,0]}
    s=geom_scores(chans, pair)
    assert s["endpoint"][0]==s["endpoint"][3]==1.0          # 主:两端
    assert abs(s["endpoint"][1]-s["endpoint"][2])<1e-9      # 中间对称
    assert s["source"][0]==1.0 and s["sink"][0]==0.0        # 敏感性
```
- [ ] **Step 2: FAIL** → **Step 3: 实现 `geom_scores`**(§3.1;重对齐 `pair.channel_names`→`channel_names`;非 U/非 valid→missing(NaN),长度/集合断言)→ **Step 4: PASS** → **Step 5: checkpoint/review**

### Task 3: 率 + SOZ 对齐(宇宙内,缺测=missing)
- [ ] **Step 1: 失败测试** — `load_rate_and_soz(perchannel,soz,universe)` 只在 U 内返回对齐 `rate_vec/y`;U 外通道不出现(非 0 填充);`sum(y)==|SOZ∩U|`。
- [ ] **Step 2: FAIL** → **Step 3: 实现**(U 内对齐;断言无 0 伪填)→ **Step 4: PASS** → **Step 5: checkpoint/review**

### Task 4: 对照 A 指标
- [ ] **Step 1: 失败测试** — 完美→AUC1.0;随机→~0.5;`|SOZ∩U|<2`→insufficient;topk 正确。
- [ ] **Step 2: FAIL** → **Step 3: `comparison_a_subject`** → **Step 4: PASS** → **Step 5: checkpoint/review**

### Task 5: 对照 A runner + 队列
**Files:** Create `scripts/run_sef_hfo_soz_localization.py`(A 部分)
- [ ] **Step 1: 失败测试**(`tests/test_soz_loc_runner.py`)— mock 跑出 `comparison_a.json`(per-subject AUC + 配对 Wilcoxon endpoint vs rate + 覆盖率)。
- [ ] **Step 2: FAIL** → **Step 3: 实现** → **Step 4: 真实 n=29**(打印中位 AUC 率/几何 + p + geom≥rate 数)→ **Step 5: checkpoint/review**

### Task 6: 每窗事件加载 + 窗口化分数(masked)
- [ ] **Step 1: 失败测试** — mock 事件(时刻+每事件 rank/bool),`window_scores(events,edges,floors)` 每窗每通道 `{count, mean_rank}`;不够地板→missing。
- [ ] **Step 2: FAIL** → **Step 3: 实现**(复用 `interictal_propagation` masked 加载;地板守门)→ **Step 4: PASS** → **Step 5: checkpoint/review**

### Task 7: 可靠性曲线 + event-count-matched + count-matched null(主分析)
- [ ] **Step 1: 失败测试** — 构造"率靶随时间换、几何不换"+"全程打散后两者都退化"的合成事件流;断言 (i) `J_to_full(M)` 几何>率,(ii) count-matched null 下率观测<null 而几何≈null,(iii) `hours_to_J08` 几何<率。
- [ ] **Step 2: FAIL** → **Step 3: 实现 `reliability_curve(events, y, M_grid, d_grid, n_null)`**(§5.1/5.2/5.3:同批事件算率/几何靶、`J_to_full`、连续窗 vs 打散重抽样 null、hours_to_J08)→ **Step 4: PASS** → **Step 5: checkpoint/review**

### Task 8: 主分析 runner + 队列(reliability.json)
- [ ] **Step 1: 失败测试** — mock 跑出 `reliability.json`(per-subject 曲线 + null 比较 + 队列 hours_to_J08 配对 + claim 闸门标记)。
- [ ] **Step 2: FAIL** → **Step 3: 实现** → **Step 4: 真实 n=29**(打印队列 hours_to_J08 率/几何 + null 闸门结果:率漂移是否>抽样噪声)→ **Step 5: checkpoint/review**

### Task 9: 补充 B1–B4 + runner(supplements_b.json)
- [ ] **Step 1: 失败测试** — B1 离散度/B2 Δt/B3 昼夜(语言代理+依赖 null)/B4 跨天(<2 天 skip)。
- [ ] **Step 2: FAIL** → **Step 3: 实现 `stability_supplements`** → **Step 4: 真实 n=29** → **Step 5: checkpoint/review**

### Task 10: 论文级图 + README
- [ ] **Step 1:** §8 三主面板 + 二补充(自包含无代号)渲染 → **Step 2:** user 目视 → 修 → 重渲染 → **Step 3:** 中文 README(逐图 + `**关注点**:`)→ **Step 4: checkpoint/review**

### Task 11: archive doc + provenance
**Files:** Create `docs/archive/topic4/sef_hfo/soz_localization_2026-06-06.md`
- [ ] **Step 1:** 三段式朴素 abstract + 全量表 + 护栏 + 判读锁 + **Topic 3 补 yuquan 重建 provenance**(9→12,p 0.05→0.11)。
- [ ] **Step 2:** 主文档**暂不改**(D-topic3-doc / 待结果稳定)→ **Step 3: checkpoint/review**

---

## 10. Self-Review(spec 覆盖)
- 核心主张改条件式 / 去"证明":✓ §0 Goal/spine,§4/§5 判读锁,B3 语言降级。
- 公平(同事件数预算):✓ §5.1,§6 护栏 3,Task 7。
- 采样时长曲线为主:✓ §5.2,Task 7/8,面板 3。
- 抽样噪声 count-matched null:✓ §5.3,§6 护栏 4,Task 7 闸门。
- 几何 endpoint 主、source/sink 敏感性:✓ §3.1,Task 2。
- 缺测=missing + 通道宇宙 + SOZ 覆盖率:✓ §3.0,§6 护栏 2,Task 1/3。
- 无自动 commit(checkpoint/review):✓ 全 Task。
- 昼夜=代理、yuquan tz 冻结、Topic 3 doc 不动:✓ §1 决策,Task 1/11。
- **已知风险(archive 写明)**:几何短窗稀疏(覆盖率/censored 报);n=29 配对功效;6 yuquan 缺检测(D-cohort-35);几何 SOZ 覆盖率可能 < 1(几何测不到部分 SOZ = 真实局限)。
- **count-matched null 的归因边界**:连续窗比随机-M 更不一致,只证"超出事件数的真实时间结构",该结构可能是慢状态漂移**或**短时事件自相关/聚集;null 本身分不开。"状态"措辞须靠 §5.3 null(证有时间结构)+ §5.5 B3 昼夜代理(narrow 到状态)联合,且仍只是代理(无真实 sleep stage)。archive 须把这条归因链写明,不可由 null 直接跳"状态驱动"。

---

## 11. Execution Handoff
建议先跑 **Task 1–5(队列+宇宙+对照 A,低风险近期出数)**:看几何定位是否不输率 + SOZ 覆盖率多少 → 决定主分析窗策略。再 Task 6–8(可靠性曲线 + null,脊梁)。补充 B1–B4 最后。执行方式:subagent-driven(每 Task 派新 subagent + 评审)或 inline executing-plans;**均不自动 commit,checkpoint 处暂停给 user**。
