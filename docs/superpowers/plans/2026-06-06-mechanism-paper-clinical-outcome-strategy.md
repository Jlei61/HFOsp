# 机制论文策略 + 临床结局分析计划（2026-06-06）

> 状态：strategy lock draft。来源 = 2026-06-06 用户长对话（老文章 NC desk-reject → 重定位为机制论文 + 结局预测临床收尾）。
> 本文件是 paper-level 策略与数据资产合同；具体统计合同（TDD + 预先指定检验）在数据到位后另立。

---

## 0. 一句话

老文章（KONWAC：Kuramoto 振荡器 + Hebbian replay + 间期→发作相变，`docs/paper/Replay of Interictal Sequential Activity...md`）被 NC **desk-reject**（编辑判影响力/新意不够，未送审）。新数据（Topic 1–4）**否掉了它的机制/模型层，但保住了现象层**。新论文 = "破旧机制 + 立正确机制（空间易激场模型）+ 临床收尾（间期模板是否标出该切的组织）"。

## 1. 什么活下来 / 什么塌了

| 层 | 老文章主张 | 新数据结论 |
|---|---|---|
| **现象（活）** | 稳定可复现传播模板（MI 17/18, 20/20） | ✅ 复现 + 加强（stable_k=2、23/30 强复现、簇内 identity bias 92%） |
| **现象（活）** | 正反模板对 | ✅ 复现（forward/reverse） |
| **现象（活）** | HFO 率定位 SOZ（AUC 0.857） | ✅ 复现（玉泉 refined 0.874） |
| 解释（塌） | IEI 幂律 | ❌ 实为 lognormal（30/30） |
| 解释（塌） | ~2Hz = 内禀振荡器 → Kuramoto 前提 | ❌ 不应期 renewal + 慢漂；振荡器前提没了 |
| 解释（塌） | Hopfield-Kuramoto "复现"序列 = 机制 | ❌ 循环论证（模式预先编码进 A_ij），BHPN-toy 已 SUPERSEDED |
| 解释（塌） | 间期→发作相变 / 发作前同步上升 14/20 | ❌ 同步 pre/post NULL；发作邻近几何 NULL；新框架不声称解释临床发作起始 |
| 解释（弱） | 稳定模板定位 SOZ 细结构 | ⚠ 端点→临床 SOZ 锚定 NULL（p≈0.39） |

诚实定性：**v1 发现了现象但过度建模；v2 把机制做对**。这是正常科学弧线，不是自我否定。

## 2. 论文脊椎（三幕 + 临床收尾）

1. **破**：~2Hz 不是振荡器、间隔不是幂律、SOZ 时序"特殊性"主要是选择偏差。（Topic 2 + Topic 3，已完成、可发表）
2. **立 + 机制**：间期事件 = 固定各向异性"高速路"上的自限可激传播；空间易激场模型（SEF-HFO）生成它；承重判别（方向随连接轴转、随电极杆旋转不变）通过。（Topic 1 现象 + Topic 4 模型）
3. **临床收尾（新核心）**：间期模板核心触点是否标出"切了才无发作"的组织 —— **Yuquan 触点级 + 结局为主，Epilepsiae 区域级为佐证**。

## 3. 数据资产图

| | 长期结局(Engel) | 切除/毁损靶区 | SOZ |
|---|---|---|---|
| **Epilepsiae**（公开） | ✅ `follow_up.outcome`（如 IIIa），18/20 模板病人有手术+结局；分布 Engel I=12 / II–IV=6 | ❌ 仅脑叶级（`surgerylocalisation` 如 't-r'） | 区域级 focus + per-contact `focus_rel` i/l/e |
| **Yuquan**（用户病例 `/mnt/yuquan_data/yuquan_24h_bingli/`，18 人 `颅内eeg.doc`） | ⏳ doc 内**无**长期随访；用户**努力去取**（不保证） | ✅ **触点级**（热凝表 = 被毁损触点；部分含开颅切除） | ✅ **触点级**（发作起始触点 + 间期放电触点 + 电刺激定位） |

**关键事实（2026-06-06 实测）**：Epilepsiae 区域级"模板源 ∈ 切除脑叶"≈15–16/18 重叠 → **二值预测变量退化、队列检验开局即死**。原因：侧化局灶癫痫里 模板源≈致痫灶≈切除脑叶；区域级编码分不开"切中模板核心"和"切了那个叶"。⇒ **Epilepsiae 只能当佐证；核心临床结果需 Yuquan 触点级 + 结局。**

## 4. 核心临床分析设计（Yuquan 触点级，数据到位后执行）

- **预测变量**：模板"起止端核心触点"(source∪sink，masked valid rank top/bottom-k) 与 被切/毁损触点 的重叠/覆盖比例（触点级）。
- **结局**：无发作(Engel I) vs 未无发作(II–IV)，**随访 ≥12 个月**为主分析，<12m 作 sensitivity。
- **承重纪律（v1 没做对、必须守）**：
  1. **反循环**：不能只测"切模板核心→好结局"（大家都切致痫灶）。必须测它在 **"切了 SOZ"之上的增量**预测力（SOZ 触点已有、热凝触点已有，可直接比）。
  2. **不假设 PR-6 null 自动去混淆**：模板对临床 SOZ 的增量要**实测**，不能用 p≈0.39 当独立性证明。
  3. **功效**：n≈18（约 12:6）只够**一个预先指定的非参检验**（如 Mann-Whitney on 模板覆盖比例 good vs poor），**不**做多协变量 logistic。
  4. **discordant case series**：模板与临床 SOZ 指向不同、结局跟着模板走的个案，对机制论文比脆弱队列 p 值更有说服力，作并列主证据。
- **基准对照**：模板覆盖比例 vs "HFO 率核心覆盖比例"(标准 resection-ratio 文献基线) vs "SOZ 覆盖比例" —— 看模板有没有增量。

## 5. Epilepsiae 支撑 pilot（区域级，现在就做）

- **角色**：① 给用户与医生谈随访时摆证据；② 演示分辨率上限 → 论证"为什么必须要 Yuquan 触点级"。
- **设计**：二值退化已证 → 改**连续预测变量** = 模板端点落在切除脑叶的**比例**（可能有 spread）；Engel I vs II–IV 的 Mann-Whitney；并报 SOZ(`focus_rel='i'`)/HFO率 覆盖比例作 context。
- **诚实口径**：区域级切除（假设切除叶内全部触点被切 → 高估、加噪）+ n=18 + 随访长短不一 → **sensitivity 级佐证，不当核心；正负都不改脊椎**。null 反而强化"需要 Yuquan 触点级"的论证。

### 5.1 Epilepsiae pilot 实测结果（2026-06-06）= 双路 NULL，确认 Epilepsiae 不能承载临床结果

跑了两条路，都死：
1. **resection-overlap → outcome**：**结构性退化**。propagation 模板算在 refined / focus-restricted 的 ~6–16 通道池上（如 1073 仅 6 ch、1096 仅 7 ch），通道池 ⊆ focus；Epilepsiae 切除编码到整叶 → "模板核心 ∈ 切除"≈1 对所有人成立，任何分辨率都无对比度。
2. **template-feature → outcome**（不需切除掩膜：stable_k / 刻板性 MI / 通道数 / 事件数 vs Engel I vs II–IV）：**无分离**。全部 MWU p>0.4（stable_k p=0.67、MI p=0.85）；唯一 sub-threshold 是 nch@≥12m p=0.066，但 n=7-vs-3 欠功效且方向反直觉（好结局通道更多）= 噪声。**未继续试变体凑显著（p-hacking 禁止）**。

**结论**：Epilepsiae（公开）**结构上无法回答临床问题**——(a) 模板通道池已 focus-限定、(b) 切除仅整叶级、(c) 模板特征 n=18 不分离结局。⇒ **临床结果只能靠 Yuquan 触点级 RF 热凝（只毁损部分 focus 触点 → focus 内有对比度）+ 结局。** 这正是给临床团队要随访数据的科学理由（见 §8）。Engel/follow-up/nch 原始数 = `results/`（pilot 未落盘，可复跑）。

### 5.2 Yuquan 热凝覆盖度检查（2026-06-06，回应"热凝范围似乎也很大"质疑）

用户质疑：热凝范围似乎也覆盖整个 focus，是否与 Epilepsiae 同样退化。实测 18 个有模板+病例 doc 的 Yuquan subject，覆盖率 = 模板通道中被热凝触点覆盖的比例：

- **覆盖率 0.10–1.00，中位 0.89；7/18 全覆盖（模板⊆热凝，多是 4–8 通道小模板，如 chengshuai），11/18 部分覆盖（模板伸出热凝外，如 huanghanwen 0.10 / songzishuo 0.13 / hanyuxuan 0.36 / zhangkexuan 0.42）。**
- **与 Epilepsiae 的本质区别**：Epilepsiae 脑叶级 + 模板算 focus 通道 → 所有人 ≈1.00 结构性退化；Yuquan 触点级 → 11/18 有真实对比度 ⇒ **overlap/coverage 设计在 Yuquan 非退化、值得用真实结局跑**。
- **两个 caveat**：(a) 覆盖率与模板大小耦合（小模板易全覆盖；大/分散模板覆盖低）—— 本身是"网络越分散越难毁干净→越易复发"的假设，但**必须对模板大小/植入范围归一化**，不能裸用覆盖率；(b) 热凝解析需逐 doc 清洗。

#### 5.2.1 清洗后结果（2026-06-06，`scripts/yuquan_template_ablation_coverage.py` → `results/template_ablation_coverage/yuquan_coverage_prep.csv`）
- **带撇电极 bug 已修**：songzishuo 0.13（解析漏 C′/D′，撇号 U+2019）→ **0.447**（正常化撇号后）。
- **三个预测变量候选**：
  - `template_coverage` = |模板∩热凝|/|模板|：**0.10–1.00，中位 0.89，11/18 部分**（有连续 spread，但 size-confounded）。
  - `source_ablated_frac` = 传播**源头**(top-3 最早通道)被热凝比例（size-robust，机制动机=驱动点是否被毁）：**~16/18 = 1.0（驱动点几乎都被毁）+ 2/18 = 0.0（huanghanwen / hanyuxuan 源头完全没毁）** → 近饱和、对比度低，但 2 个干净"漏毁源头"个案是 discordant-case 主证据候选。
  - `onset_coverage` = |临床起始∩热凝|/|临床起始|：0.32–0.87（SOZ baseline，用于"模板增量于 SOZ"）。注：onset 解析来自自由文本、近似，正式分析要逐 doc 核对。
- **设计含义**：判别信号将来自 **coverage 完整度梯度 + 少数 discordant 个案（源头/模板大段没毁的人，如 huanghanwen cov=0.10/源头0.0、hanyuxuan 0.36/源头0.0）是否结局更差**，而非 source_ablated 队列检验（近饱和）。
- gaolan/sunyuanxin = 有模板无病例 doc（不在 18 bingli folder）。
- **纪律**：有对比度 = 分析跑得动，**≠ 覆盖率一定预测结局**；merge 结局后才是结果。

### 5.3 Design B（热凝前后放电变化）= NOT FEASIBLE（2026-06-06 实测）

原设想"毁损前 vs 毁损后 24h 间期放电下降"作为不等随访的 in-hand 触点级结果。**实测不可行**：Yuquan EDF 数据集只含**术前 24h 监测**，录制日期**早于热凝 3–8 天**（chengshuai 录制 2019-09-20→21 / 热凝 09-24；zhangkexuan 08-13/08-20；huanghanwen 04-21/04-24；xuxinyi 08-31/09-06；zhaojinrui 08-14/08-22；litengsheng 11-19/11-22；pengzihang 11-28/12-03）。doc 里"第一次热凝后24h IID"是当时院内评估，**信号不在 `/mnt/yuquan_data`**。⇒ Design B 移除；in-hand 只剩 §5.2 coverage prep（需结局才成结果）。复算：`results/dataset_inventory/yuquan_block_inventory.csv` block_start_epoch vs doc 热凝日期。

### 5.4 热凝报告 case-review（2026-06-06）= 暴露覆盖率指标的 size confound + 早期失败标记可提取

用户问："从热凝报告能否看出源头没毁的 subject 术后是否仍有癫痫样波形"。挖报告文本（`/tmp/yqdoc/*颅内eeg.txt`，libreoffice 转）结论：

**(1) 术后波形读不到 + prime case 缺术后段**：术后间期/发作脑电在报告里是 **EEG 截图（图片）**，文本提取不到（章节头如 `热凝后IID` / `热凝后1H..24h` / `热凝后SZ2` 在、内容是图）。最干净的两个"源头完全没毁"病人 **huanghanwen / hanyuxuan 报告无术后段**（止于"术后切除照片"）→ 报告读不到它们术后状态。

**(2) 可提取的"早期失败"文字标记（3 例）**：
- **张珂萱(zhangkexuan)**：致痫区右岛叶-岛盖；模板覆盖 **0.42（网络大段没毁）**、源头(E1/E2/F1)毁了 → 报告"**热凝后发作2**"（术后仍发作）。支持"要毁整个网络不只源头"。
- **赵金蕊(zhaojinrui)**：起源 **F6-7** → 扩散 C4-7/D7-8/K9-14；我的覆盖 1.0/源头毁 **= 虚高**（模板仅 4 通道=源头那撮，漏掉扩散网络）→ 报告"**第4天第二次热凝**"(在 C) + "第2次热凝后发作/IID"。
- **陈子扬(chenziyang)**：临床发作**起源 E1-5/D13-14**，但我的模板**源头是 G1/G2/G6（对不上）**；覆盖 1.0 → 报告"**热凝后SZ2**"。模板定位与临床起源分歧（PR-6 endpoint≠SOZ 阴性的个案体现）。

**(3) 两条承重方法学修正（写入 §4 核心设计）**：
- **覆盖率被模板大小污染（具体证据=赵金蕊）**：小模板 trivially cov=1.0 但漏真实扩散网络 → "全覆盖"照样复发。⇒ **"该毁网络"应改用报告的"起源+扩散"触点定义，不只 refined-pool 模板**；coverage 必须对模板大小/网络范围归一化。
- **早期失败在"网络保留"(张珂萱)和"模板全覆盖"(陈子扬/赵金蕊)都出现** → "毁模板→治愈"过简；模板-源头 vs 临床-起源分歧是核心待答问题，需结局裁决。

**(4) TODO**：① 从 .doc 提取有术后段病人的 EEG 截图（zhangbichen 有 1H→第三天完整序列；chenziyang/zhangkexuan/zhaojinrui 有术后发作/IID 图）→ 图像级判读术后是否残留棘波/快波（huanghanwen/hanyuxuan 无图）；② 多次热凝病人(zhaojinrui)解析要分第一次 vs 第二次（当前 §5.2.1 parser 把两次 pooled → 覆盖虚高）；③ coverage 重定义为"报告临床网络(起源+扩散) ∩ 实际毁损/切除"。

## 6. 机制模型并行线（= NC 脊椎核心，不依赖结局）

SEF-HFO 生成模型**闭合**：异质核(Step 3) / spiking(Step 4) 的生成存在性 + 合成走真实 pipeline + 对照(isotropic/aligned-shaft) 过不了。当前状态：Step 0 机制尺度 + 各向异性判别 PASS；同质率场存在性 = NULL（移交异质核/spiking）。详见 `docs/topic4_sef_itp_framework.md` + `docs/archive/topic4/sef_itp_phase4_v2/`。

## 7. 期刊定位（诚实版）

- **关上**：顶级临床刊(Brain/Annals/Lancet Neurol，缺结局精度) + Nature/Science/Neuron 正刊。
- **NC（Nature Communications）← 模型闭合 + 锐利纠错 framing**（desk 死于影响力不够 → 模型闭合 + "纠正一整片文献"是过 desk 的钩子）。
- **Brain ← Yuquan 触点级 + 结局**（强临床版；门槛 = 拿到随访）。
- **地板（保底）**：机制 + 纠错论文投学会刊（J Neurosci / Network Neuroscience / Brain Communications）。

## 8. Yuquan 随访取数 spec（给医生的清单）

每人结构化：① Engel/ILAE（末次随访）；② **随访时长（月）**（主分析要 ≥12m）；③ 处理方式（热凝 / 热凝+切除 / 切除）+ 实际处理的触点（开颅切除要补切除触点/区域）；④ 手术日期 + 末次发作/复发日期。

## 9. 风险 / 开放问题

- Yuquan 随访**不保证拿得到**（用户努力）；拿不到则临床为辅、机制为主。
- Yuquan {有可用模板} ∩ {有结局} 交集需像 Epilepsiae 那样核对实际 n。
- 模型闭合**可能失败**（异质核/spiking 也放不出生成存在性）→ 退回"框架可证伪 + 判别通过"的地板版。
- Yuquan 多为 RF 热凝（非开颅切除），结局语义与 Epilepsiae 开颅切除不完全可比 → 分开报、不混池。

## 10. 下一步

1. ✅ 本计划文档。
2. ✅ Epilepsiae 支撑 pilot（§5.1）= 双路 NULL（结构性退化）；Yuquan 覆盖 prep（§5.2.1）+ 图 done；Design B（§5.3）= 不可行；热凝 case-review（§5.4）done。
3. ⏳ **用户取 Yuquan 随访（§8 spec）= 关键路径**。
4. ⏳ **覆盖率指标重定义**（§5.4 修正）：从 refined-pool 模板 → 报告"起源+扩散"临床网络 ∩ 实际毁损/切除；对模板大小归一化；多次热凝分第一/二次。这步不等随访就能做（清洗报告网络触点）。
5. ⏳（可选，回答"术后是否仍有癫痫样波形"）从 .doc 提取有术后段病人的 EEG 截图 → 图像级判读残留棘波/快波。
6. ⏳ 随访回来 → Yuquan {模板}∩{结局} 交集核对 → 正式统计合同（TDD + 预先指定检验 + discordant case series，先过 advisor 再落代码）。
7. ∥ 机制模型闭合（§6，独立推进）。
8. ⏳ 设计决定（用户）：每病人确定 SOZ/临床网络触点清单 vs 逐 doc 人工核对。
