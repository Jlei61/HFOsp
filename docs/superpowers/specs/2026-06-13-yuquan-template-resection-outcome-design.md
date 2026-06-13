# Yuquan 触点级"传播模板覆盖 × 术后结局" exploratory capstone — 设计 spec

> 日期：2026-06-13　状态：design（待用户审 spec）　性质：exploratory clinical capstone（Topic 5 pivot 计划 Track E1）
> 上游：`docs/archive/topic5/network_axis_pivot_plan_2026-06-13.md` §五-E；`docs/superpowers/plans/2026-06-06-mechanism-paper-clinical-outcome-strategy.md` §4/§5.4/§8
> 本 spec 是**预测变量 + schema + 待标签统计计划**的冻结合同；预测变量这轮就算，统计待结局标签到位再跑。

---

## 0. 一句话

我们想问一件有临床新意的事：**把平时高频小放电那条传播路线的"网络"切/毁得越完整，病人术后越不容易再发作吗——而且这条网络指标，比单纯的"HFO 高发率区"或"临床起始区"更贴近结局吗？** 这一轮先把"切/毁了多少网络"这一侧（预测变量）全部算出来并冻结结局表；结局标签（谁无发作、谁复发）不在 repo 里、病例文档也抠不出来，必须另外去医院要，所以"覆盖 ↔ 结局"的关联**这轮不跑**。

---

## 1. 范围与边界（先钉死，避免被读成预测模型）

- 这是 **exploratory capstone**，不是 confirmatory prediction，不做多变量 logistic 模型。
- n 很小（覆盖度 ok 18 人、有模板端点指标 14 人、有 HFO 基线 12 人），统计只够**预先指定的非参检验 + case-series**，头条科学权重放在 **discordant 个案**，不是任何单个 p 值。
- **RF 热凝（点状毁损）和开颅切除（整块切除）生物学上不是同一种处理**：可以同表，但 `surgery_type` 必须分层可见，绝不揉成一个"treated"。`complete coverage` ≠ `complete resection`——热凝是点状毁损，不是整块切除。
- 文献定位（Brain 2025 多中心 HFO study）：完整切除 HFO area ↔ seizure-freedom 有关联但非神话（pooled ILAE1 73%、forest OR 2.67、specificity 仅 0.39、中心间差异大）。⇒ framing = 传播模板**可能**是比 HFO-rate-area 更细的网络指标，值得 exploratory 探，不预设它一定赢。
- 与 pivot §6"只预注册一个 primary"纪律的关系：§6 的单 primary 锁是对**发作-轴检验**（A primary；B/C/D exploratory）。**Track E 本就在那把锁之外**，pivot 明确把 E 标为"临床收口、被结局标签 gating、exploratory"。所以这里 6 个覆盖量**对称汇报、全标 exploratory**与 §6 不冲突——前提是每个关联都按 exploratory 报（effect size + 精确 p + 显式多重比较透明），**绝不写"预测"或"显著"**。

---

## 2. Cohort 分层（实测，不是假设）

覆盖度 ok = 病例文档可解析出热凝触点的 18 人（剔除 `gaolan`/`sunyuanxin` = 无病例 doc）。各数据源覆盖**不一致**，必须分层可见、缺失不静默丢：

| 层 | n | 缺的人 | 数据源 |
|---|---|---|---|
| 覆盖度 ok（有热凝解析） | **18** | — | 病例 doc `/tmp/yqdoc/*.txt` |
| 有模板端点指标（template_anchoring） | **14** | huangwanling, zhangjinhan, zhaojinrui, zhourongxuan | `results/interictal_propagation_masked/template_anchoring/per_subject/yuquan_*.json` |
| 有方向互换判定（swap_class） | **18** | — | `results/interictal_propagation_masked/rank_displacement/per_subject/yuquan_*.json` |
| 有 HFO 率基线 | **12** | songzishuo, zhangbichen, zhangkexuan, zhaochenxi, zhaojinrui, zhourongxuan | `results/spatial_modulation/per_channel_metrics/yuquan/*_perchannel.json` |
| 有临床 SOZ | **18** | — | `results/yuquan_soz_core_channels.json` |
| 有临床网络（起源+扩散） | **18** | — | `results/lagpat_broad/yuquan_clinical_networks.json` |

**统计纪律**：任何两个量的对比用 **intersection / pairwise-complete cohort**，不能拿 template 14 人和 HFO 12 人直接比（template∩HFO ≈ 10 人）。每个量缺失带 `*_status` / `*_exit_reason`，不补值、不用旧 dominant-source 充数。

---

## 3. "treated"（被处理触点集）定义

- `ablated` = 病例 doc 热凝段（`热凝` 与 `术后`/`报告医生` 之间）解析出的触点（monopolar，撇号 U+2019/2018/02BC → ASCII `'`，bipolar `E8-9`→{E8,E9}）。**复用** `scripts/yuquan_template_ablation_coverage.py` 的 `ablated_contacts()`。
- `resected` = 开颅切除区触点。**实测结论：切除区在病例 doc 里是图片（"术后切除照片"段），文本抠不出来** → `n_resected = NA`、`resected_status = image_only_unavailable`。**绝不记 0**：`0` = "确认没有切除触点"，`NA` = "切除触点未知"，两者科学含义完全不同；记 0 会把"未知切除"误读成"没切除"。开颅切除触点化留作后续（需图像判读或医院补数据）。
- `treated_total` = `ablated ∪ resected`。当前切除未知 → `treated_total_current = ablated_only`、`treated_total_status = ablation_text_only`、`n_treated_total` 仅计 ablated。所有覆盖比例这轮的 treated 实际 = ablated_only；结局解读时必须记得**切除侧是未知、不是无**。
- **多次热凝**（如 zhaojinrui 第 4 天第二次热凝）：当前 parser 把多次 pooled → 覆盖虚高。这轮先用**累计毁损集**，加 `multi_session=True` 标记，case-series 单独警示；分次解析留作 refinement，不这轮硬塞。

---

## 4. 预测变量（科学合同核心）

每个量都是"该被处理的网络，实际被 treated 毁掉了多少" = `|该网络 ∩ treated| / |该网络|`，分母 size 并排报（§5.4 size-confound 必须可见）。问的"该处理网络"不同。

### 4.1 三个 template 主指标（进 3+3 主表）

统一用 PR-6 primary rank endpoint（`per_template[]`），**不**混用 `per_template_coreness`（后者降为 sensitivity，见 4.3）。

1. **模板端点被毁** `template_endpoint_coverage`
   - 朴素：整条传播高速路的"两头"（起点站+终点站）被毁了多少。
   - 定义：`union over templates of per_template[].endpoint`（= source ∪ sink，两个端点并集），∩ treated / 分母。
   - 源：template_anchoring `per_template[].endpoint`。gate：有 anchoring（14）。

2. **传播早端被毁** `early_end_coverage`
   - 朴素：传播的"起跑站/驱动点"被毁了多少。
   - 定义：`union over templates of per_template[].source`（每模板最早 top-3 source 的跨模板并集），∩ treated / 分母。
   - 源：template_anchoring `per_template[].source`。gate：有 anchoring（14）。这是旧 `source_ablated_frac`（仅主簇 top-3）的跨模板推广。

3. **共享端点核被毁** `shared_endpoint_core_coverage`
   - 朴素：两条稳定路线**都**反复用到的那撮端点骨架被毁了多少。
   - 定义：`intersection( union(source,sink) endpoint of template A, union(source,sink) endpoint of template B )`，∩ treated / 分母。
   - **gate = `stable_k >= 2`（两条稳定模板存在 = 14/14 anchoring 病人都过）**，否则 NA。
   - **命名纪律**：这测的是"两条稳定模板共享端点"，**不是**"split-half/odd-even 复现出来的路径"。真正的 `high_consistency_path_coverage`（跨数据切半复现）需要重跑/扩展 split-half endpoint 存储，**留作后续 refinement（Y，见 §11），本轮不生成、X 不冒充 Y**。
   - `forward_reverse_reproduced`（split-half OR odd-even，4/14）作为一列 **flag**，不作 gate（4/14 太薄 + 语义不匹配——它测"正反模板对跨切半复现"，不是"共享端点"）。

### 4.2 三个 baseline 对照（进 3+3 主表）

4. **临床 SOZ 覆盖** `clinical_soz_coverage`
   - 定义：`|curated SOZ ∩ treated| / |SOZ|`。源：`results/yuquan_soz_core_channels.json`（18/18，比 doc fuzzy onset 解析可靠）。
   - **反循环关键参照**：模板指标必须报"在切了 SOZ 之上的**增量**"（§5.4：大家都切致痫灶），见 §6。

5. **HFO 率 top-k（SOZ-size）覆盖** `hfo_rate_topk_sozsize_coverage`
   - 朴素：和临床 SOZ 一样大的那撮"HFO 最高发率触点"被毁了多少（size-matched HFO-rate core，**不等于** Brain 2025 里完整 HFO-area resection）。
   - 定义：`|HFO-rate-core ∩ treated| / |HFO-rate-core|`。
   - **bipolar→monopolar 桥接（载重科学合同）**：HFO per-channel 是**双极**（`ch_name="E9-E10"`），模板/热凝/SOZ 是**单极**。用项目已验证的 **first-contact 规则**桥接（双极对的第一触点；memory: geom ⊆ 首触点 ∧ ⊄ 次触点）。
   - **HFO-rate-core 规则（本轮冻结，不留实现期自由度）**：
     ```
     HFO-rate-core = top-k bridged-monopolar contacts by event_rate
     rate lookup   = src.sef_hfo_soz_localization.build_rate_lookup   # 已存在 src/sef_hfo_soz_localization.py:63
     min_ch_events = 30                                               # <30 events 的通道不进 rate-core（率不稳）
     k             = min(|clinical SOZ|, n_rate_available)            # size-matched，让覆盖对比 size 可比
     tie-break     = stable sort by (-event_rate, channel_name)
     ```
   - 源：`per_channel_metrics/yuquan/*_perchannel.json`（12/18）。NA + `hfo_rate_core_status`/`hfo_rate_core_exit_reason` for 缺的 6 人。
   - **分母列语义（载重，别搞错）**：`n_hfo_rate_core` = size-matched core 大小 = `k = min(|SOZ|, n_available)` = **coverage 的分母**；`n_hfo_rate_available` = 桥接后过事件门的 HFO-active 触点池（≥ core）。两列必须分开报，否则读者会把 池当成 core 分母（实测踩过：available=32 误写成 core=7 的位置）。
   - 若日后找到 topic3 更权威 HFO-area 定义，作 **sensitivity**，**不改**本 spec 主合同。

6. **临床网络覆盖** `clinical_network_coverage`
   - 朴素：报告里写的"起源+扩散"整张致痫网络被毁了多少（回应 §5.4"小模板漏扩散网"）。
   - 定义：`|clinical network ∩ treated| / |network|`。源：`results/lagpat_broad/yuquan_clinical_networks.json` 的 `network[]`（18/18，从报告 raw_lines 解析的起源+扩散触点）。

### 4.3 Sensitivity / supplement（不进 3+3 主表）

- `template_coreness_endpoint_coverage` = `union(per_template_coreness[].endpoint)` ∩ treated / 分母。**sensitivity**（coreness endpoint vs primary endpoint 的稳健性对照），明确标 sensitivity，不当主指标。
- `late_end_coverage`（可选 supplement）= `union(per_template[].sink)` ∩ treated / 分母。"终点站/汇"被毁比例，放 supplement。

---

## 5. 分层 / 描述列（stratifier，不是覆盖量）

- **`swap_class`**（strict/candidate/none，18/18）= 每病人"有没有真实双向轴结构"的描述标签。**当分层列，不当覆盖量**（none 病人 decision_k 饱和 ≈ n/2，按互换端点算覆盖只对 8 个 swap-positive 有意义）。实测分布：strict 5（chenziyang/hanyuxuan/wangyiyang/zhangjiaqi/zhaochenxi）、candidate 3（liyouran/zhangjinhan/zhourongxuan）、none 10。strict 5 ⊃ forward_reverse_reproduced 4，且 18/18 全有 → 是同一"双向轴"信号里更完整的版本。
- `forward_reverse_reproduced`（4/14）= flag 列。
- `surgery_type`（热凝 / 热凝+切除 / 切除）= 来自结局表，分层可见。
- `multi_session`（bool）。
- **`discordant_candidate` 规则驱动**（从本轮新 metrics 算，**不硬写结论**——实测 hanyuxuan 在新跨模板 `early_end_coverage` 下 = 0.5 不是 0，旧 dominant-source"源头完全没毁"在新口径下不成立）。满足任一即标 `True`：
  ```
  early_end_coverage == 0
  OR template_endpoint_coverage < 0.5
  OR clinical_network_coverage < 0.5
  OR (模板源 ∩ 临床起源) == ∅           # template-vs-clinical-origin discordance
  OR multi_session == True
  ```
  旧静态个案提示（huanghanwen / hanyuxuan / zhangkexuan / songzishuo / chenziyang / zhaojinrui）放 `notes_prior_case_hint` 列作线索，**不作 frozen truth**。

---

## 6. 混淆可见 + 反循环纪律（写死进表，不是事后补）

- 每个覆盖比例**并排报分母 size**：`n_endpoint`, `n_source`, `n_shared_endpoint`, `n_soz`, `n_hfo_rate_core`, `n_clinical_network`, `n_ablated`, `n_resected`, `n_treated_total`。§5.4 size-confound 必须可见；n≈15 不拟合多协变量模型，只报 size + 靠 case-series。
- **反循环（§4.1 strategy lock）**：模板指标的科学问题是"在切了 SOZ 之上的**增量**"，不是裸"切模板核心→好结局"。结局到位后，每个 template 量都要报相对 `clinical_soz_coverage` 的增量（不假设 PR-6 endpoint-SOZ null 自动去混淆——p≈0.39 不能当独立性证明，要实测）。

---

## 7. 冻结的 outcome schema（空表，字段锁死只填值，不边填边改）

文件 `results/template_resection_outcome/yuquan_outcome_labels.csv`：

```
subject, surgery_type, engel_class, ilae_class, followup_months,
recurrence, recurrence_date, procedure_date, last_followup_date,
source, confidence, notes
```

- `source` 固定 `yuquan_doc`；每个 outcome 必须有 `notes` 说明依据（Track E1 仅 Yuquan）。
- 主分析门 `followup_months >= 12`；`>= 24` 做 sensitivity。
- 结局来源：Yuquan 病例 doc / 医院随访（病人几年后再回来 = 复发，从随访记录推）。**这轮生成空表（字段锁死、行=18 subject、值留空）给用户去医院填**。

---

## 8. 待标签统计计划（写进 spec，结局标签到位前**不跑**）

- 主 endpoint：Engel I / ILAE1 vs 非 I。
- **6 个覆盖量全标 exploratory**，没有一个 confirmatory primary（这是把"对称 3+3"和抗分叉纪律调和的诚实方式）。
- 检验：小样本只做 Mann-Whitney U / exact Fisher / permutation；报 effect size + **精确 p**；**显式标注多重比较、不下"显著"声明**。
- baseline 对比用 intersection cohort（pairwise-complete）：template-vs-HFO 用 template∩HFO ≈ 10 人。
- **头条科学权重 = discordant case series**（§5 候选）：比脆弱队列 p 值更有说服力，作并列主证据。
- 边界重申：不写"预测模型"，写"coverage 与结局的 exploratory 关联"；覆盖高但复发的反例必须并列写、不藏。

---

## 9. 产出物

```
results/template_resection_outcome/
├── yuquan_template_resection_metrics.csv   ← 这轮就算（预测变量，每病人一行）
├── yuquan_outcome_labels.csv               ← 这轮冻结成空表（字段锁死，用户去填）
├── yuquan_outcome_merged_metrics.csv       ← 脚本生成；标签到位前 = metrics + 空 outcome 列 + `outcome_status` 列
│                                              **纯表格，不在 CSV 顶部塞 metadata**（标准 reader 会炸）；总状态写 cohort_summary.json
├── cohort_summary.json                     ← cohort 分层计数 + swap_class 分布 + 覆盖梯度 + discordant 候选 + `outcome_status: "labels_missing_not_analyzed"`
└── figures/
    ├── README.md                           ← 中文逐图说明（图实际生成后写）
    └── (覆盖梯度图 + swap_class 分层图 + discordant 候选图；**结局图等标签到位才画**)
```

新脚本：`scripts/run_template_resection_outcome.py`（复用 `yuquan_template_ablation_coverage.py` 的热凝解析 + 端点/网络读取）。旧 `yuquan_coverage_prep.csv` 保留作 legacy descriptive prep，**不进** E1 主 metric。

### `yuquan_template_resection_metrics.csv` 列清单

```
subject,
# 三 template 主指标 + 分母
template_endpoint_coverage, n_endpoint,
early_end_coverage, n_source,
shared_endpoint_core_coverage, n_shared_endpoint,
# 三 baseline + 分母
clinical_soz_coverage, n_soz,
hfo_rate_topk_sozsize_coverage, n_hfo_rate_core, n_hfo_rate_available,
clinical_network_coverage, n_clinical_network,
# sensitivity / supplement
template_coreness_endpoint_coverage,
late_end_coverage,
# treated 集（n_resected 永远 NA，不是 0）
n_ablated, n_resected, resected_status, n_treated_total, treated_total_status, multi_session,
# 分层 / 描述
swap_class, forward_reverse_reproduced, surgery_type,
# 状态（缺失不静默）
template_anchor_status, template_anchor_exit_reason, has_template_endpoint_metric,
hfo_rate_core_status, hfo_rate_core_exit_reason,
metric_status, exit_reason,
# discordant（规则驱动）+ 旧个案线索 + 自由文本
discordant_candidate, notes_prior_case_hint, notes
```

---

## 10. 实现合同（防"名义复用、实际重拼"）

- 复用 `yuquan_template_ablation_coverage.py::{ablated_contacts, expand_pairs, norm, doc_for}` 解析热凝触点（撇号正规化、bipolar 展开）。
- 端点：读 `template_anchoring/per_subject/yuquan_<s>.json` 的 `per_template[].{endpoint,source,sink}` + `audit.stable_k` + `audit.forward_reverse_reproduced`；coreness 读 `per_template_coreness[].endpoint`。
- swap：读 `rank_displacement/per_subject/yuquan_<s>.json` 的 `pairs[0].swap_sweep.swap_class`。
- SOZ：`yuquan_soz_core_channels.json[subject]`（list）。
- 临床网络：`yuquan_clinical_networks.json[subject].network`（list）。
- HFO 率：`per_channel_metrics/yuquan/<s>_perchannel.json.channel_metrics[].{ch_name,event_rate,n_events}` → **first-contact 桥接双极→单极** → HFO-rate-core monopolar 集，**按 §4.2 冻结规则**（`src.sef_hfo_soz_localization.build_rate_lookup`、`min_ch_events=30`、`k=min(|SOZ|,n_rate_available)`、tie-break `(-event_rate, channel_name)` stable sort）。
- **通道名对齐**：所有集合在比较前统一 `norm().upper()`，按 NAME 取交集，**绝不按下标对齐**。
- 缺源 → 对应 `*_status="missing"` + `*_exit_reason`，覆盖量 NA；不补值、不充数。
- 测试（TDD）：每个覆盖量一个已知小输入的单测（交集/分母/NA 路径）；first-contact 桥接一个单测；缺 anchoring / 缺 HFO 的 NA 路径各一个回归测试。

---

## 11. 开放项 / 后续 refinement

- **Y = 真正的 `high_consistency_path_coverage`**：扩展 `src/interictal_propagation.py::compute_time_split_reproducibility` 把 split-half/odd-even 的稳定端点**集合**存下来（当前只存 jaccard 标量），再算"跨切半真正复现的路径触点"被毁比例。本轮不做；X（shared_endpoint_core）不冒充 Y。
- 开颅切除触点化（图像判读 / 医院补数据）。
- 多次热凝分第一/二/最终次。
- HFO-rate-core 口径：本 spec 已冻结 size-matched top-k（§4.2）；若日后找到 topic3 更权威 HFO-area 定义，作 sensitivity，不改主合同。

---

（内部归档代号：Track E1 Yuquan-only clinical capstone；预测变量 = template_endpoint_coverage / early_end_coverage / shared_endpoint_core_coverage(gate stable_k>=2) + baseline clinical_soz_coverage / hfo_rate_topk_sozsize_coverage(bipolar→monopolar first-contact, build_rate_lookup, min_ch_events=30, k=min(|SOZ|,n_avail)) / clinical_network_coverage；sensitivity = template_coreness_endpoint_coverage / late_end_coverage；stratifier = swap_class(rank_displacement swap_sweep) / forward_reverse_reproduced / surgery_type / multi_session；Y refinement = high_consistency_path_coverage via compute_time_split_reproducibility split-half endpoint 存储。源：interictal_propagation_masked/{template_anchoring,rank_displacement}/per_subject、yuquan_soz_core_channels.json、lagpat_broad/yuquan_clinical_networks.json、spatial_modulation/per_channel_metrics/yuquan。文献：Brain 2025 多中心 HFO（ILAE1 73%、OR 2.67、spec 0.39）。上游：network_axis_pivot_plan_2026-06-13 §五-E、mechanism-paper-clinical-outcome-strategy §4/§5.4/§8。）
