# Track E2 — Epilepsiae 区域级"模板-手术-结局"可行性（no-go 落盘）

> 日期：2026-06-13　状态：feasibility artifact（read-only，不是结局分析）　性质：no-go 文档
> 与 Track E1（Yuquan 触点级）**分开立项、不混口径**。
> 产出：`results/epilepsiae_template_surgery_outcome_feasibility/{epilepsiae_outcome_surgery.csv, E2_feasibility_summary.json}`；脚本 `scripts/run_epilepsiae_e2_feasibility.py`。

---

## 一、朴素话三段

**查了什么** —— Epilepsiae 这个公开队列里，有传播模板的 20 个病人，到底有没有"术后好没好"的随访结局、以及"切了哪块脑子"的记录；如果有，这两样能不能拿来做"把传播模板那张网络切得越完整、结局越好吗"的外部验证。

**怎么查的** —— 直接读每个病人的病历数据库文件（SQL），抠三样：随访结局（Engel 分级，I 级=术后无发作）、切除区域、致痫灶位置。然后看一个关键问题：切除区域记得有多细——是精确到"哪几个触点"，还是只到"哪一整叶脑子"。

**揭示了什么** —— **结局数据是齐的、能用**：20 个里 18 个做了手术且有随访结局（随访 3–62 个月）。按主分析门"随访 ≥ 12 个月"分层：**10/18 达标，门内 Engel I 级 7 个、II–IV 级 3 个**（不分层的全 18 例是 I=12 / II–IV=6，但混进了 <12 个月的短随访）。**但切除区域只记到"整叶/区域级"**（18 例手术里 17 例有叶级码，如"右颞 t-r"；1 例占位 `958='---'`；**没有任何一例到触点级**）。注意：我们**没有做逐触点到脑叶的映射**，所以**不能说"模板被全切了"**——准确说法是：切除侧没有触点分辨率，触点级的"模板网络被切了多少"这个量**根本构造不出来**（粒度不够，不是已证明全切）。这跟之前预演（strategy §5.1）的"双路阴性"同一结论，现在落成独立 artifact。

**一句话定位**：Epilepsiae 的结局侧能用（证明这套分析流程跑得通），但它的切除记录只到叶级、没有触点分辨率，触点级的"模板网络完整度"在它上面**根本定义不出来**（不是判别力弱，是构造不出来）。**所以它不能作为 Yuquan 触点级 E1 的外部验证队列，只能当区域级弱佐证**；反过来，这正是"为什么必须去 Yuquan 医院补触点级术后结局"的数据层理由。

---

## 二、实测数

| 项 | 数 |
|---|---|
| 有传播模板的 Epilepsiae 病人 | 20 |
| 其中有手术 | 18（1146 / 635 = 仅 SEEG、无手术） |
| 有随访结局标签 | 18 |
| Engel 分布（全 18 例，含 <12mo 短随访） | I=12、II=3、III=2、IV=1 → 无发作(I) 12 / 未无发作 6 |
| **按主分析门 随访 ≥ 12mo 分层** | **达标 10/18；门内 Engel I=7 / II–IV=3** |
| 随访月数 | 3–62 |
| 切除区域粒度 | **整叶/区域级**：17/18 有叶级码（如 `t-r` / `f-l` / `t-l`），1 例占位 `958='---'`，**无任何一例触点级** |
| 模板触点池 | 5–16 通道（病灶限制的小池子）｜**未做逐触点→脑叶映射** |

（不分层的全 18 例 Engel I=12 / II–IV=6 与 mechanism-paper strategy §3 记录一致；主分析门内应取 10 例。）

## 三、为什么 no-go（触点级）

- **退化机制（粒度不足，不是"全切"）**：切除只记到叶级/区域级、**没有触点分辨率**；模板是 5–16 通道的小池子。要算"模板网络被切了百分之几"必须有触点级的切除边界——Epilepsiae 没有，所以这个比例**根本构造不出来**（不是算出来都≈1，是压根没法算）。我们**没做逐触点→脑叶映射**，**不主张"模板被完整切除"**。切除编码本身就是叶级，任何分辨率都救不回来。
- **不硬跑 p 值**（按用户指令）：predictor 既然退化成"几乎人人覆盖手术叶"，就只报 no-go，不为了出数硬做 Mann-Whitney。template-feature（stable_k / 通道数）对 Engel 不分离这条腿，strategy §5.1 已实测过（全 MWU p>0.4），此处不重跑、引用即可。

## 四、与 E1 的关系（口径隔离）

- E2（本文档）= Epilepsiae 区域级，**弱佐证 / no-go demo**，单独立项。
- E1 = Yuquan 触点级（`docs/superpowers/specs/2026-06-13-yuquan-template-resection-outcome-design.md`），结局侧被医院随访标签 gating。
- 两者**不混池、不混口径**；E2 的价值是反向论证 E1 触点级 + 结局是关键路径。

---

（内部归档代号：Track E2 = Epilepsiae region-level feasibility；源 = `/mnt/epilepsia_data/all_data_sqls/*.sql` 的 `follow_up.outcome` / `surgerylocalisation.localisation` / `eeg_focus.localisation`，subject 映射 `results/epilepsiae_subject_inventory.csv`；模板源 = `interictal_propagation_masked/per_subject/epilepsiae_*.json`。verdict = no_go_contact_level__granularity_insufficient（outcome present 18/20、>=12mo 10、Engel I=7/II-IV=3；surgery localisation 17/18 叶级 + 1 占位、无触点级；未做 channel→lobe 映射，不主张全切）。上游 dual-NULL = mechanism-paper-clinical-outcome-strategy §5.1。姊妹 = Track E1 Yuquan capstone。）
