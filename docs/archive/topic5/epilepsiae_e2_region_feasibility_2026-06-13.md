# Track E2 — Epilepsiae 区域级"模板-手术-结局"可行性（no-go 落盘）

> 日期：2026-06-13　状态：feasibility artifact（read-only，不是结局分析）　性质：no-go 文档
> 与 Track E1（Yuquan 触点级）**分开立项、不混口径**。
> 产出：`results/epilepsiae_template_surgery_outcome_feasibility/{epilepsiae_outcome_surgery.csv, E2_feasibility_summary.json}`；脚本 `scripts/run_epilepsiae_e2_feasibility.py`。

---

## 一、朴素话三段

**查了什么** —— Epilepsiae 这个公开队列里，有传播模板的 20 个病人，到底有没有"术后好没好"的随访结局、以及"切了哪块脑子"的记录；如果有，这两样能不能拿来做"把传播模板那张网络切得越完整、结局越好吗"的外部验证。

**怎么查的** —— 直接读每个病人的病历数据库文件（SQL），抠三样：随访结局（Engel 分级，I 级=术后无发作）、切除区域、致痫灶位置。然后看一个关键问题：切除区域记得有多细——是精确到"哪几个触点"，还是只到"哪一整叶脑子"。

**揭示了什么** —— **结局数据是齐的、能用**：20 个里 18 个做了手术且有随访结局（Engel I 级 12 个、II–IV 级 6 个，随访 3–62 个月）。**但切除区域只记到"整叶级"**（每人就一个粗码，像"右颞叶 t-r"），而传播模板只算在病灶里那 5–16 个触点的小池子上、整个落在被切的那叶里。**所以"模板网络被切了多少"这个比例对所有人都≈全切、没有对比度**——这跟之前预演（strategy §5.1）得到的"双路阴性"是同一个结论，现在落成了独立 artifact。

**一句话定位**：Epilepsiae 的结局侧能用（证明这套分析流程跑得通），但它的切除记录太粗（整叶级），触点级的"模板网络完整度"在它上面天生没有判别空间。**所以它只能当区域级的弱佐证，不进 Yuquan 触点级 E1 主分析**；反过来，这正是"为什么必须去 Yuquan 医院补触点级术后结局"的数据层理由。

---

## 二、实测数

| 项 | 数 |
|---|---|
| 有传播模板的 Epilepsiae 病人 | 20 |
| 其中有手术 | 18（1146 / 635 = 仅 SEEG、无手术） |
| 有随访结局标签 | 18 |
| Engel 分布 | I=12、II=3、III=2、IV=1 → **无发作(I) 12 / 未无发作(II–IV) 6** |
| 随访月数 | 3–62 |
| 切除区域粒度 | **整叶级**（每人中位 1 个 lobe 码，如 `t-r` / `f-l` / `t-l`） |
| 模板触点池 | 5–16 通道（病灶限制，⊆ 被切的那叶） |

（Engel I=12 / II–IV=6 与 mechanism-paper strategy §3 记录一致。）

## 三、为什么 no-go（触点级）

- **退化机制**：侧化局灶癫痫里 模板触点池 ⊆ 致痫灶 ⊆ 被切的那叶；切除只记到整叶 → "模板源 ∈ 切除区"对几乎所有人都成立 → 二值/比例预测变量没有对比度。任何分辨率都救不回来（切除编码本身就是叶级）。
- **不硬跑 p 值**（按用户指令）：predictor 既然退化成"几乎人人覆盖手术叶"，就只报 no-go，不为了出数硬做 Mann-Whitney。template-feature（stable_k / 通道数）对 Engel 不分离这条腿，strategy §5.1 已实测过（全 MWU p>0.4），此处不重跑、引用即可。

## 四、与 E1 的关系（口径隔离）

- E2（本文档）= Epilepsiae 区域级，**弱佐证 / no-go demo**，单独立项。
- E1 = Yuquan 触点级（`docs/superpowers/specs/2026-06-13-yuquan-template-resection-outcome-design.md`），结局侧被医院随访标签 gating。
- 两者**不混池、不混口径**；E2 的价值是反向论证 E1 触点级 + 结局是关键路径。

---

（内部归档代号：Track E2 = Epilepsiae region-level feasibility；源 = `/mnt/epilepsia_data/all_data_sqls/*.sql` 的 `follow_up.outcome` / `surgerylocalisation.localisation` / `eeg_focus.localisation`，subject 映射 `results/epilepsiae_subject_inventory.csv`；模板源 = `interictal_propagation_masked/per_subject/epilepsiae_*.json`。verdict = region_level_no_go_for_contact_level。上游 dual-NULL = mechanism-paper-clinical-outcome-strategy §5.1。姊妹 = Track E1 Yuquan capstone。）
