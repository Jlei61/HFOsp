# Topic 5 Stage-1 Ictal-Template-Echo Proxy Triage — first execution (2026-06-08)

> **Tier**: exploratory proxy triage. **NOT a standing verdict.**
> **Verdict (machine)**: `代理计算跑通·控制未闭环` (pipeline runs; controls not closed).
> **Spec**: `docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md` (v4, a713ac8)
> **Plan**: `docs/superpowers/plans/2026-06-08-topic5-ictal-template-echo-gate.md` (a412bcc + round-4 patches)
> **Code**: `src/topic5_echo_gate.py` (35 tests green), `scripts/run_topic5_echo_gate.py`
> **Results**: `results/topic5_ictal_template_echo/{b0_eligibility_audit.csv, per_subject/*.json, cohort_echo_summary.json}`

## 0. 一句话朴素话

我们用现成的发作顺序代理（ER/atlas 派生的 ictal rank，不是新仪器），问 10 个病人的发作通道点亮先后像不像各自间期那条固定模板。**代理层确实有正向倾向，但更严的"按惯常早晚分箱打乱"那个对照把它打平了（clean null 下仍成立），合并机器自检也干净——所以结论是："这个像主要来自'越靠病灶越早'的共享粗锚，不是具体传播路径被复用"。** 换句话说：**ER-derived rank 不是传播路径仪器**（与项目一贯判断一致）。subject-specificity 还判不了（特异性对照在这批数据上结构跑不起来，§3），但这不改变上面的方向结论。这是 proxy triage（不否决 Stage 2）；它的粗锚结论**主动指向**下一步该直接建 Stage 2 多特征 recruitment-time map。

## 1. Cohort

B0 audit: 22 个 (masked stable_k=2 ∩ v2.3 atlas) 候选；锁定门槛 (MIN_CH=8, broad_ER primary, atlas-quality + per-seizure tie) 后 **10 个 usable** (8 epilepsiae + 2 yuquan)。swap_class stratifier: 7 none / 2 strict / 1 candidate。

## 2. 数值（clean-shuffle 重跑后，2026-06-08 round-5）

> **重要**：round-4 的 null shuffle 有 bug（打乱整条 seizure_rank，模板无效通道值会迁入有效位置污染 null）。已修为**只在 eligible（seizure 有值 ∩ 至少一条模板有效）通道内打乱**。下表是修复后的 clean 数值。**修复同时把 bad-data 自检从 borderline 0.042 变成干净的 0.93** —— 证明那个红旗就是这个 shuffle bug。

| 量 | n | median E_s | wilcoxon p (one-sided) | 读法 |
|---|---|---|---|---|
| primary channel-shuffle | 10 | 0.568 | **0.0068** | 有正向回声（inclusive）|
| within-shaft shuffle | 10 | 0.427 | **0.010** | 不只是 shaft 间差异 |
| **anchor-matched shuffle** | 10 | 0.124 | **0.42 (FLAT)** | **关键：按惯常早晚分箱后回声消失 → 是粗锚（clean null 下仍成立）** |
| LOO de-anchor 残差 | 8 | 0.460 | 0.004 | 残差仍显著（与 anchor-matched 轻度张力，n 小 + 4 bin 粗，以 anchor-matched 的保守判读为准）|
| stratifier swap none | 7 | 1.011 | 0.016 | none 子集也回声（通用口径预期）|
| **bad-data regression（自检）** | 10 | −0.170 | **0.93 (CLEAN)** | 合并机器不自造方向性 → **机器已验证**（round-4 的 0.042 是 shuffle bug，不是噪声）|
| negative_between_subject (epi) | 0 | — | — | Null D 跑不起来（见 §3）|

**载荷信号（clean null 下稳）**：full-shuffle / within-shaft 正向，但 **anchor-matched 打平** = "发作顺序与间期模板共享一个 earliness/病灶距离粗锚，而非具体路径复用"。bad-data 自检干净，机器无偏。

## 3. 三个控制项都没闭环（为什么不能下结论）

1. **Null D（跨病人模板特异性对照）结构上跑不起来**：generic-echo 口径下，"别人模板也一样像吗"是判断"特异 vs 解剖泛化"的关键控制。但
   - **epilepsiae**：病人间通道名几乎不重叠（按名对齐后 0 overlap，n=0）→ 对照无法运行。
   - **yuquan**：A/B/C/D 是**病人内**杆编号，不是跨病人解剖标签；exact-name 跨病人对齐会把"名字碰巧一样"当"同一解剖位置" → **按 P0 已跳过**。
   - ⇒ **无法确立 subject-specificity**。要闭环需用坐标/区域标签对齐的 Null D（Stage 2）。
2. **construct-validity sentinel = pending**：尚未人工核对"ER 最早通道与 line-length/broadband/HFA 是否同向"（有形状≠是传播）。pending 时**禁止任何 standing verdict**（已写进 verdict gate）。
3. **bad-data regression：round-4 borderline 显著 (p=0.042) 已查明 = null shuffle bug**。修复（只在 eligible 通道内打乱）后干净到 p=0.93。机器已验证，此项**闭环**。

## 4. 工程结论（已闭环的部分）

- 纯数学核心 + runner：**35 tests green**；逐 task 提交。
- 路上发现并修的真问题：1D vs 2D masked-template 合同（`mask_phantom_ranks` 是 2D 事件矩阵，1D 模板用 `np.where`）；atlas 用 canonical `src.atlas_loading`（非 atlas_v2_3 figures 目录）；对齐守门改"按名查找 + MIN_CH gate"（不再误杀 yuquan partial-overlap，但 partial-overlap 是 conservative）；None/int 混合 block bug；inline spearman 6× 提速（值不变）；Null D dataset-split（yuquan name-invalid）；atlas-quality 改 joint_valid mask + per-seizure tie；verdict construct-validity gate。

## 5. 下一步（user-locked 2026-06-08 round-5）：转 Stage 2，别再榨 ER

clean-shuffle 重跑后图景稳定且清楚：**粗检验阳性、anchor-matched 打平、bad-data 自检干净**。机器已验证，红旗已查明（是 shuffle bug）。所以 proxy triage 的结论可以下了（在它能下的范围内）：

> **现成 ER/atlas 代理里的"像"，主要来自共享的"哪些通道惯常更早"粗锚（病灶距离 / 早晚优先级），不是具体传播路径被发作重放。ER-derived rank 不是传播路径仪器。**

仍未闭环的两项（construct-validity、Null D 特异性）在**这批代理数据上结构上闭不了**（Null D 病人间通道名不共享；construct-validity 要另算多特征）。但它们不改变上面的方向性结论——anchor-matched 打平已经说明问题。

**因此下一步 = 直接做 early-ictal 1–10s 多特征 recruitment-time map（Stage 2）**，**不**继续榨 ER ratio：
- per-contact onset 用 line length / broadband / HFA / CUSUM / Page-Hinkley（feature-independent，解 construct-validity）。
- 这同时是 construct-validity 检验（ER-最早 vs 多特征-最早是否一致）+ 真招募顺序仪器。
- 特异性对照改坐标/区域对齐（`seeg_coord_loader`），不靠通道名。

（P0-1：proxy triage 不否决 Stage 2；这里是 proxy 的阴性/粗锚结论**主动指向**该建真仪器。）**主文档 `docs/topic5_seizure_subtyping.md` 暂不写入任何 standing 结论**——本轮仍是 proxy triage。
