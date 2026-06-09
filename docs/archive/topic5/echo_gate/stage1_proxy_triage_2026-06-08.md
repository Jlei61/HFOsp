# Topic 5 Stage-1 Ictal-Template-Echo Proxy Triage — first execution (2026-06-08)

> **Tier**: exploratory proxy triage. **NOT a standing verdict.**
> **Verdict (machine)**: `代理计算跑通·控制未闭环` (pipeline runs; controls not closed).
> **Spec**: `docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md` (v4, a713ac8)
> **Plan**: `docs/superpowers/plans/2026-06-08-topic5-ictal-template-echo-gate.md` (a412bcc + round-4 patches)
> **Code**: `src/topic5_echo_gate.py` (35 tests green), `scripts/run_topic5_echo_gate.py`
> **Results**: `results/topic5_ictal_template_echo/{b0_eligibility_audit.csv, per_subject/*.json, cohort_echo_summary.json}`

## 0. 一句话朴素话

我们用现成的发作顺序代理（ER/atlas 派生的 ictal rank，不是新仪器），问 10 个病人的发作通道点亮先后像不像各自间期那条固定模板。**代理层确实看到一个正向倾向，但更严的"按惯常早晚分箱打乱"那个对照把它打平了——这更像"发作顺序和间期模板共享一个'越靠病灶越早'的粗锚"，不是"具体传播路径被复用"。而且三个控制项都没闭环（见 §3），所以现在不能下任何"间期模板被发作复用"的结论。**这是 proxy triage；按合同它不否决 Stage 2。

## 1. Cohort

B0 audit: 22 个 (masked stable_k=2 ∩ v2.3 atlas) 候选；锁定门槛 (MIN_CH=8, broad_ER primary, atlas-quality + per-seizure tie) 后 **10 个 usable** (8 epilepsiae + 2 yuquan)。swap_class stratifier: 7 none / 2 strict / 1 candidate。

## 2. Preliminary 数值（倾向，非结论）

| 量 | n | median E_s | wilcoxon p (one-sided) | 读法 |
|---|---|---|---|---|
| primary channel-shuffle | 10 | 0.629 | **0.005** | 有正向回声（inclusive）|
| within-shaft shuffle | 10 | 0.445 | **0.007** | 不只是 shaft 间差异 |
| **anchor-matched shuffle** | 10 | 0.157 | **0.35 (FLAT)** | **关键：按惯常早晚分箱后回声消失 → 多半是粗锚** |
| LOO de-anchor 残差 | 8 | 0.406 | 0.004 | 残差仍显著（与 anchor-matched 轻度张力，n 小噪声）|
| stratifier swap none | 7 | 0.957 | 0.016 | none 子集也回声（通用口径预期）|
| stratifier swap strict/candidate | 3 | 0.296 | 0.25 | n=3 underpowered |

**载荷信号**：full-shuffle / within-shaft 正向，但 **anchor-matched 打平** = "发作顺序与间期模板共享一个 earliness/病灶距离粗锚，而非具体路径复用"。这是本轮唯一稳的方向性观察。

## 3. 三个控制项都没闭环（为什么不能下结论）

1. **Null D（跨病人模板特异性对照）结构上跑不起来**：generic-echo 口径下，"别人模板也一样像吗"是判断"特异 vs 解剖泛化"的关键控制。但
   - **epilepsiae**：病人间通道名几乎不重叠（按名对齐后 0 overlap，n=0）→ 对照无法运行。
   - **yuquan**：A/B/C/D 是**病人内**杆编号，不是跨病人解剖标签；exact-name 跨病人对齐会把"名字碰巧一样"当"同一解剖位置" → **按 P0 已跳过**。
   - ⇒ **无法确立 subject-specificity**。要闭环需用坐标/区域标签对齐的 Null D（Stage 2）。
2. **construct-validity sentinel = pending**：尚未人工核对"ER 最早通道与 line-length/broadband/HFA 是否同向"（有形状≠是传播）。pending 时**禁止任何 standing verdict**（已写进 verdict gate）。
3. **bad-data regression borderline 显著**（medE=0.155, p=0.042）：合并机器的自检本应变平却 borderline。可能是 n=10 噪声 + 单侧检验偏 liberal，也可能是 `e_k_baddata` 单抽样方差大的小偏差。**下一轮应改成多抽样平均的 bad-data 以判定**；在判定前，primary 的"干净"未证。

## 4. 工程结论（已闭环的部分）

- 纯数学核心 + runner：**35 tests green**；逐 task 提交。
- 路上发现并修的真问题：1D vs 2D masked-template 合同（`mask_phantom_ranks` 是 2D 事件矩阵，1D 模板用 `np.where`）；atlas 用 canonical `src.atlas_loading`（非 atlas_v2_3 figures 目录）；对齐守门改"按名查找 + MIN_CH gate"（不再误杀 yuquan partial-overlap，但 partial-overlap 是 conservative）；None/int 混合 block bug；inline spearman 6× 提速（值不变）；Null D dataset-split（yuquan name-invalid）；atlas-quality 改 joint_valid mask + per-seizure tie；verdict construct-validity gate。

## 5. 下一步（控制项闭环，非 Stage 2 之前不下结论）

按依赖排序：
1. **construct-validity sentinel**：人工核 ≥5 个 sentinel seizure 的最早通道在 line-length/broadband/HFA/ER 下是否同向。这是任何 standing verdict 的前置。
2. **bad-data regression 多抽样化**：判定 p=0.042 是噪声还是偏差。
3. **Null D 改坐标/区域对齐**：用 `seeg_coord_loader` + 区域标签对齐别人模板，才能测 subject-specificity（否则 anchor-matched flat + Null D 缺失下，"非特异"是最可能但未证的解释）。
4. 以上闭环后，若 primary 仍站、anchor-matched 仍 flat → 结论是"代理层回声主要是共享粗锚"，**正是去建 Stage 2 真招募仪器 + 坐标对齐控制的动机**（P0-1：proxy 不否决 Stage 2）。

**主文档 `docs/topic5_seizure_subtyping.md` 暂不写入任何结论**——本轮是 proxy triage、控制未闭环。
