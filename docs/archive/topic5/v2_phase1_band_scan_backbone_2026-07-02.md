# Topic 5 V2 Phase 1 — 频率扫描 backbone 建成 + dev-null Gate A 初判 (handoff)

date 2026-07-02 · 分支 `topic5-v2-phase1-build`（worktree `/home/honglab/leijiaxin/HFOsp-t5v2`，off `codex/topic4-m3a-v2-2` @ `e01c08b`，27 commits）· 状态：**backbone 全部建成 + 全部过 per-task review；dev n_perm=100 broad raw 已跑，Gate A 初判为「有前景的频带特异描述性信号，但形式化 Gate A 被稀疏杆几何卡住」**

> 计划：`docs/superpowers/plans/2026-07-01-topic5-v2-phase1-band-scan-backbone.md`；设计：`docs/superpowers/specs/2026-07-01-topic5-v2-hfo-critical-mode-design.md`。SDD 账本（含每个决策/局限的完整记录）：`.superpowers/sdd/progress.md`（该 worktree 内）。

---

## 0. 摘要（朴素话）

**测了什么**：癫痫病人发作**刚起头 20 秒**里，每个频带（δ→ripple）的能量在电极阵列上"点亮"成一张空间图；间期时同一批触点各自有一条"平时谁先谁后"的 HFO 传播几何（G_HFO）。我们问：发作早期某频带的能量场，长得像不像间期这张顺序几何图，而且这种像**能不能扛过三道质疑**——(A) 只是空间涂抹？(B) 只是宽带整体招募？(C) 只是 1/f 背景？

**揭示了什么（初判，dev n_perm=100，只 broad raw，未过形式化 gate）**：
- **贴合强度 |maxab| 在所有频带上是平的**（~0.55-0.73），ripple 段并不更高——光看强度没有频带特异性。
- **但比"纯空间平滑"随机基线（spatial null）——中高频带明显超出，低频不超**：beta(13-30) Δ+0.116 p=0.010、LVFA_13_80 Δ+0.107 p=0.020、high-gamma Δ+0.112 p=0.030、ripple_full Δ+0.111 p=0.030（7/9 被试超出）；而 δ/α/宽带 Δ~+0.02-0.04 p~0.22-0.27（5/9）。即**中高频能量场贴 G_HFO 的程度超过空间平滑能给的，低频不超**——是频带特异、方向符合假设的信号。
- **过完 band 间多重比较（max-over-bands FWER）后，只有 beta + LVFA_13_80 稳住（~0.01-0.02），ripple 掉了（~0.75-0.91）**。→ 稳健信号是 **LVFA/fast(13-80Hz)，不是 ripple-specific**（对应 spec §8 的**中档**判读："HFO 几何标记一条以 LVFA/fast 招募表达的致痫通路，非 ripple-specific"）。
- **形式化 Gate A：0/11 全 weak_negative——但不是 p 值不够，是被"强度门"卡住**：min_group=4 下没有一个被试达到 within_shaft_strong（都 subject_wide_weak/distance_bin_fallback，因为 SEEG 杆稀疏），而 spec 的 P1-c 纪律规定只有 within_shaft_strong 能过形式化 Gate A。所以这是**几何/参数限制，不是干净的"对齐失败"**。

**一句话**：backbone 干净跑通，初判看到一个**有前景的、频带特异（LVFA/fast）、方向对的描述性信号**（中高频超空间平滑 null），但**形式化验证被稀疏杆几何卡住**，且**未过全量 n_perm、未跑 Gate B/C、未跑 narrow**——按 spec §1.1 证据阶梯，停在 **candidate mode，形式化 Gate A 未确立**。

---

## 1. 建成的东西（pipeline）

纯数学在 `src/topic5_v2_band_scan.py`；编排在 `scripts/run_topic5_v2_*.py` / `build_topic5_v2_*.py`；config 驱动 `config/topic5_v2_phase1.yaml`。

| 阶段 | 脚本 | 产物 |
|---|---|---|
| **硬门·legacy 复现** | `run_topic5_v2_legacy_repro.py` | 证明 v2 管线复现旧 bb/hfa align_maxab，**max\|delta\|=0.00 broad+narrow**（逐位相等） |
| **多频带 masked cache** | `build_topic5_v2_band_cache.py` | `v2_band_scan/cache/{sid}.npz`（12 band × 每 sz 的 baseline-robust-z 迹）+ sidecar（`analysis_channels` 固定掩膜 + 逐 band QC）。13 被试（broad9+narrow4）全建成 |
| **对齐表** | `run_topic5_v2_alignment.py --feature {raw\|common_resid\|aperiodic_resid}` | 窗→发作→被试中位数的 align_abs_maxab + signed 表 |
| **残差 cache（Gate B/C 输入）** | `build_topic5_v2_common_resid_cache.py`（LOBO 共场残差）、`build_topic5_v2_aperiodic_cache.py`（1/f 校正超量） | 同结构残差 cache |
| **混杂图** | `build_topic5_v2_confound_maps.py` | 每触点 hfo_rate / baseline_power / broadband / **shaft-order（非 along_axis_mm，避免自证循环）** / soz |
| **三层 null** | `run_topic5_v2_nulls.py --feature .. --n-perm ..` | perm-long parquet（max-over-bands 用）+ subject summary（null_z/empirical_p/strength） |
| **Gate 判读** | `run_topic5_v2_gates.py` | `phase1_gate_summary.csv`（Gate A/B/C flag + tier + max_over_bands_p） |

关键科学修正（详见账本）：
- **饱和质检从固定掩膜里剔除**：原设计按 band-power-z `|z|>12` 判"坏道"，会把发作时正在放电的高-ripple 通道当噪声剔掉（139 ripple 段 41→1）。改为 `analysis_channels` 只按**有效性**（有限、非 flatline）筛，饱和标记保留为旁路诊断——避免删掉要测的信号（循环论证）。cohort-wide 0 通道掉（数据干净）。
- **建 cache 提速 ~12×**：频谱图每发作只算一次、所有 band 共用（原来每 band 重算）——单被试 ~10-27min→48s；**legacy_bb 逐位仍与旧 cache 相等**（提速没动数字）。
- **order-null 用 producer nanmedian**（非 mean）匹配 G_HFO 几何（§6 边界参数一致性）。
- **shaft_position 用杆序索引**（非 along_axis_mm，后者是 G_HFO 派生的传播轴，做混杂会自证循环）。

---

## 2. 怎么跑全量（controller 未跑完的）

```bash
cd /home/honglab/leijiaxin/HFOsp-t5v2   # 该 worktree（results/ 软链到主 results）
# cache 已建（13 subj）。observed alignment：broad raw 已跑，narrow raw 已跑。
for ax in broad narrow; do
  python scripts/build_topic5_v2_confound_maps.py --substrate $ax          # 全量混杂图（dev 只 139）
  python scripts/build_topic5_v2_common_resid_cache.py --substrate $ax     # Gate B 残差 cache（未跑全量）
  python scripts/build_topic5_v2_aperiodic_cache.py --substrate $ax        # Gate C 残差 cache（~15-35min，内存 8-12GB/subj，顺序跑）
  for feat in raw common_resid aperiodic_resid; do
    python scripts/run_topic5_v2_alignment.py --substrate $ax --feature $feat
    python scripts/run_topic5_v2_nulls.py --substrate $ax --feature $feat --n-perm 1000   # ★慢：见下
  done
  python scripts/run_topic5_v2_gates.py --substrate $ax
done
```

**⚠️ 全量 null 极慢**：实测 n_perm=1000 ≈ **10.5 min/subject/feature**（E916 有 44 发作，≈2h 一个），broad ≈2.5-3.5h/feature，**全 feature×substrate ≈ 12-18h**。建议先 dev(100) 定性、final(1000) 后台排队。

---

## 3. dev-null Gate A 结果（broad raw n_perm=100，descriptive，未过形式化 gate）

见 §0 摘要 + `results/topic5_ictal_recruitment/v2_band_scan/broad/phase1_{gate_summary,null_subject_summary}.csv`。核心表（spatial null，中位 Δ=obs−null / 中位 empirical_p / #被试 obs>null）：

```
beta_LVFA_low        Δ+0.116  p=0.010  7/9   ← 超空间平滑 + family-wise 稳住(0.0099)
LVFA_13_80           Δ+0.107  p=0.020  7/9   ← 超 + family-wise 稳住(0.0198)
hg_low_ripple        Δ+0.112  p=0.030  7/9   ← 超，但 family-wise 掉(0.119)
ripple_full_80_250   Δ+0.111  p=0.030  7/9   ← 超，但 family-wise 掉(0.911)
ripple_safe_80_220   Δ+0.110  p=0.030  7/9   ← 同上
theta_preictal_PAC   Δ+0.066  p=0.040  6/9
delta/alpha/low_HYP/legacy_bb  Δ~+0.02-0.04  p~0.22-0.27  5-6/9   ← 不超
```
order null：所有频带都正（Δ+0.12-0.19）但 band-generic（非频带特异）、p 多 NS at n=100、且 gate-guarded + 轻度 anti-conservative（见 §4）→ 证据弱于 spatial。

**★ NARROW (n=7) spatial null（跨队列对照，重要）**：**所有频带都超出空间平滑 null**（Δ+0.07-0.16 p~0.01-0.06 5-6/7），**包括低频/宽带**（delta +0.132 p=0.010、low_HYP +0.162 p=0.010、legacy_bb +0.111 p=0.020）——是 **band-generic**，**不复现 broad 的中高频特异**。两点跨队列结论：
- **一致（robust）**：**两个队列 obs 都超出空间平滑 null**（broad 中高频超、narrow 全频段超，p<0.05，多数被试）→ **确实存在超出纯平滑的空间特异对齐**（描述性；形式化 Gate A 仍被 strength 门卡住）。注意"|maxab| 平"不等于"纯平滑"——null 中位数更低（~0.44），obs（~0.55）超出它。
- **不一致（NOT robust）**：**哪个频带**（频带特异性 = Gate B 的问题）——broad 频带特异（LVFA/fast），narrow band-generic（含宽带）。→ **频带特异主张不跨队列稳健**；narrow 的 band-generic 反而更像 spec 的 **broadband-recruitment 档**（G_HFO 预测宽带招募，非频带特异）。
- **诚实底线**：稳健的是"对齐超平滑"（Gate-A 描述性），不稳健的是"频带特异"（Gate-B）。整体倾向 **"G_HFO 标记一条超出平滑的空间招募通路，但频带特异性 + 形式化验证均未确立"**。

---

## 4. 已知局限 + 全量前待定项（承重，勿忽略）

1. **形式化 Gate A 被 min_group=4 卡死（0/9 within_shaft_strong）**。**关键 follow-up = min_group=3 灵敏度**（一个 flag：`nulls.min_group_for_shaft`）——密杆被试可能变 within_shaft_strong → 形式化 Gate A 才可评。现结论"形式化未过"是**几何/参数产物，非对齐失败**。
2. **Gate A cohort 显著性 = 逐被试 p 的中位数（median-of-p）——偏保守**（null 下逐被试 p~U(0,1)，中位数 ~0.5，n≥5 时 median-p<0.05 近乎不可达）。**更合适的每-band cohort permutation p 已在 `_max_over_bands_p` 里算好但没接到 spatial_p**——全量前应换（只会**加强**信号，不会削弱）。Task-14s review 的 Important 项。
3. **order-null 轻度 anti-conservative**：observed 用 producer typical_rank，null 用 event-rebuild rank（只到 corr ~0.95/0.80 复现 producer）→ strong 被试的 order-p 略乐观（gate-guarded：只 ≥0.90 的 strong 被试进决策，gap≤10%）。**spatial null 才是干净的主 Gate A 检验**；order 是 gate-保护的次要。
4. **max-T family 含 4 个与 primary 重叠的 composite**（保守，FWER 仍控住，但损 power）——全量可考虑只用 7 primary。
5. **未跑**：Gate B/C（残差 feature 全量 null）、narrow 全量、final n_perm=1000、全量 confound-adjusted。
6. **signed 方向度量不跨 substrate 稳健**：broad 低频正/高频负，narrow 多为负——因 signed 依赖 per-subject template-a 定向，跨 substrate 不可比。用**方向-不变的 |maxab|**（平的）+ **spatial-null Δ**（频带特异）作可信度量，不报 signed flip 为发现。
7. **composite band（LVFA_13_80 / ripple_full / ripple_safe）结构上封顶在 broadband_recruitment**：残差 cache 只建 7 个 primary，故 composite 的 common_resid_p/aperiodic_p 恒 NaN → 永远过不了 Gate B/C（whole-branch review 的 Important-interpretation 项）。读 gate_summary 时注意：composite 的 tier ≤ broadband_recruitment 是设计使然，非"频带不特异"的证据。Gate B/C 只在 7 primary 上判。

---

## 5. 工程状态

- 27 commits（`e01c08b..HEAD`），线性、全 `topic5-v2-`。纯测 `tests/test_topic5_v2_band_scan.py` **31 passed**；12 个 `@pytest.mark.integration` real-data smoke（各任务内验过）。
- 每个任务都由独立 subagent review（多数含手推导 + 边界测），journal 在 `.superpowers/sdd/`。cleanup pass（c9af042）补齐了饱和-nan robustness、legacy-repro teeth、null/gate/aperiodic/confound 回归测试。
- **worktree 隔离**：因主工作目录有并行 Phase-2 会话（多会话共享 HEAD 会撞车），Phase-1 全程在独立 worktree `HFOsp-t5v2` / 分支 `topic5-v2-phase1-build`，`results/` 软链共享数据。分支整合（merge/PR）待用户定。
- **未碰**：主工作目录的 Phase-2 criticality WIP + field-extrapolation WIP（互补、无文件冲突；本 Phase-1 产物正好满足 Phase-2 depcheck 的输入）。
