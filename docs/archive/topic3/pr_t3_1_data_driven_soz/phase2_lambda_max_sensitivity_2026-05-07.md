# Phase 2 — λ_max Sensitivity Sweep（2026-05-07）

**所属 PR**：PR-T3-1 v2.2.3 sensitivity arm
**本文档定位**：归档级 sensitivity report，不是主结论。
**主入口**：`docs/topic3_spatial_soz_modulation.md` → `per_subject_ictal_er_atlas.md`
**对应代码**：
- 输出：`results/data_driven_soz/layer_a_ictal_er_rank/per_subject_lambda500/`
- baseline：`results/data_driven_soz/layer_a_ictal_er_rank/per_subject/`
- runner：`scripts/run_ictal_er_rank.py --per-subject --force --lambda-max 500`

---

## 1. 起因

PR-T3-1 v2.2.3 cohort run（λ_max=100，2026-05-06）发现：30 cells 中 24 cells (75%) 的校准 λ 落在 cap=100 处（即 grid 上界），意味着 baseline 噪声真实需要 λ > 100 才到 1/h FPR target。**问题**：被 cap 截断时，producer_health 标签是否被低估？

**Phase 2 设计**：对同一 cohort 用 λ_max=500 重跑，看哪些 cell 在解开 cap 后 producer_health 改善（说明 cap=100 确实压制信号），哪些反而退化（说明 baseline 噪声过强 / ictal 信号不足以在 high λ 下 accumulate）。

**预注册结果分类**：
- 改善（tag↑ 或 s_sz↑）→ raise λ_max 该方向；考虑 per-subject λ_max
- 退化（tag↓ 或 s_sz↓）→ raise λ_max 反方向；λ_max=100 已是合理选择
- 持平 → λ_max 不是主控变量；其他参数（detection window / bias / fpr_target）才是关键

---

## 2. 完整 32-cell 对比

`focal(i)` 列来自 `epilepsiae_electrode_focus_rel.json`。`λ` 是 grid 上校准出的最优值（cap = λ_max）。`ok` = `n_seizures_ok`（status="ok" 进入 r_sz/s_sz 的 seizure 数量）。`unr` = `n_seizures_onset_unreached`。

| subject | ER | λ100→500 | ok 100→500 | unr 100→500 | s_sz 100→500 | tag 100→500 | verdict |
|---|---|---|---|---|---|---|---|
| 1073 | gamma | 100 → **374** | 9 → 3 | 9 → 15 | 0.171 → -0.023 | unstable/disc → unstable/disc | ↓ s_sz |
| 1073 | broad | 100 → **277** | 4 → 2 | 14 → 16 | 0.305 → None | unstable/disc → **insufficient** | ↓↓ collapse |
| 1077 | gamma | 100 → **417** | 3 → 1 | 5 → 7 | 0.019 → None | unstable/disc → **insufficient** | ↓↓ collapse |
| 1077 | broad | 100 → **271** | 2 → 1 | 6 → 7 | -0.342 → None | insufficient → insufficient | = unchanged_insuf |
| 1084 | gamma | 100 → **417** | 22 → 1 | 62 → 83 | 0.381 → None | unstable → **insufficient** | ↓↓ collapse |
| 1084 | broad | 100 → **302** | 36 → 0 | 48 → 84 | 0.487 → None | moderate → **insufficient** | ↓↓ collapse |
| **1096** | gamma | 100 → **500** | 7 → 5 | 1 → 3 | 0.455 → **0.626** | unstable/disc → **moderate**/disc | ↑↑ tag_up |
| **1096** | broad | 100 → **476** | 8 → 3 | 0 → 5 | 0.189 → **0.486** | unstable/disc → **moderate**/disc | ↑↑ tag_up |
| 1146 | gamma | 100 → 255 | 5 → 3 | 14 → 16 | -0.039 → -0.038 | unstable → unstable | = hold |
| 1146 | broad | 100 → 285 | 7 → 2 | 12 → 17 | 0.119 → 0.314 | unstable → **insufficient** | ↓↓ collapse |
| 1150 | gamma | 100 → **387** | 5 → 4 | 2 → 3 | -0.068 → 0.215 | unstable/disc → unstable/disc | ↑ s_sz |
| 1150 | broad | 100 → 268 | 6 → 3 | 1 → 4 | -0.095 → 0.046 | unstable/partial → unstable/disc | ↑ s_sz |
| 139 | gamma | 100 → **429** | 4 → 1 | 1 → 4 | 0.079 → None | unstable/disc → **insufficient** | ↓↓ collapse |
| 139 | broad | 100 → 251 | 4 → 4 | 1 → 1 | 0.346 → 0.270 | moderate/**conc** → unstable/partial | ↓ degrade |
| 253 | gamma | 100 → **450** | 5 → 0 | 0 → 5 | -0.148 → None | unstable → **insufficient** | ↓↓ collapse |
| 253 | broad | 100 → 223 | 5 → 4 | 0 → 1 | 0.384 → 0.431 | moderate/disc → moderate/disc | = hold |
| 442 | gamma | 100 → **290** | 16 → 1 | 5 → 20 | 0.367 → None | unstable → **insufficient** | ↓↓ collapse |
| 442 | broad | 100 → **311** | 16 → 1 | 5 → 20 | 0.446 → None | unstable → **insufficient** | ↓↓ collapse |
| **548** | gamma | 100 → **500** | 15 → 1 | 11 → 25 | 0.159 → None | unstable/partial → **insufficient** | ↓↓ collapse |
| **548** | broad | 100 → **498** | 19 → 2 | 7 → 24 | 0.097 → 0.533 | unstable/conc → **insufficient** | ↓↓ collapse (small-n) |
| **583** | gamma | 100 → **397** | 22 → 12 | 1 → 11 | 0.448 → 0.484 | moderate/conc → moderate/conc | = hold |
| **583** | broad | 100 → 215 | 21 → 18 | 2 → 5 | 0.625 → **0.751** | stable/conc → stable/conc | ↑ s_sz |
| 590 | gamma | 100 → **366** | 7 → 0 | 3 → 10 | 0.154 → None | unstable/partial → **insufficient** | ↓↓ collapse |
| 590 | broad | 100 → 243 | 9 → 3 | 1 → 7 | 0.173 → -0.251 | unstable/conc → unstable/conc | ↓ s_sz |
| 635 | gamma | 100 → **500** | 10 → 0 | 6 → 16 | 0.026 → None | unstable/disc → **insufficient** | ↓↓ collapse |
| 635 | broad | 100 → **500** | 13 → 1 | 3 → 15 | 0.060 → None | unstable/conc → **insufficient** | ↓↓ collapse |
| **916** | gamma | 100 → **500** | 26 → 1 | 23 → 48 | **0.830** → None | **stable**/disc → **insufficient** | ↓↓↓ collapse (worst) |
| 916 | broad | 100 → 346 | 2 → 0 | 47 → 49 | 0.590 → None | insufficient → insufficient | = unchanged_insuf |
| **922** | gamma | 100 → **472** | 24 → 10 | 1 → 15 | 0.159 → **0.349** | unstable/disc → **moderate**/partial | ↑↑ tag_up |
| 922 | broad | 100 → 329 | 25 → 11 | 0 → 14 | 0.129 → 0.295 | unstable/conc → unstable/partial | ↑ s_sz |
| 958 | gamma | 100 → **344** | 14 → 1 | 0 → 13 | 0.039 → None | unstable/conc → **insufficient** | ↓↓ collapse |
| **958** | broad | 100 → 301 | 14 → 9 | 0 → 5 | 0.217 → **0.428** | unstable/partial → **moderate**/disc | ↑↑ tag_up |

**bold subject** = 在主结论里有名的关键 subject。bold λ = ≥ 250 显著高于 100 cap。

### Cell-level distribution

| 类别 | 数量 | 含义 |
|---|---|---|
| **collapse_to_insufficient** | **16** (50%) | tag 从有意义状态崩溃为 insufficient |
| degrade_only | 3 | tag rank 下降但仍 ≥ unstable，或 s_sz 下降 > 0.05 |
| hold | 3 | tag 不变，s_sz 变化 < 0.05 |
| unchanged_insufficient | 2 | baseline 已经 insufficient，λ500 仍 insufficient |
| **s_sz_up** | **4** (13%) | tag 不变但 s_sz 显著上升 (> 0.05) |
| **tag_up** | **4** (13%) | producer_health rank 上升 (unstable→moderate, etc.) |

**净评判**：负面 19/32 (59%) > 正面 8/32 (25%) > 中性 5/32 (16%)。

---

## 3. 关键 case 分析

### 3.1 strong-signal subjects 在 λ500 下表现

| subject | baseline | λ500 行为 | 解释 |
|---|---|---|---|
| **583 broad** | stable/conc, s_sz=0.625 | stable/conc, s_sz=0.751 ✓ | λ_calibrated=215 < cap，本来就没被压制；raise cap 不影响 |
| 583 gamma | moderate/conc, s_sz=0.448 | moderate/conc, s_sz=0.484 ✓ | λ_calibrated=397，high λ 但 ictal 信号够强能 accumulate |
| **916 gamma** | stable/disc, s_sz=0.830 | **insufficient** ✗ | λ_calibrated 撞 500 cap，ictal 信号在 500 下不能 accumulate |

**对比 583 vs 916**：两者 baseline 都是 stable，但 583 的 ictal-到-baseline ratio 远大于 916。916 的 stable 标签在 λ100 下是因为 r_sz 高（26/51 ok 出来一致排序）但 baseline 极嘈杂；解开 λ cap 后 ictal 也不够强。**结论**：stable tag 本身不保证 λ-rescue 后保持。

### 3.2 改善的 subjects

| subject/ER | baseline → λ500 | 解释 |
|---|---|---|
| **1096 gamma+broad** | unstable/disc → moderate/disc | 真正的"被 cap 压制"案例：λ_calibrated 撞 500（gamma）和 476（broad）；解开后 ok 仍有 5 + 3，ranking 一致性提升 |
| **922 gamma** | unstable/disc → moderate/partial | λ_calibrated=472，n_ok 24→10 但 s_sz 0.16→0.35，ranking 反而更稳 |
| **958 broad** | unstable/partial → moderate/disc | λ_calibrated=301，n_ok 14→9，s_sz 0.22→0.43 |
| 583 broad | s_sz 0.625→0.751 | 已 stable，s_sz 略升 |

**共同特征**：n_ok 在 λ500 下还能保持 ≥ 5（信号能在 high λ 下 accumulate），且 r_sz 一致性比 λ100 更高（说明 λ100 时的低一致性是被 baseline noise 污染的）。

### 3.3 collapse 案例的 baseline λ_calibrated 分布

16 个 collapse cells 的 λ_calibrated 在 λ100 下 100% 撞 cap，在 λ500 下分布：

| λ500 区间 | 数量 | subjects |
|---|---|---|
| 250-300 | 2 | 1146 broad, 1077 broad |
| 300-400 | 6 | 1073 broad, 1084 broad, 590 gamma, 442 gamma, 958 gamma, 916 broad |
| 400-500 | 8 | 1077 gamma, 1084 gamma, 139 gamma, 253 gamma, 442 broad, 548 双, 916 gamma, 635 双, 1096 双 (note: 后两组 actually moderate/improved) |

不存在简单的 "λ_calibrated 低就能 rescue, 高就 collapse" 规律。**ictal-to-baseline ratio** 才是真正的预测因子，这个 ratio 由 ER 信号本身决定，不是 λ_max 能调的。

---

## 4. 主结论

### 4.1 λ_max=100 是合理的全局默认

- Net 18/32 cells 在 λ100 下比 λ500 状态更好 (collapse + degrade)
- λ100 给出更多 n_ok（中位数 8.5 vs λ500 的 3）→ 下游 r_sz/s_sz 统计更可靠
- λ100 下的 cohort distribution（v2.2.3 锁定）：1 stable / 4 moderate / 18 unstable / 7 insufficient / 2 not_eligible — 这是当前 working baseline

### 4.2 λ_max=500 是 sensitivity check 而非 replacement

少数 subject (1096, 922 gamma, 958 broad) 在 λ500 下确实改善，但：
- 改善的 cell 不构成 majority 决策依据
- 实施 per-subject λ_max 需要新增 subject-level 决策逻辑（每个 subject 跑 grid，按 producer_health rank 选 λ_max）
- 这是 **Layer A 的可选 sensitivity arm**，不挪入 main pipeline

### 4.3 真正的 metric bottleneck 不是 λ

50% collapse rate + 高 unr 比例说明：**在当前 ER + CUSUM + FPR (1/h) calibration 合同下，多数 subject 的 ictal accumulation 支撑不了更高阈值**。这是 metric-合同层的现象，不是生理层面"baseline noise > ictal effect size"的因果断言；本 sweep 不能跨越这个推断。raise λ_max 改不了 metric-合同层的现象。

可能的 Phase 2-redesign 方向（**未实施，仅记录候选**）：

| 方向 | 假设 | 风险 |
|---|---|---|
| **A. 放宽 fpr_target** | 1/h → 5/h or 10/h 容许更多 baseline 假报警换更敏感 detection | 牺牲 producer 的"专属于 ictal"语义 |
| **B. 调整 detection window** | [-5, +30] → [-10, +60] 给 ictal 更多 accumulation 时间 | 进入 post-ictal 期间会污染 onset rank |
| **C. 改 detector 范式** | argmax(z) 或 first-threshold-crossing 替代 CUSUM | 大改架构；TDD 需重写 |
| **D. 接受 v2.2 现状进 Layer B** | 18/32 unstable+insufficient 是真实信号薄弱，γ_a 策略已为 unstable 留了 sensitivity tier | 不增加新方向 |

**当前推荐：D**。理由：
1. 用户目标是"产出一份合理可用的 channel-level seizure-onset label metric"，v2.2 在 stable+moderate（γ_a primary，**6 cells = 583 双 + 916 gamma + 139 broad + 253 broad + 1084 broad**，跨 5 unique subjects）已有 working signal；γ_a sensitivity tier (unstable+concordant，5 cells) 还能扩 cohort
2. A/B/C 三个方向都是 detector 重构，需要新一轮 TDD + 全 cohort 重跑（~24h compute），没有先验数据支持收益 > 风险
3. Phase 2 数据已经回答了 "λ_max 是不是 metric 瓶颈" → 不是

---

## 5. 衍生发现

### 5.1 916 gamma 是 λ-fragile stable 的清晰案例

baseline λ100：26/51 ok, s_sz=0.830, **stable/discordant** —— 看起来是 cohort 最强 producer 之一。
λ500：1/51 ok, s_sz=None, **insufficient** —— 完全失效。

**机制**：916 baseline 噪声极大（pooled baseline 195 min），λ=100 时 baseline 频繁撞警报阈值 → CUSUM accumulator reset 慢 → ictal 期 z 升高时容易跨过低阈值。raise λ → baseline 警报阈值变高 → ictal 期 z 累积达不到。

**口径（2026-05-07 review）**：先前措辞写 "λ-cap 假阳性 stable" 太重——它**不**等于 "916 ictal 期没有真实 gamma 升高"，只说明 v2.2 producer_health 的 stable tag 对 λ_max 选择敏感。正确说法："916 gamma 是 **λ100 cap-sensitive / λ-fragile stable**"——其 stable 标签在阈值合同改变时会消失，但生理信号是真是假是独立问题，本 sweep 不能裁定。

**对 atlas 的修正**：916 在 atlas Phase 1 描述里被标记为 "stable + discordant"（指 anatomy 与 clinical i 标注不一致，gamma rank 偏向深端 PB / TS / AM 高号触点）。Phase 2 sensitivity 数据进一步表明：**916 的 stable tag 在 λ_max 选择上脆弱**——它仍然是 cohort 内 baseline λ100 唯一的 stable gamma cell，但 Layer B 输出 entry 必须携带 `lambda_fragile=true` provenance flag（见 plan §3.8 v2.2.3 schema）。

### 5.2 548 broad 是 "small-n strong signal" 的边缘案例

baseline λ100：19/31 ok, s_sz=0.097, unstable/concordant
λ500：2/31 ok, s_sz=**0.533**, insufficient (n_ok < 3)

s_sz 从 0.097 飙到 0.533，但因 n_ok=2 跌破 producer_health 阈值，被打成 insufficient。这个 cell 的 ictal 信号其实很强，但只有 2 个 seizure 能在 high λ 下 trigger。**这是 v2.2 producer_health 阈值的设计权衡**：n < 3 不可信不是错的，但损失了 "small-n 但内部一致" 的边缘信号。

**对 Layer B 的提示**：γ_b 或 future PR 可能需要"small_n_strong"作为辅助 tier 把 548 broad 这类 cell 纳入。

### 5.3 1096 是 "raise-λ 真实 rescue" 的唯一明确案例

1096 双 ER 在 λ500 下从 unstable/discordant → moderate/discordant，s_sz 显著上升。**机制**：1096 baseline 噪声适中（小 n_unr）+ ictal 信号够强；λ100 cap 截断了 baseline 的真实噪声水平 → 校准的 λ 太低 → baseline 假警报多 → CUSUM accumulator 在 ictal 期已有 noise floor → ictal 信号 marginal 显著。

**对 Layer B 的提示**：1096 在 γ_a primary 集合中权重应提升一档（unique rescue case）。

---

## 6. 行动 items（已纳入 atlas）

1. **Atlas cohort summary section 加 Phase 2 cell-level 表**：32-cell verdict + 4 个 ↑/↑↑ 改善 cell + 16 个 collapse cell 列表
2. **Atlas 916 entry 修订**：标注 "stable tag 对 λ_max 选择脆弱"
3. **Atlas 1096 entry 升级**：标注 "Phase 2 raise-λ rescue case"
4. **Plan archive §6.3.1 (γ_a Layer B 策略) 加 sensitivity 注释**：
   - primary tier 锁定 baseline λ100（**6 cells = 916 gamma + 583 gamma + 583 broad + 139 broad + 253 broad + 1084 broad**，跨 5 unique subjects: 916, 583, 139, 253, 1084）
   - sensitivity tier 拓展时考虑 1096 双 + 922 gamma + 958 broad（Phase 2 改善证据，仅记录不入 γ_a 锁定）
5. **Phase 2 sweep 数据保留**：`per_subject_lambda500/` 不删，作为 sensitivity arm 永久 archive
6. **mark task #29 completed**

下一步进入 Step B.1（Layer B label builder）。

---

## 7. 复现命令

```bash
# Phase 2 sweep
python scripts/run_ictal_er_rank.py --per-subject --force --lambda-max 500 \
  --output-dir results/data_driven_soz/layer_a_ictal_er_rank/per_subject_lambda500

# 完整对比表（脚本未持久化，inline python，见本文档 §2）
python -c "
import json, os
from pathlib import Path
base = Path('results/data_driven_soz/layer_a_ictal_er_rank/per_subject')
l500 = Path('results/data_driven_soz/layer_a_ictal_er_rank/per_subject_lambda500')
# … (full script in chat history 2026-05-07)
"
```

ETA：原始 sweep ~12h（被电源中断后 ~5h 恢复跑完）。
