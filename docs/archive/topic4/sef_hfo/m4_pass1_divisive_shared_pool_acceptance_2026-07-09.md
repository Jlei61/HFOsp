# M4 pass1 除法共享抑制池验收记录（2026-07-09）

## 结论

M4 pass1 **可以验收为机制筛选通过**：全局除法共享抑制池 `S_G` 能把动态 `q_I` 耗竭导致的 runaway 压成一个非失控、有界、持续的第三状态候选。

但它**不能验收为完整 seizure cycle**：该状态空间宽、参数窗窄、marginal，且扰动后不能回到间期。安全表述是：`S_G` 证明了“runaway 和 transient 之间确实能出现 bounded sustained attractor”，但还没有证明“可撤回/自终止的 seizure-like 发作周期”。

## 验收边界

已验收：

- 动态除法池机制能产生非失控第三态，而不只是延迟 runaway。
- 机制对照支持“动态除法 recurrent-gain 池”是关键条件：等量减法刹车仍失控，静态 clamped 池则把活动压死。
- `k_q=0.10, alpha_G=16` 是当前最稳健锚点：40 s 多 seed 中 3/4 seed 无 runaway。

未验收：

- 不是 localized ictal core：有界态是满宽横条/宽空间占据，位置随 seed 漂移。
- 不是可撤回发作：`qI_refill` 后回到有界态，`inhibitory_pulse` 后反跳失控。
- 不是严格 Hopf 证明：连续性/经验本征模只支持“delayed-feedback oscillation / Hopf-like”措辞，不能替代全模型 Jacobian 或分岔证明。

## 关键证据

### 1. 40 s 多 seed 长确认

`results/topic4_m4_dynamic_long40k_seed*/dynamic_qi_summary.json`

| cell | 40 s 结果 | 解释 |
| --- | --- | --- |
| `kq0.10_aG16.0` | 3/4 seed 无 runaway；seed2 在 5781.3 ms 延迟失控 | 当前最稳健的 bounded 锚点，但仍是 marginal |
| `kq0.25_aG16.0` | seed1/2 无 runaway；seed3 在 36480.3 ms 失控；seed4 在 10997.2 ms 失控 | 15 s 可过，但不能写成 40 s 稳健确认 |

因此，pass1 的主证据应锚定在 `kq=0.10, alpha_G=16`，不要把 `kq=0.25, alpha_G=16` 和它写成同等级别。

### 2. 机制对照

`results/topic4_m4_dynamic_mechanism/dynamic_qi_summary.json`

| arm | 结果 | 解释 |
| --- | --- | --- |
| `mech_divisive` | no_runaway，max 97.2 Hz，tail area 0.6434 | 动态除法池可 bound |
| `mech_no_pool` | 386.3 ms one_shot_burst，max 403.1 Hz | 无池直接 runaway |
| `mech_matched_subtractive` | 406.3 ms one_shot_burst，max 270.9 Hz | 等量减法刹车不够 |
| `mech_clamped_SG` | no_runaway，但 max 0.2 Hz，tail area 0 | 静态强压制不是第三态 |

结论：pass1 支持的不是“任意强全局抑制都能 bound”，而是“活动依赖、动态、除法式 recurrent-gain 抑制池”能打开一个窄的 bounded window。

### 3. 可撤回性

`results/topic4_m4_dynamic_reversibility/dynamic_qi_summary.json`

| perturbation | 结果 | 解释 |
| --- | --- | --- |
| `aG16_qI_refill` | no_runaway，但回到有界态 | 灌满 `q_I` 不能恢复间期吸引子 |
| `aG16_inhib_pulse` | 8534.7 ms one_shot_burst，max 317.3 Hz | 强抑制脉冲后反跳失控 |

结论：当前 M4 pass1 证明的是 “bound”，不是 “terminate”。

### 4. 连续性/经验本征模

`scripts/paper_figures/analyze_m4_continuity.py` 生成：

- `results/paper-ready-figure/fig_m4_dynamic_qi/figures/fig_m4_continuity_eigenmode.png`

该图用于检查上边界附近 runaway time、振荡幅度和经验 leading mode。它支持“上边界带有延迟反馈振荡放大的 Hopf-like 迹象”，但 sigma 穿零并不干净，因此只能作为经验诊断，不能作为严格分岔证明。

## 最终可写口径

推荐写法：

> M4 pass1 validates a non-runaway bounded sustained attractor opened by an activity-dependent divisive shared inhibitory pool. This is a real intermediate state between transient IED-like activity and runaway, but it remains broad, narrow-window, marginal, and non-terminating. Therefore it is a bounded third-state mechanism screen, not a complete seizure-like cycle.

禁止升级为：

- “M4 已产生真正 seizure state”
- “模型已证明 Hopf bifurcation”
- “全局抑制池能让发作自然终止”
- “有界态是 localized ictal core”

## 下一步

1. 若目标是完整发作周期：加独立终止慢变量（例如 adaptation / `g_K` / 慢恢复负反馈），先做 cheap TDD + fast screen，再跑长确认。
2. 若目标是机制论文图：保持 pass1 为机制筛选图，配机制对照、40 s 多 seed 和可撤回性阴性结果。
3. 若要升级 Hopf 叙述：需要冻结操作点的更正式低维/全模型 Jacobian 或参数延拓，不应只靠当前经验本征模图。
