# FCXR-LC6B frozen-slow causal atlas — round 1（8 条 clamp + 4 条注册延长）

日期：2026-08-15
spec：`docs/superpowers/specs/2026-08-15-topic4-fcxr-lc6b-frozen-slow-causal-atlas-design.md`
plan：`docs/superpowers/plans/2026-08-15-topic4-fcxr-lc6b-frozen-slow-causal-atlas.md`
manifest：`config/topic4_fcxr_lc6b_frozen_slow_atlas.json`
结果根：`results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas/`

## 1. 一句话结论

在这片组织上，中间高态**确实存在**；自然轨迹之所以一路冲到顶，是因为突触疲劳 `D` 在持续加深，
而不是因为快回路本身没有可停留的分支。

正式标签：`FROZEN_SLOW_BOUNDED_BRANCH_EXISTS_D_DEPLETION_IS_THE_DRIVER`。

## 2. 测了什么、怎么测的

被测现象：一片模拟皮层进入发作后，群体放电率在 5–6 秒内从每秒几十次冲到每秒四百多次并停在那里。
两种解释在自然轨迹里长得完全一样：(a) 快回路只要点着就一定冲顶，中间没有平台；
(b) 本来有平台，但两个慢变量一直在动，把系统推过去了。

做法：取同一条已完成轨迹的两个精确时刻，各复制成四份，**只改"哪个慢变量被按住"**，
其余全部逐位相同 —— 完整的快态（膜电位、不应期、四个突触滤波器、两个延迟环、噪声变量、
随机数发生器状态）、连接图、所有参数、以及未来收到的外部噪声。然后各跑 10 秒。

两个慢变量：
- `D = 1 − z` 是**突触疲劳**：每个细胞收到的抑制被 `z` 缩放，`D` 越大抑制越无力。
- `H` 是**循环增益记忆**：把细胞最近收到的兴奋性循环输入做泄漏累积，再回头加一点点兴奋回去。

如果快回路本身没有平台，那么按住慢变量也应该照样冲顶。实测不是。

## 3. 两个源快照与 checkpoint 语义修正

LC6A 的 `checkpoint_onset_detected.npz` 与 `checkpoint_onset_plus_1s.npz` **文件逐位相同**
（都是 `t = 240000`，即 12.0 s；LC6A 自己已记录 `onset_detected.timing_error_ms = 1000.0`）。
本轮不使用它们，改用两个互相独立、时间从状态 `t × dt` 与 manifest `actual_ms` 双向核对的快照：

| 记号 | `t`(steps) | 绝对时间 | 相对 onset | 进入前 1 s 的全局率 | 该处 `D` | 该处 H 门占用 |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 260000 | 13.0 s | +2.0 s | 34.27 Hz | 0.1364 | 0.5559 |
| S4 | 300000 | 15.0 s | +4.0 s | 58.57 Hz | 0.2752 | 0.9586 |

C0 的 onset 在 11.0 s，`dt = 0.05 ms`，两处 `timing_error_ms` 均为 0。

## 4. 主结果

每条臂 10 秒，逐秒群体 E 放电率（Hz）：

**S2（13 s 出发）**

| 臂 | 逐秒 | 标签 |
|---|---|---|
| NAT（都放开） | 46.1, 58.6, 74.9, 177.6, 367.5, 420.4 | `ESCALATING_SATURATION` |
| H_CLAMP | 41.8, 47.0, 59.4, 165.6, 325.7, 397.6 | `ESCALATING_SATURATION` |
| D_CLAMP | 43.1, 49.4, 57.1, 57.7, 55.2, 59.8, 56.1, 55.2, 60.6, 55.2 | `BOUNDED_OSCILLATORY` |
| DH_CLAMP | 39.3, 39.2, 42.1, 40.7, 39.4, 43.1, 39.8, 39.6, 43.2, 39.4 | `BOUNDED_OSCILLATORY` |

**S4（15 s 出发）**

| 臂 | 逐秒 | 标签 |
|---|---|---|
| NAT | 74.9, 177.6, 367.5, 420.4, 435.7, 444.0 | `ESCALATING_SATURATION` |
| H_CLAMP | 74.4, 177.2, 366.9, 420.2, 435.6, 443.9 | `ESCALATING_SATURATION` |
| D_CLAMP | 71.3, 65.1, 68.5, 68.1, 65.3, 71.4, 65.0, 70.5, 66.0, 65.1 | `BOUNDED_OSCILLATORY` |
| DH_CLAMP | 70.8, 64.5, 67.9, 67.5, 64.7, 70.8, 64.4, 69.8, 65.4, 64.5 | `BOUNDED_OSCILLATORY` |

`NAT` / `H_CLAMP` 只跑了注册的 6 秒（它们越过注册饱和线即停）；四条有界臂走完了 6 + 4 = 10 秒。

四条有界臂在整段窗口里：全局 1 秒均值最高 71.4 Hz（注册饱和线 250 Hz）；
达到近不应期率（450 Hz）的细胞比例**每一秒都恰好为 0**；
逐细胞 q99 在 S2 的两条臂是 75–82 Hz，在 S4 的两条臂是 91–100 Hz。
两条冲顶臂的逐细胞 q99 在末秒是 446 Hz（S2）与 456–457 Hz（S4），即坐在不应期天花板上。

按 spec §10：`DH_CLAMP` bounded 且 `H_CLAMP` runaway、`D_CLAMP` bounded
→ 分支 **B** + `CONTINUED_D_DEPLETION_IS_THE_MAIN_DRIVER`（两个快照一致）。

## 5. H 的效应是分级的，不是零

`gH = rho_h_lc2 · S̃(h)`，`rho_h_lc2 = 0.54` 是硬上限。门占用在 S2 是 0.5559（还剩 44.4% 行程），
在 S4 是 0.9586（只剩 4.1%）。这在跑之前就写进 spec §5 与 manifest 的 `prior_expectation`。

实测（在按住 D 之上再按住 H 对平台高度的影响）：

| 快照 | 门的剩余行程 | 自由门实际走了 | 平台变化 |
|---|---:|---:|---:|
| S2 | 44.4% | +0.444102 | 53.72 → 40.64 Hz（**−24.4%**） |
| S4 | 4.1% | +0.041435 | 68.29 → 67.69 Hz（**−0.9%**） |

因此**不得**写"H 与升级无关"。正确表述：H 对平台高度有真实且与剩余行程成正比的贡献，
但它的执行端有硬上限，无论如何都不足以把系统推成 runaway；D 的贡献才是"有界平台"与"冲顶"之间的差别。

## 6. 有界态是爆发串，不是平滑高态

四条有界臂的末 2 秒里，**53–56% 的 20 ms 窗**落回间期带（9.74 Hz）以下。
所以标签是 `BOUNDED_OSCILLATORY` 而非 `BOUNDED_STATIONARY`。

这个区分是预注册的，原因是 FCXR-LC3 的教训：一个用 300 ms 滚动均值看起来"持续"的发作载体，
实际结构是每 86 ms 从完全静默重新点火的爆发串（57% 的时间三万两千个细胞零放电）。
本轮的群体放电率因此在 20 ms 分辨率上读，静默 bin 比例与最长静默段与标签一起报出。

## 7. 注册延长为什么改变了标签

四条有界臂在 6 秒窗上全部被判 `RIGHT_CENSORED / STILL_ESCALATING_AT_WINDOW_END`
（漂移 CI 上界 0.097–0.118，门 0.05）。spec §6.2 预注册的单次 4 秒延长后，
四条的 CI 上界变为 −0.0154 ~ −0.0001，全部翻成 `BOUNDED_OSCILLATORY`。判据一个字未改。

原因：注册的漂移判据读的是**固定 2 秒长**的尾窗，但这个尾窗的**位置**随窗口移动。
6 秒窗的尾巴里仍含钳制施加后的弛豫暂态 —— S2 的 D_CLAMP 前 3 秒从 43 涨到 57 Hz，
那是系统在新的（被按住的）慢状态下重新找平衡，不是持续升级。尾窗往后挪 4 秒就越过了它。

**本轮的一处判断更正**：执行期间曾判断"尾窗固定 2 秒，所以延长不会改变标签"。
该判断错误 —— 尾窗的长度固定，位置不固定。此处按实际记录留痕。

补充诊断（**明确标注为非注册判据**，与注册标签并列而非取代）：
逐臂记录首秒→末秒斜率与全窗 Theil–Sen。S2 的 DH_CLAMP 是 39.338 → 39.415 Hz，即 **+0.0085 Hz/s**。

## 8. 工程与硬合同

- **引擎改动**：新增 `MZSlowVarsConfig.h_lc2_frozen_E`，默认 `None`。`use_h_lc2` 保持 `True`，
  `membrane_terms` 一字未改（包括冻结场产生的 `gH`），只在 `step()` 里**跳过**状态更新 ——
  跳过而非写完再覆盖，所以不存在"膜项已经消费了移动过的 h"这一帧泄漏。
  D 侧复用引擎已有的 `z_frozen_E`（要求 `use_z=False`）。
- **byte parity**：`h_lc2_frozen_E is None` 时，改动后的引擎在**真实 LC6A C0 路径**上
  （从 `checkpoint_onset_plus_2s` 续跑 2000 步）逐位复现改动前引擎：
  spike sha256 `43db32e8…`、rate sha256 `6ee3736c…`、末状态哈希 `f5a2fc58…` 三项全同。
  因此已完成的 LC6A 产物仍然可复现。
  引擎 sha256：改前 `87d38246…` → 改后 `063d5a69…`。
- **没有复用 `classify_high_state`**：它要求 rate **且** D **且** H 三条漂移都平；
  钳住后 D 与 H 逐位恒定、Theil–Sen 斜率恒等于 0，两条判据会被**钳制动作本身**满足，
  直接复用会把干预制造成结论。另外它以"进入时刻"为起点判读，而本轮所有臂从高态内部起跑。
  新写的 `classify_clamp_window` 签名里不接受任何 D/H 轨迹，并有测试锁住这一点。
- **配对对照保真**：两个快照的 NAT 臂都**逐秒精确复现**原 C0 自然轨迹
  （S2 五个重叠秒、S4 三个重叠秒，逐位一致）。
- **G1 配对未来输入**：同一快照的四条臂在**共同窗口**上共享逐位相同的输入哈希
  （S2 `28015fc4…`，S4 `8196a63a…`）。延长臂跑的是更长的窗口，其自身哈希自然不同；
  合同在共同窗口上校验，这一点在 finalize 里显式实现。
- **操作检查**：8 个被钳住的场，跨窗变化量恰好 `+0.000000`。
- **回归**：引擎相关测试 22 个文件 494 passed / 4 failed，4 个全部是缺 fixture 文件
  （位于其他 worktree 的 `results/` 下，`results/` 被 gitignore），没有一个跑到断言。
  仓库内 126 个 `MZSlowVarsConfig(...)` 调用点经 AST 检查全部使用关键字参数，
  因此在 dataclass 中间插入字段不会移动任何调用方的参数。

## 9. 可以说与不能说

**可以说**：在 canonical seed 的 C0 图与锁定的 legacy substrate 下，把两个慢变量按在这条轨迹
13 s / 15 s 自身的值上，快回路在 10 秒内停在 40–71 Hz 的爆发式有界态，不接近注册饱和线；
只按住 D 就足以做到，只按住 H 做不到；H 的贡献与其剩余行程成正比；两个快照结论一致。

**不能说**：这是队列主张（单 seed、单图、单条轨迹的两个时刻）；有界态对弱扰动稳健
（`perturbation_return_tested: false`）；LC6B 测过 termination 或 lifecycle（都是 `NOT_TESTED`）；
"H 与升级无关"；这个有界态是平滑持续高态（它是爆发串）；这些标签适用于 6 秒窗
（6 秒窗的注册判据会把它们判成 `RIGHT_CENSORED`，见 §7）。

## 10. 下一步

spec §10 的分支 **B** 成立：`natural_path_atlas_authorized: true`，
`next_authorized_action: ENTER_CONDITIONAL_NATURAL_PATH_ATLAS`。
H-EFF / H-CAP 属于分支 A，本轮**未触发**，按 spec §12 不写不跑。

`termination`：`NOT_TESTED`。`lifecycle`：`NOT_TESTED`。`perturbation_return`：`NOT_TESTED`。
