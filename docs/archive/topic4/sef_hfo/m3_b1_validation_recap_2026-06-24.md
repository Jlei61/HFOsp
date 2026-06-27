# M3 mini-W_event — B1b/c/d validation report (2026-06-24)

## STATUS — M3B B1 收口冻结（2026-06-24, supersedes §9 的 static-μ pivot）

**这一节是面向用户的收口锁。** 2026-06-24 的 M3 拆分（M3A=慢变量发作机制 / M3B=W 场读出）之后，
B1（"点着之后火往哪传"这件事到底有没有形成一个能读的方向性传播场）正式收口为：

> **M3B B1 verdict = B-BOUNDED NEGATIVE**：当前尺度下，戳出来的有限小爆发只是一团**局部招募 / 易感性**
> 现象（形状可复现、病灶核只压低点火门槛、不改火烧形状），**不是一个方向性的场算子**——它没有打败"离源点
> 距离"，也分辨不出连接的 45° 长轴。

**朴素三段式（外部读者也能复述）**：

1. **测了什么** — 在一块模拟脑组织薄片上戳一下，激起一团"就地传一小圈然后自己平息"的有限小爆发（代表发作间期
   HFO）。看三件事：① 这个"火烧的形状"换随机种子稳不稳；② 给薄片加一个"病灶核"（一小撮更易兴奋的神经元）是
   只把"点火门槛"压低，还是连"火往哪烧"的形状也改了；③ 这个形状能不能比"单纯离戳点远近"更准地预测哪个格子先被招募。
2. **怎么测的** — 每件事都跟"如果只是各向同性的随机扩散 / 随机洗牌"比。形状稳定性：同条件换种子，实测形状相似度
   0.85–0.90，随机洗牌只有 0.24–0.25 → 稳。核改不改形状：核底物在它自己的低门槛、空白底物在它自己的高门槛，各取
   成功事件比平均形状，相似度 0.99，几乎贴着"同底物换种子"的噪声天花板 → 核没改形状。方向性：拿学到的形状和"纯距离"
   分别预测实际招募顺序，两者排序相关都 ≈0.53–0.55、差距跨 0；而且对角方向（真正 45° 长轴）的权重恰好按"纯距离该多
   远就多弱"被压低，没有任何超出距离的方向残差。
3. **揭示了什么** — 在当前网格（4mm 格宽、事件半径 ≈1 个格）下，这团有限事件看起来就是一团**各向同性的局部扩散**：
   形状可复现、核只压门槛不改形状，但它没打败"距离"、也看不出 45° 长轴。所以现在**不能**把它叫"方向性传播算子"，
   也**不能**拿它去画 μ 相图。这个"没过"是**分辨率 / 可重复性天花板**（每事件招募顺序本身只 ≈0.55 可重复）限制的，
   **不是**"模型连接各向同性"被证明，**也不是**单纯样本量小。

**两条硬禁止（本轮 + 进入相图前都成立）**：

- ❌ 不能写 "W_event 是已证明的方向性传播算子 / directional propagation operator"。当前只支持 "有限招募 / 易感性算子"。
- ❌ 不能进入 μ 相图 / 慢变量阈值机制。§9 早先建议的 static-μ（即 `h(W)→阈值 μ`）路径在 M3 拆分后已被 M3B plan
  Task 0 **降级为历史负对照 / control**；发作样慢变量机制移交 **M3A**，M3B 只做 W 读出，不发明 `s_slow`、不用 `h(W)` 阈值当机制。

**provenance 锁（每个 W_event 输出都必须带）**：attempts、success seeds、K_min/K50、source-bin 排除、event-aligned 窗、
spontaneous ignition 排除——成功条件化的 W_event 永远不能当因果传播算子来读。本轮 P4 自发点火审计干净（10 个 pilot run
里 0 个被污染 core_only seed）。

代号补注见正文 §1–§8（精度归档层，允许代号）。

---

> 朴素一句话：病理核确实**只把"点火门槛"压低、不改变点着之后火往哪烧的形状**（B1d 过）；
> 但在当前 5×5 网格 + 事件大小下，这个"火烧的形状"就是一团**各向同性的局部扩散**，跟"离源点
> 的距离"预测得一样好、看不出沿 E→E 轴的方向性（B1b、B1c 没过）。**结论：不进 μ 相图**——
> 当前的 finite event 还不是一个比"距离"更有机制意义的传播算子。

---

## 1. What was tested

Finite-amplitude, event-conditioned propagation operator `W_event` on the heterogeneous-core
SNN (small-kick linear `W_small` was already abandoned). Three claims, all OFFLINE on the
existing 5×5 mini-W pilot (no SNN re-run, no μ map, no engine change):

- **B1d** — does n17.6 only lower the finite-event threshold `K_min`, or also change the early
  propagation shape `W_event`? (matched-shape equivalence)
- **B1b** — is `W_event` anisotropic along the E→E long axis (θ=45°), or an isotropic local spread?
- **B1c** — does `W_event` predict early recruitment order better than pure distance/rate/K_min?

## 2. Inputs / git / paths

- branch `topic4-snn-m3-hub`, analysis at HEAD `7f0b563` (B1 commits `00edcbb`→`7f0b563`).
- grid: `n_bins_per_axis=5` (n_bins=25) confirmed in all 10 runs; `load_run_dir` now FAILS CLOSED
  on stale/mixed (4×4) artifacts (`tests/test_m3_load_run_dir_guard.py`, 6 tests).
- pilot data: `results/topic4_sef_hfo/m3_local_w/mini_w_event/runs/<sub>_<src>/`
  (per_seed_metrics.csv, ea_net_bins.npz, thresholds.json).
- this report's artifacts: `results/topic4_sef_hfo/m3_local_w/mini_w_event/b1_validation/`
  - `status_reproduce.md` (P0), `b1d_matched_shape/`, `b1b_axis/`, `b1c_predictivity/`,
    `core_only_seed_confound/` — each with metrics CSV + summary + figures (results gitignored).
- P0 reproduction PASS: bare center K_min=1.6, n17.6 center K_min=1.1, B1a obs 0.855/0.903,
  null 0.239/0.251 — all match.

## 3. B1d — matched-shape equivalence → **PASS**

bare center @ K_min=1.6 vs n17.6 center @ K_min=1.1, EA-local-returned successful events only
(n=10 / n=9). Equivalence judged against the within-substrate split-half ceiling (NOT p>0.05):

- cos(mean shapes) = **0.995**, pearson = 0.994, top3/top5 overlap = **1.0**, centroid dist = 0.19mm.
- cross split-half similarity (cosine) = 0.987 ≥ within-floor 0.983 → equivalent at delta=0.05/0.10/0.15.
  Same for pearson; spearman cross (0.59) ≥ within (0.51).
- Difference heatmap is ±0.02 seed-noise on a ~0.25 signal — no systematic structure.

→ **The core lowers K_min (1.6→1.1) without changing the early shape.** Caveat: the mean shape
is near-isotropic (see B1b), so the axis-angle metric is moot here; PASS rests on the
shape-magnitude metrics (cos/pearson/top-k/centroid).

## 4. B1b — axis / anisotropy → **FAIL** (isotropic at this resolution)

Center mean `W_shape` anisotropy = 1.17–1.19, but p(obs ≥ spatial-shuffle null) = **0.98**
(2000× permutation; null p95 ≈ 7–9). The shape is significantly **more isotropic** than random
weight arrangements → no E→E (45°) anisotropy is resolved at center. Off-axis R_src=4mm sources
show apparent ~45° alignment (axis err 1–2°, anisotropy 1.6–3.0) but that is an **edge artifact**
(the boundary clamps the spread diagonally inward) and is still not significant vs null →
reported as sensitivity only.

**RESOLUTION CAVEAT (load-bearing):** bins are 4mm (L=20/5) and the event r95 ≈ 5mm ≈ 1 bin, so
the shape is dominated by the 4 orthogonal immediate neighbours and **cannot resolve a 45°
diagonal axis**. FAIL is a *resolution* statement, NOT proof the (AR=2, θ=45°) connectivity is
isotropic.

## 5. B1c — ordering predictivity → **FAIL** (W_event ≈ distance: local diffusion at this scale)

Leave-one-seed-out, center bare@1.6 & n17.6@1.1, early-response-rank proxy (activation time not
emitted). W_event does **not** beat distance:

- bare:  rho_W=0.528 vs rho_dist=0.552; paired diff −0.024, bootstrap CI [−0.11, +0.05], Wilcoxon p=0.92.
- n17.6: rho_W=0.536 vs rho_dist=0.543; paired diff −0.007, CI [−0.13, +0.10], Wilcoxon p=1.0.
- top3 W (0.9) marginally > top3 distance (0.7), but overall ordering is no better than distance.

**Why the FAIL is ceiling/resolution-limited, NOT merely "small n" (review fix 2026-06-24):**
the per-event rank order is itself only ~0.55–0.61 reproducible across seeds (median pairwise
Spearman: bare 0.567, n17.6 0.609). Both predictors (rho_W≈0.53, rho_dist≈0.55) sit **at that
per-event noise ceiling** — so more seeds would *not* let W pull ahead of distance; the held-out
target is itself only ~0.55-reproducible. The cause is the ceiling + coarse bins, not statistical
power.

**Stronger, noise-free corroboration (the primary evidence):** in the mean operator, at the
center the 4 orthogonal neighbours (dist 4mm) carry weight **0.16–0.27** each, while the 4
diagonal neighbours (dist 5.66mm = the 45° E→E direction) carry only **0.001–0.05**. A genuine
45° axis would *enhance* the diagonals; instead they are **suppressed exactly as pure distance
predicts**. So even setting the underpowered paired test aside, the operator shows **zero
directional residual beyond distance**.

→ The finite event's early recruitment is predicted as well by **pure distance** as by the learned
`W_event` → **local diffusion** at the resolvable scale, not a directional propagation operator.
Consistent with B1b isotropy.

**DATA_MISSING** (`b1c_predictivity/DATA_MISSING.md`): per-bin baseline rate, per-bin K_min
susceptibility map, and per-bin activation TIME are not in current artifacts — only
W_event-vs-distance was testable. The runner would need to emit per-bin sham counts, a dense
source sweep, and per-bin onset times to test the other predictors.

## 6. P4 — per-seed spontaneous confound → clean

Reusing Step B `spontaneous_ignition_flag`: **0 spontaneous-igniting core_only seeds across all 10
pilot runs** → no B1 result depends on a contaminated seed (`core_only_seed_confound/`).

## 7. Verdict table

| claim | status |
|---|---|
| B0 finite returned event exists | **supported** |
| B2 core lowers K_min (center, probability layer) | **center-supported** (1.6→1.1) |
| B1a W_shape reproducible across seeds | **supported** (obs 0.85/0.90 ≫ null 0.24/0.25) |
| B1b W_event along E→E axis | **FAIL** (isotropic at this resolution; resolution-limited; **kick-probe only — spontaneous untested**) |
| B1c W_event beats distance/rate | **FAIL** (W_event ≈ distance; ceiling/resolution-limited; **kick injects radially → may bias W toward distance; spontaneous-event W untested**) |
| B1d core changes shape? | **PASS** (it does NOT — lowers K_min, shape unchanged) |
| B3 seizure-like bridge | **not started** |
| **M3B B1 overall (2026-06-24 收口)** | **B-BOUNDED NEGATIVE** — recruitment/susceptibility operator, NOT a directional field operator; no directional W claim, no μ phase map |

## 8. What NOT to claim

- **No μ phase map was run.** (Hard-prohibited this round; and the result says don't.)
- **No seizure-like transition claim.** B3 untouched.
- **No hub / corridor / W_escape / endpoint** fallback was used or introduced.
- **R_src=4mm off-center sources are edge-sensitive** (boundary clamps r95) → sensitivity only,
  never a "spatial range of the core effect" main claim.
- Representative event figures are **diagnostic, not statistical proof** (one median-r95 seed).
- B1b FAIL is **not** "the model's connectivity is isotropic" — it is "the 5×5 / r95~5mm
  resolution cannot resolve the 45° axis."
- B1c FAIL is **not** "W_event equals distance at infinite precision" — it is "at the resolvable
  scale, and given that per-event order is only ~0.55 reproducible, W carries no directional
  information beyond distance." The remedies in §9 (finer bins / larger events) are exactly what
  would change this.
- "B1a passed" ≠ "B1 passed." "Core lowers K_min" ≠ "core does not change W_event shape is proven
  for a directional operator" — B1d proves shape-invariance of an *isotropic local spread*, not of
  a directional propagation operator.

## 9. M3B 收口与下一步（2026-06-24, supersedes 早先的 static-μ PIVOT）

> ⚠️ 历史更正：本节早先版本（pre-M3-split）建议"进入 static-μ basin pilot"。2026-06-24 的 M3 拆分之后
> 该建议**作废**——见顶部 STATUS banner。static-μ（`h(W)→阈值 μ`）已被 M3B plan §0 / Task 0 降级为
> **历史负对照 / control**；发作样慢变量机制移交 **M3A**。本段保留为演进记录，next step 以下面为准。

**冻结结论**：M3B B1 = **B-BOUNDED NEGATIVE**。`W_event` 当前只是一个 **finite-recruitment / susceptibility
operator**（有限招募 / 易感性算子），**不是**已证明的方向性传播算子。A（W_small 线性小扰动）已否定；
B0 + B2(center) + B1a + B1d 支持；B1b/B1c 在当前 5×5 / r95≈1 格分辨率下没过（分辨率受限，非"连接各向同性"）。

**两条硬禁止（重申）**：不写 directional W claim；不进 μ 相图（机制层归 M3A，M3B 不发明 `s_slow`、不用 `h(W)` 阈值）。

**B1c rescue 基础设施（已建，未重跑，不改本轮结论）**：B1c 只测到 "W_event vs distance" 是因为旧 artifact 缺三样
东西（`b1c_predictivity/DATA_MISSING.md`）。本轮在 runner 加了一个**默认关闭**的最小补丁，把其中两样做成可选 sidecar
（`ea_aux_bins.npz`；新 flag `--emit-ea-aux`，需配合 `--emit-ea-bins`）：

- per-bin **核外底物（sham / core_only）EA 窗计数** → 让 "local rate" 能作为预测子；
- per-bin **首个 spike 的激活时刻（ms）** → 让招募顺序用**真实激活时间**，而不是当前的 early-response-RANK 代理。

默认关闭时 `ea_net_bins.npz` 及一切既有 artifact **逐字节不变**（gating 测试 `tests/test_m3_ea_aux.py`，8 个；
另跑既有 bit-parity / CLI 测试无回归）。**没有重跑任何 SNN**，B-BOUNDED NEGATIVE 不因此改变。

**下一步（PILOT-FIRST，本轮不跑）**：

1. **dense per-bin K_min 易感性图**（DATA_MISSING item 2）——需要把每个 bin 单独戳一遍的**密集源扫描**，成本更高，
   不在本补丁内，单列为下一步，**不直接全量跑**。
2. **诚实的方向性 W 重测**：当前 FAIL 是**分辨率 / 可重复性天花板**限制（非预测子缺失），单加上面两个预测子在同样
   5×5 网格上**不会**翻盘。要让 W 有机会打败距离，必须把新字段与**更细的 bin（n_bins 9/11）+ 更大的事件**配对一起测——
   这是一次新 pilot，需先出 pilot、停下 review，不是顺手全量。

仍然禁止：dynamic m(t)、hub/corridor/endpoint/W_escape 兜底、engine 改动、detector 阈值追逐、固定窗当主证据、把
R4b tonic runaway 当 seizure-like bridge。

历史 M3 状态（pivot 前演进记录）：A (W_small) rejected; B0 + B2(center) + B1a + B1d supported;
`W_event` = finite-recruitment operator at this scale。

---
artifacts: `results/topic4_sef_hfo/m3_local_w/mini_w_event/b1_validation/`
analysis scripts: `scripts/run_m3_b1{d,b,c}_*.py`, metrics `src/sef_hfo_b1_validation.py`
prior recap: [[m3_finescan_recap_2026-06-23]]
