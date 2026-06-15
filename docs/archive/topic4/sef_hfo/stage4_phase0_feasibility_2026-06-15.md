# Stage 4 Phase 0 — engine feasibility @ L≈32 (gate record)

- **Date**: 2026-06-15
- **Plan**: `docs/superpowers/plans/2026-06-15-sef-hfo-snn-stage4-extended-patch.md` (Phase 0, Tasks 0.1–0.3)
- **Verdict**: **GO at L=32** (build + memory + in-degree + dynamics all pass; sim made feasible by a bit-identical scatter optimization).

## 朴素话

我们要在一个大网格（边长 32mm、约 10 万神经元）上跑这个 SNN。先做两件事确认这事能跑：

1. **建连提速**：原来每个神经元要和全网所有细胞算一遍连不连，太慢。我们加了个"只连半径 4.3mm 内"的快捷开关（`prune_radius`），并证明在我们用的密度下，这样连出来的网和老的全量连法统计上一模一样（连接距离、延迟、方向性都对得上，远尾巴被有意截断但占比 0.3%）。
2. **仿真提速**：建网其实不是瓶颈，真正慢的是仿真积分循环里一个写法——每个时间步、每条延迟通道都做一次"覆盖全部 10 万神经元的稠密加法"。我们改成"只往真正有连接的目标上加"（按源神经元索引的稀疏 scatter），并逐位验证结果和原来完全一致。L=32 上仿真从 18 分钟/0.5 秒降到 1 分钟/0.5 秒（16.5 倍）。

结论：L=32 能跑。建网约 6 分钟（一次性、每个网络建一次复用），一段 T=4 秒的仿真约 15–20 分钟，内存 36.5GB。

## Measured numbers (L=32, density=100, AR=2, θ=45°, R=8·l_par=4.30 mm)

| Quantity | Value | Gate |
|---|---|---|
| N (neurons) | 102 400 (NE=81 920) | — |
| Placement | 0.0 s | not a bottleneck |
| Connectivity build | ~6 min | practical (one-time per network) |
| **In-degree** | AMPA edges = 81.92 M = exactly `NE·C_EE` (65.5 M E→E) + `NI·C_IE` (16.4 M) | **no edge-target starvation** — prune ball held ≥800 candidates everywhere ✓ |
| Peak RSS | 36.5 GB | fits (243 GB free) ✓ |
| **Sim 0.5 s (dense, pre-opt)** | 1091 s | the wall |
| **Sim 0.5 s (flat scatter)** | **66 s (16.5×)** | → ~15–20 min per T=4000 ms run ✓ |

## Equivalence gates (Tasks 0.1–0.2, `tests/test_snn_engine_prune.py`)

Validated at **production density 100** (where the fixed in-degree `C_EE=800` is satisfied within ~1.8 mm ≪ R=4.3 mm, so the truncated cells are never selected; low density distorts and is out of scope):

- in-degree exact within the prune ball; `prune_radius=None` is **bit-identical** to the pre-2026-06-15 path.
- partner-distance KS **effect size < 0.03** (a tail-bounded approximation cannot be p-value-identical).
- bulk delay quantiles (0.1–0.9) within 3%; extreme tail (0.99) bounded by R.
- realized E→E covariance axis ≈ 45°, elongation ratio within 10%.
- 2D tail-mass `(1+R/l_par)·exp(-R/l_par)` < 1% at R=8·l_par (the bare `exp(-R/l_par)` underestimates it — 6·l_par would be 1.7%).

## Dynamics gates

- **`oneend` smoke** (L=16/d100, real prune restriction): pruned vs naive give identical dynamics — 4 events, 4/4 readable, 100% forward (correct for `oneend_neg`), peak active fraction 0.048 vs 0.049.
- **Scatter optimization** (`tests/test_snn_engine_scatter.py`): the flat source-indexed scatter equals the dense per-bin formula on the supported entries; `simulate_kick` deterministic; **E_spk_bool / rate_E bit-identical** to the pre-opt engine on a kicked small net (18 190 E spikes). Full engine suite (11 tests) passes.

## Engineering notes carried to Phase 1/2

- **The I-related channels (I→E, E→I, I→I) are still unpruned** — they dominate the ~6 min build (the E→E prune cut its channel ~17×, but build was never the wall). Pruning them is a further *build* speedup if the ensemble needs it; **not done now** (the sim, not the build, was the binding constraint, and the scatter fix resolved it).
- **Connectivity depends only on (L, density, θ, AR, seed, prune_radius)** — it is identical across the pilot's `core_r × core_mean` grid for a fixed seed. The current runner rebuilds per invocation (12 builds for the pilot). A build-cache (build once per seed, reload) would cut the pilot's build cost ~4× — a Phase 1 efficiency option.
- **Sim floor**: the remaining 66 s/0.5 s at L=32 is the unavoidable O(N) per-step LIF integration (membrane update, `rng.poisson(size=N)`, `E_spk_bool` recording), not the scatter. Further speedup would need a full-loop vectorization (out of scope).

## Decision

**GO at L=32, density=100, R=8·l_par=4.30 mm.** Proceed to Phase 1 (extended_patch lesion + Stage-4 helpers + pilot). Expected pilot cost ≈ 12 runs × (~6 min build + ~15–20 min sim) ≈ 4–5 h (one-time, gated).
