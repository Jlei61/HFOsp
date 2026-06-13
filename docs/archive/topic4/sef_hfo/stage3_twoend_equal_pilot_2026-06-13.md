# Stage 3 `twoend_equal` pilot — diagnostic audit record

**Pilot run:** 2026-06-13 · **Doc written:** 2026-06-14
**Code:** consolidation commit `18a0144` (Task 6 runner + sidecar + block-aware controls) + summary/assert follow-up commit (this doc's sibling)
**Status:** PILOT — diagnostic only. Science gate **NOT passed** → did NOT proceed to multi-seed. Not a result for any main doc.
**Spec / plan:** `docs/superpowers/specs/2026-06-13-sef-hfo-snn-stage3-stochastic-template-train-design.md` · `docs/superpowers/plans/2026-06-13-sef-hfo-snn-stage3-stochastic-template-train.md`

---

## Abstract（白话）

第一次把"两个等强易激病灶放进同一张 cm-SNN 两端、靠噪声各自自发点火"跑成一段记录（4 秒，13 个事件），目的是在进多 seed 长跑之前，先看这个工作点能不能产出**平衡的双向**自发事件。**基础设施层通过**（旁路文件与记录严格对齐、时间戳是真实秒数）。但**科学层被门拦下**：一头（pos）几乎包办所有点火，另一头（neg）从不单独点着；近四成事件是两头同一毫秒一起被大事件扫亮的"伪对撞"；而且方向读出全是"正向"却对应"pos 端先点"——方向和源标签脱钩。结论只有一句：**这个 seed / 工作点不满足"双端等强自发生成双模板"的前提，因此不进多 seed。** 不能写成"SNN 能自发生成平衡正反模板"。

---

## §1 What was run

```bash
python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
  --L 20 --density 100 --T 4000 --lesion twoend_equal \
  --core-mean 17.0 --core-std 1.5 --core-r 1.5 \
  --seed 1 --delta-onset 30 --n-min 5 --tag pilot_te_s1 --out /tmp/stage3_pilot
```

Artifacts (copied to the canonical, on-disk-auditable path; `results/` is gitignored):
`results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/`
- `readout_pilot_te_s1.json` · `sidecar_pilot_te_s1.json`
- `record/pilot_te_s1/model_pilot_te_s1_{lagPat_withFreqCent.npz, packedTimes_withFreqCent.npy, montage.json}`

## §2 Infrastructure contract — PASS (the Task 6 goal)

| check | result |
|---|---|
| sidecar events == record columns | 13 == 13 (`lagPatRank.shape = (12, 13)`) |
| `event_id` contiguous 0..n−1 | yes |
| `raw_event_index` keeps detect order | yes |
| `packedTimes[:,0] == sidecar t_on/1000` | yes, max abs err 4.4e-16 |
| `packedTimes` shape | (13, 2) |

The record↔sidecar alignment and real-seconds time axis hold at canonical L=20. This was the deliverable of Task 6 and it is verified.

## §3 Science — gate NOT passed

**Direction vs hidden-source are different axes (do not conflate):**
- read-out **direction** (`sign`): 10 forward / 0 reverse / 3 unreadable. (`readout.n_clean_forward/reverse` = DIRECTION counts.)
- hidden **source** (`stage3_source_counts`, the gate): `neg_clean = 0`, `pos_clean = 5`, `collision = 5`, `ambiguous = 3`; **collision_rate = 0.3846**.

Three blockers:
1. **One-core dominance.** The pos-end core nucleates ~everything; the neg-end core never single-ignites (`core_onset_neg` is `None` or equals `core_onset_pos` same-bin). → only ONE source represented → a blind pipeline sees ~one template, not two. Stage 3's two-template premise fails at this seed/op-point.
2. **Collision rate 38% > the 20–30% gate.** And these are same-bin (Δ=0) pseudo-collisions: a fast sheet-wide event lights both cores' ~1% (≈6 cells) within one 1 ms bin. `--delta-onset` cannot fix Δ=0 — this is the 1%-onset threshold being too sensitive, not true simultaneous double-nucleation.
3. **Direction/source decoupled.** All 13 events read direction `+1` (forward) yet all sources are pos / collision / ambiguous. Physically a pos-end (+end) nucleation should read reverse. Unresolved: a `sign`-convention issue in `read_event`, or the AR=2 anisotropy decoupling "where it starts" from "which way the readable wave sweeps."

Other: clean-event IEI median 346 ms; `true_inter_event_floor = 0.028` (above Stage 2's <0.01 quiet gate but peak 0.067 → not saturated).

## §4 Reporting discipline

**Allowed:** "当前 pilot 在真实尺度上暴露出一核通吃、高碰撞、方向/源标签脱钩，因此不进入 multi-seed。"
**Forbidden:** ~~"双端等强病灶能在同一 SNN 里自发生成平衡正反模板。"~~ (the pilot shows the opposite at this seed/op-point.)

## §5 Open — next-round decision (user-held; no param tuning until resolved)

1. **Check the `sign` convention first** — confirm whether a pos-end (+end) nucleation should read `+1` or `−1`; the direction/source decoupling must be understood before anything is tuned.
2. Then one of: 3–4 diagnostic seeds (is pos-dominance a threshold-tail seed artifact or structural?); or rethink the `twoend_equal` design (two-core competition + anisotropy → one core wins).
3. **Paused:** multi-seed long runs; the Stage 3 Task 11 figure (a failed pilot is not figured).

## §6 Follow-on engineering (landed alongside this doc)

- `readout_*.json` now carries `stage3_source_counts` (sidecar hidden neg/pos/collision/ambiguous + collision_rate); the legacy `n_clean_forward/reverse` are kept but labeled DIRECTION counts (they feed the Stage 2 gate by name — not renamed).
- `ev_recs` time-order assert pinned at the data-production boundary (sidecar + block-aware synthetic controls assume time order).
- Pre-existing latent bug (out of scope, flagged): `read_event` early-return lacks a `sign` key → `ev_recs` KeyError at tiny L; cannot trigger at L=20.
- Worktree `stage3-exec` is SUPERSEDED by the shared-tree consolidation; left physically pending user cleanup.
