# MZ early-field bridge — STATUS

Branch `codex/topic4-mz-slowvars`. Design contract:
`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`.
8h autonomous run started 2026-07-19. Local commits only; no push/merge/rebase/PR.

## 朴素话：这一步在测什么（plain-language, CLAUDE.md §8）

同一个「病人E1146布局」的模型底物上，我们让它先安静地自发放电（慢变量全关，slow-off），
它会时不时冒出一小簇「间期样事件」，每簇事件里 15 个虚拟电极被点亮的**先后顺序**大致固定
——这是这块底物的「空间指纹」。然后我们只打开一个慢变量 `z`（抑制效力随强活动被消耗 =
去抑制），让同一底物在同一噪声实现下慢慢滑向一次「操作性失控放电」（operational runaway，
**不是临床发作**）。核心问题：失控刚点火那一瞬（0–50 ms 窗）哪些电极/源点能量最高，是不是
就是间期指纹里「最早被点亮」的那些？如果是，说明这块底物提供了一份可复用的空间响应模板，
而 `z` 掉下去只是把整体增益推高——「同一支架，不同状态」。

判据全部预注册：模板用「留出事件」验证（不能自己验证自己）；关联用 maxAB（两个方向取大）+
空间置换 null；三个 seed 只报一致性，不做 n=3 的 cohort p 值。所有措辞只写 operational
runaway / virtual-LFP energy / phenomenological z，禁止写 seizure / broadband power / 因果。

## Locked facts (verified from disk)

- HEAD `66a4d93`; uncommitted engine diff = off-by-default snapshot observer in
  `src/snn_engine/mz_slow_vars.py` (snapshot_steps=None → exact parity) + 116 test lines.
  Behavior-preserving; kept as-is; only used if optional §10 z-snapshot task is reached.
- Primary candidate `zA_q75_tz5000`: `cfg={use_z:true,use_m:false,I_th_EI:95.19851312666987,tau_z:5000.0}`.
  Multiseed (T=15000) → **runaway all 3 seeds**, `runaway_ms` = 9293.6 / 9499.3 / 9757.9 (seed 1/3/4);
  slow-off baseline returning events 38 / 40 / 39. ~5 s post-onset → complete 0–50/0–100 ms window OK.
- Sensitivity candidate `zA_q50_tz10000` (onset ~4.7–4.9 s) — ONLY if all primary deliverables done +>90 min left.
- **Event-bar bug (spec §6) located**: `run_topic4_mz_slowvars.py::_events_from_res` (L133-141) recomputes
  `bar = floor + C.CAL_FRAC*(af.max()-floor)` from each run's OWN af.max(). Fixed-bar detector must freeze
  floor+bar from slow-off once (reuse `C.active_fraction/detect_events`, `C.CAL_FRAC=0.5`, `C.BIN_MS=1.0`,
  `C.BASELINE_MS=(5,50)`) and pass that bar to both slow-off and native.
- Shaft membership (within-shaft null) = leading-letter prefix of contact name (`_shaft` in
  `scripts/paper_figures/plot_fig_subject_snn.py`). Registered plane: `reg{axis_unit,center,source_*,sink_*}`.
- Reuse surface (`src/early_recruitment_readout.py`): `early_energy_field` (§8.3 + fail-closed §8.2),
  `compare_arrival_to_energy` (§9 assoc; earliness=-rho), `permutation_null` + low-level index gens.
  **maxAB null and source-grid toroidal-shift null are NOT in the module → written new, reusing the primitives.**

## Architecture (new files)

- `config/topic4_mz_early_field_bridge.yaml` — candidate, seeds, T, windows, thresholds, null params.
- `src/topic4_mz_early_field_bridge.py` — pure functions: fixed-bar detector, 30–80 Hz peak-latency timing
  field + readable rule, odd/even held-out template, direction labels, onset markers (t120/t_recruit),
  contact/source early-energy fields, association, maxAB + within-shaft + toroidal nulls, core-excluded
  support, local-tissue participation audit, eligibility/dynamic-range.
- `scripts/run_topic4_mz_early_field_bridge.py` — per-seed runner (I/O + sim scheduling + provenance +
  `--resume` atomic per-seed writes). Reuses `PP.build_substrate`, `MZR.run_mz_cell`, `MZR.build_core_masks`,
  `LFPRecorder`, `C.*`, `M4._smooth/_first_sustained`.
- `scripts/plot_topic4_mz_early_field_bridge.py` — seed1 (Fig5 grammar) + multiseed diagnostic figures.
- `tests/test_topic4_mz_early_field_bridge.py` — the 10 required contract tests (synthetic fixtures).

## Execution plan (multi-turn, background sims)

P0 preflight/reuse-audit ✅ · P1 module+tests (TDD) · smoke (small T, 1 seed; check LFP shape/RAM/wall)
· P2 background per-seed sims (serial or ≤2 parallel by RAM; each seed slow-off + native early_stop=False,
frozen bar) · P3 templates+fields+nulls+cohort (mostly inside runner) · P4 figures + eyeball + 中文 README
· P5 tests + STATUS + archive report + local commits. Stop launching new sims by hour 6.5.

## Completion-level tracker (spec §14) — FINAL

- [x] 1 engineering complete — fixed-bar detector, ported+reused readout, 12+8+41 tests green, resumable + --readout-only.
- [x] 2 numerically eligible — held-out templates eligible 3/3 (both directions); complete non-degenerate 0-50ms fields 3/3.
- [x] 3 scientific observation — direction/effect/nulls/seed-consistency reported all 3 seeds (see cohort_summary + archive §8).
- [x] 4 bridge SUPPORTED (observation-level) — 3/3 seeds eligible held-out + positive MIRROR-INVARIANT contact
      maxAB (0.945/0.735/0.924). within-shaft null sig 2/3 (seed3 weak p=0.086). Bidirectional axis: direction is
      noise-set (NOT a success criterion, NOT a fixed focus). Source = supplementary direction-free axis engagement
      (toroidal sig 3/3). Core-exclusion UNINFORMATIVE (n_kept=15, nothing removed → no not-core-driven claim).
      NOT cohort proof, NOT seizure, NOT causal.

## FINAL cohort (pre-t120 0-50ms early recruitment, contact all-support)
Mirror-invariant rho_maxAB median 0.924, range [0.735, 0.945], n_positive 3/3. Within-shaft p: 4e-4 / 0.086 / 1e-3
(seed 1/3/4; seed3 weak/null-overlap). Source (supplementary, direction-free axis engagement) 0.651/0.546/0.585,
toroidal p 0.0087/0.012/0.017 (3/3 sig). Direction noise-set per seed (maxAB winner B_to_A this run, NOT a stable phenotype).

## Live checklist

- [x] P0 preflight + read spec/plan/runner/readout module/engine
- [x] deep-contract-verify enumerate invariants
- [x] tests (green): 8 ported lib + 10 required + 1 sanity = 19 pass
- [x] config + module + runner written (plot pending)
- [x] smoke: N=40000 NE=32000, lfp (nsteps,15), 2 shafts SCL(4)/ICL(11), RSS 6.79GB@T2000 (~11GB@T15000), ~19min/run
- [x] glue validated (synthetic): contact/source maxab + within-shaft/toroidal nulls + core-excluded
- [x] seed1 sim complete (8515s ~2.4h) — onset resolved, templates both eligible
- [x] BUG found+fixed: contact_energy_field float-window edge (t_recruit+50 vs grid 91283*0.1) mis-flagged
      complete windows incomplete. Fix = integer-step window bounds snapped to exact grid samples +
      regression test_2b. SOURCE path (integer steps) was unaffected.
- [x] --readout-only patch mode: recompute CONTACT from saved LFP (no 2.4h re-sim); source preserved;
      participation NOT recomputed (native raster not persisted — documented limitation).
- [x] seed1 contact patched: rho_maxAB=0.945 (B_to_A), within-shaft p=0.0004, source 0.651 concordant.
- [~] seeds 3,4 sims RUNNING (parallel, ~2.4h) — will --readout-only patch contact after.
- [x] plot script + seed1 figure eyeballed (fixed to show maxAB-winning direction TB; timing↔energy concordant)
- [x] seed4 complete (7384s) + patched: contact mirror-invariant maxAB=0.924, within-shaft p=0.001; source 0.585 (toroidal p=0.017, supplementary). maxAB positive.
- [~] seed3 running (native/readout phase); will patch + final-aggregate after
- [ ] ~~seeds 3,4 finish~~ → --readout-only patch contact
- [ ] cohort aggregate (--aggregate-only 1,3,4)
- [ ] multiseed figure + eyeball + figures/README.md
- [ ] STATUS final + archive report results (+ plain-language abstract/handoff)
- [ ] pytest full + git audit + commits

## Commit plan (results/ is gitignored on this branch)
- Normal commit: src/{early_recruitment_readout,topic4_mz_early_field_bridge}.py, scripts/{run,plot}_topic4_mz_early_field_bridge.py,
  config/topic4_mz_early_field_bridge.yaml, tests/{test_early_recruitment_readout,test_topic4_mz_early_field_bridge}.py,
  docs/archive/topic4/sef_hfo/mz_early_field_bridge_2026-07-19.md (+ bridge design spec + 8h prompt = executed contract).
- `git add -f` (results gitignored): figures/*.png + figures/README.md + cohort_summary.{json,csv} + STATUS.md
  (+ per_seed/*/{*.json} small). Do NOT commit the big LFP npz (intermediate; keep ignored).
- Do NOT touch the parallel session's state_conditioned_susceptibility commits (HEAD 3cda82e) or its untracked
  drafts (phase-portrait design, onset-dynamics prompt, susceptibility drafts) — not my work.

## Seed1 result (primary window 0-50ms, patched)
- onset eligible: t120=9293.6, t_recruit=9078.3 (Δ215ms). 38 returning events; B_to_A=26, A_to_B=7, unresolved=5.
- Templates BOTH eligible: B_to_A held-out median 0.995, A_to_B 0.361.
- **Contact mirror-invariant maxAB=0.945, within-shaft null p=0.0004 (10k MC).** (Core-exclusion uninformative: n_kept=15, nothing removed.)
- Source (supplementary, direction-free axis engagement) rho_maxAB=0.651, toroidal p=0.0087.
- within-trajectory pre-runaway audit: 25 events (eligible).
- Limitation: local-tissue participation audit not persisted for seed1 (readout patch); documented follow-up.

## Run facts (from smoke)
- Substrate N=40000 (NE=32000, NI=8000); contacts SCL6-9 + ICL1-11 (2 shafts).
- Within-shaft null space 4!*11! >> 50000 -> Monte-Carlo n_perm=10000 (unrestricted 15! -> MC too).
- Native uses early_stop=False (full 15s trace); slow-off + native are common-random-number replays
  (run_mz_cell resets net rng to S['seed'] each call) -> matched pair, differ only in z (design §11.1).
- Peak RSS ~11GB/run @ T=15000; free RAM 233GB; run seeds SERIALLY (1 concurrent SNN) for safety.

## Open risks

- RAM/OOM: full 15 s native early_stop=False + LFP + raster. Smoke must measure RSS before parallel.
  Default serial (peak = 1 sim). Another session may be running SNN jobs — check free RAM before launch.
- Native window completeness: run early_stop=False so t_recruit+100 ms is never truncated; else fail-closed.
- Do not consume Arm C as dose-response (quarantined). Do not overwrite fig5_snn_state_readout/.
