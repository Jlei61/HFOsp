# Axis A — E/I local-lesion as a biologically-grounded alternative to the V_th↓ stand-in (cm-SNN)

**Date:** 2026-06-13
**Topic:** Topic 4 SEF-HFO observation layer → mechanism-axis robustness of the bidirectional read-out
**Status:** DRAFT — mini-spec, pending user review before an implementation plan is written
**Origin:** advisor review 2026-06-13 (mechanism axis A: "更病理的局部机制能不能替代 `V_th↓` 生成同样双向模板")
**Sibling:** Stage 3 stochastic-template-train spec (`2026-06-13-sef-hfo-snn-stage3-stochastic-template-train-design.md` = advisor axis B)

---

## §0 Scientific framing and tier discipline

**测了什么（一句话）** — 现在让"两个端点容易点火"的办法是把那一小块的**放电门槛调低**（一个唯象代理）。这一轮问：换成**更像真实病理的局部机制**——要么那一小块的**抑制刹车失灵**（本该压住兴奋的中间神经元不工作了），要么那一小块**自己兴奋自己的回路被加粗**——能不能在同一张网、同一根几何轴、同一套电极上，**长出同样的一对互为反向、端点互换的传播模板**。

This is a **mechanism-axis robustness test**, NOT a new causal claim. It asks: is the mesoscopic SEEG read-out signature (k=2 / opposite templates / endpoint swap / readable axis / entry jitter) *specific to* threshold-lowering, or does any local hyper-excitability mechanism that sits at the two axis ends produce it? Either answer is publishable and neither upgrades V_th↓'s status:

- **If E/I lesions reproduce the read-out** → within this cm-SNN read-out layer, the E/I lesion is *sufficient* to reproduce the same read-out signature, and V_th↓ is an interchangeable phenomenological stand-in *at that read-out layer* (the paper's main-line reframe is then supported by simulation, not just asserted). This is model-read-out sufficiency — NOT a claim that patient HFOs are mechanism-generic, and NOT "通用签名".
- **If only V_th↓ closes** → the E/I axis is written as *future direction*, NOT "supported"; the paper keeps V_th↓ explicitly labelled phenomenological and does not claim a biological mechanism.

**Pre-registered tiers (fixed now, per CLAUDE.md §5):**

| Layer | Tier |
|---|---|
| Read-out structure reproduced under an E/I lesion (k=2 / opposite / swap / axis / split-half) | **primary axis-A claim**, made falsifiable by the matched-rate control below |
| Lesion-internal spike (pseudo)synchrony rising with inhibition collapse | **secondary mechanism descriptor** (echoes Schlingloff 2025 perisomatic-collapse → CA3 pyramidal pseudosynchrony — citation verified 2026-06-13, but in-vitro mouse CA3: cite for the *generator mechanism only*, NOT as evidence about SEEG-scale spatiotemporal patterns) |
| Causal status of V_th↓ vs E/I | **out of scope** — this test cannot rank mechanisms by biological truth, only by read-out sufficiency |

**Forbidden wording:** "~~axis A proved epileptic HFOs come from inhibition collapse~~". The model identifies neither inhibition nor any patient mechanism; it only tests read-out sufficiency.

---

## §1 The non-trivial engineering constraint (verified 2026-06-13)

The advisor framed axis A as a "small smoke matrix". It is **not** a smoke run — here is why, verified against the engine:

- The cm-SNN already **is** a Brunel-style E/I network: inhibitory population (`f_E=0.8`), inhibitory gain `g`, `w_EE`, `w_EI = 1.07·g·w_EE`, `w_IE`, `w_II = g·w_IE`, separate `tau_m_I`/`tau_ref_I` (`engine/params.py:34-103`). So the E/I knobs **exist**.
- **But they exist only as GLOBAL scalars.** The current lesion machinery (`build_lesion_vth` → `sample_core_field`) perturbs **`V_th` per-neuron as a spatial field**; the read-out script feeds that field in via `simulate_kick(..., V_th_per_neuron=vth)`. There is **no** equivalent per-region field for synaptic weights.
- A *global* `g↓` / `w_EE↑` is one line but **scientifically wrong for this comparison**: it heats the *whole sheet* instead of two end-cores, so it cannot be the two-end axis lesion that the V_th↓ comparison requires, and the bare-sheet-quiet / nucleate-only-from-cores invariant (read-out script docstring) is destroyed.
- The correct comparison requires a **LOCAL E/I lesion** at each axis end (same `neg_xy` / `pos_xy` offsets as `twoend_*`). That means injecting a **per-neuron spatial weight-scaling field** into the connectivity build.

**Feasibility (verified — it is tractable, just not free):** the weight build loop already computes the synaptic weight **per presynaptic neuron**:
- `connectivity.py:104` → `wval = (w_EE if a_is_E else w_IE) * jump_ampa[i]`
- `connectivity_rot.py:100` → `wval = w_EE * jump_ampa[i]`; `:123` → `wval = (w_EI if a_is_E else w_II) * jump_gaba[i]`

So a per-neuron `local_scale[i]` field multiplies in exactly where `jump_ampa[i]` already does — a surgical addition, analogous to the V_th field. **Cost:** it edits `connectivity.py` and `connectivity_rot.py`, which are in the read-out script's `_ENG_FILES` checksum set → the `assert_versions` engine guard **will trip** → requires a deliberate, logged engine re-bless (new `engine_versions.json` baseline), exactly like prior guarded-engine edits.

**Step 0 (gating, do before any lesion code):** read `build_connectivity_rot` end-to-end and pin the **indexing contract** below. The source-vs-target gating is a science decision, not a code detail:
- **Inhibition collapse (g_I→E↓ locally = perisomatic-inhibition failure):** scale `w_EI` for connections whose **target E neuron is in the core** (the E cell loses its inhibitory brake). Target-indexed.
- **Recurrent excitatory cluster (W_EE↑ locally):** scale `w_EE` for connections whose **source AND target E neurons are both in the core** (a self-exciting cluster), or at minimum source-in-core. Pin which at Step 0.
- `tau_I` / inhibitory-timing (A4) is a **global** intrinsic change, not a spatial field — flag whether a local version is even meaningful before committing to A4.

**Step 0 DONE (2026-06-14, read-only audit, verified against `results/topic4_sef_hfo/lif_snn/engine/connectivity_rot.py`):**
- Read-out runner uses the **ROT path** (`build_connectivity_rot`); the non-rot `connectivity.py` E→E branch does **not** execute on this path → only `connectivity_rot.py` needs editing.
- Loop `for i in range(NE+NI)`: **`i` = target (postsynaptic)**, **`cols` = source (presynaptic)** (verified: `a_rows`/`g_rows` use `i` as row; `jump_ampa[i]`/`jump_gaba[i]` are the target's; `cols` indexes `posE`/`posI` source coords).
- **`w_EI↓` = TARGET-indexed scalar** at `connectivity_rot.py:123` (`wval = (w_EI if a_is_E else w_II)*jump_gaba[i]`): multiply by `local_scale_EI[i]` only when `a_is_E` (scale the E target's GABA input; never touch `w_II`).
- **`w_EE↑` = EDGE-indexed, LOCKED both-in-core** at `:100` (scalar `wval`) + the AMPA append block (`:111–117`): apply the gain on the **per-edge array** `np.full(cols.size, wval)`, gated `core_mask_E[i] AND core_mask_E[cols]`, with `core_mask_E = core_mask[:NE]` and **`cols` as E-LOCAL indices** (not global). **Lock both-in-core; drop the "at minimum source-in-core" option** — source-only heats out-core targets → breaks the bare-sheet-quiet invariant (= the global-`w_EE↑` failure §1 already rejects).
- **`tau_I` (A4): NOT feasible as a local field → DEFER as GLOBAL.** Per-neuron intrinsic time constant; a local version requires an integrator change (`model.py`, also guarded), high-risk, and the *strength* dimension is already covered by `w_EI↓`. First round = `w_EI↓` / `w_EE↑` / combined only.
- **Guard asymmetry (re-bless trap):** `engine_versions.json` guards **6** files incl `model.py`; the runner's `_ENG_FILES` tuple lists only **5** (no `model.py`, provenance-log only). Re-bless must snapshot the **same 6 paths** the JSON carries, or `model.py` silently drops out of the guard.

---

## §2 Lesion matrix

Fixed across all rows (identical to the validated cm read-out and to Stage 3): same EE geometry axis (`theta_EE=45°`, `AR=2`), same two-end core positions, same montage / estimator / margin, same detector, same masked pipeline, same seeds.

| Row | Local change at each core | Pathology mapped (advisor) |
|---|---|---|
| **A0** | `V_th↓` (current `twoend_*`) | phenomenological baseline (existing asset) |
| **A1** | `w_EI ↓` for in-core E targets | perisomatic / local inhibition collapse |
| **A2** | `w_EE ↑` for in-core E↔E | recurrent principal-cell cluster |
| **A3** | `w_EI ↓` **+** `w_EE ↑` in-core | combined pathological HFO kernel |
| **A4** | `tau_I` / inhibitory timing — **DEFERRED (Step 0 verdict 2026-06-14): NOT in the first-round local-lesion matrix.** A local spatial field is not meaningful (per-neuron intrinsic time constant → would need a guarded `model.py` integrator change), and the *strength* axis is already covered by `w_EI↓`. If ever run, it is a GLOBAL intrinsic scalar, not a spatial field. | delayed restraint / altered inhibition timing |

Lesion-magnitude tuning is constrained to **produce a comparable spontaneous event regime**, not to match the read-out answer (anti-tuning discipline, same as Stage 3 §4 controls).

---

## §3 Acceptance gates (each encodes its conclusion)

The whole axis hinges on one scar this project already carries (heterogeneity rate-gate holes; Stage 3 engine-control rate caveat). State it as a gate:

| Gate | Threshold |
|---|---|
| **event-rate matched to A0** (THE decisive control; band LOCKED at spec time) | each E/I row's per-end clean-event rate must fall within **0.8×–1.25× of A0's** (±25 % relative; lesion magnitude is tuned only to hit this band, NEVER to match the read-out answer). Outside the band → re-run at a rate-matched lesion magnitude; if none exists below the over-heating point, the row is reported **"cannot rate-match → read-out comparison inconclusive"**, NOT "reproduces". The rate difference is always reported. (±25 % is the proposed lock — adjustable only at spec revision, never post-hoc; encodes the conclusion, not just existence.) |
| bare-sheet-quiet preserved | `true_inter_event_floor` stays sub-critical (events nucleate only from cores), same invariant as A0 |
| `stable_k` | == 2 |
| opposite templates / endpoint swap (PRIMARY template gate) | `swap_class` ∈ {strict, candidate} (masked rank-displacement) — the SAME gate as Stage 3 §3.1, so axis A and Stage 3 stay consistent |
| inter-cluster corr (shared active contacts) | **descriptive only, NOT a gate** — report value + `n_shared`, FLAG if `n_shared` small. Consistent with Stage 3: corr 只描述、不作判据；真正判据是 `swap_class`. |
| split-half AND odd-even | OR-rule per `forward_reverse_reproduced` contract |
| axis readable | per-event `axis_err` distribution comparable to A0 (the read-out is not degraded) |
| lesion-internal pseudosynchrony (secondary) | in-core E spike co-activation rises A0 → A1 → A3 (descriptor only, not a gate on the primary claim) |

**Conclusion wording cap:** "在读出层面，局部 E/I 病灶能 / 不能复现 V_th↓ 的双向模板（在事件率匹配的前提下）" — never "E/I 机制被证实".

---

## §4 Read-out reuse (no re-invention)

The entire read-out + structure + swap stack is unchanged from A0 — axis A only swaps the lesion builder. Reuse:
- `montage` / `valid_mask` / `read_event` / `snn_event_envelope` / `detect_events` (read-out script, unchanged).
- masked PR-2 / PR-2.5 / rank-displacement structure body (the same single-network function Stage 3 §3.1 refactors out of `pool_and_cluster_spontaneous.py`).
- Stage 3's collision/onset machinery is **not** required for axis A's primary structure claim (A1/A2/A3 can run dephased like the original `twoend_deph` to avoid collisions); it is only needed if axis A is later combined with the timing layer.

This is why axis A should land **after** Stage 3 ships the single-network structure function — it consumes it directly.

---

## §5 Code units (reuse-first)

- **Engine (guarded):** add a per-neuron synaptic weight-scaling field to `build_connectivity_rot` (and `connectivity.py` if the non-rot path is used), gated by an in-core spatial mask with the Step 0 source/target indexing contract. Re-bless `engine_versions.json` with a logged diff.
- **Lesion builder:** extend `build_lesion_vth` (or a sibling `build_lesion_ei`) to emit the weight-scale field(s) alongside `vth`, for lesion modes `twoend_inhib` / `twoend_recur` / `twoend_combined`.
- **Runner:** thread the weight-scale field into **`build_connectivity_rot`** (weights are baked at build time). **CORRECTION (Step 0 audit 2026-06-14):** do NOT pass it to `simulate_kick` analogously to `V_th_per_neuron` — `simulate_kick` consumes the already-baked `net["ampa_by_delay"]`/`["gaba_by_delay"]`, so a weight field handed to it is **silently ignored**. `V_th` works as a sim-time param (applied at integration, `kick_probe.py:162`); synaptic weights cannot.
- **Analysis:** reuse Stage 3's single-network structure function verbatim; add the in-core pseudosynchrony descriptor (small helper on `E_spk_bool` restricted to `core_mask`).
- **Orchestrator/figure:** A0–A3 side-by-side read-out + rate-matched-comparison panel.

---

## §6 Execution route (gated)

1. **Spec** (this doc) — pending user review.
2. **Stage 3 ships first** (it provides the single-network structure function this axis reuses; also it is the headline and the cleaner Topic-1 closure).
3. **Step 0 engine read** → pin source/target indexing + whether A4 has a local form.
4. **Engine extension + re-bless** (guarded edit, logged) — only after the §3 rate-match band is locked (locked at spec time: 0.8×–1.25× A0).
5. **A1/A2 first** (the advisor's minimum: does inhibition-collapse OR combined replace V_th↓), rate-matched. A3/A4 only if A1/A2 are interesting.
6. Gates + figure. Write the topic4 main-doc mechanism-axis section only after the rate-matched gate is settled.

---

（内部归档代号：axis A = advisor mechanism-axis A；A0 = `twoend_*` V_th↓ baseline；A1 `w_EI↓` perisomatic collapse / A2 `w_EE↑` recurrent / A3 combined / A4 `tau_I` **(DEFERRED, global-only — Step 0 verdict 2026-06-14)**；engine guard = `assert_versions` over `_ENG_FILES`；read-out = endpoint_centroid_axis / k_dir=3 / margin 0.10；structure = masked PR-2/PR-2.5/rank-displacement single-network function from Stage 3 §3.1；rate-match control = 此仓库 heterogeneity rate-gate / Stage 3 engine-control rate caveat 的同一教训。文献锚 Schlingloff perisomatic-collapse → pseudosynchrony 已核对引用 2026-06-13（in-vitro mouse CA3 = generator-mechanism-only，禁止外推 SEEG 跨脑区尺度）。）
