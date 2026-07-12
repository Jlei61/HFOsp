# M4-3A execution runbook (commands only — no results here)

Run order per spec §10 / plan Task 10: P0a (offline proxy) → P0b (field-derived
`u_n` lock) → P1 discovery sweep → 40s acceptance → mechanism ablation. This
file is commands + what-to-check-before-running only; scientific conclusions
belong in `docs/topic4_sef_hfo.md` / archive docs after a human eyeballs the
output, not here.

All CLI flags below were verified against `scripts/run_m4_dynamic_qi.py` and
`scripts/run_m43a_p0_calibration.py` directly (2026-07 session, Task 10) — not
copied from the plan doc's Step-3 pseudocode unchecked. Two corrections vs.
that pseudocode: the network runner (`run_m4_dynamic_qi.py`) has no
`--m43a-un0` flag (`u_n0` is a compiled-in module constant, `M43A_UN0 = 0.0`,
not CLI-settable); and see the **P0b gap** callout below — `trace_un_mean` is
not currently wired out of `--m43a-sweep`'s output.

The fixed M4-3A op-point (`k_q=0.10, alpha_G=16.0`) is controlled by
`--p1-kq`/`--p1-alpha-g` (shared with the M4-2 P1 flags; defaults already are
the intended op-point) on every command below that touches the network
(`--m43a-sweep`, `--m43a-ablation`) — omitted below since the defaults are
correct; pass them explicitly only to deviate from that op-point.

## P0a — offline proxy calibration (no network; global-rate drive)

```bash
python scripts/run_m43a_p0_calibration.py --dt 0.1 \
  --regime quiet=results/topic4_m4_dynamic_p1_sweep/p1_sweep_traces.npz:p1_arm0__rate:0:20000 \
  --regime bounded_ictal=results/topic4_m4_dynamic_p1_sweep/p1_sweep_traces.npz:p1_arm0__rate \
  --regime isolated_ied=<short-IED-run>.npz:rate --event isolated_ied=<idx> \
  --regime post_offset=<suppress-cell>.npz:rate \
  --out results/topic4_m43a_p0a/
```

- `<short-IED-run>.npz:rate` / `<suppress-cell>.npz:rate` / `<idx>` are
  placeholders — no such artifacts exist yet; produce them (a short isolated-IED
  probe, a suppress-classified cell's rate trace) before this command is real.
  `results/topic4_m4_dynamic_p1_sweep/p1_sweep_traces.npz:p1_arm0__rate` DOES
  exist today (checked this session) and is a valid `quiet`/`bounded_ictal` regime.
- Gate (spec §6.1 / Task 3): `sensor_free_pass` requires `delta_a_ied > 0` AND
  `>= 2*sigma_baseline` AND `>= 0.5%*a_max` AND `R_A >= 5` AND
  `interictal_block_pass`. `soft_gate_fail` (`delta_a_ied <= 0`) is a HARD fail
  — `R_A = inf` does NOT pass on its own.
- P0a is a **proxy** (global sheet-mean rate, not the field-derived drive) —
  `sensor_free_pass` stays `False` until P0b supplies a real `u_n0`/`a_block`
  and this command is re-run with `--u-n0 <P0b value> --a-block <IED-block a>`.

## P0b — field-derived `u_n` lock (replay Arm0, dump the real drive)

**Known gap (found verifying this runbook, Task 10 — not fixed here; Task 10
does not touch the runner):** `SpatialSlowField.step()` (Task 4) appends the
real field-derived drive to `self.trace_un_mean` every step whenever
`use_A=True` (`src/snn_engine/slow_field.py:364`), but `run_arm`'s
`dump_shunt_trace=True` block (`scripts/run_m4_dynamic_qi.py`, used by every
`--m43a-sweep` cell) only copies out `a_trace`/`n_trace` — **not**
`trace_un_mean` — into its returned dict, and `_m43a_cell_worker`/
`_run_m43a_sweep`'s npz dump follow the same `a_trace`/`n_trace`/`af`/`rate`
list. So `m43a_sweep_summary.json` / `m43a_sweep_traces.npz` do not currently
expose `trace_un_mean` at all. The command below runs and produces output, but
that output will NOT yet contain the number this step needs. Closing this gap
is a one-line addition mirroring the existing `a_trace`/`n_trace` lines
(`out["un_trace"] = np.asarray(slow.trace_un_mean, np.float32)` in `run_arm`,
plus threading `"un_trace"` through `_m43a_cell_worker`'s `_`-prefixed capture
and `_run_m43a_sweep`'s npz-dump array list) — a runner change, do that (as its
own small commit) before relying on this step.

Separately, there is no `--m43a-un0` CLI flag on the runner at all — once
`trace_un_mean` is exposed and a `u_n0` value is read off, feeding it back into
a real run needs either a new CLI flag or a direct Python call to
`run_arm(..., u_n0=<value>)`. Every command in this runbook (P1 discovery, 40s
acceptance, ablation) runs at the compiled-in default `u_n0=0.0`
(`M43A_UN0` in `scripts/run_m4_dynamic_qi.py`) until that flag exists.

```bash
python scripts/run_m4_dynamic_qi.py --m43a-sweep --confirm-run --seed 1 --T 20000 \
  --m43a-alpha-grid 0 --m43a-tau-grid 20000 --out results/topic4_m43a_p0b_arm0/
```

Once the gap above is closed: the `m43a_arm0` row's `un_trace` is the
field-derived `u_n = K_n * r_E` time series. Set `u_n0` = its long-run
(post-transient) mean; re-run P0a with `--u-n0 <that>` and
`--a-block <from a small IED-kick probe>` to certify `sensor_free_pass`.

## P1 discovery sweep (T=15000 ms), per seed in {1, 3, 4}

```bash
for s in 1 3 4; do
  python scripts/run_m4_dynamic_qi.py --m43a-sweep --confirm-run --seed $s --T 15000 \
    --m43a-alpha-grid 2,4,8 --m43a-tau-grid 5000,20000,40000 --m43a-workers 5 \
    --out results/topic4_m43a_p1_seed$s/
done
```

- Runs at `u_n0=0` (see the P0b gap above — this is the current, not yet
  P0b-corrected, operating point). Primary denominator for "does it go" =
  seeds whose `m43a_arm0` cell classifies bounded-persist (compute from each
  seed's `m43a_sweep_summary.json`, not assumed).
- `go` per cell = `termination_class=="terminate_clean" AND retrigger_early=="attenuated" AND retrigger_probe=="reignite_bounded"`.

## 40s acceptance for candidate go / boundary cells (T=40000 ms)

```bash
python scripts/run_m4_dynamic_qi.py --m43a-sweep --confirm-run --seed <s> --T 40000 \
  --m43a-alpha-grid <candidate alpha_A> --m43a-tau-grid <candidate tau_n> --m43a-workers 3 \
  --out results/topic4_m43a_accept_seed<s>/
```

`<s>` / `<candidate alpha_A>` / `<candidate tau_n>` = filled in from the P1
discovery sweep's output (a cell whose `go=True`, or a boundary cell worth
confirming at longer T). `go` requires: `terminate_clean` at 40s + no
post-offset rebound + early attenuated + late `reignite_bounded`.

## Mechanism ablation at the best candidate point

```bash
python scripts/run_m4_dynamic_qi.py --m43a-ablation --confirm-run --seed <s> --T 40000 \
  --m43a-abl-alpha <a> --m43a-abl-tau <tau> --out results/topic4_m43a_ablation/
```

Produces `m43a_ablation.json` with 3 rows (`shunt_only` / `subtractive_only` /
`hybrid`, `eta_A` for the latter two matched to `shunt_only`'s mean removed
recurrent current — see `_run_m43a_ablation`'s `calibration` block for the
exact formula). Mechanism-specificity expectation to check (not asserted here
— that's a result, not a command): `shunt_only`+`hybrid` clean,
`subtractive_only` persist/suppress/fragment.
