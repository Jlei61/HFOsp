"""Stage 3 event-triggered axial intervention probe (causal sufficiency, NOT a mechanism claim).

Q: after a large interictal-like event has already started spreading along the Stage 3 axis, can an
event-triggered intervention on the propagation corridor stop further axial spread? v1 abstracts the
intervention as an idealized E-only threshold shutoff over a time window (oracle replay-triggered).

Reuses the canonical spontaneous read-out runner's construction (build_lesion_vth / montage /
read_event / active_fraction / snn_event_envelope / build_sidecar) VERBATIM via importlib, and runs
the simulation through src.sef_hfo_axial_intervention.simulate_dynamic_vth (a parity-faithful copy of
simulate_kick + a time-dependent V_th). No engine edit, no re-bless. Spec:
docs/superpowers/specs/2026-06-25-stage3-deadzone-barrier-probe-design.md.

Arms: baseline | static_deadzone | dynamic_on_axis | dynamic_off_axis | late_on_axis | wall_only.
Emits JSON only (no figures).
"""
import os
import sys
import json
import argparse
import importlib.util

import numpy as np

sys.path.insert(0, os.getcwd())
from src.sef_hfo_axial_intervention import (                  # noqa: E402
    simulate_dynamic_vth, make_on_axis_target, make_off_axis_target,
    make_static_deadzone_schedule, core_source_raw, split_near_target_far,
    participation_ratio, exclude_target_contacts,
    baseline_eligibility, select_first_eligible_event,
    build_replay_schedule, build_late_schedule,
)

# Reuse the canonical spontaneous runner's construction helpers without editing it / packaging scripts/.
_CANON_PATH = os.path.join("scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py")
_spec = importlib.util.spec_from_file_location("_canon_spontaneous_runner", _CANON_PATH)
_canon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_canon)
Params, place_neurons = _canon.Params, _canon.place_neurons
build_connectivity_rot = _canon.build_connectivity_rot
build_lesion_vth, montage, valid_mask = _canon.build_lesion_vth, _canon.montage, _canon.valid_mask
active_fraction, read_event = _canon.active_fraction, _canon.read_event
snn_event_envelope, detect_events = _canon.snn_event_envelope, _canon.detect_events
build_sidecar = _canon.build_sidecar
_engine_guard = _canon._engine_guard
DT, BIN_MS, PART_MIN = _canon.DT, _canon.BIN_MS, _canon.PART_MIN
BASELINE_MS, CAL_FRAC = _canon.BASELINE_MS, _canon.CAL_FRAC

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe"
DYNAMIC_ARMS = ("dynamic_on_axis", "dynamic_off_axis", "late_on_axis")


def _nn(x):
    """JSON-safe: NaN/None -> None, else round to 4 dp."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    return round(float(x), 4)


def _axial_reach_mm(posE, fired_E, axis_unit, center, src_focus):
    """Max distance any fired E cell advanced PAST the source focus toward the far side (mm)."""
    if not fired_E.any():
        return 0.0
    a = np.asarray(axis_unit, float); a = a / np.linalg.norm(a)
    proj = (posE - np.asarray(center, float)) @ a
    src_proj = float((np.asarray(src_focus, float) - np.asarray(center, float)) @ a)
    src_sign = 1.0 if src_proj >= 0 else -1.0
    adv = -src_sign * (proj[fired_E] - src_proj)        # + = toward far side, past the source focus
    return float(max(0.0, adv.max()))


def _far_onset_time(seg, far_free_mask, t_on, dt):
    """First absolute time (ms) any far-side (non-clamped) E cell spikes in the event window, else None."""
    far_idx = np.flatnonzero(far_free_mask)
    if far_idx.size == 0:
        return None
    any_far = seg[:, far_idx].any(axis=1)
    if not any_far.any():
        return None
    return round(t_on + int(np.argmax(any_far)) * dt, 1)


def analyze_events(spk, m, valid, posE, axis_unit, center, foci, core_masks, NE,
                   free_E, target_thickness, delta_onset, n_min):
    """Per RETURNED event: source-stratified oracle + instrument metrics. Source side from
    core_source_raw (NOT read-out sign). near/target/far split is relative to the on-axis MIDLINE
    (consistent across arms). free_E (= non-clamped E cells) is the oracle denominator."""
    af, bin_w = active_fraction(spk, DT, BIN_MS)
    nb0, nb1 = int(BASELINE_MS[0] / bin_w), int(BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    peak = float(af.max()); bar = floor + CAL_FRAC * (peak - floor)
    events = detect_events(af, bin_w, event_on_frac=bar)
    env_f, fdt, _ = snn_event_envelope(spk, posE, m, DT)

    ev_recs = []
    for ev in events:
        rd = read_event(env_f, fdt, m, valid, (ev["t_on"], ev["t_off"]), axis_unit)
        ev_recs.append(dict(t_on=round(ev["t_on"], 1), t_off=round(ev["t_off"], 1),
                            returned=bool(ev["returned"]), n_part=rd["n_part"],
                            axis_err=rd["axis_err"], sign=rd["sign"],
                            readability=rd["readability"], ranks=rd["ranks"]))
    sidecar = build_sidecar(ev_recs, spk, core_masks, NE, dt=DT, bin_ms=BIN_MS,
                            part_min=PART_MIN, delta_onset=delta_onset, n_min=n_min)
    sc_by_raw = {s["raw_event_index"]: s for s in sidecar["events"]}
    contacts = np.asarray(m.contacts, float)

    out = []
    for raw_i, e in enumerate(ev_recs):
        if not e["returned"]:
            continue
        sc = sc_by_raw[raw_i]
        src = core_source_raw(sc["core_onset_neg"], sc["core_onset_pos"], delta_onset)
        rec_e = dict(event_id=sc["event_id"], t_on=e["t_on"], t_off=e["t_off"],
                     core_source_raw=src, hidden_source_label=sc["hidden_source_label"],
                     core_onset_neg=sc["core_onset_neg"], core_onset_pos=sc["core_onset_pos"],
                     n_part=e["n_part"], axis_err=e["axis_err"],
                     source_onset=None, far_onset_time=None,
                     oracle_far_ratio=None, oracle_near_ratio=None, oracle_reach_mm=None,
                     instr_far_ratio=None, instr_far_ratio_excl_target_contacts=None)
        if src in ("neg", "pos"):
            src_focus = foci[0] if src == "neg" else foci[1]
            rec_e["source_onset"] = sc["core_onset_neg"] if src == "neg" else sc["core_onset_pos"]
            s0, s1 = int(round(e["t_on"] / DT)), int(round(e["t_off"] / DT))
            seg = spk[s0:s1]
            fired_E = seg.any(axis=0)
            spn = split_near_target_far(posE, axis_unit, center, src_focus, target_thickness)
            rec_e["oracle_far_ratio"] = _nn(participation_ratio(fired_E, spn["far"], valid=free_E))
            rec_e["oracle_near_ratio"] = _nn(participation_ratio(fired_E, spn["near"], valid=free_E))
            rec_e["oracle_reach_mm"] = _nn(_axial_reach_mm(posE, fired_E, axis_unit, center, src_focus))
            rec_e["far_onset_time"] = _far_onset_time(seg, spn["far"] & free_E, e["t_on"], DT)
            part_c = np.array([(e["ranks"] or {}).get(nm) is not None for nm in m.names])
            spc = split_near_target_far(contacts, axis_unit, center, src_focus, target_thickness)
            rec_e["instr_far_ratio"] = _nn(participation_ratio(part_c, spc["far"], valid=valid))
            rec_e["instr_far_ratio_excl_target_contacts"] = _nn(participation_ratio(
                part_c, spc["far"], valid=exclude_target_contacts(valid, spc["target"])))
        out.append(rec_e)

    n_neg = sum(1 for e in out if e["core_source_raw"] == "neg")
    n_pos = sum(1 for e in out if e["core_source_raw"] == "pos")
    n_coll = sum(1 for e in out if e["core_source_raw"] == "collision")
    n_none = sum(1 for e in out if e["core_source_raw"] == "none")
    summary = dict(n_returned=len(out), n_neg=n_neg, n_pos=n_pos, n_collision=n_coll, n_none=n_none,
                   collision_rate=round(n_coll / max(1, len(out)), 4), events=out)
    return out, summary


def _select_and_schedule(src_events, arm, trigger_delay_ms, duration_ms, late_delay_ms,
                         cross_midline_frac=0.05):
    """First eligible single-source cross-midline event for which a valid schedule can be built.
    late_on_axis -> build_late_schedule; others -> build_replay_schedule (pre-far window). Returns
    (event, schedule) or (None, None)."""
    for e in src_events:
        if not (e.get("core_source_raw") in ("neg", "pos")
                and (e.get("oracle_far_ratio") or 0.0) > cross_midline_frac):
            continue
        try:
            if arm == "late_on_axis":
                sched = build_late_schedule(e, late_delay_ms=late_delay_ms, duration_ms=duration_ms)
            else:
                sched = build_replay_schedule(e, trigger_delay_ms=trigger_delay_ms,
                                              duration_ms=duration_ms, allow_late=False)
            return e, sched
        except ValueError:
            continue
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("baseline",) + ("static_deadzone", "wall_only") + DYNAMIC_ARMS,
                    default="baseline")
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--drive", type=float, default=0.6)
    ap.add_argument("--T", type=float, default=3000.0)
    ap.add_argument("--core-mean", type=float, default=17.0)
    ap.add_argument("--core-std", type=float, default=1.5)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--sep-frac", type=float, default=0.6)
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--delta-onset", type=float, default=30.0)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--target-thickness", type=float, default=2.5,
                    help="mm; axial intervention band thickness (> ~4*l_par=2.15mm blocks not slows)")
    ap.add_argument("--trigger-delay-ms", type=float, default=8.0)
    ap.add_argument("--duration-ms", type=float, default=40.0)
    ap.add_argument("--late-delay-ms", type=float, default=8.0)
    ap.add_argument("--offaxis-mode", choices=("lateral", "translate"), default="lateral")
    ap.add_argument("--schedule-json", default=None)
    ap.add_argument("--baseline-json", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tag = a.tag or f"{a.arm}_s{a.seed}"
    out_dir = a.out or OUT
    os.makedirs(out_dir, exist_ok=True)
    _engine_guard()

    L, theta_rad = a.L, np.deg2rad(a.theta)
    axis_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    p = Params(g=3.6, L=L, density=a.density, T=a.T, dt=DT, nu_ext_ratio=a.drive, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    N = NE + NI
    is_E = np.zeros(N, bool); is_E[:NE] = True
    center = np.array([L / 2, L / 2]); half = L / 2
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR, verbose=False)
    posE = net["pos"][:NE]

    base_vth, _core, foci, core_masks = build_lesion_vth(
        net, NE, axis_unit, center, half, "twoend_equal", a.core_mean, a.core_std, a.core_r,
        0.3, a.seed, sep_frac=a.sep_frac)
    flat_vth = np.full(N, 18.0)

    on_target = make_on_axis_target(net["pos"], is_E, axis_unit, center, a.target_thickness)
    n_on_E = int((on_target & is_E).sum())
    try:
        off_target = make_off_axis_target(net["pos"], is_E, axis_unit, center, a.target_thickness,
                                          n_on_E, core_masks, np.random.default_rng(a.seed + 99),
                                          L, mode=a.offaxis_mode)
    except ValueError:
        off_target = np.zeros(N, bool)   # tiny sheet / cannot place: off-axis arm will be degenerate
    free_E = ~((on_target | off_target)[:NE] & is_E[:NE])

    m = montage(center, a.theta, 0.0, a.nc)
    valid = valid_mask(m, posE, L, p.Rr)
    common = dict(m=m, valid=valid, posE=posE, axis_unit=axis_unit, center=center, foci=foci,
                  core_masks=core_masks, NE=NE, free_E=free_E, target_thickness=a.target_thickness,
                  delta_onset=a.delta_onset, n_min=a.n_min)
    cfg = dict(arm=a.arm, seed=a.seed, L=L, density=a.density, T=a.T, core_mean=a.core_mean,
               core_std=a.core_std, sep_frac=a.sep_frac, drive=a.drive,
               target_thickness=a.target_thickness, trigger_delay_ms=a.trigger_delay_ms,
               duration_ms=a.duration_ms, late_delay_ms=a.late_delay_ms, offaxis_mode=a.offaxis_mode,
               n_on_E=n_on_E, n_off_E=int((off_target & is_E).sum()))

    def _sim(src_vth, target_mask=None, on_ms=None, off_ms=None):
        net["rng"] = np.random.default_rng(a.seed)
        return simulate_dynamic_vth(p, net, base_vth=src_vth, target_mask=target_mask, is_E=is_E,
                                    on_ms=on_ms, off_ms=off_ms)["E_spk_bool"]

    def _write(summary):
        json.dump(summary, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=2)
        print(f"[{tag}] arm={a.arm} n_returned={summary.get('n_returned')} "
              f"neg={summary.get('n_neg')} pos={summary.get('n_pos')} coll={summary.get('n_collision')}",
              flush=True)

    if a.arm == "baseline":
        spk = _sim(base_vth)
        events, summ = analyze_events(spk, **common)
        ok, reason, flags = baseline_eligibility(summ)
        summ.update(tag=tag, arm=a.arm, config=cfg,
                    baseline_eligibility=dict(eligible=ok, reason=reason, flags=flags))
        print(f"[{tag}] baseline eligible={ok} reason={reason} "
              f"cross={flags['n_cross_midline']} opp={flags['n_trigger_opportunity']}", flush=True)
        _write(summ)
        return

    if a.arm in ("static_deadzone", "wall_only"):
        sched = make_static_deadzone_schedule()
        src_vth = base_vth if a.arm == "static_deadzone" else flat_vth
        spk = _sim(src_vth, target_mask=on_target, on_ms=sched["on_ms"], off_ms=sched["off_ms"])
        events, summ = analyze_events(spk, **common)
        summ.update(tag=tag, arm=a.arm, config=cfg, schedule=sched,
                    selected_baseline_event=None, selected_replay_event=None,
                    pre_intervention_parity=None, baseline_json=None)
        _write(summ)
        return

    # ---- dynamic arms: baseline run (parity + selection) then replay run ----
    clamp_target = off_target if a.arm == "dynamic_off_axis" else on_target
    spk_base = _sim(base_vth)
    base_events, base_summ = analyze_events(spk_base, **common)

    if a.schedule_json:
        sched = json.load(open(a.schedule_json))
        sel_ev = None
    else:
        src_events = (json.load(open(a.baseline_json))["events"] if a.baseline_json else base_events)
        sel_ev, sched = _select_and_schedule(src_events, a.arm, a.trigger_delay_ms, a.duration_ms,
                                             a.late_delay_ms)
        if sel_ev is None:
            sys.stderr.write(f"[{tag}] dynamic arm {a.arm}: no eligible event with a valid "
                             f"intervention schedule (need --schedule-json or an eligible "
                             f"--baseline-json/internal baseline).\n")
            sys.exit(2)

    spk_rep = _sim(base_vth, target_mask=clamp_target, on_ms=sched["on_ms"], off_ms=sched["off_ms"])
    rep_events, rep_summ = analyze_events(spk_rep, **common)
    s_on = int(round(sched["on_ms"] / DT))
    parity = bool(np.array_equal(spk_base[:s_on], spk_rep[:s_on]))
    sel_rep = None
    if sel_ev is not None:
        sel_rep = next((e for e in rep_events if abs(e["t_on"] - sel_ev["t_on"]) < 1e-6), None)

    rep_summ.update(tag=tag, arm=a.arm, config=cfg, schedule=sched,
                    intervention_on=sched["on_ms"], intervention_off=sched["off_ms"],
                    selected_baseline_event=sel_ev, selected_replay_event=sel_rep,
                    pre_intervention_parity=parity, baseline_json=a.baseline_json,
                    baseline_summary={k: base_summ[k] for k in
                                      ("n_returned", "n_neg", "n_pos", "n_collision", "n_none")})
    print(f"[{tag}] pre_intervention_parity={parity} on={sched['on_ms']} off={sched['off_ms']}", flush=True)
    _write(rep_summ)


if __name__ == "__main__":
    main()
