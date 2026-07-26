"""FCXR pump lifecycle — layered gate adjudicators.

Gate dependency (spec §3):  I-a -> T -> C -> S -> E, with I-b a NON-BLOCKING diagnostic.

  Gate I-a  instrument validity      HARD prerequisite for everything downstream
  Gate I-b  response-operator        diagnostic only; failing it forbids eigenmode / response-mode /
                                     susceptibility claims but does NOT stop the lifecycle
  Gate T    frozen topology + slow flow
  Gate C    causal lifecycle
  Gate S    spatial scaffold preservation
  Gate E    empirical seizure compatibility

An upstream PASS is never propagated downstream as a scientific PASS: each adjudicator consumes
only its own pre-registered evidence and returns UNRESOLVED when that evidence is missing, rather
than defaulting to a pass.

Design: docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md §4 / §14
"""
from __future__ import annotations

import numpy as np

# Pre-registered Gate I-a readout rule (spec §I3): the DIRECT pump current must not carry more
# 1-80 Hz power than the synaptic components do. If it does, any spectral conclusion drawn from
# `all_components` was painted by the slow pump term rather than produced by network activity.
CONTAMINATION_RATIO = 1.0
IDENTITY_TOL = 1e-9


def adjudicate_gate_Ia(parity, baseline, readout):
    """Gate I-a verdict. `parity`, `baseline`, `readout` are the three evidence blocks; any missing
    block yields UNRESOLVED (never a pass by omission).

    parity   : {byte_parity_pass, zmx_update_order_pass, blessed_hashes_match, causal_order_pass}
    baseline : {candidate_admissible, equivalence_all_within, n_metrics_outside, ...}
    readout  : {identifiability_status, identity_max_abs_err, band_power_pump,
                band_power_no_direct_pump}
    """
    reasons = []
    need_parity = ("byte_parity_pass", "zmx_update_order_pass", "blessed_hashes_match",
                   "causal_order_pass")
    need_base = ("candidate_admissible", "equivalence_all_within")
    need_read = ("identifiability_status", "identity_max_abs_err",
                 "band_power_pump", "band_power_no_direct_pump")
    missing = ([f"parity.{k}" for k in need_parity if parity.get(k) is None]
               + [f"baseline.{k}" for k in need_base if baseline.get(k) is None]
               + [f"readout.{k}" for k in need_read if readout.get(k) is None])
    if missing:
        return dict(status="UNRESOLVED", reasons=[f"missing evidence: {', '.join(missing)}"],
                    missing=missing)

    if not (parity["byte_parity_pass"] and parity["blessed_hashes_match"]):
        reasons.append("pump-off byte parity / blessed engine hashes failed")
        return dict(status="FAIL_PARITY", reasons=reasons)
    if not parity["zmx_update_order_pass"]:
        return dict(status="FAIL_UPDATE_ORDER", reasons=["existing Z/M/X update order changed"])
    if not parity["causal_order_pass"]:
        return dict(status="FAIL_UPDATE_ORDER",
                    reasons=["pump causal order (membrane at u(t-) -> spikes -> load update) failed"])

    if readout["identifiability_status"] != "IDENTIFIABLE_AS_PROXY":
        return dict(status="FAIL_READOUT_IDENTIFIABILITY",
                    reasons=[f"readout status {readout['identifiability_status']}"])
    if float(readout["identity_max_abs_err"]) > IDENTITY_TOL:
        return dict(status="FAIL_READOUT_IDENTIFIABILITY",
                    reasons=["component decomposition identity all - no_direct_pump == pump broken"])
    nodp = float(readout["band_power_no_direct_pump"])
    pmp = float(readout["band_power_pump"])
    if nodp <= 0.0 or pmp / nodp > CONTAMINATION_RATIO:
        return dict(status="FAIL_READOUT_CONTAMINATION",
                    reasons=[f"direct pump term carries {pmp / max(nodp, 1e-300):.3g}x the 1-80 Hz "
                             "power of the synaptic components"],
                    contamination_ratio=float(pmp / max(nodp, 1e-300)))

    if not baseline["candidate_admissible"]:
        return dict(status="FAIL_BASELINE",
                    reasons=["no (a_load, tau_N) candidate satisfied the pre-registered "
                             "visible / not-pinned / headroom clauses"])
    if not baseline["equivalence_all_within"]:
        return dict(status="FAIL_BASELINE",
                    reasons=[f"{baseline.get('n_metrics_outside', '?')} primary baseline metric(s) "
                             "outside the pre-locked equivalence margin"])
    return dict(status="PASS",
                reasons=["pump-off parity + Z/M/X order + pump causal order verified",
                         "a pre-registered load candidate is admissible on the calibration trajectory",
                         "held-out pump-on baseline is equivalent within the pre-locked margins",
                         "virtual-SEEG components are pump-separable (proxy level)"],
                contamination_ratio=float(pmp / max(nodp, 1e-300)))


def adjudicate_gate_Ib(regime, operator):
    """Gate I-b diagnostic verdict (NON-BLOCKING). PASS only when the empirical finite-time response
    operator is repeatable across amplitude, noise replay and binning; otherwise the response-mode /
    eigenmode / susceptibility claim is withdrawn while the lifecycle continues."""
    if regime is None and operator is None:
        return dict(status="NOT_RUN", reasons=["Gate I-b diagnostic not executed this sprint"],
                    response_mode_claim_allowed=False)
    if operator is None:
        return dict(status="UNRESOLVED", reasons=["regime classified but no operator estimated"],
                    regime=None if regime is None else regime.get("regime"),
                    response_mode_claim_allowed=False)
    checks = dict(epsilon_linear=bool(operator.get("epsilon_linear", False)),
                  noise_replay_repeatable=bool(operator.get("noise_replay_repeatable", False)),
                  binning_stable=bool(operator.get("binning_stable", False)))
    if all(checks.values()):
        return dict(status="PASS", checks=checks, response_mode_claim_allowed=True,
                    regime=None if regime is None else regime.get("regime"))
    return dict(status="FAIL_OPERATOR", checks=checks, response_mode_claim_allowed=False,
                regime=None if regime is None else regime.get("regime"),
                reasons=[f"{k} failed" for k, v in checks.items() if not v])


# Fast-branch label families from the accepted workpoint-relative classifier
# (src/topic4_mz_fcxr_dynamics.classify_run_workpoint). "low" is the interictal side: it includes
# ELEVATED_EVENT_TRAIN, which is an event train that never sustains >=1 s above the interictal band.
LOW_LABELS = ("INTERICTAL_WORKPOINT", "ELEVATED_EVENT_TRAIN")
TRANSIENT_LABELS = ("METASTABLE_TRANSIENT",)


def _is_high(label):
    return str(label).startswith("FINITE_HIGH")


def _is_low(label):
    return str(label) in LOW_LABELS


def adjudicate_gate_T(cells, *, field="shaped"):
    """Gate T verdict from the frozen Z x P map + branch-conditioned slow flow (spec §5).

    The question is NOT "does more pump lower the rate" -- that is trivially true. It is whether
    there exists a SELECTIVE exit corridor: a pump level at which the sustained high branch is no
    longer reachable WHILE the interictal low branch still exists at the same frozen coordinates,
    with the high branch's own slow flow pointing toward that corridor.

    Verdicts:
      PASS             selective exit corridor + low branch preserved + flow toward the exit
      TOPOLOGY_NO_GO   the pump removes low and high together, or never removes high
      NO_HIGH_BRANCH   the impaired-Z corner has no sustained high branch to exit from
      UNSAFE           deciding cells hit a conductance cap / non-finite rate / runaway early stop
      UNRESOLVED       the grid does not contain the cells needed to decide
    """
    rows = [c for c in cells if c.get("field", "shaped") == field]
    if not rows:
        return dict(status="UNRESOLVED", reasons=[f"no cells for field {field!r}"])
    by = {}
    for c in rows:
        by.setdefault((round(float(c["D"]), 6), round(float(c["rho_u"]), 6)), {})[c["ic"]] = c
    Ds = sorted({k[0] for k in by})
    rhos = sorted({k[1] for k in by})
    if not Ds or not rhos:
        return dict(status="UNRESOLVED", reasons=["empty grid"])
    D_heal, D_imp = Ds[0], Ds[-1]

    unsafe = [c for c in rows if c.get("numerical", {}).get("numerical_unsafe")
              or c.get("numerical", {}).get("runaway_early_stop_ms") is not None]
    if unsafe:
        return dict(status="UNSAFE", n_unsafe=len(unsafe),
                    reasons=["frozen cells hit a cap / non-finite rate / runaway early stop; the "
                             "topology would be a numerical artifact"],
                    unsafe_cells=[dict(D=c["D"], rho_u=c["rho_u"], ic=c["ic"]) for c in unsafe[:8]])

    missing = [k for k in by if not {"low", "high"} <= set(by[k])]
    if missing:
        return dict(status="UNRESOLVED",
                    reasons=[f"{len(missing)} grid cells lack both low and high initial conditions"])

    healthy_low = all(_is_low(by[(D_heal, r)]["low"]["label"]) for r in rhos)
    imp = [(r, by[(D_imp, r)]) for r in rhos]
    high_at_min_P = _is_high(imp[0][1]["high"]["label"])
    if not high_at_min_P:
        return dict(status="NO_HIGH_BRANCH", healthy_low_preserved=bool(healthy_low),
                    reasons=[f"at D={D_imp} and the lowest pump load the high initial condition "
                             f"settles to {imp[0][1]['high']['label']}, not a sustained high branch"],
                    impaired_labels={str(r): c["high"]["label"] for r, c in imp})

    exit_row = next((c for r, c in imp[1:] if not _is_high(c["high"]["label"])), None)
    if exit_row is None:
        return dict(status="TOPOLOGY_NO_GO", healthy_low_preserved=bool(healthy_low),
                    reasons=["the sustained high branch survives every pump level on the grid: no "
                             "exit corridor"],
                    impaired_labels={str(r): c["high"]["label"] for r, c in imp})
    low_survives = _is_low(exit_row["low"]["label"])
    if not low_survives:
        return dict(status="TOPOLOGY_NO_GO", healthy_low_preserved=bool(healthy_low),
                    exit=dict(D=exit_row["high"]["D"], rho_u=exit_row["high"]["rho_u"],
                              P=exit_row["high"]["P"], high_label=exit_row["high"]["label"],
                              low_label=exit_row["low"]["label"]),
                    reasons=["at the pump level that removes the high branch the low branch is also "
                             "gone: the pump suppresses both, it does not select"])

    flows = [c["high"]["slow_flow"]["dP_dt"] for r, c in imp if _is_high(c["high"]["label"])]
    flow_to_exit = bool(flows) and all(f > 0 for f in flows)
    status = "PASS" if (healthy_low and flow_to_exit) else "UNRESOLVED"
    reasons = []
    if not healthy_low:
        reasons.append("the healthy-Z low branch is not preserved across the pump axis")
    if not flow_to_exit:
        reasons.append("the high branch's own load flow does not point toward the exit corridor "
                       "(dP/dt <= 0 on at least one high cell)")
    if status == "PASS":
        reasons = ["a selective exit corridor exists: the sustained high branch is removed while the "
                   "interictal low branch survives at the same frozen coordinates",
                   "the high branch accumulates load, so its own slow flow moves it into that corridor",
                   "the healthy-Z low branch is preserved at every pump level"]
    return dict(status=status, healthy_low_preserved=bool(healthy_low),
                high_branch_dP_dt=flows, flow_toward_exit=flow_to_exit,
                exit=dict(D=exit_row["high"]["D"], rho_u=exit_row["high"]["rho_u"],
                          P=exit_row["high"]["P"], high_label=exit_row["high"]["label"],
                          low_label=exit_row["low"]["label"]),
                impaired_labels={str(r): c["high"]["label"] for r, c in imp}, reasons=reasons)


def compare_field_controls(cells, kinds=("shaped", "uniform", "shuffle")):
    """Spec §T2/§S7: an activity-shaped pump field must be distinguishable from a mean-EXCESS-matched
    uniform field and a value-matched spatial shuffle. Same abscissa, different spatial arrangement --
    if every arm exits at the same pump level, the spatial structure is not load-bearing."""
    out = {}
    for k in kinds:
        v = adjudicate_gate_T(cells, field=k)
        out[k] = dict(status=v["status"], exit_P=(v.get("exit") or {}).get("P"))
    present = [k for k in kinds if any(c.get("field") == k for c in cells)]
    exits = [out[k]["exit_P"] for k in present if out[k].get("exit_P") is not None]
    return dict(per_field=out, fields_present=present,
                distinguishable=bool(len(exits) >= 2 and max(exits) - min(exits) > 1e-9),
                note="fewer than two arms with an exit corridor -> not comparable"
                if len(exits) < 2 else "")


def gate_conclusion_language(gates):
    """The single sentence a given set of gate verdicts entitles us to (spec §14). Engineering-green,
    one pretty trajectory, two agreeing seeds, a falling rate or a rendered figure never upgrade it."""
    ia = gates.get("Ia", {}).get("status")
    if ia != "PASS":
        return ("Gate I-a did not pass: the load/pump instrument and baseline compensation are not "
                "established; no topology or lifecycle claim is available.")
    order = [("T", "frozen fast topology and branch-conditioned slow flow"),
             ("C", "causal termination and postictal memory"),
             ("S", "spatial scaffold preservation"),
             ("E", "empirical E1146 waveform compatibility")]
    passed = []
    for key, _ in order:
        if gates.get(key, {}).get("status") == "PASS":
            passed.append(key)
        else:
            break
    if not passed:
        return ("Gate I-a only: the load/pump instrument and baseline compensation pass; topology "
                "and lifecycle are not yet demonstrated.")
    if passed == ["T"]:
        return ("Gate I-a+T: frozen fast topology and branch-conditioned slow flow are characterised; "
                "the dynamic causal loop is not demonstrated.")
    if passed == ["T", "C"]:
        return ("Gate I-a+T+C: the pump's role in termination and postictal memory is characterised; "
                "the spatial scaffold still requires Gate S.")
    if passed == ["T", "C", "S"]:
        return ("Gate I-a+T+C+S: a lifecycle scaffold preserving the E1146 spatial scaffold is "
                "obtained on holdout stochastic trajectories; real-waveform compatibility awaits Gate E.")
    return ("Gate I-a+T+C+S+E: a recoverable spatiotemporal seizure-like lifecycle candidate "
            "compatible with the real E1146 state sequence.")


def prefix_hashes(res, slow, n_steps):
    """Paired-counterfactual prefix fingerprint: every arm must be byte-identical BEFORE its
    scheduled intervention (spec §C4). Uses coarse observables that are cheap to store."""
    import hashlib
    rate = np.asarray(res["rate_E"], float)[:n_steps]
    spk = res["E_spk_bool"][:n_steps]
    out = dict(rate_sha=hashlib.sha1(rate.tobytes()).hexdigest()[:16],
               spk_sha=hashlib.sha1(np.ascontiguousarray(spk).tobytes()).hexdigest()[:16],
               n_spikes=int(spk.sum()), n_steps=int(n_steps))
    for name, tr in (("z", getattr(slow, "trace_z_mean", None)),
                     ("u", getattr(slow, "trace_u_mean", None))):
        if tr:
            arr = np.asarray(tr, float)[:n_steps]
            out[f"{name}_sha"] = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
    return out
