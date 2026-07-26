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
