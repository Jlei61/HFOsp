"""Pilot: 间期 broad 传播场能否预测隐身电极 (broad∖narrow) 发作 z-ER 招募序。

逐被试 → compute_f_c + null_F + radial_baseline → 四格判读。描述性 per-subject。
Spec: docs/superpowers/specs/2026-06-30-topic5-interictal-field-broad-extrapolation-design.md
"""
import argparse
import json
from pathlib import Path

from src.topic5_field_extrapolation import (
    load_broad_axis_record, channel_names_from_pool, broad_minus_narrow,
    ictal_zer_ranks, ictal_reliability, compute_f_c, null_F, radial_baseline_corr,
    ictal_bb_auc_by_seizure, compute_f_c_activation, null_F_activation,
    ictal_paired_features, compute_c2_perchannel_energy,
    DEF_AXIS_DIR, DEF_BROAD_POOL, DEF_NARROW_POOL,
)

OUT = Path("results/topic5_ictal_recruitment/field_extrapolation")


def classify(F, C, p, radial, *, a=0.05):
    f_ok = (p is not None and p == p and p < a) and (F is not None and F == F and F > 0) \
        and (radial is None or radial != radial or F > radial)
    c_ok = (C is not None and C == C and C > 0) and (radial is None or radial != radial or C > radial)
    if f_ok and not c_ok:
        return "field_wins"
    if f_ok and c_ok:
        return "channel_already_enough"
    if (not f_ok) and (not c_ok):
        return "science_negative"
    return "field_method_misspecified"


def run_subject(ds_sid, *, min_valid=3, n_null=2000, sigma_xy=None):
    axis_f = Path(DEF_AXIS_DIR) / f"{ds_sid}_t_a.json"
    rel = ictal_reliability(ds_sid)
    if not axis_f.exists():
        return {"subject": ds_sid, "verdict": "no_broad_geometry", "ictal_reliability": rel}
    rec = load_broad_axis_record(ds_sid)
    broad = channel_names_from_pool(ds_sid, DEF_BROAD_POOL)
    narrow = channel_names_from_pool(ds_sid, DEF_NARROW_POOL)
    hidden = broad_minus_narrow(broad, narrow)
    ictal = ictal_zer_ranks(ds_sid, min_valid_count=min_valid)
    fc = compute_f_c(rec, hidden, ictal, loo=True, sigma_xy=sigma_xy)
    nd = null_F(rec, hidden, ictal, n=n_null, sigma_xy=sigma_xy)
    radial = radial_baseline_corr(rec, hidden, ictal)
    verdict = classify(fc["F"], fc["C"], nd["p_value"], radial)
    if not rel["reliable"]:
        verdict = "ictal_ordering_unstable"  # 发作排序本身不稳 → 无稳定 ground-truth
    return {"subject": ds_sid, "n_broad": len(broad), "n_narrow": len(narrow),
            "n_hidden_total": len(hidden), "n_hidden_eval": fc["n_hidden"],
            "n_ictal_zer": len(ictal), "ictal_reliability": rel,
            "F": fc["F"], "C": fc["C"], "F_p_value": nd["p_value"],
            "F_null_p95": nd["p95"], "radial_baseline": radial,
            "verdict": verdict, "detail": fc}


def run_subject_activation(ds_sid, *, n_null=2000, sigma_xy=None, activation="bb_auc", delta_fc=0.03):
    """发作侧 = 能量 (field_concordance 显著口径)；per-seizure |corr|→中位数。
    F = A/B 两间期模板取 max (无优劣)；F_core_only = 场只用 narrow core 建 (review: 证核心非 hidden 互借)；
    C1 = 自身顺序、C2 = 自身能量 fingerprint。screen 用锁死 margin δ_FC=0.03 (review P1)。"""
    rec_a = load_broad_axis_record(ds_sid, template="t_a")
    if rec_a is None:
        return {"subject": ds_sid, "verdict": "no_broad_geometry"}
    rec_b = load_broad_axis_record(ds_sid, template="t_b")   # None 时退化单模板
    narrow = channel_names_from_pool(ds_sid, DEF_NARROW_POOL)
    hidden = broad_minus_narrow(channel_names_from_pool(ds_sid, DEF_BROAD_POOL), narrow)
    cache_ch, sz = ictal_bb_auc_by_seizure(ds_sid, activation=activation)
    fc = compute_f_c_activation(rec_a, hidden, cache_ch, sz, record_b=rec_b, loo=True, sigma_xy=sigma_xy)
    nd = null_F_activation(rec_a, hidden, cache_ch, sz, record_b=rec_b, n=n_null, sigma_xy=sigma_xy)
    fco = compute_f_c_activation(rec_a, hidden, cache_ch, sz, record_b=rec_b, loo=True,
                                 sigma_xy=sigma_xy, core_names=set(narrow))
    ndco = null_F_activation(rec_a, hidden, cache_ch, sz, record_b=rec_b, n=n_null,
                             sigma_xy=sigma_xy, core_names=set(narrow))
    cache2, paired = ictal_paired_features(ds_sid, "bact", activation)
    c2 = compute_c2_perchannel_energy(rec_a, hidden, cache2, paired)["C2"]
    F, C1, p = fc["F"], fc["C"], nd["p_value"]
    Fco, pco = fco["F"], ndco["p_value"]

    def _gt(a, b):
        return a == a and b == b and a > b + delta_fc
    # 主张以 F_core_only 为准 (review: F broad-LOO 含 hidden 互借)；F 的 screen 仅作 secondary 参考
    screen_core = (pco == pco and pco < 0.05) and _gt(Fco, C1) and _gt(Fco, c2)
    screen_loo = (p == p and p < 0.05) and _gt(F, C1) and _gt(F, c2)
    return {"subject": ds_sid, "ictal_basis": activation, "n_templates": fc["n_templates"],
            "n_hidden_eval": fc["n_hidden"], "n_hidden_total": len(hidden),
            "n_seizures_used": fc["n_seizures_used"],
            "F": F, "C1": C1, "C2": c2, "F_core_only": Fco,
            "F_p_value": p, "F_core_only_p": pco, "F_null_p95": nd["p95"],
            "delta_fc": delta_fc,
            "screen_channel_c1c2_only": bool(screen_core),     # 仅 channel null + C1/C2 margin (非完整场优势)
            "screen_loo_secondary": bool(screen_loo), "detail": fc}


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) and v == v else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="+", help="e.g. epilepsiae_583 epilepsiae_1077")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--min-valid", type=int, default=3)
    ap.add_argument("--sigma-xy", type=float, default=None)
    ap.add_argument("--ictal", choices=["order", "activation"], default="order",
                    help="order=z-ER 招募序(不稳); activation=能量(显著口径)")
    ap.add_argument("--activation", choices=["bb_auc", "hfa_auc"], default="bb_auc",
                    help="bb_auc=broadband 1-45Hz 能量(primary); hfa_auc=60-100Hz(sensitivity)")
    args = ap.parse_args()
    sub = "per_subject" if args.ictal == "order" else f"per_subject_{args.activation}"
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in args.subjects:
        if args.ictal == "activation":
            r = run_subject_activation(sid, n_null=args.n_null, sigma_xy=args.sigma_xy,
                                       activation=args.activation)
            json.dump(r, open(OUT / sub / f"{sid}.json", "w"), indent=2)
            rows.append(r)
            if "F" not in r:
                print(f"{sid}: {r.get('verdict', '?')}")
                continue
            print(f"{sid}: hid={r['n_hidden_eval']:>2}/{r['n_hidden_total']:<2} sz={r['n_seizures_used']:>2} "
                  f"T={r['n_templates']} | F={_fmt(r['F']):>6} Fco={_fmt(r['F_core_only']):>6} "
                  f"C1={_fmt(r['C1']):>6} C2={_fmt(r['C2']):>6} | F_p={_fmt(r['F_p_value']):>6} "
                  f"| c1c2scr={'Y' if r['screen_channel_c1c2_only'] else '-'}")
            continue
        r = run_subject(sid, min_valid=args.min_valid, n_null=args.n_null, sigma_xy=args.sigma_xy)
        json.dump(r, open(OUT / sub / f"{sid}.json", "w"), indent=2)
        rows.append(r)
        rl = r.get("ictal_reliability", {})
        if "F" not in r:
            print(f"{sid}: {r['verdict']:>22} | health={rl.get('health')} s_sz={rl.get('s_sz')}")
            continue
        print(f"{sid}: hidden={r['n_hidden_eval']:>2}/{r['n_hidden_total']:<2} "
              f"| F={_fmt(r['F']):>7} C={_fmt(r['C']):>7} "
              f"| F_p={_fmt(r['F_p_value']):>6} radial={_fmt(r['radial_baseline']):>7} "
              f"| health={str(rl.get('health')):>8} s_sz={_fmt(rl.get('s_sz')):>5} | {r['verdict']}")
    out_name = "pilot_summary.json" if args.ictal == "order" else f"pilot_summary_{args.activation}.json"
    json.dump(rows, open(OUT / out_name, "w"), indent=2)


if __name__ == "__main__":
    main()
