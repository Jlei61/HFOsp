"""Stage-2 spontaneous bidirectional read-out sweep WITH per-end gates (review 2026-06-11).

For each lesion parameter cell we run BOTH ends (oneend_neg expecting forward, oneend_pos expecting
reverse) through the aligned ∥/⊥ read-out, then gate EACH end before pooling — we must not infer the
+end from the −end rate map. A cell enters the real masked pipeline ONLY if BOTH ends pass:
  - expected-direction clean events >= N_CLEAN_MIN (clean already requires returned & correct sign &
    axis_err<25 & n_part>=PART_MIN — the runner's _clean gate),
  - true inter-event baseline < TRUE_FLOOR_MAX (core quiet between events, not quasi-continuous).
Then pool_and_cluster runs the real masked PR-2/PR-2.5/rank-displacement. Cells that fail the gate
(e.g. the over-heated 16.5 wide, expected to fragment) are reported as such, NOT pooled.

Matrix (review-approved): main=17.0 wide, low-abnormality=17.5 wide, over-heated=16.5 wide,
variance-control=17.0 narrow. (16.0 narrow deliberately excluded — strong-mean branch, off-story.)
"""
import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
READOUT = "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"
POOL = "scripts/pool_and_cluster_spontaneous.py"
T = 3500.0
N_CLEAN_MIN, TRUE_FLOOR_MAX = 10, 0.01
# (mean, std, spread, role)
CONFIGS = [(17.0, 1.5, "wide", "main"),
           (17.5, 1.5, "wide", "low_abnormality"),
           (16.5, 1.5, "wide", "overheated"),
           (17.0, 0.5, "narrow", "variance_control")]
ENDS = [("oneend_neg", "forward", "n_clean_forward"),
        ("oneend_pos", "reverse", "n_clean_reverse")]
ENV = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")


def run_end(mean, std, role, lesion):
    tag = f"stage2_{role}_{lesion.split('_')[1]}"     # stage2_main_neg / stage2_main_pos
    cmd = ["python3", READOUT, "--T", str(T), "--lesion", lesion, "--core-mean", str(mean),
           "--core-std", str(std), "--seed", "1", "--tag", tag]
    with open(os.path.join(OUT, f"{tag}.log"), "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False, env=ENV)
    p = os.path.join(OUT, f"readout_{tag}.json")
    return tag, (json.load(open(p)) if os.path.exists(p) else None)


def gate_end(summ, clean_key):
    if summ is None:
        return dict(ok=False, reason="no_output")
    n_clean = summ[clean_key]
    floor = summ["detector"].get("true_inter_event_floor")
    ok = (n_clean >= N_CLEAN_MIN) and (floor is not None and floor < TRUE_FLOOR_MAX)
    reason = "pass" if ok else (
        f"n_clean={n_clean}<{N_CLEAN_MIN}" if n_clean < N_CLEAN_MIN
        else f"true_floor={floor}>={TRUE_FLOOR_MAX}")
    return dict(ok=ok, n_clean=n_clean, true_floor=floor, reason=reason)


def main():
    os.makedirs(OUT, exist_ok=True)
    # run all 8 ends concurrently
    jobs = [(m, s, role, lesion) for (m, s, sp, role) in CONFIGS for (lesion, _, _) in ENDS]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = dict(ex.map(lambda a: run_end(*a), jobs))   # {tag: summary}

    report = []
    for (mean, std, sp, role) in CONFIGS:
        neg = results.get(f"stage2_{role}_neg"); pos = results.get(f"stage2_{role}_pos")
        gneg = gate_end(neg, "n_clean_forward"); gpos = gate_end(pos, "n_clean_reverse")
        both = gneg["ok"] and gpos["ok"]
        cell = dict(role=role, mean=mean, std=std, spread=sp,
                    neg_gate=gneg, pos_gate=gpos, both_ends_pass=both, pooled=None)
        if both:
            ptag = f"stage2_{role}_bidir"
            cmd = ["python3", POOL, "--fwd", f"stage2_{role}_neg", "--rev", f"stage2_{role}_pos",
                   "--tag", ptag]
            with open(os.path.join(OUT, f"{ptag}.log"), "w") as lf:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False, env=ENV)
            mp = os.path.join(OUT, "pooled_bidir", ptag, "masked_pipeline_summary.json")
            if os.path.exists(mp):
                m = json.load(open(mp))
                cell["pooled"] = dict(stable_k=m["pr2"]["stable_k"],
                                      cluster_sizes=m["pr2"]["cluster_sizes"],
                                      inter_cluster_corr=m["pr2"]["inter_cluster_corr_offdiag"],
                                      pr25=m["pr25"]["forward_reverse_reproduced"],
                                      swap_class=m["rank_displacement"]["swap_class"],
                                      decision_k=m["rank_displacement"]["decision_k"])
        report.append(cell)
        ps = "POOLED" if both else "GATE-FAIL"
        print(f"[{role}] mean={mean} {sp}: neg({gneg['reason']}) pos({gpos['reason']}) -> {ps}"
              + (f"  stable_k={cell['pooled']['stable_k']} swap={cell['pooled']['swap_class']} "
                 f"sizes={cell['pooled']['cluster_sizes']}" if cell.get("pooled") else ""), flush=True)

    json.dump(dict(config=dict(T=T, n_clean_min=N_CLEAN_MIN, true_floor_max=TRUE_FLOOR_MAX,
                               matrix=[dict(mean=m, std=s, spread=sp, role=r) for (m, s, sp, r) in CONFIGS]),
                   report=report),
              open(os.path.join(OUT, "stage2_summary.json"), "w"), indent=2)
    print("\nwrote stage2_summary.json")


if __name__ == "__main__":
    main()
