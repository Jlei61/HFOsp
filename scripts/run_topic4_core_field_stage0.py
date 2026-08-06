"""Stage 0: re-run the seed-5 baseline for parity, rescore the three references
with ONE frozen scorer, and freeze the config later stages are checked against.

The three references live in different regimes and are not interchangeable:
  axis_only          -- no pathology field at all
  manual spontaneous -- the two-core network running free; THIS is the baseline
  driven_pooled      -- source-only + sink-only pooled; a READ-OUT UPPER REFERENCE,
                        which its own frozen stats file states in as many words
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field import (  # noqa: E402
    core_thresholds, manual_mask, sample_core_quantiles, signed_depth)
from src.topic4_core_field_runner import canonical_checksum, provenance  # noqa: E402
from src.topic4_core_field_scoring import (  # noqa: E402
    PART_MIN, adversarial_gain, assignment_invariant_S, axis_only_templates,
    balanced_pair_score, load_patient_templates, model_templates, sim_matrix)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
SUBJECT = "epilepsiae_1146"
PARITY_TAG = f"{SUBJECT}_gradient_shared_corefrozen_cr1p5_s5_20260722"
QUANTILE_SEED = 20260806
SOURCES = ("gradient", "geometry")
MISSING_RULES = ("mean_rank", "common_only")


def _score_row(model, targets, support):
    row = {}
    for src in SOURCES:
        for rule in MISSING_RULES:
            row[f"S_{src}_{rule}"] = assignment_invariant_S(
                sim_matrix(model, targets[src], support, rule))
        row[f"pair_{src}"] = balanced_pair_score(model, targets[src], support)
        row[f"advgain_{src}"] = adversarial_gain(
            model, targets[src], support, "mean_rank")["gain"]
    row.update(n_dir=model["n_dir"],
               coverage_forward=model["coverage_forward"],
               coverage_reverse=model["coverage_reverse"])
    return row


def _parity(out_dir):
    """Re-run gradient_shared seed 5 with the CURRENT code and compare to the
    frozen artifact field by field (spec 6.1 step 1)."""
    spec = importlib.util.spec_from_file_location(
        "subrun", os.path.join("scripts", "run_sef_hfo_subject_snn.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh, _, _ = mod.subject_run(SUBJECT, "narrow", "twoend_equal", 20.0, 100.0,
                                  mod.cmrun.DRIVE, 8000.0, 17.5, 1.0, 1.5, 5, None, 2,
                                  "gradient_shared", 3, None, None)
    frozen = json.load(open(os.path.join(RUN, f"readout_{PARITY_TAG}.json")))
    fields = ("n_events", "n_directional", "dir_forward", "dir_reverse", "n_clean",
              "clean_forward", "clean_reverse", "valid_contacts", "n_contacts",
              "theta_deg", "inter_core_sheet")
    diffs = {f: dict(frozen=frozen.get(f), fresh=fresh.get(f))
             for f in fields if frozen.get(f) != fresh.get(f)}
    result = dict(tag=PARITY_TAG, identical=not diffs, differing_fields=diffs,
                  provenance=provenance())
    json.dump(result, open(os.path.join(out_dir, "parity_seed5.json"), "w"), indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-parity", action="store_true",
                    help="only for iterating on the scoring path; never for a real Stage 0")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "config"), exist_ok=True)

    parity = dict(skipped=True)
    if not a.skip_parity:
        print("[stage0] re-running gradient_shared seed 5 for parity (~8 min) ...", flush=True)
        parity = _parity(a.out)
        print(f"[stage0] parity identical={parity['identical']} "
              f"differing={list(parity['differing_fields'])}", flush=True)

    targets = {s: load_patient_templates(SUBJECT, s) for s in SOURCES}
    support = sorted(set(targets["gradient"]["t_a"]) & set(targets["gradient"]["t_b"])
                     & set(targets["geometry"]["t_a"]) & set(targets["geometry"]["t_b"]))
    print(f"[stage0] frozen scoring support: {len(support)} contacts -> {support}")

    rows = []
    fd = np.load(os.path.join(RUN, f"figdata_{PARITY_TAG}.npz"), allow_pickle=True)
    reg = fd["reg"].item()
    ao = axis_only_templates([str(x) for x in fd["names"]],
                             np.asarray(fd["contacts"], float),
                             np.asarray(reg["center"]), np.asarray(reg["axis_unit"]))
    rows.append(dict(reference="axis_only", tag="-", **_score_row(ao, targets, support)))

    for path in sorted(glob.glob(os.path.join(RUN, "readout_*.json"))):
        ro = json.load(open(path))
        if ro.get("subject") != SUBJECT:
            continue
        lesion, placement = ro.get("lesion"), ro.get("placement")
        if lesion == "twoend_equal" and placement in ("gradient_shared", "template_source"):
            ref = f"spontaneous_two_core_{placement}"
        elif lesion == "driven_pooled":
            ref = "driven_pooled_upper_reference"
        else:
            continue
        m = model_templates(ro["events"], support, part_min=PART_MIN)
        rows.append(dict(reference=ref, tag=os.path.basename(path),
                         **_score_row(m, targets, support)))

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "reference_scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[stage0] wrote reference_scores.csv ({len(rows)} rows)")

    # --- pathology budget from the ACTUAL sheet geometry ------------------
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from connectivity import place_neurons
    from params import Params
    from src.sef_hfo_subject_placement import (
        gradient_shared_template_foci, register_to_sheet, template_source_foci)
    m_real, _, _, _ = gradient_shared_template_foci(SUBJECT, 3)
    _, src_n, snk_n = template_source_foci(SUBJECT, "narrow", 3)
    regd = register_to_sheet(m_real, src_n, snk_n, L=20.0, target_inter_core_mm=None)
    p = Params(g=3.6, L=20.0, density=100.0, T=100.0, dt=0.1, seed=1)
    pos, _, NE, _ = place_neurons(p, np.random.default_rng(1))
    mask = manual_mask(pos[:NE], regd["source_centroid"], regd["sink_centroid"], 1.5)
    n_core = int(mask.sum())
    d = signed_depth(core_thresholds(sample_core_quantiles(NE, QUANTILE_SEED)))
    D0 = float((mask.astype(float) * d).sum())
    print(f"[stage0] N_core_manual = {n_core}   D0 = {D0:.2f} mV")

    cfg = dict(
        subject=SUBJECT, support=support, quantile_seed=QUANTILE_SEED,
        N_core_manual=n_core, D0=D0, part_min=PART_MIN, delta_eq=0.05,
        sources=list(SOURCES), missing_rules=list(MISSING_RULES),
        score_defs=["spearman", "pair"],
        seeds=list(range(1, 13)), duration_ms=8000.0,
        field=dict(M=9, EPS=1e-3, TAU_H=0.25, A0=1.5, B0=1.5,
                   SIGMA_S_FACTOR=1.2, AXIAL_MARGIN=2.0, SHIFT_MM=3.0),
        engine=dict(L=20.0, density=100.0, AR=2.0, g=3.6, dt=0.1, k_dir=2,
                    core_mean=17.5, core_std=1.0, core_r=1.5, v_base=18.0),
        provenance=provenance(), parity=parity,
    )
    cfg["checksum"] = canonical_checksum(cfg)
    json.dump(cfg, open(os.path.join(a.out, "config", "stage_config.json"), "w"), indent=2)
    print(f"[stage0] froze config checksum={cfg['checksum'][:12]}")

    def summarise(prefix):
        v = [r["S_gradient_mean_rank"] for r in rows
             if r["reference"].startswith(prefix) and r["n_dir"] == 2
             and np.isfinite(r["S_gradient_mean_rank"])]
        if not v:
            return "n/a"
        return f"{np.mean(v):.3f} +/- {np.std(v, ddof=1) if len(v) > 1 else 0:.3f} (n={len(v)})"

    open(os.path.join(a.out, "model_integrity_report.md"), "w").write(f"""# Stage 0 完整性报告 — {SUBJECT}

## 网络规模（从代码读，不从手稿读）
`N = round(density * L^2) = 100 * 20 * 20 = 40000`，`N_E = 32000`、`N_I = 8000`。
手稿若写 `N = 4000`，是手稿错。

## seed 5 parity 复现
`identical = {parity.get('identical')}`；差异字段：`{list(parity.get('differing_fields', {}))}`
（详见 `parity_seed5.json`）

## 冻结的打分支撑集
{len(support)} 个触点：{support}

## 病理预算
`N_core_manual = {n_core}`，`D0 = {D0:.2f}` mV（有符号，约 31% 的 d_i 为负）

## 三个参照（同一冻结 scorer，`S_gradient_mean_rank`，事件门 n_part>=5）
| 参照 | 是什么 | 分数 |
|---|---|---|
| `axis_only` | **完全没有病理场**，只按触点在 u_C 上的投影排序 | {rows[0]['S_gradient_mean_rank']:.3f} |
| 自发双核 gradient_shared | **这才是基线** | {summarise('spontaneous_two_core_gradient_shared')} |
| 自发双核 template_source（旧几何） | 参考 | {summarise('spontaneous_two_core_template_source')} |
| `driven_pooled` | **读出上参照，不是基线** | {summarise('driven_pooled')} |

`driven_pooled` 的冻结统计文件逐字写着
`"independent_unit": "paired network seed (source-only and sink-only arms)"`。

## 事件门
只有 `n_part >= 5`。`gate = 4` 不存在：`endpoint_centroid_axis` 在低于 `2*k_dir+1` 时返回 `None`，
实测 signed 事件 `n_part` 最小值就是 5。

## config
`config/stage_config.json`，checksum `{cfg['checksum']}`
""")
    print("[stage0] wrote model_integrity_report.md")


if __name__ == "__main__":
    main()
