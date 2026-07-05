"""Cohort field-swap subject-SNN summary + verdict (in-repo, reproducible).

Reads the cohort config + each subject's twoend readout, recomputes the scientific
gate fields, and writes cohort_summary.{csv,json} + a contact-sheet README under
results/paper-ready-figure/_cohort_field_swap_snn/.

Verdict logic (computed, not eyeballed):
  fig4a_ok          : Fig4A png exists
  fig4b_generated   : Fig4B png exists (fails when <2 clean directional events)
  n_forward/n_reverse : directional event counts (sign-based)
  bidirectional     : n_forward >= MIN_DIR and n_reverse >= MIN_DIR   (matrix is meaningful)
  balanced          : minority_frac >= BAL_FRAC
  model_real_pass   : bidirectional AND similarity diag (fwd~t_a, rev~t_b) both > 0 AND
                      off-diag both < 0 AND at least one diagonal perm p < 0.05  (true swap match)
  geom_ok           : no hard geometry flag (cores_close_wide_cloud / large_extent...)
  final_verdict     : keep | keep_geom_caveat | bidir_no_real_match |
                      one_direction_only | fig4b_failed

Main-figure candidates = final_verdict == "keep".
"""
import csv, glob, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"
OUT = ROOT / "results/paper-ready-figure/_cohort_field_swap_snn"
FIGROOT = ROOT / "results/paper-ready-figure"
CONFIG = OUT / "cohort_config.json"

MIN_DIR = 3          # both directions need >= this many events for a valid fwd-rev x t_a/t_b matrix
BAL_FRAC = 0.25      # minority-direction fraction for "balanced"
HARD_GEOM_FLAGS = {"cores_close_wide_cloud", "large_extent_maybe_cross_region"}

sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from scripts.paper_figures.plot_fig_subject_snn_realvsmodel import _sim_matrix, _real_templates  # noqa: E402


def _cohort_readout(subject):
    """Return (readout_dict, core_r) for the subject's cohort run, or (None, None)."""
    g = sorted(glob.glob(str(RUN_DIR / f"readout_{subject}_cohort_cr*_s*.json")))
    if not g:
        return None, None
    fn = os.path.basename(g[0])           # readout_<subject>_cohort_cr<CR>_s<seed>.json
    cr = fn.split("_cohort_cr")[1].split("_s")[0]
    return json.load(open(g[0])), cr


def _sign_meanrank(ro, sign):
    """per-channel mean within-event rank over clean directional events of one sign."""
    kd = ro.get("k_dir", 2)
    evs = [e for e in ro["events"] if e.get("sign") is not None and e.get("n_part", 0) >= 2 * kd
           and (e["sign"] > 0) == (sign > 0)]
    acc = {}
    for e in evs:
        for n, v in (e.get("ranks") or {}).items():
            if v is not None:
                acc.setdefault(n, []).append(v)
    return {n: float(np.mean(vs)) for n, vs in acc.items()}


def summarize():
    cfg = [c for c in json.load(open(CONFIG)) if c["montage"]]
    rows = []
    for c in cfg:
        s, m, flags = c["subject"], c["montage"], list(c["chosen"].get("flags", []))
        ro, core_r = _cohort_readout(s)
        fig_dir = FIGROOT / f"fig_subject_snn_{s}/figures"
        fig4a_ok = (fig_dir / f"fig_subject_snn_{s}.png").exists()
        fig4b_gen = (fig_dir / f"fig_subject_snn_{s}_kmeans2.png").exists()
        row = dict(subject=s, montage=m, fig4a_ok=fig4a_ok, fig4b_generated=fig4b_gen,
                   flags=";".join(flags) or "-")
        if ro is None:
            row.update(final_verdict="no_readout"); rows.append(row); continue
        vc = max(1, ro["valid_contacts"]); kd = ro.get("k_dir", 2)
        clean = [e for e in ro["events"] if e.get("sign") is not None and e.get("n_part", 0) >= 2 * kd]
        npart = np.array([e["n_part"] for e in clean]) if clean else np.array([0])
        union = set()
        for e in clean:
            union |= {n for n, v in (e.get("ranks") or {}).items() if v is not None}
        nf, nr = ro["dir_forward"], ro["dir_reverse"]
        minf = round(min(nf, nr) / (nf + nr), 2) if (nf + nr) else 0.0
        bidir = (nf >= MIN_DIR) and (nr >= MIN_DIR)
        row.update(core_r=core_r, n_forward=nf, n_reverse=nr, n_clean=len(clean),
                   per_event_cov=round(float(npart.mean() / vc), 2),
                   union_cov=round(len(union) / vc, 2), minority_frac=minf,
                   bidirectional=bidir, balanced=(minf >= BAL_FRAC))
        # model-vs-real (sign-built templates, gated)
        mr_pass, mr_detail = False, "not_bidirectional"
        if bidir:
            mfwd, mrev = _sign_meanrank(ro, +1), _sign_meanrank(ro, -1)
            try:
                M, P = _sim_matrix({"forward": mfwd, "reverse": mrev}, _real_templates(s, m), B=2000)
                diag_pos = (M[0, 0] > 0) and (M[1, 1] > 0)
                off_neg = (M[0, 1] < 0) and (M[1, 0] < 0)
                diag_sig = (P[0, 0] < 0.05) or (P[1, 1] < 0.05)
                mr_pass = bool(diag_pos and off_neg and diag_sig)
                mr_detail = (f"fa={M[0,0]:.2f}(p{P[0,0]:.3f}) rb={M[1,1]:.2f}(p{P[1,1]:.3f}) "
                             f"fb={M[0,1]:.2f} ra={M[1,0]:.2f}")
            except Exception as e:
                mr_detail = f"err:{str(e)[:40]}"
        row.update(model_real_pass=mr_pass, model_real_detail=mr_detail)
        geom_ok = not (set(flags) & HARD_GEOM_FLAGS)
        row["geom_ok"] = geom_ok
        if not fig4b_gen:
            v = "fig4b_failed"
        elif not bidir:
            v = "one_direction_only"
        elif mr_pass and geom_ok:
            v = "keep"
        elif mr_pass and not geom_ok:
            v = "keep_geom_caveat"
        else:
            v = "bidir_no_real_match"
        row["final_verdict"] = v
        rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["subject", "montage", "core_r", "fig4a_ok", "fig4b_generated", "n_forward", "n_reverse",
            "n_clean", "per_event_cov", "union_cov", "minority_frac", "bidirectional", "balanced",
            "model_real_pass", "geom_ok", "flags", "model_real_detail", "final_verdict"]
    with open(OUT / "cohort_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(rows, open(OUT / "cohort_summary.json", "w"), indent=2)

    from collections import Counter
    vc_count = Counter(r["final_verdict"] for r in rows)
    denom = dict(run=len(rows),
                 fig4a=sum(r["fig4a_ok"] for r in rows),
                 fig4b_generated=sum(r["fig4b_generated"] for r in rows),
                 bidirectional=sum(bool(r.get("bidirectional")) for r in rows),
                 model_real_pass=sum(bool(r.get("model_real_pass")) for r in rows))
    _write_readme(rows, denom, vc_count)
    _contact_sheets(rows)
    print("DENOMINATORS:", denom)
    print("VERDICTS:", dict(vc_count))
    return rows, denom, vc_count


_VERDICT_ORDER = ["keep", "keep_geom_caveat", "bidir_no_real_match", "one_direction_only",
                  "fig4b_failed", "no_readout"]


def _sorted(rows):
    return sorted(rows, key=lambda r: (_VERDICT_ORDER.index(r["final_verdict"])
                                       if r["final_verdict"] in _VERDICT_ORDER else 9,
                                       -(r.get("per_event_cov") or 0)))


def _write_readme(rows, denom, vc):
    keep = [r for r in rows if r["final_verdict"] in ("keep", "keep_geom_caveat")]
    md = ["# Cohort field-swap subject-SNN — summary + verdict\n",
          "Place a subject-SNN with two low-V_th cores at the two interictal template-source regions "
          "(earliest-3 channels of each template), spontaneous twoend readout; ask whether the model "
          "readout reproduces the patient's TWO real interictal templates (forward~t_a, reverse~t_b).\n",
          "## Accounting (no conflation of file-exists vs scientifically-interpretable)",
          f"- **{denom['run']}** subjects run (stable_k=2, two distinct template-source cores)",
          f"- **{denom['fig4a']}** Fig4A generated",
          f"- **{denom['fig4b_generated']}** Fig4B generated  ({denom['run']-denom['fig4b_generated']} failed: too few clean events)",
          f"- **{denom['bidirectional']}** bidirectional (>= {MIN_DIR} forward AND >= {MIN_DIR} reverse events -> similarity matrix is meaningful)",
          f"- **{denom['model_real_pass']}** model-vs-real PASS (bidirectional AND fwd~t_a & rev~t_b both >0, off-diag <0, >=1 diagonal perm p<0.05)\n",
          f"### final verdicts: {dict(vc)}\n",
          "**Main-figure candidates (keep) = bidirectional + model-vs-real swap match. "
          "Do NOT pick by coverage ranking** — e.g. epilepsiae_620 has cov 0.88 but forward does NOT "
          "match t_a (fa<0), so it fails.\n",
          "| subject | montage | core_r | fwd/rev | cov | union | bidir | model-real | verdict | model-real detail |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in _sorted(rows):
        if r["final_verdict"] == "no_readout":
            md.append(f"| {r['subject']} | {r['montage']} | - | - | - | - | - | - | no_readout | - |"); continue
        md.append(f"| {r['subject']} | {r['montage']} | {r.get('core_r','-')} | "
                  f"{r.get('n_forward','?')}/{r.get('n_reverse','?')} | {r.get('per_event_cov','-')} | "
                  f"{r.get('union_cov','-')} | {'Y' if r.get('bidirectional') else '-'} | "
                  f"{'Y' if r.get('model_real_pass') else '-'} | {r['final_verdict']} | "
                  f"{r.get('model_real_detail','')} |")
    md += ["\n## 诚实口径",
           "- **coverage↔balance 权衡只在 E1146 上真扫了 core_r/drive**（见 cohort_field_swap_snn_coverage_tradeoff archive）；"
           "其余被试是单 seed 单参数 screen。所以只能说 *E1146 calibration + cohort screen suggests* 该权衡，"
           "不是 cohort 层面证实。",
           "- one_direction_only 的 12 个被试：自发 readout 几乎单向，fwd/rev × t_a/t_b 矩阵无意义 -> 标 N/A，"
           "只作 one-direction diagnostic。",
           "- fig4b_failed 的 4 个（1125/384/635/1096）：cr 下事件太局部/太少；救援不是当前最高优先级。",
           f"- keep 候选：{', '.join(r['subject'] for r in keep) or '(none)'}（主图从这里挑，不按 coverage 排名）。"]
    OUT.joinpath("README.md").write_text("\n".join(md))


def _contact_sheet(rows, which, fname, title):
    use = [r for r in rows if r.get(f"fig4{which.lower()}" + ("_ok" if which == "A" else "_generated"))]
    pairs = []
    for r in _sorted(use):
        stem = f"fig_subject_snn_{r['subject']}" + ("_kmeans2" if which == "B" else "")
        png = FIGROOT / f"fig_subject_snn_{r['subject']}/figures/{stem}.png"
        if png.exists():
            pairs.append((r, png))
    if not pairs:
        return
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(20, 3.0 * n), facecolor="white")
    if n == 1:
        axes = [axes]
    for ax, (r, png) in zip(axes, pairs):
        ax.imshow(mpimg.imread(str(png))); ax.axis("off")
        lab = (f"[{r['final_verdict']}]  {r['subject']} ({r['montage']}, cr={r.get('core_r','?')})  "
               f"fwd/rev={r.get('n_forward','?')}/{r.get('n_reverse','?')}  cov={r.get('per_event_cov','?')} "
               f"union={r.get('union_cov','?')}  model-real={'Y' if r.get('model_real_pass') else '-'}  "
               f"{r['flags'] if r['flags']!='-' else ''}")
        ax.set_title(lab, fontsize=11, loc="left",
                     color=("#0a0" if r["final_verdict"].startswith("keep") else "#333"))
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _contact_sheets(rows):
    _contact_sheet(rows, "A", "contact_sheet_fig4A.png",
                   "Fig4A (mechanism | tempA | tempB | readout) — verdict-ordered (keep first)")
    _contact_sheet(rows, "B", "contact_sheet_fig4B.png",
                   "Fig4B (KMeans=2 + model-vs-real, gated) — verdict-ordered (keep first)")


if __name__ == "__main__":
    summarize()
