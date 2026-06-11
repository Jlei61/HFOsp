#!/usr/bin/env python3
"""Yuquan template × 热凝(RF-thermocoagulation) coverage prep.

Builds the per-subject table that the clinical outcome analysis will merge with
surgical outcome (Engel) once it is collected. This is PREP, not a locked
contract: it produces the predictor candidates + the confound-visible columns.

Predictors (per subject):
  - template_coverage    = |template ∩ ablated| / |template|         (size-confounded)
  - source_ablated_frac  = fraction of top-k propagation SOURCE channels ablated
                           (size-ROBUST, mechanism-motivated: was the driver hit)
  - onset_coverage       = |onset ∩ ablated| / |onset|               (clinical SOZ baseline)
Confound-visible columns: n_template, n_ablated, n_onset (advisor: coverage
correlates with template size; report sizes so it can be controlled downstream).

Parsing notes:
  - 热凝 section = text between '热凝' and '术后'|'报告医生'|EOF.
  - Prime electrodes (C', D', ...) use U+2019/U+2018; normalized to ASCII ' in
    BOTH ablated contacts and propagation channel names so they match.
  - Bipolar pair 'E8-9' -> monopolar {E8, E9}.
  - onset contacts parsed from the ictal/summary free-text lines (fuzzy → flagged).

NOT a result: contrast existing != coverage predicts outcome. Output is a CSV
ready for outcome merge.
"""
import json, os, re, glob, csv

PROP_DIR = "results/interictal_propagation_masked/per_subject"
BING = "/mnt/yuquan_data/yuquan_24h_bingli"
DOCTXT = "/tmp/yqdoc"  # libreoffice-converted .doc -> .txt
OUT = "results/template_ablation_coverage"

PRIME = {"’": "'", "‘": "'", "ʼ": "'"}
def norm(s):
    for k, v in PRIME.items():
        s = s.replace(k, v)
    return s

# electrode token: 1-3 letters + optional prime + number ; bipolar has -number
TOK = re.compile(r"([A-Za-z]{1,3}'?)(\d+)(?:-(\d+))?")

def expand_pairs(text):
    """Return set of monopolar contact names from a text region (handles A-B pairs)."""
    out = set()
    for el, a, b in TOK.findall(norm(text)):
        el = el.upper()
        out.add(f"{el}{a}")
        if b:
            out.add(f"{el}{b}")
    return out

def section(t, start, ends):
    m = re.search(start + r"(.*?)(" + "|".join(ends) + r"|\Z)", t, re.S)
    return m.group(1) if m else ""

def ablated_contacts(txt):
    t = open(txt, encoding="utf-8", errors="ignore").read()
    seg = section(t, "热凝", ["术后", "报告医生"])
    # restrict to lines that look like ablation rows: electrode-pair + power(W)/time(S) nearby
    return expand_pairs(seg), seg

def onset_contacts(txt):
    t = open(txt, encoding="utf-8", errors="ignore").read()
    # ictal onset + interictal summary lines
    lines = re.findall(r"(?:脑电图发作|发作期|间歇期|间期|总结|起始|起源)[：:].{0,120}", t)
    return expand_pairs(" ".join(lines))

def template_info(pin):
    f = os.path.join(PROP_DIR, f"yuquan_{pin}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    ch = [norm(c).upper() for c in d["channel_names"]]
    # source = earliest (smallest valid rank) in dominant adaptive cluster
    ac = d.get("adaptive_cluster", {})
    cl = ac.get("clusters")
    src = None
    try:
        if isinstance(cl, list) and cl:
            dom = max(cl, key=lambda c: c.get("n_events", 0))
            tr = dom.get("template_rank"); vm = dom.get("template_valid_mask")
            pairs = [(r, c) for r, c, m in zip(tr, ch, (vm or [True]*len(ch)))
                     if m and r is not None]
            pairs.sort()
            src = [c for _, c in pairs[:3]]
    except Exception:
        src = None
    return set(ch), (set(src) if src else set())

def doc_for(pin):
    cand = [f for f in glob.glob(f"{BING}/{pin}/*颅内eeg.doc")
            if not os.path.basename(f).startswith(".~")]
    if not cand:
        return None
    base = os.path.basename(cand[0])[:-4]
    txt = os.path.join(DOCTXT, base + ".txt")
    return txt if os.path.exists(txt) else None

def jacc_cov(a, b):
    return (len(a & b) / len(a)) if a else None

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(PROP_DIR, "yuquan_*.json"))):
        pin = os.path.basename(f).replace("yuquan_", "").replace(".json", "")
        ti = template_info(pin)
        if ti is None:
            continue
        templ, src = ti
        txt = doc_for(pin)
        if not txt:
            rows.append(dict(subject=pin, n_template=len(templ), status="no_doc"))
            continue
        abl, _ = ablated_contacts(txt)
        onset = onset_contacts(txt)
        cov = jacc_cov(templ, abl)
        src_cov = jacc_cov(src, abl) if src else None
        onset_cov = jacc_cov(onset, abl) if onset else None
        # template-source channels NOT in clinical onset = template's incremental target
        src_beyond_onset = sorted(src - onset) if src else []
        rows.append(dict(
            subject=pin, status="ok",
            n_template=len(templ), n_ablated=len(abl), n_onset=len(onset), n_source=len(src),
            template_coverage=round(cov, 3) if cov is not None else None,
            source_ablated_frac=round(src_cov, 3) if src_cov is not None else None,
            onset_coverage=round(onset_cov, 3) if onset_cov is not None else None,
            template_not_ablated=";".join(sorted(templ - abl)),
            source_channels=";".join(sorted(src)),
            source_beyond_onset=";".join(src_beyond_onset),
        ))
    cols = ["subject", "status", "n_template", "n_ablated", "n_onset", "n_source",
            "template_coverage", "source_ablated_frac", "onset_coverage",
            "template_not_ablated", "source_channels", "source_beyond_onset"]
    out_csv = os.path.join(OUT, "yuquan_coverage_prep.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    # console summary
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{'subj':<14}{'nT':<4}{'nAbl':<6}{'cov':<7}{'srcAbl':<8}{'onsetCov':<9}{'srcChans'}")
    for r in sorted(ok, key=lambda x: (x["template_coverage"] is None, x["template_coverage"] or 0)):
        print(f"{r['subject']:<14}{r['n_template']:<4}{r['n_ablated']:<6}"
              f"{str(r['template_coverage']):<7}{str(r['source_ablated_frac']):<8}"
              f"{str(r['onset_coverage']):<9}{r['source_channels']}")
    import numpy as np
    covs = [r["template_coverage"] for r in ok if r["template_coverage"] is not None]
    if covs:
        print(f"\ncoverage: median={np.median(covs):.2f} range {min(covs):.2f}-{max(covs):.2f}; "
              f"partial(<1.0)={sum(c < 0.999 for c in covs)}/{len(covs)}")
    print(f"no_doc: {[r['subject'] for r in rows if r['status']=='no_doc']}")
    print(f"\nwrote {out_csv}")

if __name__ == "__main__":
    main()
