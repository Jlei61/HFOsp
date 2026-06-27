"""Build the stable_k=2 field-swap subject-SNN cohort config.

For each subject with adaptive_cluster.stable_k==2 AND propagation_geometry templates,
pick ONE montage (narrow preferred; broad when narrow degenerate) and emit a geometry
diagnostic + eligibility verdict. SKIP when the two template-source cores share channels
(no endpoint swap to place two cores on).

Output: results/paper-ready-figure/_cohort_field_swap_snn/cohort_config.json
"""
import json, glob, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper-ready-figure/_cohort_field_swap_snn"
sys.path.insert(0, str(ROOT))
from src.sef_hfo_subject_placement import template_source_foci  # noqa: E402


def stable_k(path):
    try:
        return json.load(open(path))["adaptive_cluster"]["stable_k"]
    except Exception:
        return None


def diag(subj, montage):
    m, ca, cb = template_source_foci(subj, montage, k_early=3)
    C = np.asarray(m.contacts, float)
    ext = (C.max(0) - C.min(0))

    def cen(names):
        idx = [m.names.index(n) for n in names]; return C[idx].mean(0)
    ax = cen(cb) - cen(ca); ic = float(np.linalg.norm(ax))
    u = ax / (ic + 1e-9); perp = np.array([-u[1], u[0]])
    proj = C @ u; pp = C @ perp
    aspect = float((pp.max() - pp.min()) / ((proj.max() - proj.min()) + 1e-9))
    return dict(n_ch=len(m.names), overlap=len(set(ca) & set(cb)), raw_ic_mm=round(ic, 1),
                raw_ext_mm=round(float(ext.max()), 1), aspect=round(aspect, 2))


def eligible(d):
    if d["overlap"] > 0 or d["raw_ic_mm"] < 5.0:
        return False, "cores_not_distinct"
    return True, "ok"


def flags(d):
    fl = []
    if d["aspect"] < 0.10:
        fl.append("near_1D_axis~shaft")
    if d["raw_ext_mm"] > 90.0:
        fl.append("large_extent_maybe_cross_region")
    if d["n_ch"] < 7:
        fl.append("few_channels")
    return fl


def main():
    subjects = {}
    for montage, dirn in [("narrow", "interictal_propagation_masked"),
                          ("broad", "interictal_propagation_masked_broad")]:
        base = (ROOT / ("results/spatial_modulation/propagation_geometry"
                        + ("_broad" if montage == "broad" else "") + "/observation_readout/real_subjects"))
        for f in glob.glob(str(ROOT / f"results/{dirn}/per_subject/*.json")):
            subj = os.path.basename(f)[:-5]
            if subj.startswith("pr"):
                continue
            if stable_k(f) != 2:
                continue
            if not ((base / f"{subj}_t_a.json").exists() and (base / f"{subj}_t_b.json").exists()):
                continue
            try:
                d = diag(subj, montage); ok, why = eligible(d)
                subjects.setdefault(subj, {})[montage] = dict(**d, eligible=ok, why=why, flags=flags(d))
            except Exception as e:
                subjects.setdefault(subj, {})[montage] = dict(eligible=False, why=f"err:{str(e)[:40]}")

    config = []
    for subj in sorted(subjects):
        r = subjects[subj]
        pick = next((mo for mo in ("narrow", "broad") if mo in r and r[mo].get("eligible")), None)
        config.append(dict(subject=subj, montage=pick, chosen=(r[pick] if pick else None), candidates=r))

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(config, open(OUT / "cohort_config.json", "w"), indent=2)
    run = [c for c in config if c["montage"]]
    print(f"{len(config)} subjects; {len(run)} RUN, {len(config) - len(run)} SKIP -> {OUT}/cohort_config.json")


if __name__ == "__main__":
    main()
