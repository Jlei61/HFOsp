"""Cohort field-swap SNN batch — spontaneous twoend, per-subject core_r.

Calibration (E1146 sweep+probe+driven test) established a FUNDAMENTAL tradeoff:
high per-event coverage needs big/far waves, but that loses directional cleanliness
(big cores -> one-core dominance OR mixed-direction nucleation; higher drive -> events
vanish). This holds for twoend AND single-core, so neither spontaneous nor driven-pooled
escapes it. => use the approved spontaneous twoend with a per-subject core_r scaled to the
subject's geometry (bigger field -> bigger core, the "mind core size per subject" ask),
clamped BELOW the balance-collapse point. Report coverage + balance; flag tradeoff cases.

Per subject: ONE twoend run at rule core_r (T=FINAL_T) -> single-run Fig4A + montage-aware
Fig4B. Memory-gated, polite concurrency (shared machine, user's a2 loop runs concurrently).
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/home/honglab/leijiaxin/HFOsp"
RUN_DIR = f"{ROOT}/results/topic4_sef_hfo/field_swap_subject_snn"
CONFIG = f"{ROOT}/results/paper-ready-figure/_cohort_field_swap_snn/cohort_config.json"
MASTER_LOG = f"{ROOT}/results/paper-ready-figure/_cohort_field_swap_snn/cohort_batch.log"
INDEX = f"{ROOT}/results/paper-ready-figure/_cohort_field_swap_snn/cohort_index.json"

# ---- tunables (calibrated on E1146: balance holds to ~cr2.5-3.0, collapses by cr3.5+) ----
K_FRAC = 0.22        # core_r = clamp(K_FRAC * inter_core_sheet, CR_LO, CR_HI)
CR_LO, CR_HI = 2.5, 3.5
DRIVE = 0.6          # probe: drive>0.6 destroys discrete events
FINAL_T = 8000.0
SEED = 3
CORE_MEAN, CORE_STD, K_DIR = 17.5, 1.0, 2
CONCURRENCY = 10
MIN_FREE_GB = 40

os.chdir(ROOT)
sys.path.insert(0, ROOT)
from src.sef_hfo_subject_placement import template_source_foci, register_to_sheet  # noqa: E402


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(MASTER_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024 / 1024
    return 0.0


def _wait_for_mem():
    waited = 0
    while _avail_gb() < MIN_FREE_GB:
        time.sleep(20); waited += 20
        if waited % 300 == 0:
            log(f"  [mem-gate] {_avail_gb():.0f}GB < {MIN_FREE_GB}GB, waiting")


def core_r_for(subject, montage):
    m, ca, cb = template_source_foci(subject, montage, k_early=3)
    ic = register_to_sheet(m, ca, cb)["inter_core_mm_sheet"]
    cr = max(CR_LO, min(CR_HI, K_FRAC * ic))
    cr = min(cr, 0.42 * ic)        # no-merge cap: keep the two cores spatially distinct
    cr = max(cr, 1.0)              # must contain enough neurons to ignite
    flag = "cores_close_wide_cloud" if (0.42 * ic < CR_LO) else None   # plane-fit shrank inter-core
    return round(cr, 2), round(ic, 2), flag


def run_twoend(subject, montage, core_r, tag):
    ro = f"{RUN_DIR}/readout_{tag}.json"
    if os.path.exists(ro):
        return json.load(open(ro))
    _wait_for_mem()
    cmd = ["python", "scripts/run_sef_hfo_subject_snn.py",
           "--subject", subject, "--montage", montage, "--lesion", "twoend_equal",
           "--placement", "template_source", "--k-early", "3",
           "--core-mean", str(CORE_MEAN), "--core-std", str(CORE_STD), "--k-dir", str(K_DIR),
           "--core-r", str(core_r), "--drive", str(DRIVE), "--T", str(FINAL_T),
           "--seed", str(SEED), "--tag", tag]
    env = dict(os.environ, MPLCONFIGDIR="/tmp/mpl", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(ro):
        log(f"  FAIL sim {tag}: rc={r.returncode} {r.stderr.strip()[-200:]}")
        return None
    return json.load(open(ro))


def metrics(ro):
    import numpy as np
    vc = max(1, ro["valid_contacts"]); kd = ro.get("k_dir", 2)
    clean = [e for e in ro["events"] if e.get("sign") is not None and e.get("n_part", 0) >= 2 * kd]
    npart = np.array([e["n_part"] for e in clean]) if clean else np.array([0])
    union = set()
    for e in clean:
        union |= {n for n, v in (e.get("ranks") or {}).items() if v is not None}
    f, rv = ro["dir_forward"], ro["dir_reverse"]
    minfrac = (min(f, rv) / (f + rv)) if (f + rv) else 0.0
    return dict(valid=vc, n_clean=len(clean), fwd=f, rev=rv, minority_frac=round(minfrac, 2),
                per_event_cov=round(float(npart.mean() / vc), 2),
                max_n_part=int(npart.max()), union_cov=round(len(union) / vc, 2))


def main():
    cfg = [c for c in json.load(open(CONFIG)) if c["montage"]]
    log(f"=== COHORT BATCH (spontaneous twoend) START: {len(cfg)} subjects, conc={CONCURRENCY}, "
        f"K_FRAC={K_FRAC} clamp[{CR_LO},{CR_HI}] drive={DRIVE} T={FINAL_T} seed={SEED} ===")

    plan = []
    for c in cfg:
        s, m = c["subject"], c["montage"]
        try:
            cr, ic, cflag = core_r_for(s, m)
        except Exception as e:
            log(f"  rule ERR {s}: {e}"); continue
        flags = list(c["chosen"].get("flags", [])) + ([cflag] if cflag else [])
        plan.append(dict(subject=s, montage=m, core_r=cr, inter_core=ic,
                         tag=f"{s}_cohort_cr{cr}_s{SEED}", flags=flags))

    # ---- Phase A: one twoend run per subject ----
    log("--- Phase A: twoend runs ---")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        fut = {ex.submit(run_twoend, p["subject"], p["montage"], p["core_r"], p["tag"]): p for p in plan}
        for f in as_completed(fut):
            p = fut[f]; ro = f.result()
            if ro is None:
                p["status"] = "sim_failed"; log(f"  done {p['subject']}: SIM FAILED")
            else:
                p["metrics"] = metrics(ro); p["status"] = "sim_ok"
                mm = p["metrics"]
                log(f"  done {p['subject']} cr={p['core_r']}: cov={mm['per_event_cov']} "
                    f"union={mm['union_cov']} {mm['fwd']}/{mm['rev']} minfrac={mm['minority_frac']} "
                    f"clean={mm['n_clean']}")

    # ---- Phase B: figures ----
    log("--- Phase B: Fig4A(single-run) + Fig4B ---")
    env = dict(os.environ, MPLCONFIGDIR="/tmp/mpl")
    for p in plan:
        if p.get("status") != "sim_ok":
            continue
        fig_name = f"fig_subject_snn_{p['subject']}"
        a = subprocess.run(["python", "scripts/paper_figures/plot_fig_subject_snn.py",
                            "--twoend-tag", p["tag"], "--fig-name", fig_name,
                            "--label", f"{p['subject']} ({p['montage']}, core_r={p['core_r']})"],
                           capture_output=True, text=True, env=env)
        b = subprocess.run(["python", "scripts/paper_figures/plot_fig_subject_snn_kmeans2.py",
                            "--tag", p["tag"], "--fig-name", fig_name, "--montage", p["montage"]],
                           capture_output=True, text=True, env=env)
        p["fig4A"] = (a.returncode == 0); p["fig4B"] = (b.returncode == 0)
        log(f"  figs {p['subject']}: A={'ok' if p['fig4A'] else 'FAIL '+a.stderr.strip()[-120:]} "
            f"B={'ok' if p['fig4B'] else 'FAIL '+b.stderr.strip()[-120:]}")

    json.dump(plan, open(INDEX, "w"), indent=2)
    log(f"=== COHORT BATCH DONE; index -> {INDEX} ===")


if __name__ == "__main__":
    main()
