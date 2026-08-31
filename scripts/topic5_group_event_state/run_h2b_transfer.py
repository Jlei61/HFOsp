#!/usr/bin/env python3
"""H2b: can an interictal-only group-event state transfer to the seizure task?

The state model is **frozen**: this script loads a trained checkpoint, replays the
patient's interictal stream with no gradient and no label, and only then lets a
post-hoc probe see seizure times.  Nothing about seizures ever reaches the
encoder, the state dynamics, or the state trajectory.

Risk sets are seizure-level and chronological, never event-level: with ~15
seizures and tens of thousands of events, an event-level AUC would mostly measure
how autocorrelated the event stream is.

H2b is a cross-task check, not a gate on H1.  A development-set positive is
written as development-only and never as cohort confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.dataset import SubjectSequence  # noqa: E402
from src.topic5_group_event_state.source_audit import write_json_atomic  # noqa: E402
from src.topic5_group_event_state.train import (  # noqa: E402
    GroupEventStateModel,
    TrainConfig,
    _auto_chunk,
    _data_shape,
    _load_geometry,
    _to_device,
    build_arms,
    estimate_stats,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"

# A control must be far from the next seizure and clear of the previous one's
# aftermath.  The first version demanded 4 h on both sides, which for a patient
# with eight seizures in 24 h left a pool of 147 events all sitting in one gap --
# and every one of them fell on the same side of the split, so the probe trained
# on four cases and zero controls and returned exactly 0.5 for everything.
CONTROL_MIN_LEAD_SEC = 2 * 3600.0
CONTROL_MIN_POSTICTAL_SEC = 1 * 3600.0
CONTROLS_PER_CASE = 20
MIN_SEIZURES = 6
MIN_CONTROLS_PER_SIDE = 5
# With ~50 training points a 96-dimensional state separates the labels perfectly
# and carries nothing to held-out time: on synthetic data a no-signal probe still
# scored 0.735.  Features are projected onto a few train-fitted components, and
# the label-permuted score is reported next to every number as the calibration.
MAX_PROBE_COMPONENTS = 12
N_PERMUTATIONS = 200


@torch.no_grad()
def dump_states(
    model: GroupEventStateModel, seq: SubjectSequence, device: torch.device, cfg: TrainConfig
) -> dict[str, np.ndarray]:
    """Pre-event state and current-event code for every interictal event."""

    model.eval()
    states, codes = [], []
    fast = slow = None
    for lo, hi, starts_session in seq.chunks(0, len(seq), cfg.chunk_events):
        batch = _to_device(seq.gather(lo, hi), device)
        n = hi - lo
        dt = batch["dt_prev"]
        dt_safe = torch.where(torch.isfinite(dt), dt, torch.zeros_like(dt))
        event_emb, _tok = model.encoder(batch)
        event_emb = event_emb.float()
        if starts_session or fast is None:
            fast, slow = model.state.initial(1, device)
        taus = model.state.taus()
        if model.background is not None:
            age = batch["background_age"]
            valid = torch.isfinite(age)
            bg_f, bg_s = model.background.encode(
                batch["background"], torch.where(valid, age, torch.zeros_like(age)), valid
            )
        chunk_states = []
        for step in range(n):
            f_e, s_e = model.state.evolve(fast, slow, dt_safe[step : step + 1], taus)
            if model.background is not None:
                f_e = f_e + bg_f[step : step + 1]
                s_e = s_e + bg_s[step : step + 1]
            chunk_states.append(torch.cat([f_e, s_e], dim=-1))
            fast, slow = model.state.update(f_e, s_e, event_emb[step : step + 1])
        states.append(torch.cat(chunk_states, 0).cpu().numpy())
        codes.append(event_emb.cpu().numpy())
    return {
        "state": np.concatenate(states, 0).astype(np.float32),
        "event_code": np.concatenate(codes, 0).astype(np.float32),
    }


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(
        (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    )


def _project(x_train: np.ndarray, x_test: np.ndarray, n_components: int):
    """Train-fitted standardisation + PCA. Nothing about test time is used."""

    mu, sd = x_train.mean(0), x_train.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    zt = (x_train - mu) / sd
    ze = (x_test - mu) / sd
    k = int(max(1, min(n_components, zt.shape[1], zt.shape[0] - 1)))
    if k >= zt.shape[1]:
        return zt, ze
    _u, _s, vt = np.linalg.svd(zt - zt.mean(0), full_matrices=False)
    basis = vt[:k].T
    return zt @ basis, ze @ basis


def _logistic_scores(x_train, y_train, x_test, alpha=1.0, iters=300) -> np.ndarray:
    xt = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    xe = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
    w = np.zeros(xt.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(xt @ w, -30, 30)))
        grad = xt.T @ (p - y_train) / y_train.size + alpha * np.r_[w[:-1], 0.0]
        hess_diag = (xt**2).T @ (p * (1 - p)) / y_train.size + alpha
        w -= grad / np.maximum(hess_diag, 1e-6)
    return xe @ w


def _probe_auc(x_train, y_train, x_test, y_test, seed: int) -> dict:
    """Held-out AUC together with the AUC this probe reaches on permuted labels.

    The permuted figure is not decoration: with a handful of cases and a state of
    this width, a probe fitted on shuffled labels still scores well above 0.5, so
    only the gap between the two numbers means anything."""

    zt, ze = _project(x_train, x_test, MAX_PROBE_COMPONENTS)
    observed = _auc(_logistic_scores(zt, y_train, ze), y_test)
    rng = np.random.default_rng(seed)
    null = np.array([
        _auc(_logistic_scores(zt, rng.permutation(y_train), ze), y_test)
        for _ in range(N_PERMUTATIONS)
    ])
    null = null[np.isfinite(null)]
    return {
        "auc": observed,
        "permuted_auc_median": float(np.median(null)) if null.size else float("nan"),
        "auc_minus_permuted": (
            observed - float(np.median(null)) if null.size and np.isfinite(observed) else float("nan")
        ),
        "permutation_p": (
            float((null >= observed).mean()) if null.size and np.isfinite(observed) else float("nan")
        ),
        "n_components": int(zt.shape[1]),
    }


def analyse_patient(run_dir: Path, dataset_root: Path, device: torch.device, seed: int) -> dict | None:
    result = json.loads((run_dir / "result.json").read_text())
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        return None
    subject = result["subject"]
    seq = SubjectSequence(dataset_root / subject)
    seizures = seq.index.get("seizures", [])
    if len(seizures) < 1:
        return {"subject": subject, "arm": result["arm"], "seed": result["seed"],
                "status": "no_seizures", "n_seizures": 0}

    arm = build_arms()[result["arm"]]
    cfg = TrainConfig(chunk_events=_auto_chunk(seq, 128))
    train_lo, train_hi = seq.split_slice("train")
    input_stats, target_stats = estimate_stats(seq, train_lo, train_hi, seed=int(result["seed"]))
    geometry = _load_geometry(seq) if arm.encoder.use_geometry else None
    model = GroupEventStateModel(
        arm, _data_shape(seq), geometry.to(device) if geometry is not None else None,
        seq.history.shape[1], None, input_stats, target_stats,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    for p in model.parameters():
        p.requires_grad_(False)

    dumped = dump_states(model, seq, device, cfg)
    t = seq.t_abs
    onsets = np.sort(np.array([float(s["onset_epoch"]) for s in seizures]))
    offsets = np.sort(np.array([float(s["offset_epoch"]) for s in seizures]))
    patterns = [str(s.get("pattern", "")) for s in seizures]

    case_idx, case_time = [], []
    for onset in onsets:
        before = np.flatnonzero(t < onset)
        if before.size >= 100:
            case_idx.append(int(before[-1]))
            case_time.append(float(onset))
    if len(case_idx) < 2:
        return {"subject": subject, "arm": result["arm"], "seed": result["seed"],
                "status": "too_few_usable_seizures", "n_seizures": len(seizures)}

    dist_next = np.full(t.size, np.inf)
    dist_prev = np.full(t.size, np.inf)
    for i, ti in enumerate(t):
        fut = onsets[onsets >= ti]
        past = offsets[offsets <= ti]
        if fut.size:
            dist_next[i] = fut[0] - ti
        if past.size:
            dist_prev[i] = ti - past[-1]
    far = (dist_next > CONTROL_MIN_LEAD_SEC) & (dist_prev > CONTROL_MIN_POSTICTAL_SEC)
    control_pool = np.flatnonzero(far)
    if control_pool.size < CONTROLS_PER_CASE:
        return {"subject": subject, "arm": result["arm"], "seed": result["seed"],
                "status": "no_control_pool", "n_seizures": len(seizures)}
    rng = np.random.default_rng(seed)
    controls = rng.choice(
        control_pool, size=min(control_pool.size, CONTROLS_PER_CASE * len(case_idx)), replace=False
    )

    idx = np.concatenate([np.array(case_idx), controls])
    label = np.concatenate([np.ones(len(case_idx)), np.zeros(controls.size)])
    times = t[idx]
    # Split on the median time of the risk set itself, not on the median seizure
    # time: when the seizures cluster at one end, the latter puts every control on
    # one side and the probe trains on cases alone.
    cut = float(np.median(times))
    is_train = times < cut
    both_sides = (
        label[is_train].sum() >= 1
        and label[~is_train].sum() >= 1
        and (is_train.sum() - label[is_train].sum()) >= MIN_CONTROLS_PER_SIDE
        and ((~is_train).sum() - label[~is_train].sum()) >= MIN_CONTROLS_PER_SIDE
    )
    if not both_sides:
        return {
            "subject": subject, "arm": result["arm"], "seed": result["seed"],
            "status": "chronological_split_degenerate", "n_seizures": len(seizures),
            "n_cases": len(case_idx), "n_controls": int(controls.size),
            "n_train": int(is_train.sum()), "n_train_cases": int(label[is_train].sum()),
            "n_test": int((~is_train).sum()), "n_test_cases": int(label[~is_train].sum()),
        }

    feature_sets = {
        "history_only": seq.history[idx],
        "current_observation": np.stack(
            [
                np.nan_to_num(seq.dt_prev[idx]),
                np.asarray(seq.arrays["participation"][seq.order[idx]]).sum(1),
                np.nan_to_num(np.asarray(seq.arrays["relative_delay"][seq.order[idx]])).max(1),
            ],
            axis=1,
        ),
        "memoryless_event_code": dumped["event_code"][idx],
        "persistent_group_event_state": dumped["state"][idx],
    }
    probes = {}
    for name, x in feature_sets.items():
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        probes[name] = _probe_auc(
            x[is_train], label[is_train], x[~is_train], label[~is_train], seed
        )
    aucs = {k: v["auc"] for k, v in probes.items()}
    nulls = {k: v["permuted_auc_median"] for k, v in probes.items()}
    excess = {k: v["auc_minus_permuted"] for k, v in probes.items()}
    perm_p = {k: v["permutation_p"] for k, v in probes.items()}

    by_pattern: dict[str, int] = {}
    for p in patterns:
        by_pattern[p or "unlabelled"] = by_pattern.get(p or "unlabelled", 0) + 1
    return {
        "subject": subject,
        "arm": result["arm"],
        "seed": result["seed"],
        "status": "ok" if len(case_idx) >= MIN_SEIZURES else "underpowered_few_seizures",
        "n_seizures": len(seizures),
        "n_cases": len(case_idx),
        "n_controls": int(controls.size),
        "n_cases_test": int(label[~is_train].sum()),
        "auc": aucs,
        "label_permuted_auc": nulls,
        "auc_minus_permuted": excess,
        "permutation_p": perm_p,
        "n_probe_components": {k: v["n_components"] for k, v in probes.items()},
        "seizure_patterns": by_pattern,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=V0_1 / "runs")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--arms", nargs="+", default=["a4_full_multimodal_state"])
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_1/dataset"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=V0_1 / "h2b_transfer.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for run_dir in sorted((args.runs_root / args.tag).glob("*")):
        if not run_dir.is_dir() or not (run_dir / "result.json").exists():
            continue
        if not any(f"__{a}__" in run_dir.name for a in args.arms):
            continue
        try:
            rec = analyse_patient(run_dir, args.dataset_root, device, args.seed)
        except Exception as exc:
            rec = {"run": run_dir.name, "status": f"error:{type(exc).__name__}", "detail": str(exc)[:200]}
        if rec:
            records.append(rec)
            print(f"{rec.get('subject', run_dir.name)}: {rec['status']} "
                  + (json.dumps({k: round(v, 3) for k, v in rec['auc'].items()}) if rec.get("auc") else ""),
                  flush=True)

    usable = [r for r in records if r.get("status") == "ok"]
    summary: dict = {"n_runs": len(records), "n_patients_ok": len({r["subject"] for r in usable})}
    if usable:
        names = list(usable[0]["auc"])
        per_patient: dict[str, dict[str, list[float]]] = {}
        for r in usable:
            per_patient.setdefault(r["subject"], {n: [] for n in names})
            for n in names:
                per_patient[r["subject"]][n].append(r["auc"][n])
        excess_by_patient: dict[str, dict[str, list[float]]] = {}
        for r in usable:
            excess_by_patient.setdefault(r["subject"], {n: [] for n in names})
            for n in names:
                excess_by_patient[r["subject"]][n].append(r["auc_minus_permuted"][n])
        summary["patient_first_auc"] = {}
        summary["patient_first_auc_minus_permuted"] = {}
        for n in names:
            vals = np.array([float(np.median(v[n])) for v in per_patient.values()])
            vals = vals[np.isfinite(vals)]
            summary["patient_first_auc"][n] = {
                "n_patients": int(vals.size),
                "median": float(np.median(vals)) if vals.size else float("nan"),
                "n_above_0.5": int((vals > 0.5).sum()),
            }
            ex = np.array([float(np.median(v[n])) for v in excess_by_patient.values()])
            ex = ex[np.isfinite(ex)]
            summary["patient_first_auc_minus_permuted"][n] = {
                "n_patients": int(ex.size),
                "median": float(np.median(ex)) if ex.size else float("nan"),
                "n_above_zero": int((ex > 0).sum()),
            }
        state = np.array([float(np.median(v["persistent_group_event_state"])) for v in excess_by_patient.values()])
        code = np.array([float(np.median(v["memoryless_event_code"])) for v in excess_by_patient.values()])
        ok = np.isfinite(state) & np.isfinite(code)
        summary["state_minus_memoryless_on_permutation_calibrated_excess"] = {
            "n_patients": int(ok.sum()),
            "median": float(np.median(state[ok] - code[ok])) if ok.any() else float("nan"),
            "n_state_better": int((state[ok] > code[ok]).sum()),
        }
    if usable:
        try:
            from scipy.stats import binomtest, wilcoxon

            for n in names:
                ex = np.array([float(np.median(v[n])) for v in excess_by_patient.values()])
                ex = ex[np.isfinite(ex)]
                if ex.size >= 5:
                    nz = ex[ex != 0]
                    summary["patient_first_auc_minus_permuted"][n]["sign_test_p"] = float(
                        binomtest(int((nz > 0).sum()), int(nz.size), 0.5).pvalue
                    ) if nz.size else float("nan")
                    summary["patient_first_auc_minus_permuted"][n]["wilcoxon_p"] = float(
                        wilcoxon(nz).pvalue
                    ) if nz.size >= 5 else float("nan")
        except Exception:
            pass
    summary["power_note"] = (
        "calibration on synthetic data with a known answer: with 6 cases per side the "
        "permutation-calibrated excess is -0.08 under no signal and only +0.06 under a "
        "2.5-SD signal (per-patient permutation p = 0.34). Per patient this probe can see "
        "only a large effect; a per-patient null means 'not visible at this size', not "
        "'not there'. The load-bearing statistic is the patient-first sign test."
    )
    summary["note"] = (
        "development-only; H2b is a cross-task check, not a gate on H1, and a "
        "positive here is not cohort confirmation"
    )
    write_json_atomic({"summary": summary, "runs": records}, args.out)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
