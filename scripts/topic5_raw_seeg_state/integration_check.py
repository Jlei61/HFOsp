#!/usr/bin/env python
"""Cross-worker integration gate for Raw-SEEG state R0.1.

Owner: main agent. The four workers each unit-tested their own module; the
errors that actually destroy a conclusion live at the seams between them, and
those seams are only exercised on real data. Execution plan section 10 lists
the eight checks below; all eight must PASS on at least one real subject before
any pilot training run is allowed to start.

    python scripts/topic5_raw_seeg_state/integration_check.py --subject yuquan_huanghanwen

Checks 3, 4, 6 and 7 are the ones that catch silent pollution. A FAIL on any of
them is not overridable by "the other six passed".
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.topic5_raw_seeg_state import contract  # noqa: E402


class Gate:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def run(self, num: int, name: str, critical: bool, fn) -> None:
        try:
            detail = fn()
            ok, note = True, detail
        except AssertionError as exc:
            ok, note = False, f"ASSERT: {exc}"
        except Exception as exc:  # noqa: BLE001
            ok, note = False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
        self.rows.append({"n": num, "name": name, "critical": critical,
                          "pass": ok, "detail": note})
        flag = "PASS" if ok else ("FAIL*" if critical else "FAIL")
        first = str(note).splitlines()[0] if note else ""
        print(f"  [{flag}] {num}. {name} :: {first}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-train", action="store_true",
                    help="skip check 8 (the end-to-end resume test), which needs a GPU")
    args = ap.parse_args(argv)
    subject = args.subject

    import pandas as pd
    import torch

    import zarr

    from src.topic5_raw_seeg_state import (analysis, model as M, raw_cache as RC,
                                           spectral_target as ST, train as T, windows as W)

    def open_target(subj):
        return zarr.open_array(str(contract.spectral_target_path(subj)), mode="r")

    print(f"integration gate :: {subject}", flush=True)
    g = Gate()

    data_dir = contract.DATA_DIR
    win = pd.read_parquet(data_dir / "window_index.parquet")
    con = pd.read_parquet(data_dir / "contact_metadata.parquet")
    win_s = win[win.subject == subject].sort_values("minute_index").reset_index(drop=True)
    con_s = con[con.subject == subject].sort_values("channel_index").reset_index(drop=True)
    dev_end = contract.dev_end_epoch(subject)

    # ---------------------------------------------------------------- 1
    def check1():
        f = np.fft.rfftfreq(contract.TARGET_WELCH_NPERSEG, 1.0 / contract.ANALYSIS_RATE_HZ)
        bands = contract.band_indices(f)
        assert len(bands) == contract.N_FREQ_BINS
        assert all(len(b) >= 1 for b in bands)
        assert contract.WINDOW_SAMPLES % contract.PATCH_SAMPLES == 0
        assert contract.MINUTE_SAMPLES == contract.WINDOW_SAMPLES * contract.WINDOWS_PER_MINUTE
        assert contract.MINUTE_SAMPLES % contract.PATCH_SAMPLES == 0
        return (f"{contract.N_FREQ_BINS} bands, min {min(len(b) for b in bands)} FFT bins; "
                f"patch {contract.PATCH_SAMPLES} | window {contract.WINDOW_SAMPLES} | "
                f"minute {contract.MINUTE_SAMPLES}")

    g.run(1, "contract self-consistency", False, check1)

    # ---------------------------------------------------------------- 2
    def check2():
        need = ["dataset_manifest.parquet", "contact_metadata.parquet",
                "window_index.parquet", "eligibility_summary.csv", "split_manifest.json",
                "data_audit.json"]
        missing = [n for n in need if not (data_dir / n).exists()]
        assert not missing, f"missing {missing}"
        audit = json.loads((data_dir / "data_audit.json").read_text())
        elig = pd.read_csv(data_dir / "eligibility_summary.csv")
        assert len(elig) == 34, f"eligibility has {len(elig)} rows, expected 34"
        cohort = audit.get("cohort", {})
        checks = cohort.get("checks", {})
        failed = [k for k, v in checks.items()
                  if isinstance(v, dict) and v.get("status") not in (None, "PASS",
                                                                     "NOT_CHECKABLE_AT_STAGE_A")]
        assert not failed, f"cohort audit FAIL on {failed}"
        return f"34 subjects; cohort checks {len(checks)} entries, 0 FAIL"

    g.run(2, "data-contract artifacts complete", False, check2)

    # ---------------------------------------------------------------- 3
    def check3():
        """Recompute a cached minute's Welch field from the cache itself."""
        raw, scale = RC.load_cache(subject)
        scale = np.asarray(scale, dtype=np.float64)
        tgt = open_target(subject)
        cidx = pd.read_parquet(contract.cache_dir(subject) / "cache_index.parquet")
        cached = cidx[(cidx.cached) & (~cidx.filled)]["minute_index"].to_numpy()
        assert cached.size, "no cached minutes"
        rng = np.random.default_rng(0)
        picks = rng.choice(cached, size=min(5, cached.size), replace=False)
        worst = 0.0
        for mi in picks:
            a = int(mi) * contract.MINUTE_SAMPLES
            x = np.asarray(raw[a:a + contract.MINUTE_SAMPLES, :], dtype=np.float64) * scale
            got = ST.minute_spectral_field(x)
            want = np.asarray(tgt[int(mi)], dtype=np.float64)
            fin = np.isfinite(got) & np.isfinite(want)
            assert fin.any(), f"minute {mi} target is all NaN"
            d = np.abs(got[fin] - want[fin]) / np.maximum(np.abs(want[fin]), 1e-9)
            worst = max(worst, float(d.max()))
        assert worst < 1e-4, f"max relative deviation {worst:.3e}"
        return f"{len(picks)} random cached minutes, max relative deviation {worst:.2e}"

    g.run(3, "cache <-> spectral target time alignment", True, check3)

    # ---------------------------------------------------------------- 4
    def check4():
        """Cache columns follow contact_metadata.channel_index.

        v2: the first version correlated the per-contact variance of the RAW
        int16 cache against the target. That test was worthless: the cache
        scales every contact so 6*MAD maps to a fixed count, which flattens the
        int16 variance across contacts by construction (measured r = 0.44 on a
        perfectly aligned subject). The scale must be applied first. And a test
        that cannot detect a permutation is not a test, so this one also proves
        it discriminates: the same statistic under a cyclic shift of the contact
        axis must be clearly worse than under the true alignment.
        """
        raw, scale = RC.load_cache(subject)
        scale = np.asarray(scale, dtype=np.float64)
        tgt = open_target(subject)
        assert raw.shape[1] == len(con_s), (
            f"cache has {raw.shape[1]} columns, contact_metadata has {len(con_s)} rows")
        assert tgt.shape[1] == len(con_s), "target contact axis disagrees with metadata"
        cidx = pd.read_parquet(contract.cache_dir(subject) / "cache_index.parquet")
        cached = cidx[(cidx.cached) & (~cidx.filled)]["minute_index"].to_numpy()
        mi = int(cached[len(cached) // 2])
        a = mi * contract.MINUTE_SAMPLES
        x = np.asarray(raw[a:a + contract.MINUTE_SAMPLES, :], dtype=np.float64) * scale
        v_raw = np.log10(x.var(axis=0) + 1e-12)
        v_tgt = np.asarray(tgt[mi], dtype=np.float64).mean(axis=1)
        ok = np.isfinite(v_raw) & np.isfinite(v_tgt)
        assert ok.sum() >= 8, "too few finite contacts to test the ordering"
        r = float(np.corrcoef(v_raw[ok], v_tgt[ok])[0, 1])
        # discrimination: the same statistic with the contact axis rolled
        shifts = [k for k in (1, 2, 3, len(con_s) // 2) if 0 < k < int(ok.sum())]
        r_shift = max(
            float(np.corrcoef(v_raw[ok], np.roll(v_tgt[ok], k))[0, 1]) for k in shifts)
        assert r > 0.9, (
            f"per-contact power correlates only r={r:.3f} between the raw cache and "
            "the spectral target -- the two contact axes are not the same order")
        assert r - r_shift > 0.25, (
            f"the ordering test does not discriminate: true r={r:.3f} but a rolled "
            f"contact axis reaches r={r_shift:.3f}, so it would not have caught a "
            "permutation")
        return (f"{raw.shape[1]} columns; per-contact power r={r:.3f} on minute {mi}, "
                f"best rolled r={r_shift:.3f} (margin {r - r_shift:.3f})")

    g.run(4, "channel order raw cache == target == contact_metadata", True, check4)

    # ---------------------------------------------------------------- 5
    def check5():
        idx = W.eligible_indices(subject, "validation", win_s,
                                 horizons=contract.HORIZONS_MIN, require_all=True,
                                 cache_index_path=contract.cache_dir(subject) / "cache_index.parquet")
        if idx.size == 0:
            idx = W.eligible_indices(subject, "train", win_s, horizons=contract.HORIZONS_MIN,
                                     require_all=True,
                                     cache_index_path=contract.cache_dir(subject) / "cache_index.parquet")
        assert idx.size, "no eligible window at all four horizons"
        ds = W.SubjectWindowDataset(subject, "train", win_s, con_s,
                                    horizons=contract.HORIZONS_MIN)
        item = ds[0]
        enc = {k: item[k] for k in contract.ALLOWED_INPUT_KEYS}
        contract.assert_no_forbidden_inputs(enc)
        n_c = int(item["raw"].shape[0])
        n_shafts = int(np.asarray(item["shaft_id"]).max()) + 1
        net = M.RawSeegStateModel(n_contacts=n_c, n_shafts=max(n_shafts, 1))
        net.eval()
        batch = {k: torch.as_tensor(np.asarray(v))[None] for k, v in enc.items()}
        with torch.no_grad():
            out = net(**batch)
        for h in contract.HORIZONS_MIN:
            p = out["pred"][h]
            assert tuple(p.shape) == (1, n_c, contract.N_FREQ_BINS), f"h={h} shape {tuple(p.shape)}"
            assert torch.isfinite(p).all(), f"h={h} non-finite prediction"
        try:
            net(**batch, soz=torch.zeros(1))
        except ValueError:
            pass
        else:
            raise AssertionError("forbidden-input gate did not fire on soz=")
        return (f"C={n_c}, {len(contract.HORIZONS_MIN)} horizons finite; "
                f"forbidden-input gate fired; {idx.size} all-horizon windows")

    g.run(5, "dataset item feeds the model, forbidden inputs rejected", False, check5)

    # ---------------------------------------------------------------- 6
    def check6():
        """Every timestamp in every artifact must sit strictly before the seal."""
        worst = -np.inf
        where = ""
        # Every row of the grid must START before the seal ...
        assert float(win_s.minute_start_epoch.max()) < dev_end, (
            "a window_index minute starts at or after dev_end_epoch")
        # ... and every minute that is actually USABLE must also END before it.
        # The one grid row that would straddle the seal is deliberately kept in
        # the index with covered=False, so the end-bound check belongs on the
        # usable subset, not on every row.
        usable = win_s[win_s.minute_usable.astype(bool)]
        assert len(usable), "no usable minute at all"
        assert float(usable.minute_start_epoch.max()) + 60.0 <= dev_end + 1e-6, (
            "a USABLE window_index minute ends at or after dev_end_epoch")
        worst, where = float(usable.minute_start_epoch.max()), "window_index(usable)"
        man = pd.read_parquet(data_dir / "dataset_manifest.parquet")
        man_s = man[(man.subject == subject) & (man.split != "sealed")]
        if len(man_s):
            v = float(man_s.block_start_epoch.max())
            if v > worst:
                worst, where = v, "dataset_manifest(non-sealed)"
            assert v < dev_end, "a non-sealed block starts at/after dev_end_epoch"
        cidx = pd.read_parquet(contract.cache_dir(subject) / "cache_index.parquet")
        cc = cidx[cidx.cached]
        if len(cc):
            v = float(cc.minute_start_epoch.max())
            assert v + 60.0 <= dev_end + 1e-6, "a cached minute reaches into the sealed span"
            if v > worst:
                worst, where = v, "cache_index"
        contract.assert_not_sealed(subject, [worst])
        return (f"max epoch {worst:.1f} ({where}) is {dev_end - worst:.1f} s "
                f"before dev_end {dev_end:.1f}")

    g.run(6, "sealed partition untouched everywhere", True, check6)

    # ---------------------------------------------------------------- 7
    def check7():
        """Normalisation must be a function of the artifact-clean TRAIN minutes.

        v2: the basis changed. target_mean/target_std are now estimated over the
        train contact-minutes that survive the artifact rule, because the
        artifact tail was carrying most of the variance and inflating the
        denominator by up to 9x. This check mirrors that basis, and still proves
        it bites by showing a train+validation estimate would move.
        """
        stats = json.loads(contract.subject_stats_path(subject).read_text())
        assert stats.get("standardisation_basis") == "artifact_clean_train_minutes", (
            "train_stats.json was not produced by the artifact-aware second pass; "
            f"basis = {stats.get('standardisation_basis')!r}")
        mean = np.asarray(stats["target_mean"], dtype=np.float64)
        std = np.asarray(stats["target_std"], dtype=np.float64)
        tgt = open_target(subject)
        bad_z = zarr.open_array(
            str(contract.cache_dir(subject) / "artifact_mask.zarr"), mode="r")
        cidx = pd.read_parquet(contract.cache_dir(subject) / "cache_index.parquet")
        tr = cidx[(cidx.cached) & (~cidx.filled) & (cidx.split == "train")]["minute_index"].to_numpy()
        va = cidx[(cidx.cached) & (~cidx.filled) & (cidx.split == "validation")]["minute_index"].to_numpy()
        assert tr.size, "no cached train minute"
        tr = np.sort(tr)
        arr = np.asarray(tgt[tr], dtype=np.float64)
        bad = np.asarray(bad_z[tr], dtype=bool)
        clean = np.where(~bad[:, :, None], arr, np.nan)
        with np.errstate(invalid="ignore"):
            m_ref = np.nanmean(clean, axis=0)
            s_ref = np.nanstd(clean, axis=0)
        dm = float(np.nanmax(np.abs(m_ref - mean)))
        ds_ = float(np.nanmax(np.abs(s_ref - std)))
        assert dm < 1e-6 and ds_ < 1e-6, (
            f"train stats differ from an artifact-clean train-only recomputation: "
            f"dmean={dm:.2e} dstd={ds_:.2e}")
        # the standardised clean train population must have unit second moment
        unit = float(np.nanmean(((clean - mean) / std) ** 2))
        assert 0.98 <= unit <= 1.02, f"clean train mean z^2 is {unit:.4f}, not ~1"
        # and the check must bite: folding in validation has to move the estimate
        assert va.size, "no cached validation minute to test independence against"
        arrv = np.asarray(tgt[np.sort(va)], dtype=np.float64) * 10.0
        moved = float(np.nanmax(np.abs(
            np.nanmean(np.concatenate([clean, arrv], axis=0), axis=0) - mean)))
        assert moved > 1e-9, ("scaling the validation half x10 did not move a "
                              "train+validation mean -- the independence test is vacuous")
        audit = stats.get("standardisation_audit", {})
        return (f"artifact-clean train recomputation matches (dmean {dm:.1e}, dstd {ds_:.1e}); "
                f"clean train mean z^2 = {unit:.4f}; a train+val mean would have moved "
                f"by {moved:.2f}; std inflation removed "
                f"{audit.get('std_inflation_median', float('nan')):.2f}x median")

    g.run(7, "normalisation uses train minutes only", True, check7)

    # ---------------------------------------------------------------- 8
    def check8():
        if args.skip_train:
            return "skipped by --skip-train"
        import copy
        import tempfile
        idx = W.eligible_indices(subject, "train", win_s, horizons=(1,), require_all=False)
        assert idx.size >= 12, f"only {idx.size} train windows"
        with tempfile.TemporaryDirectory() as td:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            # val_windows_final must be capped too. Leaving it at the production
            # default made this 10-step smoke test encode up to 3000 validation
            # windows in its final pass: 18 GB of reads and 23 minutes before it
            # was killed. One sample is 10 min x C contacts x 15360 samples x
            # 2 bytes, i.e. 9.5 MB at C=31 and 43 MB at C=139, so the final pass
            # is by far the most expensive thing in a short run.
            base = dict(subject=subject, arm="integration", seed=0, horizons=(1,),
                        batch_size=1, grad_accum=1, max_epochs=1, max_steps_per_epoch=10,
                        train_windows_per_epoch=10, val_windows_per_epoch=4,
                        val_windows_final=8, num_workers=0, device=dev)
            cfg_a = T.TrainConfig(out_dir=Path(td) / "a", **base)
            cfg_b = T.TrainConfig(out_dir=Path(td) / "b", **base)
            ra = T.train_subject(copy.deepcopy(cfg_a))
            rb = T.train_subject(copy.deepcopy(cfg_b))
            assert ra["status"] == "ok" and rb["status"] == "ok"
            assert np.isfinite(ra["best_val_forecast_loss"])
        return (f"2 x 10 steps on {dev}, status ok, val loss "
                f"{ra['best_val_forecast_loss']:.4f}; second run "
                f"{rb['best_val_forecast_loss']:.4f}")

    g.run(8, "end-to-end training runs and is finite", False, check8)

    # ---------------------------------------------------------------- report
    crit_fail = [r for r in g.rows if r["critical"] and not r["pass"]]
    any_fail = [r for r in g.rows if not r["pass"]]
    payload = {
        "subject": subject,
        "contract_version": contract.CONTRACT_VERSION,
        "code_revision": contract.code_revision(),
        "checks": g.rows,
        "n_pass": sum(1 for r in g.rows if r["pass"]),
        "n_fail": len(any_fail),
        "critical_failures": [r["name"] for r in crit_fail],
        "verdict": "PASS" if not any_fail else ("BLOCKED" if crit_fail else "FAIL_NONCRITICAL"),
    }
    out = Path(args.out) if args.out else (contract.RESULT_ROOT / "manifests" /
                                           f"INTEGRATION_CHECK_{subject}.json")
    contract.atomic_write_json(out, payload)
    print(f"\nverdict: {payload['verdict']}  ({payload['n_pass']}/{len(g.rows)} pass)  -> {out}")
    if crit_fail:
        print("CRITICAL failures block the pilot: " + ", ".join(payload["critical_failures"]),
              file=sys.stderr)
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
