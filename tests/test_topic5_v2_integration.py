# tests/test_topic5_v2_integration.py
import subprocess, sys, csv
import pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
@pytest.mark.integration
@pytest.mark.parametrize("axis", ["broad", "narrow"])
def test_legacy_reproduction_within_tolerance(axis, tmp_path):
    r = subprocess.run([sys.executable, "scripts/run_topic5_v2_legacy_repro.py",
                        "--substrate", axis, "--outdir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(tmp_path / axis / "phase1_qc_legacy_reproduction.csv")))
    assert rows and all("n_seizures" in x for x in rows)
    for x in rows: assert abs(float(x["delta"])) <= 0.02, f"{x['subject']} {x['band']} {x['delta']}"


@pytest.mark.integration
def test_iter_subject_seizure_windows_yields_for_epilepsiae_139(monkeypatch):
    """Task 6a: the factored-out seizure-window loader yields >=1 (idx, sw, eeg_rel) for
    epilepsiae_139 and reproduces the committed long-cache window params (read-only; no rebuild,
    no write to the shared ictal_field_long_cache)."""
    import json
    monkeypatch.chdir(ROOT)  # _inventory_rows() reads results/ via a cwd-relative path
    from scripts.build_topic5_ictal_field_long_cache import iter_subject_seizure_windows
    items = list(iter_subject_seizure_windows("epilepsiae_139", "broad"))
    assert items, "expected >=1 seizure window for epilepsiae_139"
    for idx, sw, eeg_rel in items:
        assert isinstance(idx, int)
        for attr in ("pre_sec", "post_sec", "fs", "ch_names", "seizure_id"):
            assert hasattr(sw, attr), f"sw missing {attr}"
        assert eeg_rel is None or isinstance(eeg_rel, float)
    # Behavior-preservation vs the committed cache (139 has drops=[], so loader-pass == eligible_idxs).
    meta = json.loads((ROOT / "results/topic5_ictal_recruitment"
                       / "ictal_field_long_cache/epilepsiae_139.json").read_text())
    yielded = {idx: sw for idx, sw, _ in items}
    assert set(yielded) == {int(k) for k in meta["seizure"]}, "yielded idxs != committed eligible_idxs"
    for k, s in meta["seizure"].items():
        sw = yielded[int(k)]
        assert abs(float(sw.pre_sec) - s["pre_sec"]) < 1e-6, f"pre_sec drift sz{k}"
        assert abs(float(sw.post_sec) - s["post_sec"]) < 1e-6, f"post_sec drift sz{k}"
