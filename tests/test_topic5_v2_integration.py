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


@pytest.mark.integration
def test_band_cache_smoke_epilepsiae_139(tmp_path):
    """Task 6: build the multi-band masked band-power cache for one 512 Hz subject (2 bands)
    and check the npz + sidecar contract. Writes to an isolated tmp dir (never the shared tree).

    ``ripple_full_80_250`` on epilepsiae_139 (fs=512): line-noise harmonics (100/150/200/250)
    fall inside [80,250] -> masked out -> ``eff_frac < 1``; hi 250 > fs512_hi_safe 220 ->
    ``fs_edge_flag`` True. A subject-level ``analysis_channels`` fixed mask is written."""
    import json
    import numpy as np
    r = subprocess.run([sys.executable, "scripts/build_topic5_v2_band_cache.py",
                        "--subjects", "epilepsiae_139",
                        "--bands", "legacy_bb_1_45", "ripple_full_80_250",
                        "--outdir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    npz = tmp_path / "cache" / "epilepsiae_139.npz"
    sidecar = tmp_path / "cache" / "epilepsiae_139.json"
    assert npz.exists(), f"missing npz\n{r.stdout}\n{r.stderr}"
    assert sidecar.exists(), f"missing sidecar json\n{r.stdout}"
    meta = json.loads(sidecar.read_text())
    rip = meta["bands"]["ripple_full_80_250"]
    assert rip, "no ripple_full_80_250 (band,seizure) QC entries"
    for idx, qc in rip.items():
        assert qc["eff_frac"] < 1.0, f"sz{idx} eff_frac={qc['eff_frac']} (line-noise mask should bite)"
        assert qc["fs_edge_flag"] is True, f"sz{idx} fs_edge_flag={qc['fs_edge_flag']} (250>220 on 512Hz)"
        assert "n_band_bins" in qc and "low_baseline_channels" in qc and "bad_channels" in qc
    # validity-only good-set (issue #8, task 6b): saturation on genuinely-seizing high-ripple
    # channels no longer drives the fixed mask, so analysis_channels must not collapse to empty.
    assert isinstance(meta["analysis_channels"], list) and meta["analysis_channels"], \
        f"analysis_channels must be a non-empty list, got {meta['analysis_channels']!r}"
    z = np.load(npz, allow_pickle=True)
    assert "channels" in z
    some_idx = next(iter(rip))
    assert f"ripple_full_80_250__zt__{some_idx}" in z, "missing per-(band,seizure) z trace"
    assert f"ripple_full_80_250__relt__{some_idx}" in z, "missing per-(band,seizure) relt vector"


@pytest.mark.integration
def test_order_null_depcheck_broad_strength_domain(tmp_path):
    """Task 9 (relaxed, issue #12): the order-null dependency check runs for --substrate broad,
    writes the depcheck CSV with the required columns, and EVERY order_null_strength is one of
    {strong, weak_downgrade, missing}. Does NOT assert ">=1 strong" -- that is a QC print; the
    script exits 0 (with a stderr warning) even if no subject reaches strong."""
    r = subprocess.run([sys.executable, "scripts/run_topic5_v2_order_null_depcheck.py",
                        "--substrate", "broad", "--outdir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    csv_path = tmp_path / "broad" / "phase1_order_null_depcheck.csv"
    assert csv_path.exists(), f"missing depcheck CSV\n{r.stdout}\n{r.stderr}"
    rows = list(csv.DictReader(open(csv_path)))
    assert rows, "empty depcheck CSV"
    required = {"subject", "axis_set", "has_event_data_a", "has_event_data_b",
                "corr_rebuilt_vs_geo_a", "corr_rebuilt_vs_geo_b", "order_null_strength"}
    for x in rows:
        assert required <= set(x), f"missing cols {required - set(x)} for {x.get('subject')}"
        assert x["axis_set"] == "broad", f"{x['subject']} axis_set={x['axis_set']!r}"
        assert x["order_null_strength"] in {"strong", "weak_downgrade", "missing"}, \
            f"{x['subject']} bad strength {x['order_null_strength']!r}"


@pytest.mark.integration
def test_v2_alignment_raw_smoke_epilepsiae_139(tmp_path):
    """Task 7: raw early-ictal alignment tables for epilepsiae_139 (broad substrate).

    Subject-fixed analysis mask (same contacts across every band); window -> seizure
    median -> subject median; broad/narrow never pooled. Asserts the subject_summary
    carries a PRIMARY composite band (``ripple_full_80_250``) AND the legacy reproduction
    band (``legacy_bb_1_45``), all rows ``axis_set=broad`` on the PRIMARY fixed-mask path
    (``used_fixed_mask=True``), and that legacy_bb carries the QC-2 record columns."""
    r = subprocess.run([sys.executable, "scripts/run_topic5_v2_alignment.py",
                        "--feature", "raw", "--substrate", "broad",
                        "--subjects", "epilepsiae_139", "--outdir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    subj_csv = tmp_path / "broad" / "phase1_alignment_raw_subject_summary.csv"
    assert subj_csv.exists(), f"missing subject_summary\n{r.stdout}\n{r.stderr}"
    rows = list(csv.DictReader(open(subj_csv)))
    assert rows, "empty subject_summary"
    bands = {x["band"] for x in rows}
    assert "ripple_full_80_250" in bands, f"missing ripple_full_80_250 row; bands={bands}"
    assert "legacy_bb_1_45" in bands, f"missing legacy_bb_1_45 row; bands={bands}"
    for x in rows:
        assert x["axis_set"] == "broad", f"{x['band']} axis_set={x['axis_set']!r}"
        assert x["used_fixed_mask"] == "True", f"{x['band']} used_fixed_mask={x['used_fixed_mask']!r}"
        assert x["feature"] == "raw", f"{x['band']} feature={x['feature']!r}"
    legacy = next(x for x in rows if x["band"] == "legacy_bb_1_45")
    # QC-2 (P1-d) record-only cross-check columns must be populated for the legacy band.
    assert legacy["fixed_mask_delta"] != "", "legacy_bb missing fixed_mask_delta (QC-2 record)"
    assert legacy["n_channels_dropped_by_fixed_mask"] != "", "legacy_bb missing n_channels_dropped_by_fixed_mask"


@pytest.mark.integration
def test_confound_maps_smoke_epilepsiae_139(tmp_path):
    """Task 12a: build the per-contact confound covariate maps for one subject (broad).

    These maps let downstream residualize G_HFO against confounds (HFO rate, baseline power,
    shaft position) so alignment claims are about TIMING GEOMETRY, not rate/power topography.
    Asserts the JSON exists and carries non-empty per-contact hfo_rate / baseline_band_power /
    shaft_position maps for epilepsiae_139."""
    import json
    r = subprocess.run([sys.executable, "scripts/build_topic5_v2_confound_maps.py",
                        "--subjects", "epilepsiae_139", "--substrate", "broad",
                        "--outdir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    out = tmp_path / "broad" / "phase1_confound_maps.json"
    assert out.exists(), f"missing confound maps json\n{r.stdout}\n{r.stderr}"
    data = json.loads(out.read_text())
    assert "epilepsiae_139" in data, f"no epilepsiae_139 entry; keys={list(data)}"
    m = data["epilepsiae_139"]
    for key in ("hfo_rate", "baseline_band_power", "shaft_position"):
        assert key in m, f"missing {key} map"
        assert isinstance(m[key], dict) and m[key], f"{key} map is empty"
    # hfo_rate is a per-channel interictal event-count topography (>=1 positive count).
    assert any(float(v) > 0 for v in m["hfo_rate"].values()), "hfo_rate all zero"


@pytest.mark.integration
def test_common_resid_cache_and_alignment_epilepsiae_139(tmp_path):
    """Task 10b (Gate B input): build the leave-one-band-out (LOBO) common-field residual cache
    for epilepsiae_139, then run the alignment with ``--feature common_resid`` pointing at that
    residual cache and assert the ``common_resid`` subject_summary exists with PRIMARY-band rows.

    The residual cache has the SAME npz structure/keys as the raw band cache (``{B}__zt__{idx}`` /
    ``{B}__relt__{idx}`` / ``channels``) plus the reused ``analysis_channels`` sidecar, so the
    alignment reads it exactly like the raw cache (same fixed-mask + ``primary_bands_validity``
    basis asserts). Fully isolated: residual cache and alignment CSVs both go to tmp dirs."""
    import json
    import numpy as np
    cache_dir = tmp_path / "common_resid_cache"
    rb = subprocess.run([sys.executable, "scripts/build_topic5_v2_common_resid_cache.py",
                         "--subjects", "epilepsiae_139", "--substrate", "broad",
                         "--outdir", str(cache_dir)], cwd=ROOT, capture_output=True, text=True)
    assert rb.returncode == 0, f"{rb.stdout}\n{rb.stderr}"
    npz = cache_dir / "epilepsiae_139.npz"
    sidecar = cache_dir / "epilepsiae_139.json"
    assert npz.exists(), f"missing residual npz\n{rb.stdout}\n{rb.stderr}"
    assert sidecar.exists(), f"missing residual sidecar\n{rb.stdout}"
    z = np.load(npz, allow_pickle=True)
    assert "channels" in z, "residual cache missing channels array"
    # residual cache carries per-(primary band, seizure) zt/relt keys — SAME structure as band cache.
    assert any(k.startswith("gamma_LVFA__zt__") for k in z.files), f"no primary residual zt; {list(z.files)[:6]}"
    assert any(k.startswith("gamma_LVFA__relt__") for k in z.files), "no primary residual relt"
    # reused sidecar keeps the fixed-mask contract the alignment asserts on.
    side = json.loads(sidecar.read_text())
    assert side["analysis_channels_basis"] == "primary_bands_validity", side.get("analysis_channels_basis")
    assert side["analysis_channels"], "residual sidecar analysis_channels empty"

    ra = subprocess.run([sys.executable, "scripts/run_topic5_v2_alignment.py",
                         "--feature", "common_resid", "--substrate", "broad",
                         "--subjects", "epilepsiae_139", "--feature-cache-dir", str(cache_dir),
                         "--outdir", str(tmp_path / "align_out")], cwd=ROOT, capture_output=True, text=True)
    assert ra.returncode == 0, f"{ra.stdout}\n{ra.stderr}"
    subj_csv = tmp_path / "align_out" / "broad" / "phase1_alignment_common_resid_subject_summary.csv"
    assert subj_csv.exists(), f"missing common_resid subject_summary\n{ra.stdout}\n{ra.stderr}"
    rows = list(csv.DictReader(open(subj_csv)))
    assert rows, "empty common_resid subject_summary"
    primary = {"delta_HYP_slow", "theta_preictal_PAC", "alpha_sharp_leq13", "beta_LVFA_low",
               "gamma_LVFA", "hg_low_ripple", "ripple_high"}
    bands = {x["band"] for x in rows}
    assert bands & primary, f"no primary band rows in common_resid summary; bands={bands}"
    for x in rows:
        assert x["feature"] == "common_resid", f"{x['band']} feature={x['feature']!r}"
        assert x["axis_set"] == "broad", f"{x['band']} axis_set={x['axis_set']!r}"


def test_aperiodic_vectorized_excess_matches_helper():
    """Task 11b core math: the build script's VECTORIZED per-(channel,time-bin) band excess equals
    the scalar Task-11 helper ``aperiodic_corrected_excess_power`` cell-by-cell (the vectorization
    is a perf refactor — fit the log-log 1/f ONCE per (c,tt), reuse for every band — NOT a different
    computation). Synthetic pure-1/f PSD + a ripple bump on one (channel,bin); fast (no subprocess)."""
    import numpy as np
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.build_topic5_v2_aperiodic_cache import _excess_traces, FIT_LO, FIT_HI, MIN_R2
    from src.topic5_v2_band_scan import aperiodic_corrected_excess_power, line_noise_bin_mask
    rng = np.random.default_rng(0)
    f = np.arange(0, 257, 1.0)                                  # 1 Hz bins to 256 (fs=512-like grid)
    n_ch, n_time = 3, 5
    base = np.where(f > 0, f, 1.0) ** (-1.6)                    # pure aperiodic 1/f
    Sxx = np.empty((n_ch, f.size, n_time))
    for c in range(n_ch):
        for tt in range(n_time):
            Sxx[c, :, tt] = base * (1.0 + 0.05 * rng.standard_normal(f.size))
    Sxx = np.abs(Sxx) + 1e-9
    Sxx[1, (f >= 150) & (f < 250), 2] += 3.0 * base[(f >= 150) & (f < 250)]  # a real ripple bump on one cell
    lm = line_noise_bin_mask(f, [50, 100, 150, 200, 250], 2.0)
    specs = [("gamma_LVFA", 30.0, 80.0), ("hg_low_ripple", 80.0, 150.0), ("ripple_high", 150.0, 250.0)]
    out = _excess_traces(f, Sxx, lm, specs, FIT_LO, FIT_HI, MIN_R2)
    for name, lo, hi in specs:
        assert out[name].shape == (n_ch, n_time)
        for c in range(n_ch):
            for tt in range(n_time):
                ref = aperiodic_corrected_excess_power(
                    f, Sxx[c, :, tt], lo, hi, lm, fit_lo=FIT_LO, fit_hi=FIT_HI,
                    min_r2=MIN_R2, half_open=True)["excess_power"]
                got = float(out[name][c, tt])
                if np.isnan(ref):
                    assert np.isnan(got), f"{name} c{c} t{tt}: helper nan, vectorized {got}"
                else:
                    assert abs(got - ref) <= 1e-6 * abs(ref) + 1e-9, \
                        f"{name} c{c} t{tt}: vectorized {got} != helper {ref}"
    # the bump cell must carry clearly more ripple excess than the other cells (sanity, not tautology).
    # ripple-band 1/f PSD is tiny (~f**-1.6), so compare RELATIVELY, not with an absolute margin.
    other_max = float(np.nanmax(np.delete(out["ripple_high"], 2, axis=1)[1]))
    assert out["ripple_high"][1, 2] > 10.0 * (other_max + 1e-9)


@pytest.mark.integration
def test_aperiodic_cache_and_alignment_epilepsiae_139(tmp_path):
    """Task 11b (Gate C input): build the aperiodic-residual (1/f-corrected band excess) cache for
    epilepsiae_139, then run the alignment with ``--feature aperiodic_resid`` pointing at that cache
    and assert the ``aperiodic_resid`` subject_summary exists with PRIMARY-band rows.

    Gate C asks: does a band carry OSCILLATORY EXCESS above the 1/f background (aligned to G_HFO)?
    The cache has the SAME npz structure/keys as the raw band cache (``{B}__zt__{idx}`` /
    ``{B}__relt__{idx}`` / ``channels``) plus the reused ``analysis_channels`` sidecar, so the
    alignment reads it exactly like the raw cache. Fully isolated: cache + alignment CSVs -> tmp."""
    import json
    import numpy as np
    cache_dir = tmp_path / "aperiodic_resid_cache"
    rb = subprocess.run([sys.executable, "scripts/build_topic5_v2_aperiodic_cache.py",
                         "--subjects", "epilepsiae_139", "--substrate", "broad",
                         "--outdir", str(cache_dir)], cwd=ROOT, capture_output=True, text=True)
    assert rb.returncode == 0, f"{rb.stdout}\n{rb.stderr}"
    npz = cache_dir / "epilepsiae_139.npz"
    sidecar = cache_dir / "epilepsiae_139.json"
    assert npz.exists(), f"missing aperiodic npz\n{rb.stdout}\n{rb.stderr}"
    assert sidecar.exists(), f"missing aperiodic sidecar\n{rb.stdout}"
    z = np.load(npz, allow_pickle=True)
    assert "channels" in z, "aperiodic cache missing channels array"
    # per-(primary band, seizure) zt/relt keys — SAME structure as the raw band cache.
    assert any(k.startswith("gamma_LVFA__zt__") for k in z.files), f"no primary zt; {list(z.files)[:6]}"
    assert any(k.startswith("gamma_LVFA__relt__") for k in z.files), "no primary relt"
    side = json.loads(sidecar.read_text())
    assert side["analysis_channels_basis"] == "primary_bands_validity", side.get("analysis_channels_basis")
    assert side["analysis_channels"], "aperiodic sidecar analysis_channels empty"
    assert side.get("feature") == "aperiodic_resid", side.get("feature")

    ra = subprocess.run([sys.executable, "scripts/run_topic5_v2_alignment.py",
                         "--feature", "aperiodic_resid", "--substrate", "broad",
                         "--subjects", "epilepsiae_139", "--feature-cache-dir", str(cache_dir),
                         "--outdir", str(tmp_path / "align_out")], cwd=ROOT, capture_output=True, text=True)
    assert ra.returncode == 0, f"{ra.stdout}\n{ra.stderr}"
    subj_csv = tmp_path / "align_out" / "broad" / "phase1_alignment_aperiodic_resid_subject_summary.csv"
    assert subj_csv.exists(), f"missing aperiodic_resid subject_summary\n{ra.stdout}\n{ra.stderr}"
    rows = list(csv.DictReader(open(subj_csv)))
    assert rows, "empty aperiodic_resid subject_summary"
    primary = {"delta_HYP_slow", "theta_preictal_PAC", "alpha_sharp_leq13", "beta_LVFA_low",
               "gamma_LVFA", "hg_low_ripple", "ripple_high"}
    bands = {x["band"] for x in rows}
    assert bands & primary, f"no primary band rows in aperiodic_resid summary; bands={bands}"
    for x in rows:
        assert x["feature"] == "aperiodic_resid", f"{x['band']} feature={x['feature']!r}"
        assert x["axis_set"] == "broad", f"{x['band']} axis_set={x['axis_set']!r}"
