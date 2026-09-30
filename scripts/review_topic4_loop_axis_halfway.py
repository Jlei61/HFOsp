#!/usr/bin/env python3
"""Fixed60s descriptive prefix review; never a completed autonomous outcome."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import review_topic4_loop_native_states as view
import analyze_topic4_interictal_recurrence as audit
import run_topic4_rhythm_preserving_feedback as rhythm

ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
RUN_ROOT = ROOT / 'axis_controls/native_runs/isotropic'
FOLDER = RUN_ROOT / 'runs/isotropic_s9108405'
OUT = ROOT / 'axis_controls/native_runs/prefix_checkpoints/isotropic60s'
END_STEP = 600000
MANIFEST = []


def load(subdir, keys):
    parts = {key: [] for key in keys}
    last = 0
    for path in sorted((FOLDER / subdir).glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as chunk:
            # Sidecar streams encode committed step bounds in filenames;
            # only the main observer also stores them as NPZ scalars.
            start, stop = map(int, path.stem.split('_'))
            assert start == last
            if 'start_step' in chunk:
                assert int(chunk['start_step']) == start
                assert int(chunk['end_step']) == stop
            last = stop
            assert last <= END_STEP, 'Fixed prefix must end at a committed block.'
            for key in keys:
                parts[key].append(chunk[key])
        MANIFEST.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if last == END_STEP:
            break
    assert last == END_STEP, (subdir, last)
    return {key: np.concatenate(values) for key, values in parts.items()}


def main():
    d = load('chunks', ['spikes_1ms', 'regions_1ms', 'field_5ms', 'raster', 'slow_time_ms', 'Z'])
    fb = load('feedback_chunks', ['time_ms', 'K_mean', 'G_raw', 'G_applied_mean'])
    zb = load('z_budget_chunks', ['time_ms', 'values'])
    causal = load('mechanism_chunks', ['time_ms', 'global_E_rate_Hz'])
    response = load('global_response_chunks', ['time_ms', 'q'])
    contacts = load('actual_current_chunks', ['time_ms', 'contact_current'])
    with np.load(RUN_ROOT / 'geometry.npz') as g:
        geo = {k: g[k] for k in g.files}
    raw = np.c_[d['spikes_1ms'][:, 0], d['regions_1ms'][:, :3]]
    counts = np.r_[32000, geo['region_counts'][:3]]
    r5 = raw.reshape(-1, 5, 4).sum(1) / counts / .005
    r10 = raw.reshape(-1, 10, 4).sum(1) / counts / .01
    assert len(raw) == 60000
    assert np.array_equal(d['field_5ms'].sum(1), raw[:, 0].reshape(-1, 5).sum(1))
    assert np.array_equal(d['regions_1ms'][:, :3].sum(1), raw[:, 0])
    assert np.array_equal(d['slow_time_ms'], fb['time_ms'])
    assert np.array_equal(causal['time_ms'], response['time_ms'])
    b = zb['values']; tb = zb['time_ms'] / 1000
    assert np.max(np.abs(b[:, :, 5])) < 1e-10
    assert np.max(np.abs(b[1:, :, 0] - b[:-1, :, 1])) < 1e-10
    events = rhythm.strict_events(r10, 60.)
    primary = audit.temporal_audit(r10)
    rows = []
    for label, lo, hi in [('initial', .5, 8.), ('late', 50., 60.)]:
        part = audit.interval_events(events, lo, hi)
        metrics = [view.event_metrics(dict(rates=r5), e) for e in part['brief_events']]
        keep = (tb - .02 >= lo - 1e-9) & (tb <= hi + 1e-9)
        selected = b[keep]
        gain = .02 * selected[:, :, 2].sum(0)
        loss = .02 * selected[:, :, 3].sum(0)
        delta = selected[-1, :, 1] - selected[0, :, 0]
        error = delta - gain + loss
        assert np.max(np.abs(error)) < 1e-9
        tc = causal['time_ms'] / 1000
        native = (tc >= lo) & (tc < hi)
        t = d['slow_time_ms'] / 1000
        slow = (t >= lo) & (t < hi)
        rate = r10[round(lo * 100):round(hi * 100)]
        R = causal['global_E_rate_Hz'][native]
        q = response['q'][native]
        row = dict(label=label, window_s=[lo, hi], events=part, event_metrics=metrics,
                   core_recruitment=view.summarize(metrics) if metrics else None,
                   jointquiet_fraction=float(np.mean(rate[:, :3].max(1) < 5)),
                   causal_R_mean_peak_Hz=[float(R.mean()), float(R.max())],
                   fraction_causal_R_at_or_below5=float(np.mean(R <= 5)),
                   fraction_feedback_gate_positive=float(np.mean(q > 0)),
                   Z_mean_all_A_B=d['Z'][slow][:, [0, 5, 6]].mean(0).tolist(),
                   Z_range_all_A_B=np.stack([d['Z'][slow][:, [0, 5, 6]].min(0),
                                            d['Z'][slow][:, [0, 5, 6]].max(0)]).tolist(),
                   feedback_summary={k: dict(mean=float(fb[k][slow].mean()),
                                               minimum=float(fb[k][slow].min()),
                                               maximum=float(fb[k][slow].max()))
                                     for k in ['K_mean', 'G_raw', 'G_applied_mean']},
                   native_Z_budget=dict(region_order=['all_E', 'core_A', 'core_B', 'other_E'],
                       exact_window_s=[float(tb[keep][0] - .02), float(tb[keep][-1])],
                       gain=gain.tolist(), loss=loss.tolist(), delta=delta.tolist(),
                       balance_error=error.tolist()))
        rows.append(row)
    figures = OUT / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    late = rows[1]
    assert late['events']['brief_events']
    late['example'] = view.draw_event(FOLDER, geo, d, r5, contacts, 0.,
        late['events']['brief_events'][0], 'isotropic_50_60s', figures)
    payload = dict(status='FIXED60S_PARTIAL_PREFIX', source=str(FOLDER),
        observation_window_s=[0, 60], planned_horizon_s=120,
        operational_entries=primary['entries'], operational_exits=primary['low_activity_exits'],
        complete_activity_episodes=events, windows=rows, source_manifest=MANIFEST,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        viewer_sha256=view.PRODUCER_SHA, field_spike_integrity='PASS',
        interpretation='Descriptive comparison of two fixed windows in one evolving trajectory. '
                       'The late events are retained short activity, not a post-ictal return: '
                       'no operational entry precedes them in this prefix. '
                       'Feedback alignment is not an ablation or proof of feedback necessity. '
                       'Regional recruitment order alone does not identify a unique source.',
        human_review='PENDING')
    (OUT / 'analysis.json').write_text(json.dumps(payload, indent=2) + '\n')
    (figures / 'README.md').write_text(
        '### isotropic_50_60s_first_brief.png\n\n'
        '固定50–60秒窗口中按时间顺序第一个完整短事件。沿用原生审阅布局，同时展示固定80细胞raster、区域放电率、六帧5ms空间计数和同一15触点电流代理；未按形态挑选。'
        '这是120秒仿真的前60秒中途读出，前缀内尚无操作性进入，不能称为发作后返回。\n\n'
        '**关注点**：后段短事件是否保留core募集及空间传播；区域先后顺序不能单独证明唯一传播源，触点代理不是HFO能量验证。图待人工审阅。\n')
    print(json.dumps(dict(status=payload['status'], windows=[
        {k: row[k] for k in ['label', 'core_recruitment', 'jointquiet_fraction',
                             'causal_R_mean_peak_Hz', 'fraction_feedback_gate_positive',
                             'Z_mean_all_A_B', 'feedback_summary', 'native_Z_budget']}
        for row in rows]), indent=2))


if __name__ == '__main__':
    main()
