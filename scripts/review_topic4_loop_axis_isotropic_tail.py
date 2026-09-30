#!/usr/bin/env python3
"""Completed isotropic120s: unchanged observer, fixed final10s spatial review."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import numpy as np
import review_topic4_loop_axis_halfway as prefix
import review_topic4_loop_native_states as view
import analyze_topic4_interictal_recurrence as audit
import run_topic4_rhythm_preserving_feedback as rhythm


def main():
    result_path = prefix.FOLDER / 'result.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'COMPLETE' and result['end_s'] == 120.
    prefix.END_STEP = 1200000
    prefix.MANIFEST.clear()
    d = prefix.load('chunks', ['spikes_1ms', 'regions_1ms', 'field_5ms',
                              'raster', 'slow_time_ms', 'Z'])
    fb = prefix.load('feedback_chunks', ['time_ms', 'K_mean', 'G_raw', 'G_applied_mean'])
    zb = prefix.load('z_budget_chunks', ['time_ms', 'values'])
    contacts = prefix.load('actual_current_chunks', ['time_ms', 'contact_current'])
    causal = prefix.load('mechanism_chunks', ['time_ms', 'global_E_rate_Hz'])
    response = prefix.load('global_response_chunks', ['time_ms', 'q'])
    with np.load(prefix.RUN_ROOT / 'geometry.npz') as g:
        geo = {k: g[k] for k in g.files}
    raw = np.c_[d['spikes_1ms'][:, 0], d['regions_1ms'][:, :3]]
    counts = np.r_[32000, geo['region_counts'][:3]]
    assert len(raw) == 120000
    assert np.array_equal(raw[:, 0], raw[:, 1:].sum(1))
    assert np.array_equal(d['field_5ms'].sum(1), raw[:, 0].reshape(-1, 5).sum(1))
    assert np.array_equal(d['slow_time_ms'], fb['time_ms'])
    assert np.array_equal(causal['time_ms'], response['time_ms'])
    r5 = raw.reshape(-1, 5, 4).sum(1) / counts / .005
    r10 = raw.reshape(-1, 10, 4).sum(1) / counts / .01
    events = rhythm.strict_events(r10, 120.)
    primary = audit.temporal_audit(r10)
    part = audit.interval_events(events, 110., 120.)
    metrics = [view.event_metrics(dict(rates=r5), e) for e in part['brief_events']]
    tb = zb['time_ms'] / 1000; b = zb['values']
    assert np.max(abs(b[:, :, 5])) < 1e-10
    assert np.max(abs(b[1:, :, 0] - b[:-1, :, 1])) < 1e-10
    keep = (tb > 110. + 1e-9) & (tb <= 120. + 1e-9)
    selected = b[keep]; assert len(selected) == 500
    gain = .02 * selected[:, :, 2].sum(0)
    loss = .02 * selected[:, :, 3].sum(0)
    delta = selected[-1, :, 1] - selected[0, :, 0]
    error = delta - gain + loss
    assert np.max(abs(error)) < 1e-9
    slow = (d['slow_time_ms'] >= 110000) & (d['slow_time_ms'] < 120000)
    fast = (causal['time_ms'] >= 110000) & (causal['time_ms'] < 120000)
    out = prefix.ROOT / 'axis_controls/native_runs/isotropic_completed_tail'
    figures = out / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    example = view.draw_event(prefix.FOLDER, geo, d, r5, contacts, 0.,
        part['brief_events'][0] if metrics else None, 'isotropic_110_120s', figures,
        **({} if metrics else {'fixed_window': (119.7, 120.), 'fixed_position': 'final'}))
    payload = dict(status='COMPLETE120S_FIXED_TAIL', source=str(prefix.FOLDER),
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        window_s=[110., 120.], observed_s=120., operational_entries=primary['entries'],
        operational_exits=primary['low_activity_exits'], total_complete_activity_episodes=len(events),
        all_complete_activity_durations_s=[e['duration_s'] for e in events],
        events=part, core_recruitment=view.summarize(metrics) if metrics else None,
        event_metrics=metrics, example=example,
        jointquiet_fraction=float(np.mean(r10[11000:12000, :3].max(1) < 5)),
        mean_Hz_all_A_B_other=r10[11000:12000].mean(0).tolist(),
        Z_mean_all_A_B=d['Z'][slow][:, [0, 5, 6]].mean(0).tolist(),
        Z_range_all_A_B=np.stack([d['Z'][slow][:, [0, 5, 6]].min(0),
                                 d['Z'][slow][:, [0, 5, 6]].max(0)]).tolist(),
        native_Z_budget=dict(region_order=['all_E', 'core_A', 'core_B', 'other_E'],
            window_s=[110., 120.], gain=gain.tolist(), loss=loss.tolist(), delta=delta.tolist(),
            balance_error=error.tolist()),
        feedback_summary={k: dict(mean=float(fb[k][slow].mean()), minimum=float(fb[k][slow].min()),
                                 maximum=float(fb[k][slow].max())) for k in ['K_mean', 'G_raw', 'G_applied_mean']},
        fraction_feedback_gate_positive=float(np.mean(response['q'][fast] > 0)),
        source_manifest=prefix.MANIFEST, field_spike_integrity='PASS',
        producer_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        loader_sha256=hashlib.sha256(open(prefix.__file__, 'rb').read()).hexdigest(),
        viewer_sha256=view.PRODUCER_SHA, human_review='PENDING',
        interpretation='Fixed final10s of a completed120s native trajectory, first chronological completebrief. '
                       'When no operational entry precedes these events, they are retained interictal-like activity, '
                       'not post-ictal return. Finite-window persistence does not certify a stable attractor; '
                       'one trajectory and no pure-orientation intervention.')
    (out / 'analysis.json').write_text(json.dumps(payload, indent=2) + '\n')
    (figures / 'README.md').write_text('### isotropic_110_120s_first_brief.png\n\n各向同性结构完整120秒轨迹的最后110–120秒窗口，按时间顺序取首个完整短事件。沿用固定80细胞raster、区域率、六帧原生5ms空间计数及15触点电流代理，没有按形态挑选。整个观察窗未达到持续高活动进入标准，因此末段短事件属于保留的间期样活动，不计作发作后返回。\n\n**关注点**：末段是否仍有core募集与传播后停止；此图不证明稳定吸引子、纯方向因果或临床HFO对应，待作者目视。\n')
    print(json.dumps({k: payload[k] for k in ['status', 'total_complete_activity_episodes',
          'core_recruitment', 'jointquiet_fraction', 'Z_mean_all_A_B', 'native_Z_budget']}))


if __name__ == '__main__':
    main()
