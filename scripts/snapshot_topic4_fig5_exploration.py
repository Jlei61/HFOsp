#!/usr/bin/env python3
"""Read-only inventory for the bounded Fig5 exploration and its inherited jobs.

No simulation dispatch, stopping, or scientific reclassification is performed.
"""
from datetime import datetime
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/topic4_sef_hfo'
WINDOW = BASE / 'fig5_m_overnight_exploration_20260913'
OUT = WINDOW / 'comprehensive_inventory'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def select(value, keys):
    return {key: value[key] for key in keys if key in value}


def process(pid):
    if not isinstance(pid, int):
        return None
    try:
        proc = psutil.Process(pid)
        return dict(pid=pid, state=proc.status(), command=proc.cmdline(),
                    start_time=proc.create_time(),
                    cpu_seconds=sum(proc.cpu_times()[:2]),
                    memory_GiB=proc.memory_info().rss / 2**30)
    except psutil.NoSuchProcess:
        return dict(pid=pid, state='NOT_LIVE')


def batch(folder):
    status = read(folder / 'status.json')
    runs = []
    for run in sorted((folder / 'runs').glob('*')):
        if not run.is_dir() or run.name.startswith('qa'):
            continue
        progress = read(run / 'progress.json')
        result = read(run / 'result.json')
        item = dict(name=run.name, path=str(run),
                    result_file_exists=bool(result),
                    result=select(result, ['status', 'end_s', 'job']),
                    progress=select(progress, ['status', 'pid', 'time_s', 'end_s',
                        'entries', 'recoveries', 'recurrence_onset_s', 'recovered',
                        'phase', 'restore_s', 'release_s', 'Z', 'adaptation_current']),
                    tracker=select(result.get('tracker', {}), ['phase', 'entries',
                        'recoveries', 'restore_s', 'release_s', 'stop_reason']),
                    process=process(progress.get('pid')) if not result else None)
        if (run / 'computational_variant.json').exists():
            item['computational_variant'] = read(run / 'computational_variant.json')
        checkpoint = run / 'checkpoint.pkl'
        if checkpoint.exists():
            item['checkpoint'] = dict(path=str(checkpoint),
                                      modified=checkpoint.stat().st_mtime)
        # Immutable filenames provide committed times without unpickling a
        # potentially large state; live progress may be further ahead.
        ends = []
        for chunk in (run / 'chunks').glob('*.npz'):
            parts = chunk.stem.split('_')
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                ends.append(int(parts[1]) * .0001)
        item['last_committed_chunk_s'] = max(ends) if ends else None
        runs.append(item)
    return dict(path=str(folder), controller=select(status, ['status', 'pid',
        'supervisor_pid', 'phase', 'total', 'completed', 'pending', 'failed']),
        result_files=sum(r['result_file_exists'] for r in runs),
        live_workers=sum(bool(r['process']) and
                         r['process']['state'] not in ['NOT_LIVE', 'zombie']
                         for r in runs), runs=runs)


def figures():
    paths = [WINDOW / 'completed_Z_only_1000s_fig5/fig5_metadata.json',
             WINDOW / 'existing_mode_figures/weak_fast/fig5_metadata.json',
             WINDOW / 'existing_mode_figures/weak_20s/fig5_metadata.json']
    for prefix in read(WINDOW / 'ongoing_fig5_prefixes/index.json').values():
        paths.append(Path(prefix['figure']).parent.parent / 'fig5_metadata.json')
    paths.extend((BASE / 'm_parameter_modes_fig5_20260913').glob(
        'candidates/*/fig5_metadata.json'))
    paths.extend((WINDOW / 'early_z_refill_branches').glob(
        'candidates/*/fig5_metadata.json'))
    paths.append(WINDOW / 'early_Z_lookup_dense/candidate/fig5_metadata.json')
    paths.append(WINDOW / 'early_energy_high_resolution_replay/candidate/fig5_metadata.json')
    paths.append(WINDOW / 'early_energy_high_resolution_replay_seed9108402/candidate/fig5_metadata.json')
    late=read(WINDOW / 'late_Z_refill_fig5/latest.json')
    if late:
        paths.append(Path(late['figure']).parent.parent / 'fig5_metadata.json')
    rows = []
    for path in sorted(set(paths)):
        if not path.exists():
            continue
        d = read(path)
        metrics = d.get('metrics', {})
        job = d.get('job', {})
        rows.append(dict(metadata=str(path), figure=str(path.parent / 'figures/fig5.png'),
            trajectory_key=json.dumps(select(job, ['name', 'seed', 'eta_m', 'tau_M_s',
                'early_refill_branch', 'source_run']), sort_keys=True),
            source_duration_s=d.get('source_duration_s'),
            mode=metrics.get('mode'), full_1_to_5_observed=metrics.get('full_1_to_5_observed'),
            entries=metrics.get('entries'), recoveries=metrics.get('recoveries'),
            finite_events_before_first_high=metrics.get('finite_events_before_first_high'),
            finite_events_after_return=metrics.get('finite_events_after_return'),
            sustained_oscillation_status=metrics.get('sustained_oscillation_status'),
            early_energy=select(d.get('E2', {}), ['status', 'target_s',
                'model_patient_rho', 'n_contacts', 'n_model_contacts_above_baseline',
                'native_cells_with_increased_power', 'native_valid_baseline_cells']),
            full_figure_acceptance=d.get('full_figure_acceptance'),
            full_scientific_acceptance=d.get('full_scientific_acceptance'),
            agent_visual_review=d.get('agent_visual_review'), human_review='PENDING'))
    return rows


def main():
    window = read(WINDOW / 'window.json')
    inherited = [BASE / name for name in window['inherited_batches']]
    extras = [WINDOW / name for name in ['fast_state_pilot', 'high_state_M_gain_probe',
              'early_z_refill_branches', 'early_Z_lookup_dense', 'early_Z_lookup_dense_figures',
              'early_energy_high_resolution_replay', 'early_energy_high_resolution_replay_seed9108402']]
    now = datetime.now().astimezone()
    deadline = datetime.fromisoformat(window['deadline'])
    inventory = dict(time=now.isoformat(), deadline=window['deadline'],
        exploration_window_ended=now >= deadline,
        no_simulations_dispatched_or_stopped_by_this_report=True,
        batches={path.name: batch(path) for path in inherited + extras},
        numerical_replica_not_independent_sample=['early_Z_lookup_dense',
            'early_Z_lookup_dense_figures', 'early_energy_high_resolution_replay',
            'early_energy_high_resolution_replay_seed9108402'],
        figures=figures(),
        controls_not_new_M40_samples=True,
        source_progress_parity=read(WINDOW / 'early_Z_lookup_dense/source_progress_parity.json'),
        source_observation_parity=select(read(WINDOW / 'early_Z_lookup_dense/full_observation_parity_through20s.json'),
            ['status', 'independently_executed_interval_s', 'full_membrane_delay_RNG_state_at20s_compared']),
        recurrence_count_verification=read(WINDOW / 'early_Z_lookup_dense/recurrence_prefix_qa.json'),
        high_resolution_measurement_replay_QA=select(read(WINDOW / 'early_energy_high_resolution_replay/qa.json'),
            ['status', 'true_dt_ms', 'recording_end_s', 'raw_to_1ms_field_exact',
             'measurement_replay_not_new_trial']),
        high_resolution_measurement_analysis=read(WINDOW / 'early_energy_high_resolution_replay/analysis_status.json'),
        seed2_high_resolution_measurement_QA=select(read(WINDOW / 'early_energy_high_resolution_replay_seed9108402/qa.json'),
            ['status', 'true_dt_ms', 'recording_end_s', 'raw_to_1ms_field_exact', 'measurement_replay_not_new_trial']),
        seed2_high_resolution_measurement_analysis=read(WINDOW / 'early_energy_high_resolution_replay_seed9108402/analysis_status.json'),
        seed2_recurrence_count_verification=read(WINDOW / 'early_z_refill_branches/seed2_recurring_prefix_qa.json'),
        late_Z_refill_figure=read(WINDOW / 'late_Z_refill_fig5/latest.json'),
        late_Z_refill_analysis_status=read(WINDOW / 'late_Z_refill_fig5/analysis_status.json'),
        late_Z_refill_postprocessor=process(window.get('late_Z_refill_postprocessing',{}).get('pid')),
        matched_reset90=select(read(WINDOW / 'reset_matched_90s/analysis.json'),
            ['status', 'full_fast_pilot_complete', 'pilot_subarm_dispatch_allowed',
             'external_input_summary_bitwise_equal_all_arms', 'agent_visual_review']),
        prefix_watcher=read(WINDOW / 'ongoing_fig5_prefixes/status.json'),
        high_gain_matched_prefix=select(read(WINDOW / 'high_state_M_gain_probe/analysis_matched_prefix.json'),
            ['status', 'post_intervention_observation_s', 'full_5s_interventions_complete', 'records']),
        high_gain_complete=select(read(WINDOW / 'high_state_M_gain_probe/analysis.json'),
            ['status', 'post_intervention_observation_s', 'full_5s_interventions_complete',
             'records', 'constant_parameter_native_cycle_demonstrated', 'agent_visual_review']),
        available_memory_GiB=psutil.virtual_memory().available / 2**30,
        human_scientific_acceptance='PENDING')
    inventory['unique_live_worker_pids'] = sorted({r['process']['pid']
        for d in inventory['batches'].values() for r in d['runs']
        if r['process'] and r['process']['state'] not in ['NOT_LIVE', 'zombie']})
    inventory['unique_trajectories_with_five_states'] = len({d['trajectory_key']
        for d in inventory['figures'] if d['full_1_to_5_observed'] is True})
    inventory['noise_seeds_with_five_states'] = sorted({json.loads(d['trajectory_key'])['seed']
        for d in inventory['figures'] if d['full_1_to_5_observed'] is True
        and 'seed' in json.loads(d['trajectory_key'])})
    OUT.mkdir(exist_ok=True)
    tmp = OUT / 'latest.tmp.json'
    tmp.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(OUT / 'latest.json')
    lines = [f'# Fig5 探索清单 · {now:%Y-%m-%d %H:%M}', '',
        '这是当前文件与进程的只读快照。完整图的排版、完整仿真随访和科学验收分别记录；运行中的未进入/未返回不能记为最终阴性。', '',
        '| 批次 | 已有结果文件 | 活跃 worker | 计划总数 |', '|---|---:|---:|---:|']
    for name, d in inventory['batches'].items():
        lines.append(f'| {name} | {d["result_files"]} | {d["live_workers"]} | {d["controller"].get("total", "—")} |')
    lines += ['', 'dense 与 dense_figures 是同一个数值副本的计算/观察目录；高分辨率重放也只是同条件的观测核查，均不新增生物学样本。同一轨迹的前缀图和完成图不重复计算五状态轨迹数。', '',
        '| 已有完整布局 | 实际时长(s) | 实际模式 | ①–⑤全部观察 |', '|---|---:|---|---|']
    for d in inventory['figures']:
        lines.append(f'| [{Path(d["metadata"]).parent.name}]({d["figure"]}) | {d["source_duration_s"]} | {d["mode"]} | {d["full_1_to_5_observed"]} |')
    lines += ['', f'新增探索截止：{window["deadline"]}。已启动有限任务及继承批次按既定终点保留续跑，不因图尚不完整而截短观察窗。',
        '统计单位是独立噪声轨迹；同一轨迹的事件、神经元、二维网格及数值副本均不能作为额外独立样本。所有新图仍待用户目视与科学验收。']
    (OUT / 'latest.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(time=inventory['time'],
        batches={name: select(d, ['result_files', 'live_workers'])
                 for name, d in inventory['batches'].items()},
        full_layouts=len(inventory['figures']),
        unique_trajectories_with_five_states=inventory['unique_trajectories_with_five_states'],
        report=str(OUT / 'latest.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()
