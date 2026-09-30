#!/usr/bin/env python3
"""Bounded G strength x response-time test on the preserved Fig5 SNN."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import copy
import fcntl
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import run_topic4_quiet_tail_recovery as quiet

SOURCE = Path('/data/hfosp/topic4_sef_hfo/fig5_quiet_tail_confirmation_20260918')
OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923')
window, parent, base, budget = quiet.window, quiet.parent, quiet.base, quiet.budget
fixed, load, write, rate_data = window.fixed, quiet.load, quiet.write, quiet.rate_data
SEED = quiet.SEED
SCREEN_SEEDS = [9108402, 9108403]
SETTINGS = [(30., 0.), (15., .5), (30., .5)]
OriginalAnalyze = quiet.analyze


class GlobalResponseSlow(quiet.QuietTailSlow):
    """tau_G ds/dt=q-s; only added G uses s, K still uses instantaneous q.

    s starts at zero. Currents use pre-step s and q; s then advances using that
    causal q. tau_G=0 is the exact original fast law, without an extra delay.
    """
    global_tau_ms = 0.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_state = 0.
        self.global_response_records = []

    def apply_currents(self, ie, ii, labels=None, rec=None):
        # Same native/local and recording operations as RhythmSlow. The sole
        # physical change in this method is raw = G * s instead of G * q.
        self.gate = float(np.clip((self.r_global-parent.RATE_START)/(parent.RATE_FULL-parent.RATE_START), 0., 1.))
        drive = self.gate if self.global_tau_ms == 0 else self.global_state
        raw = self.global_gain * drive
        self.g_global[:] = raw * self.z[:self.NE]
        total = ii
        if raw:
            total = ii.copy()
            total[:self.NE] += raw * (18.-self.global_reversal)
        self.delivered = total
        self._I_I_last = total
        self.raw_mean = float(ii[:self.NE].mean())
        value = ie-self.z*ii-self.cfg.eta_m*self.m
        if np.any(self.g_k):
            value[:self.NE] += self.g_k*(parent.EK-self.global_reversal)
        if self.gate > 0 and (self.global_gain > 0 or self.sahp_gain > 0) and self.first_feedback_s is None:
            self.first_feedback_s = self._step_index*.0001
        v = self.voltage[:self.NE] if self.voltage is not None else np.zeros(self.NE)
        global_current = self.g_global*(v-self.global_reversal)
        if self._step_index % 10 == 0:
            self.extra_records.append([self._step_index*.1, self.r_global, raw,
                self.g_global.mean(), 0., 0., global_current.mean()])
            self.k_records.append([self._step_index*.1, self.g_k.mean(), self.g_k.max(),
                                   (self.g_k*(v-parent.EK)).mean()])
            self.global_response_records.append([self._step_index*.1, self.gate, drive, raw])
        if self._step_index % 20 == 0 and self.current_recorder is not None:
            absolute = np.abs(ie[:self.NE])+np.abs(self.z[:self.NE]*ii[:self.NE])+np.abs(global_current)
            self.contact_records.append(np.array([np.dot(w, absolute[ix]) for ix, w in zip(self.current_recorder._idx, self.current_recorder._w)]))
            self.field_records.append(np.bincount(self.current_cells, weights=absolute, minlength=400)/self.current_cell_counts)
            self.current_record_times.append(self._step_index*.1)
        if self._step_index % 200 == 0:
            row = [self._step_index*.1]
            for ix in self.region_groups():
                row.extend([self.g_k[ix].mean(), self.g_global[ix].mean(), v[ix].mean(),
                    (self.g_k[ix]*(v[ix]-parent.EK)).mean(), global_current[ix].mean(),
                    ie[ix].mean(), (self.z[:self.NE][ix]*ii[:self.NE][ix]).mean(),
                    self.cfg.eta_m*self.m[:self.NE][ix].mean(), self.z[:self.NE][ix].mean(), self.m[:self.NE][ix].mean()])
            self.regional_records.append(row)
            self.feedback_records.append([self._step_index*.1, self.r_global, self.gate, raw,
                                          self.g_global.mean(), self.g_k.mean()])
        return value

    def step(self, spk, labels, dt):
        q = self.gate
        super().step(spk, labels, dt)
        if self.global_tau_ms:
            decay = np.exp(-dt/self.global_tau_ms)
            self.global_state = self.global_state*decay+q*(1.-decay)
        else:
            self.global_state = q


def extra_state(obj):
    return dict(global_state=float(obj.global_state), global_tau_ms=float(obj.global_tau_ms),
                first_feedback_s=obj.first_feedback_s)


def restore_extra(state, obj):
    extra = state['global_feedback_response']
    assert extra['global_tau_ms'] == obj.global_tau_ms
    obj.global_state = extra['global_state']
    obj.first_feedback_s = extra['first_feedback_s']


def configure():
    # Reuse the accepted observation chain; route all paths to this new batch.
    for module in [quiet, window, parent, budget]:
        module.OUT = OUT
    quiet.prepare = window.prepare = parent.prepare = prepare
    quiet.reference = window.reference = parent.reference = reference
    quiet.configure = window.configure = configure
    parent.previous.OUT = OUT


def reference(seed):
    path = OUT/'references'/f'native_s{seed}.npz'
    return path if path.exists() else None


def make_job(gain, tau, seed, stage='screen', horizon=120., index=0):
    job = parent.make_job(gain, 40., .5, seed, index, stage=stage, horizon=horizon)
    job.update(name=f'G{gain:g}_response{tau:g}_s{seed}', global_tau_s=tau,
        off_tau_s=5., on_tau_s=.5, retention_below_Hz=5., full_state_snapshots=True,
        mechanism='G_raw=G500*s; tau_G*ds/dt=q-s (tau_G=0: s=q). K increment remains0.16*q; K tau0.5s except5s at causal R_G<=5Hz.')
    return job


def reuse_control(job):
    source = SOURCE/'runs'/f'quiet5_off5_s{job["seed"]}'
    dest = OUT/'runs'/job['name']
    dest.mkdir(parents=True, exist_ok=True)
    result = base.read(source/'result.json')
    for key in ['seed', 'eta_m', 'tau_M_s', 'tau_Z_s', 'threshold', 'gamma', 'global_gain',
                'global_resource', 'sahp_gain', 'sahp_tau_s', 'off_tau_s', 'retention_below_Hz']:
        assert job[key] == result['job'][key], key
    assert result['status'] == 'COMPLETE' and result['end_s'] == 120.
    for folder in source.iterdir():
        if folder.is_dir() and folder.name.endswith('chunks'):
            shutil.copytree(folder, dest/folder.name, copy_function=os.link)
    for name in ['rhythm_preservation.json', 'feedback_activation.json']:
        shutil.copy2(source/name, dest/name)
    applied = base.read(source/'applied_configuration.json')
    applied['job'] = job
    write(dest/'applied_configuration.json', applied)
    result.update(job=job, reused_from=str(source))
    write(dest/'result.json', result)
    write(dest/'reuse_provenance.json', dict(source=str(source), new_independent_realization=False,
          source_result_sha256=base.sha(source/'result.json'), identity=result['identity'], tau_G0_equivalent_law=True))


def prepare():
    if (OUT/'protocol.json').exists():
        return base.read(OUT/'protocol.json')
    old = base.read(SOURCE/'protocol.json')
    for path, expected in old['source_hashes'].items():
        assert base.sha(path) == expected, path
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE/'geometry.npz', OUT/'geometry.npz')
    shutil.copytree(SOURCE/'references', OUT/'references', dirs_exist_ok=True)
    jobs = [make_job(g, tau, seed, index=i*2+k) for i, (g, tau) in enumerate(SETTINGS)
            for k, seed in enumerate(SCREEN_SEEDS)]
    screen_names = [j['name'] for j in jobs]
    jobs += [make_job(15., 0., seed, stage='reused_control', index=k) for k, seed in enumerate(SCREEN_SEEDS)]
    for job in jobs:
        write(OUT/'jobs'/f'{job["name"]}.json', job)
        if job['stage'] == 'reused_control':
            reuse_control(job)
    p = copy.deepcopy(old)
    for key in ['launched_epoch', 'confirmation_selected', 'selection', 'active_names']:
        p.pop(key, None)
    p.update(initial_jobs=jobs, initial_screen_names=screen_names, branch_jobs=[],
        created_epoch=time.time(), deadline_epoch=time.time()+24*3600, wall_budget_hours=24.,
        max_workers=6, max_workers_per_device=3, devices=[0, 1], memory_reserve_GiB=70.,
        confirmation_decided=False, source_round=str(SOURCE), runner_sha256=base.sha(__file__),
        authorization='User2026-09-23: set up and start the next round, approving the preceding bounded G strength x response-time proposal.',
        question='Can stronger and/or temporally accumulated global feedback cause reliable autonomous exit without losing native interictal events or Z recovery?',
        experiment='6new120s screens (3settings x2known failing noises);2reused controls. At most1setting passing both screens gets3cold-start240s confirmations (old seed9108401 and new9108404/9108405), plus2native8s reference jobs.',
        confirmation_policy='Both development seeds must complete120s and pass full_sequence_screen. Rank by minimum returned-event span across seeds, then minimum brief-event count, then smaller gain and shorter tau. At most one setting; no later expansion.',
        new_fixed_equation=dict(q='clip((causal15ms_global_E_rate-200)/300,0,1)',
            G_state='tau_G*ds/dt=q-s; s(0)=0; tau_G0 uses q directly. Exact exponential update with pre-step q at dt0.1ms.',
            G='G500*s*Z_i', J='Original local II_i+(18-EG)*G500*s for E; I unchanged',
            K='K_i*=exp(-dt/tauK); each E spike adds0.16*q; tauK5s iff pre-step R_G<=5Hz, otherwise0.5s',
            unchanged='Native Z/M equations, all physical substrate/input parameters, K gain, local inhibition; no protected pathway, forced quiet window, external intervention or Z target/reset.'),
        scientific_scope='One fixed manual two-core topology; new temporal G state is a hypothesis, not a reproduced Liou mechanism. New noises only confirm within this topology.',
        dispatch_policy='Keep both GPUs working with up to3workers/GPU;30s monitoring; stop new dispatch on failure or1h before24h deadline. Workers checkpoint and finish at deadline. No auto expansion beyond this protocol.',
        figure_contract='Accepted five-row two-column interictal preservation / exit zoom; every exit separately. Native spatial GIFs of the first reference event and first return event, or failed-high window. Human review pending.',
        full_Fig5_acceptance='NOT_ESTABLISHED', human_review='PENDING')
    p['source_hashes'][str(Path(__file__).resolve())] = base.sha(__file__)
    for path in [Path(__file__).with_name('report_topic4_global_feedback_response.py'),
                 Path(__file__).with_name('qa_topic4_global_feedback_response.py'),
                 Path(__file__).with_name('plot_topic4_recovery_rhythm_exit.py')]:
        p['source_hashes'][str(path.resolve())] = base.sha(path)
    write(OUT/'protocol.json', p)
    impl = OUT/'implementation'
    impl.mkdir(exist_ok=True)
    for path in [__file__, quiet.__file__, window.__file__, parent.__file__,
                 Path(__file__).with_name('report_topic4_global_feedback_response.py'),
                 Path(__file__).with_name('qa_topic4_global_feedback_response.py'),
                 Path(__file__).with_name('plot_topic4_recovery_rhythm_exit.py')]:
        shutil.copy2(path, impl/Path(path).name)
    (OUT/'design.md').write_text('''# 全局反馈强度与响应时间：自主退出复核

用户2026-09-23授权开始前一轮提出的有界方案。固定当前手放双核原生SNN及Z/M；两核原间期、进入、自主退出、Z充分恢复、同类群体事件返回按顺序验收。新增的连续全局响应状态不是原Liou方程，也不是患者机制结论。

全局高率响应q仍由原15ms因果E率生成，q=clip((R_G-200)/300,0,1)。增加s(0)=0、tau_G ds/dt=q-s；tau_G=0严格走原q。G_raw=G500*s，G_applied=Z_i*G_raw，资源负荷同步按J=原II+(18-EG)*G_raw更新。仅G使用s；K仍每E spike加0.16*q，R_G<=5Hz时按5秒消退，其余0.5秒。所有规律从t=0固定，无按已检测onset/exit触发的干预，Z参考值不进入物理更新。

初筛G500={15,30}、tau_G={0,0.5s}，用失败种子9108402/9108403各120秒；G15/tau0两个完整旧结果经身份核验复用，新增6条。新方程必须通过tau0逐位等价、固定q稳态/阶跃响应、K驱动分离、Z资源预算和新状态断点保留检查；每条冷启动前8秒与同噪声原间期逐位比较，失败停止追加派发。

两种子均完成120秒且通过原间期→进入→退出→两核达到各自8秒参考Z→至少10个短事件跨5秒、短事件比例>=0.8、时长/间隔/全E峰中位数为参考0.5–2倍，才进入候选排序。先最大化两个种子中较小的返回事件跨度，再比较较小事件数，再优先较小G和较短tau。最多选1个条件，在原成功种子9108401及两个未见噪声9108404/9108405冷启动各240秒，观察第二次退出及返回。新噪声另各跑8秒原生参考；总计最多9条科学轨迹、2条参考，新增1456仿真秒。新噪声进入前若已触发新增反馈而破坏前8秒参考，也明确判失败。

同时报告两核恢复/消耗/净Z、实际Z门控G、K、共同低活动、原生空间和两核招募；只静默、阻断进入、宽爆发、低Z振荡均不算通过。没有两种子完整候选就停止，不继续提高G或自动扩大矩阵；确认失败不再替换条件。强度增大若只加重Z消耗，按负证据报告。

最多6worker、每GPU3个，内存保留70GiB、磁盘保留50GiB，24小时墙钟；每30秒监测、定期自动出图，最后1小时停止新增派发，运行任务在完整检查点截尾。原始连续状态和每10秒完整快照保留。图沿用“间期保留及退出放大”五行双列，右侧显示实际退出/恢复/返回，无退出明确注明；配套原生二维场GIF。科学与人工Fig5验收仍分开，正式图不替换。
''')
    return p


def analyze(name):
    configure()
    job = base.read(OUT/'jobs'/f'{name}.json')
    if job['stage'] == 'native_reference':
        return None
    row = OriginalAnalyze(name)
    if row is None:
        return None
    episodes = [e for e in row['absolute_Z_recovery']['episodes'] if e['sustained_return_after_absolute_recovery']]
    row['confirmation'] = dict(stage=job['stage'], absolute_return_episodes=len(episodes),
        repeated_return_screen=len(episodes) >= 2, fixed_topology=True, human_spatial_acceptance='PENDING')
    if episodes:
        rr = rate_data(OUT/'runs'/name)
        field = load(OUT/'runs'/name, keys=['field_5ms'])['field_5ms']
        ref = row['recovery_window']['reference_features']
        desc = []
        for ep in episodes:
            features = window.event_features(ep['post_absolute_recovery_events']['brief_events'], rr, field)
            desc.append(dict(exit_start_s=ep['exit_start_s'], features=features,
                core_peak_ratios=np.array(features['core_peak_Hz'])/ref['core_peak_Hz'],
                spatial_coverage_ratio=features['active_cell_fraction']/ref['active_cell_fraction']))
        row['confirmation']['native_return_descriptors'] = desc
    write(OUT/'analysis'/f'{name}.json', row)
    live = base.read(OUT/'runs'/name/'live_status.json')
    live.update(absolute_return_episodes=len(episodes), repeated_return_screen=len(episodes) >= 2)
    write(OUT/'runs'/name/'live_status.json', live)
    return row


def export_reference(job):
    folder = OUT/'runs'/job['name']
    result = base.read(folder/'result.json')
    assert result['status'] == 'COMPLETE' and result['end_s'] == 8.
    keys = ['spikes_1ms', 'regions_1ms', 'raster', 'slow_time_ms', 'Z', 'M', 'inputs', 'field_5ms']
    data = load(folder, keys=keys)
    assert len(data['spikes_1ms']) == 8000
    fb = load(folder, 'feedback_chunks', keys=['G_raw', 'K_mean'])
    assert np.all(fb['G_raw'] == 0.) and np.all(fb['K_mean'] == 0.)
    dest = OUT/'references'/f'native_s{job["seed"]}.npz'
    tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **data)
    tmp.replace(dest)
    write(dest.with_suffix('.json'), dict(source=str(folder), seed=job['seed'], observed_s=8.,
        native_added_G_K_zero=True, identity=result['identity'], source_result_sha256=base.sha(folder/'result.json')))


def audit_counts(name):
    folder = OUT/'runs'/name
    end = 0
    for path in sorted((folder/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as data:
            assert data['start_step'] == end, (name, path)
            end = int(data['end_step'])
            assert data['spikes_1ms'][:, 0].sum() == data['regions_1ms'][:, :3].sum() == data['field_5ms'].sum()
            assert data['spikes_1ms'][:, 1].sum() == data['regions_1ms'][:, 3:].sum()
    assert end*.0001 == base.read(folder/'result.json')['end_s']
    write(folder/'integrity.json', dict(status='PASS', observed_s=end*.0001, continuous=True, E_I_counts_conserved=True))


def worker(name):
    p = prepare()
    assert p['runner_sha256'] == base.sha(__file__)
    assert base.read(OUT/'mechanism_qa.json')['status'] == 'PASS'
    configure()
    job = base.read(OUT/'jobs'/f'{name}.json')
    folder = OUT/'runs'/name
    cls = GlobalResponseSlow
    cls.C_R = 0.; cls.feedback_form = 'conductance'
    cls.sahp_gain = job['sahp_gain']; cls.sahp_tau_ms = 500.
    cls.off_tau_ms = 5000.; cls.retention_below_Hz = 5.
    cls.global_tau_ms = job['global_tau_s']*1000.
    cls.record_regions = True; cls.z_gate_off = False; cls.k_freeze = False
    import checkpoint
    capture0, restore0 = checkpoint.capture, checkpoint.restore_slow
    old = fixed.OUT, fixed.prepare, fixed.TerminationSlow, fixed.observation_sink
    sink0 = fixed.observation_sink

    def capture(**kwargs):
        state = capture0(**kwargs)
        obj = kwargs['slow']
        state['global_feedback_response'] = extra_state(obj)
        return state

    def restore(state, obj):
        restore0(state, obj)
        restore_extra(state, obj)

    def factory(sink, j, deadline):
        full = sink0(sink, j, deadline)
        def observe(step, state):
            try:
                return full(step, state)
            finally:
                config = base.read(folder/'applied_configuration.json')
                config.update(native_local_GABA_unchanged=True, native_Z_function_preserved=True,
                    native_GABA_and_Z_unchanged=job['G500'] == 0.,
                    native_Z_input='Original II+(18-EG)*G500*s; same s as membrane G',
                    new_fixed_feedback_law=p['new_fixed_equation'], global_tau_s=job['global_tau_s'])
                write(folder/'applied_configuration.json', config)
                window.flush_window(folder, step)
                obj = budget.RecoveryBudgetSlow.instance
                if obj.global_response_records:
                    rec = np.asarray(obj.global_response_records)
                    dest = folder/'global_response_chunks'
                    dest.mkdir(exist_ok=True)
                    path = dest/f'{round(rec[0,0]*10):010d}_{step:010d}.npz'
                    tmp = path.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp, time_ms=rec[:, 0], q=rec[:, 1], s=rec[:, 2], G_raw=rec[:, 3])
                    tmp.replace(path)
                    obj.global_response_records.clear()
                if job['stage'] != 'native_reference':
                    parent.prefix_check(name)
                    row = analyze(name)
                else:
                    row = None
                # Engine state already includes G memory, all native states and RNGs.
                snap = folder/'states'
                snap.mkdir(exist_ok=True)
                wanted = [f't{step*.0001:g}s.pkl'] if step % 100000 == 0 else []
                if row:
                    for kind, events in [('entry', row['primary']['entries']), ('exit', row['primary']['low_activity_exits'])]:
                        wanted += [f'{kind}{i+1}_checkpoint.pkl' for i in range(len(events))]
                for label in wanted:
                    if not (snap/label).exists():
                        shutil.copy2(folder/'checkpoint.pkl', snap/label)
        return observe

    checkpoint.capture, checkpoint.restore_slow = capture, restore
    fixed.OUT, fixed.prepare, fixed.TerminationSlow, fixed.observation_sink = OUT, lambda: p, cls, factory
    try:
        fixed.worker(name)
    finally:
        checkpoint.capture, checkpoint.restore_slow = capture0, restore0
        fixed.OUT, fixed.prepare, fixed.TerminationSlow, fixed.observation_sink = old
    result = base.read(folder/'result.json')
    result['display_stop_s'] = result['end_s']
    if result['status'] != 'CENSORED_WALL_DEADLINE':
        result['tracker']['stop_reason'] = 'SIMULATION_HORIZON'
    result['new_fixed_feedback_law'] = p['new_fixed_equation']
    write(folder/'result.json', result)
    write(folder/'progress.json', result)
    audit_counts(name)
    if job['stage'] == 'native_reference':
        if result['status'] == 'COMPLETE':
            export_reference(job)
    else:
        analyze(name)


def choose_confirmation(p):
    rows = {n: base.read(OUT/'analysis'/f'{n}.json') for n in p['initial_screen_names']}
    candidates = []
    for gain, tau in SETTINGS:
        pair = [rows[make_job(gain, tau, seed)['name']] for seed in SCREEN_SEEDS]
        if not all(r['run_status'] == 'COMPLETE' and r['full_sequence_screen'] for r in pair):
            continue
        metrics = []
        for row in pair:
            first = next(e for e in row['absolute_Z_recovery']['episodes'] if e['sustained_return_after_absolute_recovery'])
            events = first['post_absolute_recovery_events']
            metrics.append((events['event_span_s'], events['brief_count']))
        rank = (min(v[0] for v in metrics), min(v[1] for v in metrics), -gain, -tau)
        candidates.append(dict(gain=gain, tau=tau, rank=rank))
    p['confirmation_decided'] = True
    selection = dict(candidates=candidates, selected=None, rule=p['confirmation_policy'])
    if candidates and time.time() < p['deadline_epoch']-3600:
        selected = max(candidates, key=lambda c: c['rank'])
        selection['selected'] = selected
        # New-seed references are dependencies, not outcome samples.
        for index, seed in enumerate([9108404, 9108405]):
            ref = make_job(0., 0., seed, stage='native_reference', horizon=8., index=index)
            ref.update(name=f'native_reference_s{seed}', sahp_gain=0., K500=0., k100=0., added_K_increment_when_gate1=0.)
            p['initial_jobs'].append(ref)
        for index, seed in enumerate([9108401, 9108404, 9108405]):
            p['initial_jobs'].append(make_job(selected['gain'], selected['tau'], seed,
                stage='confirmation', horizon=240., index=index))
        for job in p['initial_jobs']:
            write(OUT/'jobs'/f'{job["name"]}.json', job)
    else:
        selection['reason'] = 'No paired full-return candidate' if not candidates else 'Wall budget does not allow additional dispatch'
    write(OUT/'selection.json', selection)
    write(OUT/'protocol.json', p)


def report(final=False):
    logs = OUT/'logs'
    logs.mkdir(exist_ok=True)
    with (logs/'report.log').open('a') as handle:
        cmd = [sys.executable, '-u', str(Path(__file__).with_name('report_topic4_global_feedback_response.py'))]
        if final:
            cmd.append('--final')
        code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT)
    write(OUT/'report_status.json', dict(exit_code=code, time=time.time(), human_review='PENDING'))
    if code:
        raise RuntimeError(f'Report failed with exit code{code}; see logs/report.log')


def resource_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
    gpus = {}
    for line in result.stdout.splitlines():
        device, utilization, free = [float(x.strip()) for x in line.split(',')]
        gpus[int(device)] = dict(utilization_percent=utilization, free_MiB=free)
    return dict(available_memory_GiB=psutil.virtual_memory().available/2**30,
        free_disk_GiB=shutil.disk_usage(OUT).free/2**30, gpus=gpus)


def current_workers():
    found = {}
    runner = str(Path(__file__).resolve())
    for proc in psutil.process_iter(['pid', 'cmdline']):
        cmd = proc.info['cmdline'] or []
        if runner in cmd and 'worker' in cmd and '--name' in cmd:
            name = cmd[cmd.index('--name')+1]
            if name in found:
                raise RuntimeError(f'Duplicate worker: {name}')
            found[name] = proc
    return found


def supervise():
    p = prepare(); configure()
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert base.read(OUT/'mechanism_qa.json')['status'] == 'PASS'
    if 'launched_epoch' not in p:
        p['launched_epoch'] = time.time()
        p['deadline_epoch'] = time.time()+24*3600
        write(OUT/'protocol.json', p)
    running = current_workers()
    children = {}
    failures = []
    last_report = time.time()
    last_done = -1
    (OUT/'logs').mkdir(exist_ok=True)
    for job in p['initial_jobs']:
        if job['stage'] == 'reused_control' and not (OUT/'analysis'/f'{job["name"]}.json').exists():
            audit_counts(job['name']); analyze(job['name'])
    while True:
        now = time.time()
        jobs = {j['name']: j for j in p['initial_jobs']}
        actual = current_workers()
        for name, proc in list(running.items()):
            if name in actual:
                continue
            code = None
            if name in children:
                child, handle = children.pop(name)
                code = child.wait(); handle.close()
            if code not in [None, 0] or not (OUT/'runs'/name/'result.json').exists() or (OUT/'runs'/name/'failure.json').exists():
                failures.append(dict(name=name, pid=proc.pid, exit_code=code, time=now))
        running = actual
        done = [n for n in jobs if n not in running and (OUT/'runs'/n/'result.json').exists() and not (OUT/'runs'/n/'failure.json').exists()]
        if not p['confirmation_decided'] and all(n in done for n in p['initial_screen_names']):
            choose_confirmation(p)
            jobs = {j['name']: j for j in p['initial_jobs']}
        failed = {f['name'] for f in failures}
        pending = [j for n, j in jobs.items() if n not in done and n not in running and n not in failed]
        resources = resource_snapshot()
        can_start = now < p['deadline_epoch']-3600 and not failures
        counts = {d: sum(jobs[n]['device'] == d for n in running) for d in [0, 1]}
        for job in list(pending):
            if not can_start or len(running) >= p['max_workers']:
                break
            device = job['device']
            if counts[device] >= 3 or resources['gpus'][device]['free_MiB'] < 4096 or resources['available_memory_GiB'] < 70 or resources['free_disk_GiB'] < 50:
                continue
            if job['stage'] != 'native_reference' and reference(job['seed']) is None:
                continue
            name = job['name']
            handle = (OUT/'logs'/f'{name}.log').open('a')
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--name', name],
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children[name] = child, handle
            running[name] = psutil.Process(child.pid)
            counts[device] += 1
            pending.remove(job)
            print('START', name, child.pid, 'GPU', device, flush=True)
        progress = {}
        warnings = []
        for name, proc in running.items():
            file = OUT/'runs'/name/'progress.json'
            state = base.read(file) if file.exists() else {}
            age = now-file.stat().st_mtime if file.exists() else now-proc.create_time()
            progress[name] = dict(pid=proc.pid, device=jobs[name]['device'],
                simulated_s=state.get('time_s', state.get('end_s')), target_s=jobs[name]['horizon_s'],
                status=state.get('status', 'STARTING'), update_age_s=age)
            if age > 1800:
                warnings.append(dict(name=name, reason='No progress update for30min', age_s=age))
        write(OUT/'status.json', dict(updated_epoch=now, pid=os.getpid(), running={n: pr.pid for n, pr in running.items()},
            queued=[j['name'] for j in pending], finished=done, failures=failures,
            deadline_epoch=p['deadline_epoch'], confirmation_decided=p['confirmation_decided'],
            resource_snapshot=resources, progress=progress, warnings=warnings))
        # Do not repeatedly render untouched reused controls during startup.
        if (len(done) != last_done and len(done) > 2) or time.time()-last_report >= 3600:
            report(); last_report = time.time(); last_done = len(done)
        if not running and (not pending or failures or not can_start):
            break
        time.sleep(30)
    missing = [j['name'] for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    censored = [j['name'] for j in p['initial_jobs'] if j['name'] not in missing and
                base.read(OUT/'runs'/j['name']/'result.json')['end_s'] < j['horizon_s']]
    write(OUT/'batch_complete.json', dict(status='FAILED' if failures else 'CENSORED' if missing or censored else 'FINISHED',
        finished_epoch=time.time(), failures=failures, missing_results=missing, censored=censored,
        human_review='PENDING', full_Fig5_acceptance='NOT_ESTABLISHED'))
    report(final=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'worker', 'supervise', 'analyze', 'report'])
    parser.add_argument('--name')
    args = parser.parse_args()
    try:
        globals()[args.action](args.name) if args.action in ['worker', 'analyze'] else globals()[args.action]()
    except Exception as exc:
        dest = OUT/'runs'/args.name/'failure.json' if args.name else OUT/f'{args.action}_failure.json'
        write(dest, dict(error=repr(exc), time=time.time()))
        raise
