"""Descriptive CAL references at each simulation's actual event count.

Consecutive events stay within their original recording block. This is neither
an independent patient test nor a confidence interval, and does not change BO.
"""
from pathlib import Path
import argparse, sys, warnings
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as a
from src.topic4_three_observable_objective import normalized_ranks

DEST = a.A / 'patient_count_block_reference'
SEED = 2026091501
DRAWS = 256


def prepare():
    DEST.mkdir(exist_ok=True)
    ev, names, _ = a.patient()
    obj = a.load_objective()
    identity = a.A / 'patient_event_identity.npz'
    with np.load(identity) as z:
        ids = z['cal_chronological_indices']
        blocks = z['blocks'][ids]
        clock = z['event_abs_time'][ids]
    t = ev.patient[ids]
    labels = obj.labels(t)
    ranks, mask = normalized_ranks(t)
    ranks = np.where(mask, ranks, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        lag = np.nanmedian(t[:, obj.scl], axis=1) - np.nanmedian(t[:, obj.icl], axis=1)
    pairs = np.concatenate(obj.pairs)
    delta = t[:, pairs[:, 1]] - t[:, pairs[:, 0]]
    a.rt.write(DEST / 'method.json', dict(
        version='consecutive_CAL_actual_N_v1', seed=SEED, draws=DRAWS,
        patient_identity=str(identity), identity_sha256=a.rt.sha(identity),
        objective_sha256=a.rt.sha(a.A / 'training_objective.pkl'),
        source='Frozen CAL development events, never relabelled or balanced',
        sampling='Uniform among all length-N consecutive windows wholly within one original CAL block; with replacement; window draws saved',
        exclusions='Blocks shorter than N supply no complete window. Coverage and block weights reported for every N. N greater than the longest CAL block is NOT_ESTIMABLE.',
        comparison_unit='A full simulation trajectory versus a descriptive distribution of patient N-event windows',
        interpretation='5-95% reference ranges, not confidence intervals, independent replicates, significance thresholds or new optimization gates. Matches event count and within-block order, not recording duration.',
        temporal_limits='Overlapping windows remain dependent; CAL has already informed development. Original block boundaries are never crossed.',
        names=names, modes=dict(TA=1, TB=0), pair_names=[f'{names[i]} to {names[j]}' for i, j in pairs],
        blocks=[dict(block=int(b), N=int((blocks == b).sum())) for b in np.unique(blocks)]))
    return dict(obj=obj, t=t, ids=ids, blocks=blocks, clock=clock,
                labels=labels, ranks=ranks, mask=mask, lag=lag, delta=delta)


def summarize_samples(values):
    x = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return dict(valid_draws=np.isfinite(x).sum(axis=0),
                    q05=np.nanquantile(x, .05, axis=0),
                    median=np.nanmedian(x, axis=0),
                    q95=np.nanquantile(x, .95, axis=0))


def raw_metrics(t, lab, ranks, mask, lag, delta):
    result = dict(TB_fraction=float(np.mean(lab == 0)), N=len(t))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for name, k in [('ALL', None), ('TA', 1), ('TB', 0)]:
            ix = np.ones(len(t), bool) if k is None else lab == k
            rr, mm, ll, dd = ranks[ix], mask[ix], lag[ix], delta[ix]
            valid = np.isfinite(dd)
            order = np.where(valid, (dd > 0) + .5 * (dd == 0), np.nan)
            prefix = name + '/'
            result[prefix + 'N'] = int(ix.sum())
            result[prefix + 'participation'] = mm.mean(0) if len(mm) else np.full(mask.shape[1], np.nan)
            result[prefix + 'rank_mean'] = np.nanmean(rr, axis=0)
            result[prefix + 'rank_variance'] = np.nanvar(rr, axis=0)
            result[prefix + 'pair_order_probability'] = np.nanmean(order, axis=0)
            result[prefix + 'pair_lag_mean_ms'] = np.nanmean(dd, axis=0)
            result[prefix + 'pair_lag_variance_ms2'] = np.nanvar(dd, axis=0)
            result[prefix + 'both_rods'] = float(np.isfinite(ll).mean()) if len(ll) else np.nan
            result[prefix + 'rod_lag_N'] = int(np.isfinite(ll).sum())
            for metric, fun in [('mean_ms', np.nanmean), ('median_ms', np.nanmedian), ('variance_ms2', np.nanvar)]:
                result[prefix + 'rod_lag_' + metric] = float(fun(ll))
    return result


def reference(n, prepared):
    path = DEST / f'N_{n}.json'
    if path.exists():
        return a.rt.read(path)
    d = prepared
    candidates = []
    for b in np.unique(d['blocks']):
        ix = np.flatnonzero(d['blocks'] == b)
        if len(ix) >= n:
            assert np.all(np.diff(ix) == 1)
            candidates.extend((int(b), int(start)) for start in ix[:len(ix) - n + 1])
    if not candidates:
        payload = dict(N=n, status='NOT_ESTIMABLE', reason='No CAL block contains this many events')
        a.rt.write(path, payload)
        return payload
    rng = np.random.default_rng(np.random.SeedSequence([SEED, n]))
    chosen = rng.integers(len(candidates), size=DRAWS)
    draws = []
    arrays = {}
    for j in chosen:
        b, start = candidates[j]
        ix = np.arange(start, start + n)
        assert np.all(d['blocks'][ix] == b)
        assert np.all(np.diff(d['clock'][ix]) >= 0)
        draws.append(dict(block=b, CAL_start=start,
                          first_parent_id=int(d['ids'][start]), last_parent_id=int(d['ids'][start + n - 1]),
                          duration_s=float(d['clock'][ix[-1]] - d['clock'][ix[0]])))
        metrics = raw_metrics(*(d[key][ix] for key in ['t', 'labels', 'ranks', 'mask', 'lag', 'delta']))
        for key, value in metrics.items():
            arrays.setdefault(key, []).append(value)
    eligible = {b for b, _ in candidates}
    payload = dict(N=n, status='DESCRIPTIVE_REFERENCE', available_windows=len(candidates),
                   eligible_blocks=sorted(eligible), eligible_events=int(np.isin(d['blocks'], list(eligible)).sum()),
                   CAL_events=len(d['t']),
                   block_draw_probability={str(b):sum(bb == b for bb, _ in candidates)/len(candidates) for b in sorted(eligible)},
                   draws=draws, metrics={key:summarize_samples(value) for key, value in arrays.items()},
                   no_TB_draws=int(sum(x == 0 for x in arrays['TB/N'])),
                   no_TA_draws=int(sum(x == 0 for x in arrays['TA/N'])),
                   duration_s=summarize_samples([x['duration_s'] for x in draws]))
    a.rt.atomic_npz(DEST / f'N_{n}_draw_statistics.npz', **{key:np.asarray(value) for key, value in arrays.items()})
    a.rt.write(path, payload)
    print(f'PATIENT_COUNT_REFERENCE N={n} windows={len(candidates)}', flush=True)
    return a.rt.read(path)


def update():
    d = prepare()
    comparisons = []
    for r in a.records():
        if r['N'] < 2 or r['physical_status'] != 'COMPLETE_NO_RUNAWAY':
            continue
        ref = reference(r['N'], d)
        _, t, lab, ids, _ = a.load_small(Path(r['source']))
        t, lab = t[ids], lab[ids]
        ranks, mask = normalized_ranks(t)
        ranks = np.where(mask, ranks, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lag = np.nanmedian(t[:, d['obj'].scl], axis=1) - np.nanmedian(t[:, d['obj'].icl], axis=1)
        pairs = np.concatenate(d['obj'].pairs)
        delta = t[:, pairs[:, 1]] - t[:, pairs[:, 0]]
        measured = raw_metrics(t, lab, ranks, mask, lag, delta)
        comparisons.append(dict(candidate=r['candidate'], stage=r['stage'], topology=r['topology'], noise=r['noise'],
                                N=r['N'], actual=measured, reference=str(DEST/f"N_{r['N']}.json"), reference_status=ref['status']))
    a.rt.write(DEST / 'run_comparisons.json', dict(runs=comparisons))
    return comparisons


def figure(rows, confirmation=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.topic4_pdf_font_guard import install
    from scripts.topic4_three_observable_plot_labels import plot_label, parameter_table
    install()
    plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10, 'pdf.fonttype':3})
    from scripts import control_topic4_three_observable_bo as ctl
    conditions = sorted([r for r in ctl.training_data() if r['scorable']], key=lambda r:r['J'])
    refid = a.rt.read(a.OUT/'plan.json')['reference_id']
    ids = (a.rt.read(a.OUT/'nomination.json')['ids'] if confirmation else
           [refid] + [r['candidate'] for r in conditions if r['candidate'] != refid][:2])
    stages = ['confirmation'] if confirmation else ['initial', 'adaptive']
    chosen = [r for cid in ids for r in sorted(rows, key=lambda q:(q['topology'],q['noise']))
              if r['candidate']==cid and r['stage'] in stages]
    metrics = [('TB_fraction','TB事件占比'), ('TB/rod_lag_median_ms','TB杆间时差中位数 (ms)'),
               ('TB/rod_lag_variance_ms2','TB杆间时差方差 (ms²)'), ('TA/rod_lag_median_ms','TA杆间时差中位数 (ms)'),
               ('TA/rod_lag_variance_ms2','TA杆间时差方差 (ms²)'), ('TB/both_rods','TB两杆同时参与比例')]
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    for j, r in enumerate(chosen):
        p = a.rt.read(r['reference'])
        if p['status'] != 'DESCRIPTIVE_REFERENCE':
            continue
        for ax,(key,title) in zip(axes.ravel(),metrics):
            v = p['metrics'][key]
            if v['median'] is not None:
                ax.plot([j,j],[v['q05'],v['q95']],color='#bbbbbb',lw=5,zorder=1)
                ax.scatter(j,v['median'],marker='_',color='black',s=90,zorder=2)
            ax.scatter(j,r['actual'][key],color='#327aaa' if r['noise'] in [847401,849401] else '#d07e35',s=40,zorder=3)
            ax.set_title(title);ax.grid(alpha=.15)
    def label(r):
        cid=r['candidate']
        short='固定参考' if cid==refid else plot_label(cid)
        return short+f"\n{r['topology']}/{r['noise']}\nN={r['N']}"
    labels=[label(r) for r in chosen]
    for ax in axes.ravel():
        ax.set(xticks=range(len(chosen)),xticklabels=labels);ax.tick_params(axis='x',labelsize=7)
    phase = '新网络确认' if confirmation else '训练网络'
    fig.suptitle(phase+'：与每条模型运行的实际事件数匹配的患者CAL连续窗参照\n灰条=患者窗统计量5–95%，黑横线=中位数，彩点=模型；横轴第二行为网络/噪声种子\n不是置信区间或新的通过门槛；患者窗与模型没有匹配记录时长',fontsize=12)
    fig.tight_layout(rect=(0,.18,1,.92));parameter_table(fig,ids)
    f = DEST/'figures';f.mkdir(exist_ok=True)
    stem = 'patient_actual_N_confirmation_reference' if confirmation else 'patient_actual_N_reference'
    for ext in ['png','pdf']:
        fig.savefig(f/f'{stem}.{ext}',dpi=150)
    plt.close(fig)
    mapping='\n'.join(f'- 候选{i}：{a.point_label(cid)}（`{cid}`）' for i,cid in enumerate(ids) if i)
    with (f/'README.md').open('a' if confirmation else 'w') as out:
        for ext in ['png','pdf']:
            out.write(f'### {stem}.{ext}\n'+phase+'按模型实际N，抽取同一患者CAL块内的连续N事件；每次保留自然标签比例。灰条描述患者窗口统计量的变化，既非独立样本置信区间，也非全患者变异的完整覆盖；短于N的块不贡献窗口。训练与确认分别成图，避免运行标签重叠。\n**关注点**：不能把低事件数当成机制失败；同时检查模式比例、杆间时差的中心及散布，而非只看loss。\n\n')
        out.write(mapping+'\n')
    if not confirmation and any(r['stage']=='confirmation' for r in rows):
        figure(rows, confirmation=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figure',action='store_true');args=parser.parse_args()
    rows=update()
    if args.figure:figure(rows)
