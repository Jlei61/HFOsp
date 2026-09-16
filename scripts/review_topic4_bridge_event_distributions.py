"""Show complete event distributions for predeclared new-noise XY bridges.

No new metric for optimization: expose the existing rod and fixed-contact delays
without pooling runs. Input is frozen FIT and completed primary event arrays.
"""
from pathlib import Path
import sys, json, hashlib, time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts import analyze_topic4_shape_output_response as an
from src.topic4_pdf_font_guard import install

BASE = Path('/data/hfosp/topic4_sef_hfo')
N = BASE/'core_multiseed_response_curves_20260913'
OUT = BASE/'overnight_exploration_20260913/bridge_event_distributions'
CIDS = ['bridge_circle_out125', 'bridge_circle_out125_xminus075']
COLORS = {2511: '#666666', 2711: '#3478b8', 2712: '#cf7b2b'}
MODES = [('ALL', None), ('TA', 1), ('TB', 0)]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())


def quantities(x, names):
    scl = np.char.startswith(names, 'SCL'); icl = np.char.startswith(names, 'ICL')
    joint = np.isfinite(x[:, scl]).any(1) & np.isfinite(x[:, icl]).any(1)
    rod = np.nanmedian(x[joint][:, scl], 1) - np.nanmedian(x[joint][:, icl], 1)
    i, j = list(names).index('ICL11'), list(names).index('ICL9')
    pair = x[:, j] - x[:, i]; pair = pair[np.isfinite(pair)]
    return {'rod_median_difference_ms': rod, 'ICL9_minus_ICL11_ms': pair}


def main():
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    parent = an.rt.read(an.run.base.PARENT); ev = an.rt.load_evaluator(parent)
    names = np.asarray(an.rt.load_observation_contract(parent)['contact_names'])
    px = np.asarray(ev.fit); labels = np.asarray(ev.fit_labels)
    refs = {m: quantities(px if k is None else px[labels == k], names) for m, k in MODES}
    ref_saved = read(N/'analysis/patient_reference.json')
    for m, _ in MODES:
        assert np.isclose(np.median(refs[m]['rod_median_difference_ms']), ref_saved['modes'][m]['SCL_minus_ICL_lag_median_ms'])
    runs = []; sources = []
    for p in sorted((N/'analysis/units').glob('*/result.json')):
        r = read(p); c = r['counts']
        if c['candidate'] not in CIDS or c['noise'] != 847401: continue
        if c['physical_status'] != 'COMPLETE_NO_RUNAWAY': continue
        raw = Path(r['source']).with_suffix('.npz')
        assert sha(raw) == r['source_sha256']
        with np.load(raw) as a:
            n = a['contact_names'].astype(str)
            order = [list(n).index(q) for q in names]
            ids = [e['event'] for e in r['events'] if e['primary']]
            x = a['centroid_ms'][ids][:, order]; labs = a['event_mode'][ids]
        assert len(x) == c['primary']
        values = {}
        for mode, k in MODES:
            values[mode] = quantities(x if k is None else x[labs == k], names)
            saved = next(q for q in r['observations'] if q['layer'] == 'primary' and q['mode'] == mode)
            v = values[mode]['rod_median_difference_ms']
            if len(v): assert np.isclose(np.median(v), saved['SCL_minus_ICL_lag_median_ms'])
        runs.append((c, values))
        sources.append(dict(analysis=str(p), analysis_sha256=sha(p), arrays=str(raw), arrays_sha256=r['source_sha256']))
    stats = []
    for condition, values in [(dict(candidate='patient_FIT', topology=None, noise=None), refs)] + runs:
        for mode, _ in MODES:
            for key, v in values[mode].items():
                q = np.quantile(v, [.05, .5, .95]) if len(v) else [None]*3
                stats.append(dict(candidate=condition['candidate'], topology=condition['topology'], noise=condition['noise'],
                    mode=mode, observable=key, n=len(v), mean_ms=float(np.mean(v)) if len(v) else None,
                    variance_ms2=float(np.var(v, ddof=1)) if len(v)>1 else None, q05_ms=q[0], median_ms=q[1], q95_ms=q[2]))
    pd.DataFrame(stats).to_csv(OUT/'event_distribution_statistics.csv', index=False)
    install(); plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10, 'pdf.fonttype':3})
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.17, hspace=.4, wspace=.22)
    metrics = [('rod_median_difference_ms', '每事件 SCL−ICL 杆间质心差 (ms)'),
               ('ICL9_minus_ICL11_ms', '每事件 ICL9−ICL11 质心差 (ms)')]
    limits = {}
    for key, _ in metrics:
        all_v = np.concatenate([val[m][key] for _, val in [(None, refs)]+runs for m, _ in MODES])
        limits[key] = (min(-100, 10*np.floor(all_v.min()/10)), max(140, 10*np.ceil(all_v.max()/10)))
    for row, (mode, _) in enumerate(MODES):
        for col, (key, title) in enumerate(metrics):
            ax = axes[row, col]
            for c, val in runs:
                v = np.sort(val[mode][key]); style = '--' if c['candidate'] == CIDS[0] else '-'
                if len(v): ax.step(v, np.arange(1, len(v)+1)/len(v), where='post', color=COLORS[c['topology']], ls=style, lw=1.5)
            v = np.sort(refs[mode][key])
            ax.step(v, np.arange(1, len(v)+1)/len(v), where='post', color='black', lw=1.6)
            ax.axvline(0, c='.7', lw=.8)
            ax.set(title=mode, xlabel=title, ylabel='累计事件比例', ylim=(-.02,1.02), xlim=limits[key])
            ax.grid(alpha=.14)
    used = sorted({c['topology'] for c, _ in runs})
    handles = [Line2D([],[],c='black',label='患者FIT')]+[Line2D([],[],c=COLORS[t],label=f'网络{t}') for t in used]+[
        Line2D([],[],c='.4',ls='--',label='原位置'),Line2D([],[],c='.4',ls='-',label='左移0.75mm')]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.52,.955),frameon=False)
    fig.suptitle('中位数背后：新噪声下的完整事件时差分布',fontsize=18,y=.992)
    completed = '; '.join(f'{t} 原/左移 '+ '/'.join(str(next((c['primary'] for c,_ in runs if c['topology']==t and c['candidate']==cid),'待完成')) for cid in CIDS) for t in used)
    fig.text(.08,.045,'固定首条新噪声847401，所有已完成的原位置／左移条件；每条彩线是一条60秒运行，事件不作为独立网络重复。\n'
        '杆间差：先取每杆参与触点质心的中位数，再相减；右列只使用ICL9、ICL11均参与的事件，正值表示ICL11更早。\n'
        '两类分别及ALL保留全部合格事件，不按路径好坏筛选；这些是两个一维边缘分布，不代表完整联合路径恢复。\n'
        '已完成合格事件数：'+completed+'。每项有效n、均值、中位数、方差、5–95%范围见CSV。\n'
        '患者FIT也参与冻结目标；当前图不作独立患者验证、显著性检验或新loss，未完成曲线不补值。',fontsize=9)
    for ext in ['png','pdf']: fig.savefig(OUT/'figures'/f'new_noise_event_delay_distributions.{ext}',dpi=150)
    plt.close(fig)
    (OUT/'figures/README.md').write_text('### new_noise_event_delay_distributions.png\nALL、TA、TB分别展示杆间质心差与固定ICL接触对时差的经验累计分布。颜色区分网络，虚线原位置、实线左移，固定首条新噪声且不混池；未完成的条件不填值。\n**关注点**：训练分数或中位数改变是否伴分布更接近患者，以及模型是否仍在患者不集中的延迟范围形成窄峰。\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(created_unix=time.time(),completed_runs=len(runs),sources=sources,
        patient_parent=str(an.run.base.PARENT),patient_parent_sha256=sha(an.run.base.PARENT),
        patient_n=len(px),definition_check='Patient and every model rod median match existing frozen observer numerically',
        producer=str(Path(__file__)),producer_sha256=sha(Path(__file__)),physical_runs_added=0,score_changes=0),ensure_ascii=False,indent=2))
    print(dict(completed_runs=len(runs),output=str(OUT)),flush=True)


if __name__ == '__main__': main()
