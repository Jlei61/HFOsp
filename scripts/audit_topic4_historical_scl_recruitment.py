"""Read saved historical radius runs; no simulation and no relabelling as TA/TB."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/topic4_sef_hfo/field_swap_subject_snn'
OUT = ROOT / 'results/topic4_sef_hfo/core_connectivity_search_design_20260910'

def main():
    figdir = OUT / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    rows, inputs, arrays = [], [], {}
    for radius in [1.5, 2.5, 4.0, 6.0]:
        stem = f'epilepsiae_1146_sweep_cr{radius:.1f}'
        jp, zp = SOURCE / f'readout_{stem}.json', SOURCE / f'figdata_{stem}.npz'
        result = json.loads(jp.read_text())
        with np.load(zp, allow_pickle=True) as archive:
            a = {k: archive[k] for k in archive.files}
        arrays[radius] = a
        for path in [jp, zp]:
            inputs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        events = result['events']
        assert len(events) == result['n_events']
        names = a['names'].astype(str)
        scl = [str(n) for n in names if n.startswith('SCL')]
        counts = {n: sum(e['ranks'].get(n) is not None for e in events) for n in scl}
        row = dict(radius_mm=radius, seed=result['seed'], n_detected=len(events),
                   contact_event_counts=counts,
                   any_SCL=sum(any(e['ranks'].get(n) is not None for n in scl) for e in events),
                   n_E=len(a['vth']), n_lowered=int((a['vth'] < 18).sum()),
                   n_raised=int((a['vth'] > 18).sum()),
                   centers_mm=a['foci'].tolist(), theta_deg=float(a['theta_deg']),
                   recorded_last_time_ms=float(a['times'][-1]),
                   legacy_direction_counts=dict(forward=result['dir_forward'], reverse=result['dir_reverse']),
                   representative_local_activity={})
        for repkey in ['rep_fwd', 'rep_rev']:
            rep = a[repkey].item()
            checks = {}
            for name in scl:
                xy = a['contacts'][list(names).index(name)]
                near = np.linalg.norm(a['posE'] - xy, axis=1) <= .278
                checks[name] = dict(n_local_E=int(near.sum()),
                    n_active_E=int((near & np.isfinite(rep['onset'])).sum()))
            row['representative_local_activity'][repkey] = checks
        rows.append(row)
    summary = dict(status='OFFLINE_HISTORICAL_AUDIT_COMPLETE', sources=inputs, rows=rows,
        statistical_unit='one historical topology/noise seed; events nested in each approximately 4 s run',
        caveats=['Historical direction signs are not patient TA/TB.',
                 'Representatives were selected by the old producer for participation/readability.',
                 'Local native activity is assessed within 0.278 mm, not the current readout footprint.',
                 'Historical coordinate registration and graph sampler differ from the current baseline.',
                 'Radius changes both support and total threshold modulation; large cores cross boundaries.'])
    (OUT / 'historical_scl_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout='constrained')
    norm = matplotlib.colors.Normalize(0, 40)
    for ax, radius in zip(axes[:2], [1.5, 6.0]):
        a = arrays[radius]; onset = a['rep_fwd'].item()['onset']; ok = np.isfinite(onset)
        ax.scatter(*a['posE'][ok].T, c=onset[ok], cmap='viridis', norm=norm, s=2, rasterized=True)
        for center in a['foci']:
            ax.add_patch(Circle(center, radius, fill=False, ec='#c14646', lw=1.5))
        names = a['names'].astype(str)
        for shaft, color in [('SCL','#00b7c4'), ('ICL','#df9427')]:
            ids = sorted([i for i,n in enumerate(names) if n.startswith(shaft)], key=lambda i:int(names[i][3:]))
            ax.plot(*a['contacts'][ids].T, '-o', color=color, ms=3, lw=1)
            for i in ids:
                ax.annotate(names[i], a['contacts'][i], xytext=(2,4), textcoords='offset points', fontsize=6)
        r = next(r for r in rows if r['radius_mm']==radius)
        ax.set(xlim=(0,20), ylim=(0,20), aspect='equal', xlabel='x (mm)', ylabel='y (mm)',
               title=f'旧手放双核：半径 {radius:g} mm\n降阈值 E 细胞 {r["n_lowered"]:,} 个')
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap='viridis'), ax=axes[:2], shrink=.75, pad=.015)
    bar.set_label('保存示例内首发时间 (ms)')
    ax = axes[2]
    x = np.arange(4)
    for n, display in [('SCL9','SCL9 = SCL6'), ('SCL8','SCL8 = SCL7')]:
        values=[r['contact_event_counts'][n]/r['n_detected'] for r in rows]
        ax.plot(x, values, '-o', label=display, lw=1.5)
    ax.set(xticks=x, xticklabels=[str(r['radius_mm']) for r in rows], ylim=(-.03,.52),
           xlabel='双核半径 (mm)', ylabel='全部检测事件中的参与比例', title='同一旧 seed 的全部检测事件')
    for i, r in enumerate(rows):
        ax.text(i,.47,f'N={r["n_detected"]}',ha='center',fontsize=9)
    ax.legend(ncol=1,loc='upper left',bbox_to_anchor=(0,.86),fontsize=9)
    fig.suptitle('历史证据：扩大核能增加 SCL 招募；同时改变了支持范围和阈值总量',fontsize=14)
    for ext in ['png','pdf']:
        fig.savefig(figdir/f'historical_scl_radius_evidence.{ext}',dpi=180)
    plt.close(fig)
    (figdir/'README.md').write_text('### historical_scl_radius_evidence.png\n\n左两图直接显示旧半径1.5与6 mm运行保存的同向代表窗内原生E首发时间，沿用各自真实历史SEEG布局；红圈为阈值核，橙色为ICL，青色为SCL。右图使用该半径下全部检测事件，分母为14、16、7、11；这不是患者TA/TB分类，也不是跨网络统计。旧代表窗由原producer按参与数及可读性选取，不能据两例判断典型性或传播因果；当前坐标注册不同，不与当前位置逐点叠加。\n\n**关注点**：6 mm条件确有上部SCL附近原生放电，但降阈值细胞数和边界截断同时改变，不能据此声称半径已是唯一原因。\n\n### historical_scl_radius_evidence.pdf\n\n同名PNG的PDF版本；神经元散点栅格化，其余元素保留矢量。数据与选例完全相同。\n\n**关注点**：颜色为单窗首发时间，不能当成连续前沿或自然模式身份。\n')
    print(json.dumps([dict(radius=r['radius_mm'],N=r['n_detected'],SCL=r['contact_event_counts'],lowered=r['n_lowered']) for r in rows]))

if __name__ == '__main__':
    main()
