"""Full conditional bifurcation figure incorporating the independent recheck.

Line styles interpolate only between equal-status anchors on the same monotone
continuation segment. A known fold or contradictory sample breaks interpolation.
Unclassified portions remain dotted. Source model and simulation data are unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_periodic_completion import orbit_field
from model_zm import ZMSpatialRate, DEST, OLD, read, write
from figures_and_report import spatial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

AUDIT = DEST / 'recheck_20260917'
PERIODIC = DEST / 'periodic_completion'
ORANGE, GREEN, RED, GRAY = '#ca8325', '#208975', '#ad3748', '#8c8c8c'


def orbit(path):
    row = read(path)
    row['_id'] = path.stem
    row['_source'] = str(path)
    return row


def prepare():
    root = PERIODIC / 'orbits'
    verified = read(PERIODIC / 'stability_by_orbit.json')['rows']
    selected = {}
    for row in verified:
        if row['status'] not in ['STABLE', 'UNSTABLE']:
            continue
        key = Path(row['source']).stem
        if key not in selected or row['dt_ms'] < selected[key]['dt_ms']:
            selected[key] = row
    audit = read(AUDIT / 'review_status.json')
    assert audit['middle_orbit_stability'] == 'UNSTABLE_SUPPORTED_BY_TWO_TIME_STEPS'
    selected['burstUp_0031_N512'] = dict(status='UNSTABLE',
        source=str(root / 'burstUp_0031_N512.npz'),
        evidence=audit['middle_orbit_Floquet'], dt_ms=.025)
    arc = [orbit(p) for p in sorted(root.glob('burstUp_*N512.json'))]
    low = sorted([orbit(p) for p in root.glob('physicalLow*.json')] +
                 [orbit(root / f'burst_D{d:.9f}_N512.json') for d in [.18, .181]],
                 key=lambda q: q['D'])
    small = sorted([orbit(p) for p in (AUDIT / 'orbits').glob('LPC_small_*N512.json')]
                   + arc[27:35], key=lambda q: q['mean_rates_hz'][1])
    first, second = [orbit(AUDIT / 'orbits' / f'LPC_small_{k}_N512.json')
                     for k in ['max', 'min']]
    main = orbit(root / 'LPC_burst_N1024.json')
    # Core B mean is a local monotone coordinate here, including all three folds.
    # Do not sort by D: it reverses at each fold and labels would be spliced.
    approach = arc[35:42] + [orbit(root / 'LPC_burst_before_N1024.json'), main,
                            arc[42], orbit(root / 'LPC_burst_after_N1024.json')]
    approach.sort(key=lambda q: q['mean_rates_hz'][1])
    branch = low + arc[:27] + small + approach + arc[43:]
    fold_ids = {q['_id'] for q in [first, second, main]}
    anchors = [(i, selected[q['_id']]['status']) for i, q in enumerate(branch)
               if q['_id'] in selected]
    styles = ['UNCLASSIFIED'] * (len(branch)-1)
    for (left, a), (right, b) in zip(anchors[:-1], anchors[1:]):
        interior = branch[left:right+1]
        crosses_fold = any(q['_id'] in fold_ids for q in interior)
        if a == b and not crosses_fold:
            styles[left:right] = [a] * (right-left)
    assert branch[0]['D'] < 1e-10
    for fid in fold_ids:
        i = next(i for i,q in enumerate(branch) if q['_id'] == fid)
        assert styles[i-1] == styles[i] == 'UNCLASSIFIED'
    weak = next(i for i,q in enumerate(branch) if q['_id'] == 'burstUp_0031_N512')
    assert styles[weak-1] == styles[weak] == 'UNCLASSIFIED'
    return branch, styles, selected, [first, second, main], audit


def periodic_lines(ax, branch, styles, lw=1.6):
    colors = {'STABLE': ORANGE, 'UNSTABLE': ORANGE, 'UNCLASSIFIED': GRAY}
    lines = {'STABLE': '-', 'UNSTABLE': '--', 'UNCLASSIFIED': ':'}
    # Draw each contiguous status range in one call so dash phase is meaningful.
    start = 0
    for stop in range(1, len(styles)+1):
        if stop < len(styles) and styles[stop] == styles[start]:
            continue
        rows = branch[start:stop+1]
        ax.plot([q['D'] for q in rows], [q['global_mean_hz'] for q in rows],
                ls=lines[styles[start]], color=colors[styles[start]], lw=lw, zorder=3)
        start = stop


def equilibrium(ax):
    names = ['D_arclength_lower', 'D_arclength_upper_focused',
        'D_arclength_upper_onset_range', 'D_arclength_upper_onset_range_v2',
        'D_arclength_upper_onset_range_v3', 'D_arclength_upper_onset_range_v4',
        'D_gap_lower_guarded', 'D_gap_middle_down', 'D_gap_middle_up', 'D_gap_middle_to_low']
    for name in names:
        rows = [q for q in read(OLD/'g20'/name/'result.json')['rows'] if q.get('converged', True)]
        ax.plot([q['D'] for q in rows], [q['global_E_hz'] for q in rows],
                '.', color=GRAY, ms=1.2, zorder=1)
        path = PERIODIC/'equilibrium_classification'/f'{name}.json'
        if path.exists():
            checked = read(path)['rows']
            for left, right in zip(checked[:-1], checked[1:]):
                if left['status'] == right['status'] == 'UNSTABLE':
                    section = [q for q in rows if left['index'] <= q['index'] <= right['index']]
                    ax.plot([q['D'] for q in section], [q['global_E_hz'] for q in section],
                            '--', color='#333333', lw=.8, zorder=2)
    tail = [q for q in read(OLD/'g20/conditional_dynamic_M/result.json')['rows']
            if q['converged'] and q['direction']=='decreasing' and q['D']>=.4]
    ax.plot([q['D'] for q in tail], [q['global_E_hz'] for q in tail], color='#333333', lw=1.3)
    for path in sorted((PERIODIC/'equilibrium_counts').glob('*.json')):
        q = read(path)
        if q['status']=='RESOLVED' and q['unstable_roots']==0:
            ax.plot(q['D'], q['global_E_hz'], 'o', ms=3, color='#333333')
    for q in read(DEST/'critical_spectra/result.json')['rows']:
        if not q['label'].startswith('SN'):
            continue
        ax.plot(q['D'], q['global_E_hz'], 'D', ms=4, mfc='white', mec='#494949')
        offset = {'SN1': (18,-10), 'SN2': (18,10), 'SN7': (12,13), 'SN6': (12,-20)}
        if q['label'] in offset:
            ax.annotate('SN3–6' if q['label']=='SN6' else q['label'],
                        (q['D'], q['global_E_hz']), xytext=offset[q['label']],
                        textcoords='offset points', fontsize=9,
                        arrowprops=dict(arrowstyle='-', lw=.6))


def main():
    s = ZMSpatialRate()
    branch, styles, selected, folds, audit = prepare()
    root = PERIODIC/'orbits'
    fig = plt.figure(figsize=(14.4,9.7))
    gs = fig.add_gridspec(3,2,width_ratios=[2.95,1],hspace=.48,wspace=.36)
    ax = fig.add_subplot(gs[:,0])
    equilibrium(ax)
    periodic_lines(ax,branch,styles)
    high = sorted([orbit(p) for p in root.glob('highHopf_A*_N64.json')],key=lambda q:q['D'])
    hopf = read(PERIODIC/'hopf_high.json')
    ax.plot([q['D'] for q in high]+[hopf['D']], [q['global_mean_hz'] for q in high]+[hopf['global_E_hz']],
            ':',color=GRAY,lw=1.4)
    ax.plot(hopf['D'],hopf['global_E_hz'],'^',color='#8355a4',ms=7,zorder=8)
    ax.annotate('H',(hopf['D'],hopf['global_E_hz']),xytext=(-22,-25),textcoords='offset points',
                color='#8355a4',arrowprops=dict(arrowstyle='-',lw=.7,color='#8355a4'))
    for key,checked in selected.items():
        q=read(root/f'{key}.json');stable=checked['status']=='STABLE'
        ax.plot([q['D']]*2,[q['global_min_hz'],q['global_max_hz']],'s',
                mfc=GREEN if stable else 'white',mec=GREEN,mew=.9,ms=4.2,zorder=5)
        ax.plot(q['D'],q['global_mean_hz'],'o',ms=3.4,mec=ORANGE,
                mfc=ORANGE if stable else 'white',mew=.9,zorder=6)
    for fold in folds:
        ax.plot(fold['D'],fold['global_mean_hz'],'*',color=RED,ms=11,zorder=7)
    ax.annotate('LPC1–3',(folds[-1]['D'],folds[-1]['global_mean_hz']),xytext=(20,-8),
                textcoords='offset points',color=RED,fontsize=10,
                arrowprops=dict(arrowstyle='-',lw=.7,color=RED))
    q=read(root/'burst_D0.180000000_N512.json')
    ax.annotate('1',(q['D'],q['global_mean_hz']),xytext=(-24,-28),textcoords='offset points',color=ORANGE,
                arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.6))
    ax.annotate('2',(folds[-1]['D'],folds[-1]['global_mean_hz']),xytext=(-7,19),
                textcoords='offset points',color=RED)
    post=next(q for q in read(PERIODIC/'crossing_summary.json')['rows'] if q['D']==.1887)
    ax.plot(.1887,post['last_2000ms_mean_hz'],'d',color='#7654a0',ms=6,zorder=6)
    ax.annotate('3',(.1887,post['last_2000ms_mean_hz']),xytext=(12,0),textcoords='offset points',color='#7654a0')
    for path in sorted((DEST/'runs').glob('main_D*/result.json')):
        q=read(path)
        if q['D_initial']>=.2:
            ax.plot(q['D_initial'],q['dynamics'][0]['mean_hz'],'d',color='#7654a0',ms=4)
    ax.set(xlim=(0,1),ylim=(.045,550),yscale='log',xlabel=r'$D=1-\langle Z_E\rangle$',
           ylabel='Global E rate (Hz / neuron)')
    ax.set_yticks([.05,.1,1,10,100,500]);ax.set_yticklabels(['0.05','0.1','1','10','100','500'])
    ax.text(-.11,1.015,'A',transform=ax.transAxes,fontsize=18,fontweight='bold')
    # Both zooms use exactly the same source rows and classification map.
    zooms=[ax.inset_axes([.17,.17,.36,.24]),ax.inset_axes([.62,.17,.36,.24])]
    for zax in zooms:
        periodic_lines(zax,branch,styles,1.3)
        for i,fold in enumerate(folds):
            zax.plot(fold['D'],fold['global_mean_hz'],'*',color=RED,ms=9,zorder=7)
        q=next(q for q in branch if q['_id']=='burstUp_0031_N512')
        zax.plot(q['D'],q['global_mean_hz'],'o',mfc='white',mec=ORANGE,ms=5,zorder=8)
        zax.set_xlabel('D',fontsize=10);zax.tick_params(labelsize=8)
        zax.xaxis.set_major_formatter(FormatStrFormatter('%.7f'))
        zax.tick_params(axis='x',rotation=22)
    zooms[0].set(xlim=(.18848,.18865),ylim=(16.755,17.04),xticks=[.18849,.18864],yticks=[16.8,16.9,17.0])
    zooms[0].set_ylabel('Period mean (Hz)',fontsize=10)
    zooms[0].annotate('LPC1,2',(folds[0]['D'],folds[0]['global_mean_hz']),xytext=(12,27),
                      textcoords='offset points',fontsize=8,color=RED,arrowprops=dict(arrowstyle='-',lw=.5,color=RED))
    zooms[0].annotate('LPC3',(folds[2]['D'],folds[2]['global_mean_hz']),xytext=(-49,-5),
                      textcoords='offset points',fontsize=8,color=RED)
    zooms[1].set(xlim=(.18852393,.18852433),ylim=(16.801,16.854),
                 xticks=[.1885240,.1885243],yticks=[16.81,16.83,16.85])
    q=folds[0];zooms[1].annotate('LPC1',(q['D'],q['global_mean_hz']),xytext=(-42,7),
                               textcoords='offset points',fontsize=8,color=RED)
    q=folds[1];zooms[1].annotate('LPC2',(q['D'],q['global_mean_hz']),xytext=(.27,.04),
                               textcoords='axes fraction',fontsize=8,color=RED)
    handles=[Line2D([],[],color='#333333',label='Equilibrium: stable'),
        Line2D([],[],color='#333333',ls='--',label='Equilibrium: unstable'),
        Line2D([],[],color=ORANGE,label='Period mean: stable samples joined'),
        Line2D([],[],color=ORANGE,ls='--',label='Period mean: unstable samples joined'),
        Line2D([],[],color=GRAY,ls=':',label='Stability unclassified'),
        Line2D([],[],color=ORANGE,marker='o',mfc='white',ls='none',ms=4,label='Unstable period mean: checked sample'),
        Line2D([],[],marker='s',ls='none',color=GREEN,label='Periodic extrema: stable'),
        Line2D([],[],marker='s',ls='none',mfc='white',mec=GREEN,label='Periodic extrema: unstable'),
        Line2D([],[],marker='D',ls='none',mfc='white',mec='#494949',label='Equilibrium fold (SN)'),
        Line2D([],[],marker='*',ls='none',color=RED,ms=9,label='Fold of cycles (LPC)'),
        Line2D([],[],marker='^',ls='none',color='#8355a4',label='Supercritical Hopf (H)'),
        Line2D([],[],marker='d',ls='none',color='#7654a0',label='Finite-time mean')]
    ax.legend(handles=handles,loc='center right',bbox_to_anchor=(1,.63),frameon=False,
              fontsize=9,handlelength=2.7,labelspacing=.55)
    fields=[]
    for j,path in enumerate([root/'burst_D0.180000000_N512.npz',root/'LPC_burst_N1024.npz',None]):
        if path is not None:
            field,meta=orbit_field(s,path)
        else:
            path=PERIODIC/'crossing_runs/D0.188700000/trajectory.npz';z=np.load(path)
            idx=len(z['time_ms'])-500;field=z['field_E_hz'][idx-25:idx+25].mean(0)
            meta=dict(source=str(path),window_ms=[int(z['time_ms'][idx-25]),int(z['time_ms'][idx+24])],D=.1887)
        bx=fig.add_subplot(gs[j,1]);im=spatial(bx,field,s)
        bx.text(-.22,1.09,chr(66+j),transform=bx.transAxes,fontsize=17,fontweight='bold')
        bx.text(.5,1.04,f'{j+1}    D = {meta["D"]:.6f}',transform=bx.transAxes,ha='center',fontsize=11)
        fields.append(meta)
    fig.subplots_adjust(right=.90)
    fig.colorbar(im,cax=fig.add_axes([.925,.18,.014,.62]),label='E rate (Hz / neuron)')
    out=AUDIT/'figures';out.mkdir(exist_ok=True)
    name='fig_zm_periodic_bifurcation_spatial_reviewed'
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)
    write(out/f'{name}_metadata.json',dict(
        model='Frozen 935-group, 1 mm spatial rate DDE',J_EE_core=1,Z='held spatial path',M='dynamic',
        D_domain=[0,1],folds=folds,selected_Floquet=selected,spatial_panels=fields,
        periodic_segments=[dict(left=a['_id'],right=b['_id'],status=status)
                           for a,b,status in zip(branch[:-1],branch[1:],styles)],
        line_rule='Equal-status sampled anchors joined only without an intervening known fold or contradictory sample. Not exhaustive stability proof.',
        unknown_regions='Includes small-fold neighborhood, approach to each fold, and the unclassified remote return branch.',
        high_Hopf_periodic_curve='Unclassified geometry with stable A16 sample only; local supercritical Hopf retained',
        equilibrium_style='Same prior sampled classification; high-D stable line supported by representative contour counts',
        audit_source=str(AUDIT/'review_status.json'),SNN_equivalence='NOT_VALIDATED',
        bifurcation_completeness='INCOMPLETE',human_visual_acceptance='PENDING',
        numerical_model_or_parameter_changes=False))
    print(out/f'{name}.png')


if __name__=='__main__':
    main()
