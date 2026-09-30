#!/usr/bin/env python3
"""Full-window native versus distribution approximation, without promotion."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from campaign import ROOT, REPO, read, write, sha
sys.path.insert(0,str(REPO/'scripts/topic4_zm_runaway_mechanism/frozen_v3'))
from native_readouts import readouts, window_stats

OUT=ROOT/'density_spatial_resolution'
BASE=REPO/'results/topic4_sef_hfo/fig5_zm_rate_v3_20260918'
REPLAY=REPO/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay'
WINDOWS=[(500,3000),(1000,4000),(4000,8000),(8000,9420),(1000,9420)]


def sources():
    rows=[]
    count=np.load(REPLAY/'geometry.npz')['cell_e_counts']
    for seed in [9108401,9108402]:
        p=BASE/f'native_reference/seed{seed}_readouts.npz'
        with np.load(p) as z:
            t=z['t'];field=z['rate_cells'].astype(float)
        slow=[];zz=[]
        for f in sorted((REPLAY/f'runs/eta0.0005_s{seed}/chunks').glob('*.npz')):
            with np.load(f) as z:slow.append(z['slow_time_ms']);zz.append(z['Z'][:,0])
        rows.append(dict(name=f'native{seed}',label=f'Native {str(seed)[-4:]}',kind='native',
            source=str(p),t=t,field=field,counts=count,zt=np.concatenate(slow),Z=np.concatenate(zz)))
    for R,seed,folder in [(2048,927611,ROOT/'density_spatial_onset/R2048_num927611'),
                           (2048,927612,ROOT/'density_spatial_onset/R2048_num927612'),
                           (8192,927611,OUT)]:
        assert read(folder/'result.json')['status']=='COMPLETE'
        with np.load(folder/'trajectory.npz') as z:
            assert np.array_equal(count,z['cell_counts'])
            t=z['time_ms'];field=z['field_E_Hz'].astype(float)
            meanz=np.average(z['group_Z'][:,z['population_E']],weights=z['group_sizes'][z['population_E']],axis=1)
        rows.append(dict(name=f'R{R}_num{seed}',label=f'Density {R}, stream {seed%10}',kind='density',
            source=str(folder/'trajectory.npz'),t=t,field=field,counts=count,zt=t,Z=meanz))
    return rows


def compare():
    rows=sources();result=[];nativeD=read(BASE/'native_reference/checkpoint_projections.json')['9870']['D']
    for row in rows:
        ev,s,rate,sm=readouts(row['t'],row['field'],row['counts'],row['name'])
        row.update(events=ev,summary=s,sm=sm)
        quiet_bounded=[]
        for e in ev:
            a=int(np.searchsorted(row['t'],e['start_ms']));b=a+int(e['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():quiet_bounded.append(e)
        primary=window_stats(ev,1000,9420);n=primary['n']
        D=float(1-np.interp(9870,row['zt'],row['Z']))
        gate=dict(self_limited_events=bool(n and 50<=primary['median_duration_ms']<=200 and s['quiet_fraction']>=.15),
            two_core_participation=bool(n and primary['both_cores']/n>=.5),
            surround_recruitment=bool(n and .3<=primary['median_area']<=1.),
            propagation=bool(n and primary['forward']>0 and primary['reverse']>0 and 5<=primary['median_extent_mm']<=20),
            entry=bool(s['high_onset_ms'] is not None and 7000<=s['high_onset_ms']<=13000),
            D_track=bool(abs(D-nativeD)<=.05))
        result.append(dict(name=row['name'],kind=row['kind'],source=row['source'],summary=s,
            original_A4_checks=gate,n_original_A4_pass=sum(gate.values()),D9870=D,
            windows={f'{a}-{b}':window_stats(ev,a,b) for a,b in WINDOWS},
            complete_quiet_bounded_windows={f'{a}-{b}':window_stats([e for e in quiet_bounded if e['start_ms']+e['duration_ms']<=b],a,b) for a,b in WINDOWS},
            events_crossing_primary_window_end=sum(1000<=e['start_ms']<9420 and e['start_ms']+e['duration_ms']>9420 for e in ev),
            quiet_by_window={f'{a}-{b}':float((sm[(row['t']>=a)&(row['t']<b)]<5).mean()) for a,b in WINDOWS},
            events=[{k:v for k,v in e.items() if k!='onset'} for e in ev]))
    native_delta=abs(result[1]['summary']['high_onset_ms']-result[0]['summary']['high_onset_ms'])
    stream_delta=abs(result[3]['summary']['high_onset_ms']-result[2]['summary']['high_onset_ms'])
    resolution_onsets=[result[j]['summary']['high_onset_ms'] for j in [2,4]]
    resolution_delta=None if None in resolution_onsets else float(resolution_onsets[1]-resolution_onsets[0])
    write(OUT/'comparison.json',dict(status='COMPLETE',rows=result,created_epoch=time.time(),producer_sha256=sha(__file__),
        original_contract=str(BASE/'a4_contract.json'),native_D9870_from_exact_checkpoint=nativeD,
        entry_differences_ms=dict(two_existing_native_seeds=native_delta,two_2048_numerical_streams=stream_delta,
                                  nested_8192_minus_2048=resolution_delta),
        scope='Same native event observer and unchanged original broadA4criteria; actual native seed differences also retained. Density streams are numerical quadrature realizations under one recorded external forcing, not independent native seeds. A single4xresolution comparison is not an extrapolated zero-noise density limit.',
        time_convention='Native1msrate bins are labelled centers .5,1.5,...; density bins retain recorded end labels1,2,... . No event matching is imposed; subms label difference does not explain a seconds-scale onset difference.',
        contact_validation='Separate identical raw-current observer replay; gated-current trajectory not used as native rawLFP.',
        model_promoted=False,formal_bifurcation_allowed=False,human_review='PENDING'))
    plot(rows)
    print([(r['name'],r['summary']['high_onset_ms'],r['D9870'],r['n_original_A4_pass']) for r in result],flush=True)


def plot(rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,7.3),layout='constrained')
    colors=['#272727','#8e8e8e','#9968b4','#cbb5d9','#167d91']
    styles=['-','--','-','--','-']
    for j,(row,c,ls) in enumerate(zip(rows,colors,styles)):
        axs[0,0].plot(row['t']/1000,row['sm'],c=c,ls=ls,lw=1,label=row['label'])
        axs[0,1].plot(row['zt']/1000,row['Z'],c=c,ls=ls,lw=1.4)
        onset=row['summary']['high_onset_ms']
        if onset is not None:axs[0,0].plot(onset/1000,200,'o',c=c,ms=4)
        ev=[e for e in row['events'] if 1000<=e['start_ms']<9420]
        for ax,key in zip(axs[1],['duration_ms','area_fraction']):
            val=np.array([e[key] for e in ev]);offset=np.linspace(-.18,.18,len(val))
            crossing=np.array([e['start_ms']+e['duration_ms']>9420 for e in ev])
            ax.scatter((j+offset)[~crossing],val[~crossing],s=12,color=c,alpha=.55,edgecolor='none')
            ax.scatter((j+offset)[crossing],val[crossing],s=32,facecolor='none',edgecolor=c,marker='^')
            ax.plot([j-.25,j+.25],[np.median(val)]*2,color=c,lw=2.5)
    axs[0,0].set(xlim=(0,12.5),ylim=(0,510),xlabel='Time (s)',ylabel='All-E rate (Hz)',title='A  Interictal activity and entry')
    axs[0,1].set(xlim=(0,12.5),ylim=(.3,1.01),xlabel='Time (s)',ylabel='Mean Z',title='B  Resource depletion')
    for ax,title,ylabel in zip(axs[1],['C  Event durations','D  Recruited area'],['Duration (ms)','E-cell fraction']):
        ax.set(title=title,ylabel=ylabel,xticks=range(5),xticklabels=['Native\n8401','Native\n8402','2048\nstream1','2048\nstream2','8192\nstream1'])
    axs[1,0].set_yscale('log');axs[1,0].set_yticks([30,100,300,1000,3000],labels=['30','100','300','1000','3000'])
    axs[1,0].text(.97,.95,'△ extends beyond 9.42 s',transform=axs[1,0].transAxes,ha='right',va='top',fontsize=8)
    for ax in axs.ravel():ax.grid(axis='y',alpha=.15)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='outside upper center',frameon=False)
    fig.supxlabel('Distribution approximation with original Z/M; added G/K disabled. Event starts: 1–9.42 s; triangles cross the window end.\nDensity streams and particle counts describe numerical sampling, not native seed replication.',fontsize=9)
    folder=ROOT/'figures'
    for suffix in ['png','svg']:fig.savefig(folder/f'density_baseline_correspondence.{suffix}',dpi=180)
    plt.close(fig)
    write(folder/'density_baseline_correspondence_metadata.json',dict(source=str(OUT/'comparison.json'),
        producer_sha256=sha(__file__),agent_visual_review='PENDING',human_review='PENDING',formal_bifurcation=False,
        event_censoring='OriginalA4uses start-time inclusion. Triangles mark events continuing beyond9.42s, including native onset-connected activity; these are not completed interictal events. Separate quiet-bounded complete-window statistics retained in comparisonJSON.'))
    p=folder/'README.md';text=p.read_text();marker='### density_baseline_correspondence.png / density_baseline_correspondence.svg\n'
    section=marker+'原生两条参考与保留局部状态分布的新近似模型比较：上排为完整进入前后全E活动及Z耗竭，下排为1–9.42秒全部事件的持续时间和招募范围。两条2048粒子轨迹使用不同数值流，8192与第一条共享嵌套数值流；所有近似轨迹使用同一记录外源，且本图G/K关闭，尚不检验闭环。\n**关注点**：分辨率改变是否改变进入时间，原生与近似的Z消耗、短事件形态是否存在系统差异；原六项宽容差通过不能代替完整空间和闭环对应。\n'
    if marker in text:
        before,after=text.split(marker,1);tail=after.find('\n### ');text=before+section+(after[tail:] if tail>=0 else '')
    else:text=text.rstrip()+'\n\n'+section
    p.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
    while args.wait and not (OUT/'result.json').exists():time.sleep(30)
    compare()
