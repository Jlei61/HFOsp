#!/usr/bin/env python3
"""Compare the bounded immediate-clamp density checks with completed native data."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time,shutil
import numpy as np
from campaign import ROOT,read,write,sha
from density_bracket_protocol import OUT,SOURCE,NAMES
from coupled_density_exit import ADAPTED
from analyze_actual_G_history import load as native_load
from analyze_high_state_continuation import load as density_load


def main(wait):
    dest=OUT/'comparison';dest.mkdir(exist_ok=True)
    assert not (dest/'result.json').exists()
    while True:
        ready=all((OUT/n/'result.json').exists() for n in NAMES)
        native_ready=(SOURCE/'extended_analysis_summary.json').exists()
        if ready and native_ready:break
        if (OUT/'status.json').exists() and read(OUT/'status.json')['status']=='FAILED':
            write(dest/'progress.json',dict(status='STOPPED_ON_WORKER_FAILURE'));return
        write(dest/'progress.json',dict(status='WAITING_COMPLETE_DENSITY_AND_NATIVE',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(SOURCE/'extended_analysis_summary.json')['status']=='COMPLETE'
    geo=dict(np.load(ADAPTED/'geometry.npz'));E=geo['population']==0
    counts=np.bincount(geo['group_cell'][E],weights=geo['group_size'][E],minlength=400)
    rows=[];data={}
    for name in NAMES:
        assert read(OUT/name/'result.json')['status']=='COMPLETE'
        n=native_load(SOURCE,name);d=density_load(OUT/name,geo);data[name]=(n,d)
        windows=[]
        for lo,hi in [(0,5),(5,10)]:
            nm=(n['time5']>=lo)&(n['time5']<hi)
            ng=(n['time1']>=lo)&(n['time1']<hi)
            nd=(n['drift_time']>lo)&(n['drift_time']<=hi)
            dm=(d['time_s']>lo)&(d['time_s']<=hi)
            assert nm.sum()==(hi-lo)*200 and dm.sum()==(hi-lo)*1000
            nf=n['field'][nm].mean(0);df=d['field_Hz'][dm].mean(0)
            windows.append(dict(interval_s=[lo,hi],native_rates_Hz=n['rate'][nm].mean(0).tolist(),
                density_rates_Hz=d['rate_Hz'][dm].mean(0).tolist(),
                native_Graw=float(n['G'][ng].mean()),density_Graw=float(d['Graw'][dm].mean()),
                native_counterfactual_dZ_per_s=n['drift'][nd,:,0].mean(0).tolist(),
                density_counterfactual_dZ_per_s=d['drift_per_s'][dm].mean(0).tolist(),
                spatial_field_weighted_RMS_Hz=float(np.sqrt(np.average((nf-df)**2,weights=counts)))))
        tail=(n['time5']>=20)&(n['time5']<30);earlier=(n['time5']>=5)&(n['time5']<10)
        nf=n['field'][tail].mean(0);ef=n['field'][earlier].mean(0)
        row=dict(name=name,K=n['job']['target_K'],windows=windows,
            native_tail20to30s_rates_Hz=n['rate'][tail].mean(0).tolist(),
            native_field5to10_vs20to30_RMS_Hz=float(np.sqrt(np.average((nf-ef)**2,weights=counts))),
            native_complete_30s=True,density_complete_10s=True)
        rows.append(row);print('IMMEDIATE PROTOCOL COMPARISON',row,flush=True)
        np.savez_compressed(dest/f'{name}.npz',native_time_s=n['time5'],native_rates_Hz=n['rate'],
            native_field_Hz=n['field'],native_time1_s=n['time1'],native_Graw=n['G'],
            density_time_s=d['time_s'],density_rates_Hz=d['rate_Hz'],density_field_Hz=d['field_Hz'],density_Graw=d['Graw'])
    result=dict(status='COMPLETE',rows=rows,
        protocol='Native and new density begin from the same declared12s physical initial state, sameheldZ/K fields, and same50-60s externalexpectedrate process. Their microscopic stochastic inputs differ. Previous density ramp uses a carriedK9 state andconstant expectedexternalmean; its difference cannot be attributed solely to approach history.',
        statistical_unit='One native internal history/future input; twoK interventions; one paired128replica density numerical stream. Not independent native seeds.',
        gate='No posthoc RMS acceptance threshold. Report remaining mismatch explicitly before any formal continuation.',
        formal_bifurcation_allowed=False,counts_as_autonomous_loop=False,producer_sha256=sha(__file__))
    write(dest/'result.json',result)
    plot(data,dest)
    write(dest/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))


def plot(data,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    name=NAMES[0];n,d=data[name]
    path=ROOT/'carried_exit_lower_holds/analysis/held_K9p35.npz'
    with np.load(path) as z:h={k:z[k] for k in z.files}
    series=[dict(title='Native: immediate K clamp',t=n['time5'],r=n['rate'],tg=n['time1'],g=n['G'],
                 field=n['field'][(n['time5']>=5)&(n['time5']<10)].mean(0),window=[5,10]),
            dict(title='Density: same initial history',t=d['time_s'],r=d['rate_Hz'],tg=d['time_s'],g=d['Graw'],
                 field=d['field_Hz'][5000:10000].mean(0),window=[5,10]),
            dict(title='Density: K ramp from held high state',t=h['time_s'],r=h['rate_Hz'],tg=h['time_s'],g=h['Graw'],
                 field=h['field_Hz'][-3000:].mean(0),window=[float(h['time_s'][-1]-3),float(h['time_s'][-1])])]
    colors=['#8a63b4','#d63378','#008eb3'];labels=['All E','Core A','Core B']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(13,9),layout='constrained')
    grid=fig.add_gridspec(3,4,width_ratios=[1,1,1,.035],height_ratios=[1,.65,1.3])
    geometry=np.load(SOURCE/'geometry.npz');centers=geometry['centers_mm'];radius=float(geometry['core_radius_mm'])
    for col,s in enumerate(series):
        ax=fig.add_subplot(grid[0,col]);step=4 if col==0 else 20
        nr=len(s['t'])//step*step;t=s['t'][:nr].reshape(-1,step).mean(1)
        r=s['r'][:nr].reshape(-1,step,4).mean(1)
        for j in range(3):ax.plot(t,r[:,j],color=colors[j],lw=1,label=labels[j])
        ax.set(title=s['title'],xlim=(0,10.4),ylim=(-5,510),xticks=[0,5,10],xlabel='Elapsed time (s)')
        ax.axvspan(*s['window'],color='.5',alpha=.08)
        if col==0:ax.set_ylabel('Rate (Hz)');ax.legend(frameon=False,fontsize=8,loc='center right',bbox_to_anchor=(.99,.50))
        ax=fig.add_subplot(grid[1,col]);ax.plot(s['tg'],s['g'],color='#b27228',lw=1.1)
        ax.set(xlim=(0,10.4),ylim=(-.05,5),xticks=[0,5,10],xlabel='Elapsed time (s)')
        if col==0:ax.set_ylabel(r'$G_{\rm raw}$')
        ax=fig.add_subplot(grid[2,col]);im=ax.imshow(s['field'].reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=500,interpolation='nearest')
        for label,xy in zip(['A','B'],centers):
            ax.add_patch(Circle(xy,radius,fill=False,edgecolor='#00c3c5',lw=1))
            ax.text(xy[0],xy[1]+1.75,label,ha='center',color='#00c3c5',fontsize=8)
        ax.set(title=f"Mean field: {s['window'][0]:g}–{s['window'][1]:g} s",xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if col==0:ax.set_ylabel('y (mm)')
    fig.colorbar(im,cax=fig.add_subplot(grid[2,3]),label='E rate (Hz)')
    fig.suptitle(r'Conditional exit-field check: mean $Z=0.21$, held mean $K=9.35$',weight='bold')
    fig.text(.5,-.025,'First two columns: same initial physical history and expected external input. Third: carried high state and constant expected input.\nHeld Z/K controls; these finite trajectories do not certify a bifurcation or autonomous recovery.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'native_density_K9p35_protocol.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    shutil.copy2(__file__,dest/'figure_producer.py')
    write(dest/'figure_metadata.json',dict(producer_sha256=sha(__file__),result_sha256=sha(dest/'result.json'),
        fields_and_windows=[dict(title=s['title'],window=s['window'],field_Hz=s['field'].tolist()) for s in series],
        circle_semantics='Original substrate core landmarks, radius1.5mm from geometry. Rate/core-budget groups retain original1.75mm nearest-core selection; circles are not analysis aperture boundaries.',
        agent_visual='PENDING',human_review='PENDING',formal_Fig5_replaced=False))
    readme=ROOT/'figures/README.md';heading='### native_density_K9p35_protocol.png / .svg'
    if heading not in readme.read_text():
        with readme.open('a') as f:f.write('\n'+heading+'\n固定实际退出场族的Z=0.21、K=9.35，比较原生直接钳制、从同一初始物理状态出发的密度模型，以及此前从K9高态缓慢升K的密度轨迹。上排保留全E与两核的活动，中排为G，下排以相同色标展示注明时间窗的原空间场；第三列还使用常值外源均值，故差异不能单独归因于历史。青圈为原底物1.5mm核位置标记，率及预算沿用原1.75mm近核分组，青圈不表示统计口径边界。全部为条件有限窗检查，不作为自主恢复或正式分岔认证。\n**关注点**：同参数下核心和外围的空间招募是否对应，以及此前求得的高态是否解释原生实际到达的状态。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
