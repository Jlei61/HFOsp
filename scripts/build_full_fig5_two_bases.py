#!/usr/bin/env python3
"""Complete original A-F figure for each fixed GIF substrate; no historical curves reused."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.util
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import recover_fig5_two_gif_bases as recovery
from src.topic4_xy_fig5_followup import read,write,sha
ZROOT=ROOT.parent/'topic4-dual-core-z-bifurcation'
import src
src.__path__.append(str(ZROOT/'src'))
from src.topic4_patient_zm_meanfield import build_patient_coarse_model,save_patient_coarse_model
from src.topic4_dual_core_spatial_z import build_dual_core_z_map,path_state,regional_rates_hz
from src.topic4_dual_core_spatial_z_delay import build_coarse_delay_operators,simulate_delayed_ou_trajectory

DATA=ROOT/'results/topic4_sef_hfo/fig5_full_two_bases'
FIG=ROOT/'results/paper-ready-figure/fig5_full_two_bases/figures'
WORK={}


def map_cell(task):
    s,gain,prep=task;w=WORK;out=w['out']/f's{s:.2f}_m{gain:g}_{prep}.json'
    if out.exists():return read(out)
    model,zmap,ops=w['model'],w['zmap'],w['ops']
    za,zb,zs,z=path_state(zmap,s)
    result=simulate_delayed_ou_trajectory(model,ops,np.full(2*model.n_cells,.005 if prep=='low' else .45),
        z_field=z,z_second_moment=zmap.z_second_moment_field(z_a=za,z_b=zb,z_surround=zs),
        ou_rate_e=w['zero_ou'],tail_steps=10000,eta_m=w['eta']*gain,tau_m_slow_ms=500.)
    tail=result['mean_e_rate_hz'][-10000:];drift=float(tail[5000:].mean()-tail[:5000].mean())
    regional=regional_rates_hz(model,zmap,result['tail_mean_rates'][:model.n_cells])
    mean=float(tail.mean());stationary=abs(drift)<=5
    state='tonic_runaway' if stationary and mean>=300 and min(regional.values())>=250 else ('bounded' if stationary and mean<250 else 'unresolved')
    row=dict(s=s,m_gain_scale=gain,preparation=prep,population_hz=mean,regional_hz=regional,state=state,half_tail_drift_hz=drift)
    row['arrays']=recovery.old.save_arrays(out.with_suffix('.npz'),dict(time_ms=np.arange(len(result['mean_e_rate_hz']))*.1,population_hz=result['mean_e_rate_hz'],M=result['mean_adaptation_state']))
    write(out,row);return row


def maps(cid):
    out=DATA/cid;out.mkdir(parents=True,exist_ok=True)
    substrate,cfg,regions,f,job,p=recovery.build(cid)
    model=build_patient_coarse_model(substrate,n_grid=10,threshold_groups=8)
    if model.tau_gaba_ms!=p['candidate']['dynamic_parameters']['tau_d_GABA_ms']:raise RuntimeError('GABA kinetics lost in reduction')
    zmap=build_dual_core_z_map(model,substrate.positions_e,substrate.h_e,p['candidate']['node_field']['centers_mm'])
    ops=build_coarse_delay_operators(substrate,model)
    if ops.dt_ms!=.1:raise RuntimeError('coarse native dt mismatch')
    model_record=save_patient_coarse_model(out/'coarse_model.npz',model)
    contract=dict(s_values=[round(x/10,1) for x in range(10)],m_gain_scales=[0.,1.,2.,4.],
        preparations=['low','high'],duration_ms=10000.,tail_ms=1000.,dt_ms=.1,
        preparation_contract='Uniform 5 Hz or 450 Hz E/I rates; synapses/history and M initialized consistently from rates. Finite initial-condition response, not equilibrium continuation.',
        z_path='Zcore1=Zcore2=1-s; Zsurround=1-0.7s. Frozen Z, dynamic M; zero added OU.',
        population_definition='Unweighted mean coarse E rate, matching original deterministic gate.',
        model=model_record,substrate_fingerprint=f,
        code_sources={str(q):sha(q) for q in [Path(__file__),ZROOT/'src/topic4_patient_zm_meanfield.py',ZROOT/'src/topic4_dual_core_spatial_z.py',ZROOT/'src/topic4_dual_core_spatial_z_delay.py']})
    write(out/'map_contract.json',contract)
    cells=out/'map_cells';cells.mkdir(exist_ok=True)
    WORK.update(model=model,zmap=zmap,ops=ops,out=cells,eta=job['config']['eta_m'],zero_ou=np.zeros((100000,model.n_cells),np.float32))
    tasks=[(s,g,p) for s in contract['s_values'] for g in contract['m_gain_scales'] for p in contract['preparations']]
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('fork')) as pool:
        for future in as_completed([pool.submit(map_cell,t) for t in tasks]):
            rows.append(future.result());write(out/'map_progress.json',dict(completed=len(rows),total=len(tasks)))
    write(out/'map_summary.json',dict(config=contract,rows=rows,status='COMPLETE'))


def spatial(cid):
    out=DATA/cid;out.mkdir(parents=True,exist_ok=True)
    r=read(recovery.DATA/cid/'trajectory.json')
    if not r['trajectory']['qualified_pretransition']:raise RuntimeError('full figure requires qualified transition')
    with np.load(r['arrays']['path']) as a:trajectory={k:a[k] for k in a.files}
    from scipy.signal import find_peaks
    rates=trajectory['rates_hz'][:,0];times=trajectory['time_ms'];onset=r['trajectory']['earliest_regional_recruitment_ms']
    peaks,_=find_peaks(rates,prominence=5,distance=5)
    peaks=[i for i in peaks if 600<=times[i]<min(2000,onset-500) and rates[i]<250 and np.min(rates[i+1:i+16])<.2*rates[i]]
    if not peaks:raise RuntimeError('no early returned population excursion for panel C')
    peak=peaks[0];windows={'interictal':[float(times[peak]-100),float(times[peak]+100)],'pre_onset':[onset-250,onset-50],'early_runaway':[onset,onset+200]}
    write(out/'spatial_contract.json',dict(windows_ms=windows,selection='First returned population-rate peak after 600 ms and before 2 s; diagnostic excursion, no TA/TB label.'))
    s,cfg,regions,f,job,p=recovery.build(cid)
    fields={};checks={}
    result,slow=recovery.old.simulate(s,cfg,regions,job,duration=windows['interictal'][1])
    if not np.array_equal(result['lfp_trace'].reshape(-1,10,len(s.contact_names)).mean(1),trajectory['lfp'][:len(result['lfp_trace'])//10]):raise RuntimeError('interictal replay changed')
    start=int(round(windows['interictal'][0]/.1));fields['interictal']=result['E_spk_bool'][start:].sum(0)*5.
    del result,slow
    state=recovery.old.load_checkpoint(r['checkpoints']['pre_onset']['path'])
    result,slow=recovery.old.simulate(s,cfg,regions,job,resume=state,duration=450.)
    begin=int(round((onset-250)/1.))
    current=result['lfp_trace'].reshape(-1,10,len(s.contact_names)).mean(1)
    if not np.array_equal(current,trajectory['lfp'][begin:begin+450]):raise RuntimeError('transition spatial replay changed')
    fields['pre_onset']=result['E_spk_bool'][:2000].sum(0)*5.
    fields['early_runaway']=result['E_spk_bool'][2500:4500].sum(0)*5.
    arrays=dict(positions_E=s.positions_e,contact_xy=s.contact_xy,**fields)
    record=dict(windows_ms=windows,replay_exact=True,arrays=recovery.old.save_arrays(out/'spatial.npz',arrays))
    write(out/'spatial.json',record)


def producer():
    # Reuse original A/B, shading, geometry, panel titles and F rendering.
    import scripts.paper_figures
    scripts.paper_figures.__path__.append(str(ZROOT/'scripts/paper_figures'))
    spec=importlib.util.spec_from_file_location('original_fig5',ZROOT/'scripts/paper_figures/build_fig5_transition_susceptibility.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def render(cid):
    original=producer();plt=original.plt
    from matplotlib.colors import PowerNorm,SymLogNorm
    out=DATA/cid;r=read(recovery.DATA/cid/'trajectory.json');p=read(recovery.DATA/cid/'protocol.json')
    a=original.load_npz(r['arrays']['path']);sp=read(out/'spatial.json');sa=original.load_npz(sp['arrays']['path'])
    pr=read(recovery.DATA/cid/'probe.json');pa=original.load_npz(pr['arrays']['path']);summary=read(out/'map_summary.json')
    for record in [r,sp,pr]:
        if sha(record['arrays']['path'])!=record['arrays']['sha256']:raise RuntimeError('figure input changed')
    onset=r['trajectory']['earliest_regional_recruitment_ms'];centers=np.asarray(p['candidate']['node_field']['centers_mm'])
    stages=dict(onset_ms=onset,display_ms=[0,min(15000,onset+2500)],returned_interictal_ms=[sp['windows_ms']['interictal']],pre_onset_ms=sp['windows_ms']['pre_onset'])
    adapted=dict(a,transition_lfp_trace=a['lfp'],transition_lfp_dt_ms=1.,transition_spatial_frame_time_ms=a['time_ms'],transition_rate_E_hz_20ms=a['rates_hz'][:,0])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig=plt.figure(figsize=(14.2,10.4),facecolor='white')
    fig.text(.06,.985,'From recurrent interictal events to tonic runaway',fontsize=16,fontweight='bold',va='top')
    fig.text(.06,.957,f"Fixed GIF base: {cid} · GABA decay {p['candidate']['dynamic_parameters']['tau_d_GABA_ms']:g} ms",fontsize=9)
    original.title(fig,(.065,.924),'A','Spontaneous transition in the two-core SNN')
    ma=original.plot_a(fig.add_axes([.065,.65,.485,.237]),adapted,stages)
    ma['filter']='1-ms box average of native current proxy; no bandpass'
    original.title(fig,(.065,.604),'B','Population activity and inhibitory state')
    bax=[fig.add_axes([.065,y,.485,.058]) for y in [.509,.439,.369]]
    original.plot_b(bax,adapted,stages,p['job']['config']['eta_m']);bax[0].set_ylim(0,max(480,float(a['rates_hz'][:,0].max())*1.03));bax[1].set_ylim(0,1.04)
    original.title(fig,(.62,.924),'C','Spatial recruitment along the same trajectory')
    edges=np.arange(21);pos=sa['positions_E'];occupancy=np.histogram2d(pos[:,0],pos[:,1],bins=(edges,edges))[0]
    maps=[]
    for name in ['interictal','pre_onset','early_runaway']:
        counts=np.histogram2d(pos[:,0],pos[:,1],bins=(edges,edges),weights=sa[name])[0]
        maps.append(np.divide(counts,occupancy,out=np.zeros_like(counts),where=occupancy>0))
    vmax=max(x.max() for x in maps)
    for i,(field,label) in enumerate(zip(maps,['Interictal','Pre-onset','Early runaway'])):
        ax=fig.add_axes([.62+i*.115,.673,.095,.207]);im=ax.imshow(field.T,origin='lower',extent=(0,20,0,20),cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=vmax));original.geometry(ax,sa['contact_xy'],centers);ax.set_title(label,fontsize=9)
    fig.colorbar(im,cax=fig.add_axes([.957,.75,.009,.13])).set_label('E rate (Hz)',fontsize=8)
    original.title(fig,(.62,.604),'D','Paired stimulation-site susceptibility')
    low=pa['reference_full_field'].sum(1);pre=pa['pre_onset_full_field'].sum(1);v=max(1,np.abs(np.r_[low,pre]).max());norm=SymLogNorm(linthresh=1,vmin=-v,vmax=v)
    for i,(values,label) in enumerate(zip([low,pre],['Reference','Pre-onset'])):
        ax=fig.add_axes([.62+i*.128,.394,.106,.16]);original.geometry(ax,sa['contact_xy'],centers);im=ax.scatter(pa['sites_mm'][:,0],pa['sites_mm'][:,1],c=values,cmap='RdBu_r',norm=norm,s=55,zorder=7,edgecolor='.4',lw=.5);ax.set_title(label,fontsize=9)
    ax=fig.add_axes([.901,.394,.059,.16])
    for x,y in zip(low,pre):ax.plot([0,1],[x,y],color='.7',lw=.7)
    ax.scatter(np.zeros(9),low,s=12,color=original.COL['z']);ax.scatter(np.ones(9),pre,s=12,color=original.COL['pre']);ax.set_xticks([0,1],['Ref.','Pre']);ax.set_yscale('symlog',linthresh=1);ax.axhline(0,color='.5',lw=.5);ax.set_title('9 sites',fontsize=9);original.style(ax)
    fig.colorbar(im,cax=fig.add_axes([.637,.353,.198,.008]),orientation='horizontal').set_label('Extra descendant spikes · 0–200 ms',fontsize=8)
    original.title(fig,(.065,.293),'E','Recruitment as inhibition weakens')
    ax=fig.add_axes([.065,.075,.485,.176])
    for prep,ls in [('low','-'),('high','--')]:
        rows=sorted([x for x in summary['rows'] if x['m_gain_scale']==1 and x['preparation']==prep],key=lambda x:x['s'])
        ax.plot([x['s'] for x in rows],[x['population_hz'] for x in rows],ls+'o',ms=3,label=f'{prep.capitalize()} initial rates')
    ax.axhline(300,color='.6',lw=.6,ls=':');ax.set(xlabel='Core disinhibition, s = 1 − Zcore',ylabel='Terminal E rate (Hz)',xlim=(0,.9),ylim=(0,510));ax.legend(fontsize=8,frameon=False);original.style(ax)
    original.title(fig,(.62,.293),'F','Adaptation and initial-state dependence')
    fax=fig.add_axes([.62,.095,.34,.157]);original.plot_f(fax,summary);fax.set_xticks([0,.2,.4,.6,.8])
    FIG.mkdir(parents=True,exist_ok=True);stem=FIG/('fig5-original-layout-'+cid)
    outputs=original.save_all(fig,stem)
    write(stem.with_suffix('.metadata.json'),dict(candidate_id=cid,outputs=outputs,author_accepted=False,
        panel_A=ma,panel_C=sp,panel_D=pr,panel_E_F=summary['config'],
        sources={str(q):sha(q) for q in [out/'spatial.json',out/'map_summary.json',recovery.DATA/cid/'trajectory.json',recovery.DATA/cid/'probe.json',Path(__file__),ZROOT/'scripts/paper_figures/build_fig5_transition_susceptibility.py']},
        boundary='A-D one fixed full SNN trajectory. E-F separately rebuilt deterministic coarse model, frozen spatial Z and dynamic M. Finite 10 s outcomes; lines are not equilibrium branches. No old fold location reused.'))
    readme=FIG/'README.md'
    entries=[]
    for name in recovery.IDS:
        if (FIG/('fig5-original-layout-'+name+'.png')).exists():entries.append(f'### fig5-original-layout-{name}.png\n沿用原版 A–F 布局，A/B 是本基底的同轨迹读出和慢变量，C 为同轨迹空间活动，D 为精确 sham 配对扰动。E/F 从本基底的实际连接、阈值及 GABA 时间常数独立重建简化模型，展示固定空间 Z 与动态 M 的有限初值响应。\n**关注点**：E/F 不是完整 SNN 的分叉证明，旧底物阈值没有搬用；D 保留全部正负响应，待作者目视验收。\n')
    readme.write_text('\n'.join(entries))


def finish():
    while not all((recovery.DATA/c/'probe.json').exists() for c in recovery.IDS):
        status=subprocess.run(['systemctl','--user','is-active','--quiet','fig5-two-gif-bases-recovery-20260907.service'])
        if status.returncode:raise RuntimeError('upstream stopped before both probes completed')
        time.sleep(30)
    jobs=[]
    for cid in recovery.IDS:
        if not (DATA/cid/'spatial.json').exists():jobs.append(subprocess.Popen([sys.executable,__file__,'spatial',cid]))
    if any(p.wait()!=0 for p in jobs):raise RuntimeError('spatial replay failed')
    while not all((DATA/c/'map_summary.json').exists() for c in recovery.IDS):
        for i in [1,2]:
            if subprocess.run(['systemctl','--user','is-active','--quiet',f'fig5-full-map-base{i}-20260907.service']).returncode and not (DATA/recovery.IDS[i-1]/'map_summary.json').exists():raise RuntimeError('map service failed')
        time.sleep(30)
    for cid in recovery.IDS:render(cid)
    write(DATA/'completion.json',dict(status='TWO_FULL_FIGURES_RENDERED_PENDING_VISUAL_REVIEW',author_accepted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['maps','spatial','render','finish']);parser.add_argument('candidate',nargs='?',choices=recovery.IDS);args=parser.parse_args()
    if args.mode=='finish':finish()
    else:globals()[args.mode](args.candidate)
