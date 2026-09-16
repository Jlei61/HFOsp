#!/usr/bin/env python3
"""Mechanism readout for the k100 x tau_K matrix: currents, Z, K, M by region.

Reads committed chunks only (never touches simulation state). Produces per-run
current-balance windows, time-series figures and the matrix summary table.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_topic4_interictal_recurrence as audit
import run_topic4_k100_recurrence as run

old=audit.old
OUT=run.OUT
E_K=-30.


def load_run(folder):
    folder=Path(folder)
    d=old.load(folder,keys=['time_ms','spikes_1ms','regions_1ms','slow_time_ms','Z','M','currents','regional_currents','field_time_ms','field_5ms'])
    if not d:return None
    d['mechanism']=old.load(folder,'mechanism_chunks')
    d['intrinsic']=old.load(folder,'intrinsic_adaptation_chunks')
    reg=old.load(folder,'regional_chunks') if (folder/'regional_chunks').exists() else {}
    if reg:   # per-chunk metadata arrays were concatenated by the generic loader; restore the fixed schema
        reg['keys']=np.array(run.REGIONAL_KEYS);reg['region_names']=np.array(run.REGION_NAMES)
        assert reg['values'].shape[1:]==(3,len(run.REGIONAL_KEYS))
    d['regional']=reg
    applied=json.loads((folder/'applied_configuration.json').read_text());job=applied['job']
    job.setdefault('sahp_tau_s',(applied.get('added_sahp_tau_ms') or 5000.)/1000.)
    job.setdefault('k100',job['sahp_gain']*job['sahp_tau_s'])
    d['job']=job
    with np.load(OUT/'geometry.npz') as g:d['geo']={k:g[k] for k in g.files}
    return d


def rates(d,ms=10):
    n=len(d['spikes_1ms'])//ms;nr=np.r_[32000,d['geo']['region_counts'][:3]]
    all_e=d['spikes_1ms'][:n*ms,0].reshape(n,ms).sum(1);local=d['regions_1ms'][:n*ms,:3].reshape(n,ms,3).sum(1)
    return (np.arange(n)+.5)*ms*1e-3,np.column_stack([all_e,local])/nr/(ms*1e-3)


def window_balance(d,lo,hi):
    """Mean currents (mV-equivalent) in [lo,hi) s from the 20 ms slow records."""
    t=d['slow_time_ms']/1000;s=(t>=lo)&(t<hi)
    if not s.any():return None
    c=d['currents'][s].mean(0);z=d['Z'][s].mean(0);job=d['job']
    m=d['mechanism'];k=d['intrinsic']
    ms=(m['time_ms']/1000>=lo)&(m['time_ms']/1000<hi)
    ks=(k['time_ms']/1000>=lo)&(k['time_ms']/1000<hi) if k else np.zeros(0,bool)
    out=dict(window_s=[lo,hi],I_E=float(c[0]),raw_local_I=float(c[1]*(1-job['gamma'])),Z_local_I=float(c[2]-job['gamma']*job['C_R']*z[0]*m['global_E_rate_Hz'][ms].mean()) if ms.any() else None,
             Z_mean=float(z[0]),Z_coreA=float(z[5]),Z_coreB=float(z[6]),Z_other=float(z[7]),fraction_J_above_threshold=float(z[8]),
             I_M=float(job['eta_m']*d['M'][s,0].mean()),
             global_E_rate_Hz=float(m['global_E_rate_Hz'][ms].mean()) if ms.any() else None,
             g_global=float(m['global_applied_conductance_ratio'][ms].mean()) if ms.any() else None,
             I_global=float(m['global_outward_current_mV_equiv'][ms].mean()) if ms.any() else None,
             g_K=float(k['sahp_mean_conductance_ratio'][ks].mean()) if k and ks.any() else 0.,
             I_K=float(k['sahp_outward_current_mV_equiv'][ks].mean()) if k and ks.any() else 0.)
    # currents column 2 is mean(z*J) with J=(1-gamma)I_I+gamma*C_R*r_G ; subtract the global equivalent part to get the local Z-gated delivery
    tot_inh=(out['Z_local_I'] or 0.)+(out['I_global'] or 0.)+out['I_K']+out['I_M']
    out['total_inhibition_plus_adaptation']=tot_inh;out['excitation_minus_inhibition']=out['I_E']-tot_inh
    if d['regional']:
        rt=d['regional']['time_ms']/1000;rs=(rt>=lo)&(rt<hi)
        if rs.any():
            v=d['regional']['values'][rs].mean(0);keys=[str(x) for x in d['regional']['keys']]
            out['regions']={name:{k:float(v[i,j]) for j,k in enumerate(keys)} for i,name in enumerate(run.REGION_NAMES)}
    return out


def k_needed_to_hold_threshold(I_E,Z_local,I_global,I_M,g_global,threshold=18.,E_K=E_K):
    """Shunt-form steady state: V_inf=(I_E-Z_local-I_M+g_G*E_G+g_K*E_K)/(1+g_G+g_K) < threshold.
    Solve for g_K (I_global here is the applied current g_G*(V-E_G); use conductance form with E_G)."""
    num=I_E-Z_local-I_M;   # plus g_G*E_G handled via I_global at V=threshold approx
    # at V=threshold the global current is g_G*(threshold-E_G) = I_global scaled; use I_global directly
    need=(num-I_global-threshold)/(threshold-E_K)
    return float(max(need,0.))


def analyze(name,folder=None,write=True):
    folder=Path(folder) if folder else OUT/'runs'/name
    d=load_run(folder)
    if d is None:return None
    t,r=rates(d);end=len(d['spikes_1ms'])*1e-3
    prim=audit.temporal_audit(r)
    entries=prim['entries'];exits=prim['low_activity_exits']
    windows=[('baseline',.5,min(5.,end))]
    if entries:
        on=entries[0]['onset_s']
        windows+=[('pre_entry_2s',max(.5,on-2),on),('entry_ramp',on,min(on+.7,end)),('post_entry_1_5s',min(on+1,end),min(on+5,end)),('late',max(on+5,end-5),end)]
    else:
        windows+=[('late',max(.5,end-5),end)]
    for ex in exits[:1]:
        windows+=[('post_exit_0_2s',ex['confirmation_s'],min(ex['confirmation_s']+2,end)),('post_exit_2_10s',min(ex['confirmation_s']+2,end),min(ex['confirmation_s']+10,end))]
    balance={}
    for label,lo,hi in windows:
        if hi>lo:
            b=window_balance(d,lo,hi)
            if b:
                b['g_K_needed_to_hold_V_below_threshold']=k_needed_to_hold_threshold(b['I_E'],b['Z_local_I'] or 0.,b['I_global'] or 0.,b['I_M'],b['g_global'] or 0.)
                b['k100_needed_at_this_rate']=b['g_K_needed_to_hold_V_below_threshold']/max(b['global_E_rate_Hz']/100.,1e-9) if b['global_E_rate_Hz'] else None
                balance[label]=b
    # Z per region first-drop time: first time regional Z falls below 0.5
    st=d['slow_time_ms']/1000;zdrop={}
    for i,lab in [(5,'coreA'),(6,'coreB'),(7,'other'),(0,'mean')]:
        idx=np.flatnonzero(d['Z'][:,i]<.5);zdrop[lab]=float(st[idx[0]]) if len(idx) else None
    # refractory-limit check on the late window (E cells with tau_ref 2 ms -> 500 Hz)
    late_rate=float(r[-min(len(r),500):,0].mean())
    out=dict(name=name,source=str(folder),job=d['job'],observed_s=end,classification=prim['classification'],
             entries=entries,exits=exits,preentry_brief=prim['preentry']['brief_count'],
             interhigh=[dict(brief=p['brief_count'],span=p['event_span_s'],fraction=p['brief_fraction'],temporal_pass=p['temporal_pass']) for p in prim['interhigh_intervals']],
             latest_postexit=None if prim['latest_postexit'] is None else dict(brief=prim['latest_postexit']['brief_count'],span=prim['latest_postexit']['event_span_s'],fraction=prim['latest_postexit']['brief_fraction'],finite=prim['latest_postexit']['finite_count']),
             temporal_loop_pass=prim['temporal_loop_pass'],current_balance=balance,Z_below_0p5_first_s=zdrop,
             late_all_E_Hz=late_rate,late_state='REFRACTORY_LIMITED_PLATEAU' if late_rate>=450 else 'HIGH' if late_rate>=200 else 'LOW',
             final_Z=float(d['Z'][-1,0]),final_gK=float(d['intrinsic']['sahp_mean_conductance_ratio'][-1]) if d['intrinsic'] else None,
             final_I_M=float(d['job']['eta_m']*d['M'][-1,0]),regional_records=bool(d['regional']),
             high_morphology=high_state_morphology(d,entries,exits,end))
    if write:
        old.write(OUT/'analysis'/f'{name}_mechanism.json',out)
    return out,d


def plot_mechanism(name,folder=None):
    res=analyze(name,folder,write=False)
    if res is None:return None
    out,d=res;t,r=rates(d);end=out['observed_s'];job=d['job'];st=d['slow_time_ms']/1000
    m=d['mechanism'];k=d['intrinsic'];mt=m['time_ms']/1000
    plt.rcParams.update({'font.size':13,'axes.labelsize':14,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,1,figsize=(15,15),sharex=True,gridspec_kw=dict(hspace=.12,height_ratios=[1,1,1,1,1]))
    a=axes[0];a.plot(t,r[:,0],c='k',lw=.8,label='All E');a.plot(t,r[:,1],c='#d24b99',lw=.6,alpha=.8,label='Core A E');a.plot(t,r[:,2],c='#2195bc',lw=.6,alpha=.8,label='Core B E')
    a.axhline(200,c='#bc2946',ls=':',lw=1);a.axhline(500,c='#888',ls=':',lw=.8);a.set(ylabel='E rate (Hz, 10 ms)',ylim=(0,520));a.legend(loc='upper left',ncol=3,fontsize=11)
    for e in out['entries']:a.axvline(e['onset_s'],c='#bc2946',lw=1,ls='--')
    for x in out['exits']:a.axvline(x['confirmation_s'],c='#208677',lw=1,ls='--')
    b=axes[1];b.plot(st,d['Z'][:,0],c='#74398f',lw=1.6,label='Mean Z');b.plot(st,d['Z'][:,5],c='#d24b99',lw=.9,label='Core A');b.plot(st,d['Z'][:,6],c='#2195bc',lw=.9,label='Core B');b.plot(st,d['Z'][:,7],c='#888',lw=.9,label='Other')
    b.set(ylabel='Resource Z',ylim=(0,1.05));b.legend(loc='upper right',ncol=4,fontsize=11)
    c=axes[2];glob_eq=job['gamma']*job['C_R']*d['Z'][:,0]*np.interp(st,mt,m['global_E_rate_Hz'])
    c.plot(st,d['currents'][:,0],c='#c0392b',lw=1.2,label=r'$I_E$ (excitatory input)')
    c.plot(st,d['currents'][:,2]-glob_eq,c='#2c6fbb',lw=1.2,label=r'$Z\,(1-\gamma)I_I$ (local, Z-gated)')
    c.plot(mt,m['global_outward_current_mV_equiv'],c='#1b9e77',lw=1,label='global shunt current (Z-gated)')
    if k:c.plot(k['time_ms']/1000,k['sahp_outward_current_mV_equiv'],c='#a36329',lw=1.2,label='added K current')
    c.plot(st,job['eta_m']*d['M'][:,0],c='#e6ab02',lw=1,label=r'native $\eta_M M$')
    c.set(ylabel='Current (mV equiv.)',yscale='symlog',ylim=(0,3000));c.legend(loc='upper left',ncol=3,fontsize=10.5)
    e=axes[3]
    if k:e.plot(k['time_ms']/1000,k['sahp_mean_conductance_ratio'],c='#a36329',lw=1.2,label='added $g_K/g_L$ (mean E)')
    e.plot(mt,m['global_applied_conductance_ratio'],c='#1b9e77',lw=1,label='global $g_G/g_L$ (applied, Z-gated)')
    e.set(ylabel='Conductance / $g_L$');e.legend(loc='upper left',ncol=2,fontsize=11)
    f=axes[4]
    if d['regional']:
        rt=d['regional']['time_ms']/1000;keys=[str(x) for x in d['regional']['keys']];iv=keys.index('V_mV');ik=keys.index('I_K_mV')
        for i,(lab,col) in enumerate([('Core A','#d24b99'),('Core B','#2195bc'),('Other','#888')]):
            f.plot(rt,d['regional']['values'][:,i,iv],c=col,lw=1,label=f'{lab} mean V')
        f.axhline(18,c='k',ls=':',lw=.8);f.set(ylabel='Mean V (mV)');f.legend(loc='upper left',ncol=3,fontsize=11)
    else:
        f.plot(st,d['Z'][:,8],c='#555',lw=1);f.set(ylabel='Fraction J ≥ threshold')
    f.set(xlabel='Time (s)',xlim=(0,end))
    fig.suptitle(f"k100={job.get('k100',job['sahp_gain']*job['sahp_tau_s']):g}, τK={job['sahp_tau_s']:g} s (K gain {job['sahp_gain']:g}), seed {job['seed']} — {out['classification']}, {end:.0f} s",y=.995,fontsize=14)
    dest=OUT/'figures'/'mechanism';dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/f'mechanism_{name}.png',dpi=120,bbox_inches='tight');plt.close(fig)
    old.write(OUT/'analysis'/f'{name}_mechanism.json',out)
    return out


def matrix_rows():
    p=json.loads((OUT/'protocol.json').read_text());rows=[]
    for j in p['initial_jobs']:
        folder=OUT/'runs'/j['name']
        if (folder/'chunks').exists():
            res=analyze(j['name'],folder,write=False)
            if res:rows.append(dict(res[0],reused=False,running=not (folder/'result.json').exists()))
    for rr in p['reused_previous_conditions']:
        res=analyze(rr['name'],rr['previous_run'],write=False)
        if res:rows.append(dict(res[0],reused=True,running=False,previous_name=rr['previous_name']))
    return rows


def summary_table():
    rows=matrix_rows();lines=['| k100 | τK (s) | K gain | seed | 观察 (s) | 状态 | 进入前短事件 | 高活动次数 | 退出次数 | 退出后短事件 | 时序闭环 | 末段全 E (Hz) | 末 Z | 末 gK/gL |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    table=[]
    for o in sorted(rows,key=lambda o:(o['job'].get('k100',0),o['job']['sahp_tau_s'],o['job']['seed'])):
        j=o['job'];post=o['latest_postexit'];k100=j.get('k100',j['sahp_gain']*j['sahp_tau_s'])
        status='REUSED' if o['reused'] else ('RUNNING' if o['running'] else 'DONE')
        lines.append(f"| {k100:g} | {j['sahp_tau_s']:g} | {j['sahp_gain']:g} | {j['seed']} | {o['observed_s']:.0f} | {status} | {o['preentry_brief']} | {len(o['entries'])} | {len(o['exits'])} | {post['brief'] if post else '-'} | {'PASS' if o['temporal_loop_pass'] else 'no'} | {o['late_all_E_Hz']:.0f} | {o['final_Z']:.3f} | {o['final_gK'] if o['final_gK'] is None else round(o['final_gK'],3)} |")
        table.append(dict(k100=k100,tau_K_s=j['sahp_tau_s'],K_gain=j['sahp_gain'],seed=j['seed'],observed_s=o['observed_s'],status=status,name=o['name'],
                          classification=o['classification'],preentry_brief=o['preentry_brief'],n_entries=len(o['entries']),n_exits=len(o['exits']),
                          first_entry_s=o['entries'][0]['onset_s'] if o['entries'] else None,first_exit_s=o['exits'][0]['confirmation_s'] if o['exits'] else None,
                          postexit_brief=post['brief'] if post else None,temporal_loop_pass=o['temporal_loop_pass'],late_all_E_Hz=o['late_all_E_Hz'],late_state=o['late_state'],
                          final_Z=o['final_Z'],final_gK=o['final_gK'],Z_below_0p5_first_s=o['Z_below_0p5_first_s']))
    old.write(OUT/'matrix_summary.json',dict(updated_epoch=time.time(),rows=table))
    (OUT/'matrix_summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return table


def plot_branches(names,label,window_before_s=3.):
    """Overlay causal-diagnostic branches that share one source checkpoint."""
    p=json.loads((OUT/'protocol.json').read_text());jobs={j['name']:j for j in p['branch_jobs']}
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,1,figsize=(12,13),sharex=True,gridspec_kw=dict(hspace=.1))
    colors=['#b82d46','#1f77b4','#2ca02c','#9467bd','#ff7f0e'];summary=[]
    for col,name in zip(colors,names):
        folder=OUT/'runs'/name;d=load_run(folder)
        if d is None:continue
        j=jobs[name];start=j['branch']['start_s'];t,r=rates(d);st=d['slow_time_ms']/1000;k=d['intrinsic'];m=d['mechanism']
        lab=f"{j['branch']['label']}: {json.dumps(j['branch']['modification'])}"
        axes[0].plot(t,r[:,0],c=col,lw=1,label=lab)
        axes[1].plot(st,d['Z'][:,0],c=col,lw=1.2)
        if k:axes[2].plot(k['time_ms']/1000,k['sahp_mean_conductance_ratio'],c=col,lw=1.2)
        axes[3].plot(st,d['currents'][:,0],c=col,lw=1.2,ls='-')
        glob_eq=j['gamma']*j['C_R']*d['Z'][:,0]*np.interp(st,m['time_ms']/1000,m['global_E_rate_Hz'])
        axes[3].plot(st,d['currents'][:,2]-glob_eq+np.interp(st,m['time_ms']/1000,m['global_outward_current_mV_equiv'])+(np.interp(st,k['time_ms']/1000,k['sahp_outward_current_mV_equiv']) if k else 0)+j['eta_m']*d['M'][:,0],c=col,lw=1.2,ls='--')
        if d['regional']:
            rt=d['regional']['time_ms']/1000;keys=[str(x) for x in d['regional']['keys']];iv=keys.index('V_mV')
            axes[4].plot(rt,d['regional']['values'][:,0,iv],c=col,lw=1)
        end=t[-1]+.005;late=r[-min(len(r),100):,0].mean()
        tracker_exits=[x for x in audit.temporal_audit(r)['low_activity_exits'] if x['confirmation_s']>start]
        summary.append(dict(name=name,modification=j['branch']['modification'],start_s=start,observed_to_s=float(end),late_1s_all_E_Hz=float(late),
                            exit_confirmed_s=tracker_exits[0]['confirmation_s'] if tracker_exits else None,final_Z=float(d['Z'][-1,0]),final_gK=float(k['sahp_mean_conductance_ratio'][-1]) if k else None,
                            min_all_E_10ms_after_branch_Hz=float(r[t>start,0].min()) if (t>start).any() else None))
    starts=sorted(set(x['start_s'] for x in summary));start=starts[0]
    for ax in axes:
        for x0 in starts:ax.axvline(x0,c='k',ls=':',lw=1)
    axes[0].set(ylabel='All-E rate (Hz)',ylim=(0,520));axes[0].legend(fontsize=10,loc='lower left')
    axes[1].set(ylabel='Mean Z',ylim=(0,1.05));axes[2].set(ylabel='Added $g_K/g_L$')
    axes[3].set(ylabel='mV equiv.',yscale='symlog');axes[3].text(.01,.9,'solid: $I_E$; dashed: local Z-gated + global + K + M',transform=axes[3].transAxes,fontsize=10)
    axes[4].set(ylabel='Core A mean V (mV)',xlabel='Time (s)');axes[4].axhline(18,c='k',ls=':',lw=.8)
    axes[-1].set(xlim=(start-window_before_s,max(s['observed_to_s'] for s in summary)))
    fig.suptitle(f'Causal-diagnostic branches from {label} (state continued from stored checkpoint; one feedback term changed)',fontsize=12)
    dest=OUT/'figures'/'mechanism';dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/f'branches_{label}.png',dpi=120,bbox_inches='tight');plt.close(fig)
    old.write(OUT/'analysis'/f'branches_{label}.json',dict(source=label,rows=summary,diagnostic_only=True,not_counted_as_autonomous_loop=True))
    print(json.dumps(summary,indent=1))
    return summary


def high_state_morphology(d,entries,exits,end):
    """Operational morphology of each high episode: refractory-limited plateau vs sustained
    oscillation vs decaying ringing, plus spatial spread. Descriptive only."""
    from scipy.signal import welch
    t,r=rates(d);out=[]
    r1=d['spikes_1ms'][:,0]/32.   # all-E rate at 1 ms (Hz)
    nf=len(d['field_5ms'])//10;fields=d['field_5ms'][:nf*10].reshape(nf,10,400).sum(1)/d['geo']['cell_e_counts']/.05;ft=(np.arange(nf)+.5)*.05
    for e in entries:
        on=e['onset_s'];ex=next((x['confirmation_s'] for x in exits if x['confirmation_s']>on),None);stop=ex if ex else end
        s=(t>=on)&(t<stop);seg=r[s,0]
        if seg.size<20:continue
        seg1=r1[int(on*1000):int(stop*1000)]
        f,pxx=welch(seg1-seg1.mean(),fs=1000.,nperseg=min(len(seg1),4096))
        band=(f>=1)&(f<=200);fpk=float(f[band][np.argmax(pxx[band])]) if band.any() else None
        rel=float(pxx[band].max()/pxx[band].sum()) if band.any() else None
        thirds=np.array_split(seg,3);trend=[float(x.mean()) for x in thirds];cv=[float(x.std()/max(x.mean(),1e-9)) for x in thirds]
        fs_=(ft>=on)&(ft<stop);ext=(fields[fs_]>=20).mean(1) if fs_.any() else np.zeros(0)
        t90=float(ft[fs_][np.flatnonzero(ext>=.9)[0]]-on) if fs_.any() and (ext>=.9).any() else None
        # which region first reaches 200 Hz (10 ms) in the 3 s before onset
        pre=(t>=on-3)&(t<=on+.3);first={}
        for i,lab in enumerate(['allE','coreA','coreB','other']):
            idx=np.flatnonzero(r[pre,i]>=200);first[lab]=float(t[pre][idx[0]]) if len(idx) else None
        label=('REFRACTORY_LIMITED_PLATEAU' if np.mean(seg>=450)>.5 else 'DECAYING' if trend[2]<.5*trend[0] else 'SUSTAINED_OSCILLATION' if (rel and rel>.15 and np.mean(cv)>.3) else 'SUSTAINED_HIGH')
        out.append(dict(onset_s=on,end_s=stop,duration_s=stop-on,censored=ex is None,mean_Hz=float(seg.mean()),fraction_bins_ge_450Hz=float(np.mean(seg>=450)),
                        thirds_mean_Hz=trend,thirds_cv=cv,dominant_frequency_Hz=fpk,dominant_relative_power=rel,
                        spatial_fraction_active_mean=float(ext.mean()) if ext.size else None,time_to_90pct_sheet_s=t90,first_200Hz_s=first,operational_label=label))
    return out


def brief_event_profile(d,events,lo,hi):
    """Duration / peak / spatial extent / core lead of complete brief events in [lo,hi)."""
    nf=len(d['field_5ms'])//10;fields=d['field_5ms'][:nf*10].reshape(nf,10,400).sum(1)/d['geo']['cell_e_counts']/.05;ft=(np.arange(nf)+.5)*.05
    t,r=rates(d);rows=[]
    for e in events:
        if not (e['start_s']>=lo and e['end_s']<=hi and .02<=e['duration_s']<=.2):continue
        fs=(ft>=e['start_s']-.05)&(ft<=e['end_s']+.05);ext=float((fields[fs]>=20).mean(1).max()) if fs.any() else None
        s=(t>=e['start_s'])&(t<e['end_s']);seg=r[s]
        lead={}
        for i,lab in enumerate(['coreA','coreB','other']):
            idx=np.flatnonzero(seg[:,i+1]>=50);lead[lab]=float(t[s][idx[0]]-e['start_s']) if len(idx) else None
        first=min([k for k,v in lead.items() if v is not None],key=lambda k:lead[k]) if any(v is not None for v in lead.values()) else None
        ab=None
        if lead['coreA'] is not None and lead['coreB'] is not None:ab=lead['coreB']-lead['coreA']
        rows.append(dict(start_s=e['start_s'],duration_s=e['duration_s'],peak_Hz=e['peak_Hz'],max_active_fraction=ext,first_region=first,B_minus_A_lead_s=ab))
    def q(vals):
        vals=[v for v in vals if v is not None];return None if not vals else dict(n=len(vals),median=float(np.median(vals)),q25=float(np.percentile(vals,25)),q75=float(np.percentile(vals,75)))
    return dict(window_s=[lo,hi],n=len(rows),duration_s=q([x['duration_s'] for x in rows]),peak_Hz=q([x['peak_Hz'] for x in rows]),
                max_active_fraction=q([x['max_active_fraction'] for x in rows]),B_minus_A_lead_s=q([x['B_minus_A_lead_s'] for x in rows]),
                first_region_counts={k:sum(x['first_region']==k for x in rows) for k in ['coreA','coreB','other']},events=rows)


def compare_return(name,folder=None,split_s=None):
    """Pre-entry brief events versus post-exit brief events of the same trajectory."""
    folder=Path(folder) if folder else OUT/'runs'/name;d=load_run(folder);t,r=rates(d);end=len(d['spikes_1ms'])*1e-3
    prim=audit.temporal_audit(r);events=prim['events'];on=prim['entries'][0]['onset_s'] if prim['entries'] else end
    pre=brief_event_profile(d,events,.5,on)
    post=None
    if prim['low_activity_exits']:
        ex=prim['low_activity_exits'][0]['confirmation_s'];nxt=next((e['onset_s'] for e in prim['entries'] if e['onset_s']>ex),end)
        post=brief_event_profile(d,events,ex,nxt)
    out=dict(name=name,pre_entry=pre,post_exit=post)
    old.write(OUT/'analysis'/f'{name}_return_profile.json',out)
    def fmt(p):
        if not p:return 'none'
        return f"n={p['n']} dur={p['duration_s'] and round(p['duration_s']['median']*1000)} ms peak={p['peak_Hz'] and round(p['peak_Hz']['median'])} Hz extent={p['max_active_fraction'] and round(p['max_active_fraction']['median'],2)} lead(B-A)={p['B_minus_A_lead_s'] and round(p['B_minus_A_lead_s']['median']*1000)} ms first={p['first_region_counts']}"
    print(name,'\n  pre :',fmt(pre),'\n  post:',fmt(post))
    return out


def interictal_regime_table(seed=9108401,t_lo=5.,t_hi=30.):
    """Interictal regime (before any entry) per condition: brief-event statistics, Z balance, K."""
    p=json.loads((OUT/'protocol.json').read_text());rows=[]
    sources=[(j['name'],OUT/'runs'/j['name']) for j in p['initial_jobs'] if j['seed']==seed]+[(r['name'],Path(r['previous_run'])) for r in p['reused_previous_conditions']]
    for name,folder in sources:
        if not (folder/'chunks').exists():continue
        d=load_run(folder);t,r=rates(d);end=len(d['spikes_1ms'])*1e-3;prim=audit.temporal_audit(r)
        on=prim['entries'][0]['onset_s'] if prim['entries'] else end;hi=min(t_hi,on-2. if prim['entries'] else end);lo=t_lo
        if hi<=lo+5:continue
        ev=[e for e in prim['events'] if lo<=e['start_s'] and e['end_s']<=hi and .02<=e['duration_s']<=.2]
        st=d['slow_time_ms']/1000;s=(st>=lo)&(st<hi);z=d['Z'][s,0];slope=float(np.polyfit(st[s],z,1)[0]) if s.sum()>2 else None
        k=d['intrinsic'];ks=(k['time_ms']/1000>=lo)&(k['time_ms']/1000<hi) if k else None
        c=d['currents'][s].mean(0);j=d['job']
        rows.append(dict(name=name,k100=j['k100'],tau_K_s=j['sahp_tau_s'],K_gain=j['sahp_gain'],window_s=[lo,hi],entry_s=on if prim['entries'] else None,
                         brief_rate_per_s=len(ev)/(hi-lo),brief_duration_ms_median=float(np.median([e['duration_s'] for e in ev])*1000) if ev else None,
                         brief_peak_Hz_median=float(np.median([e['peak_Hz'] for e in ev])) if ev else None,
                         mean_E_rate_Hz=float(r[(t>=lo)&(t<hi),0].mean()),Z_mean=float(z.mean()),Z_slope_per_s=slope,Z_end=float(z[-1]),
                         gK_mean=float(k['sahp_mean_conductance_ratio'][ks].mean()) if k is not None and ks.any() else None,
                         I_E_mean=float(c[0]),Z_local_I_mean=float(c[2]-j['gamma']*j['C_R']*z.mean()*np.interp(st[s],d['mechanism']['time_ms']/1000,d['mechanism']['global_E_rate_Hz']).mean())))
    rows.sort(key=lambda x:(x['k100'],x['tau_K_s']))
    old.write(OUT/'analysis'/'interictal_regime_table.json',dict(seed=seed,rows=rows))
    lines=['| k100 | τK (s) | K gain | 窗口 (s) | 进入 (s) | 短事件率 (/s) | 时长中位 (ms) | 峰中位 (Hz) | 平均E率 (Hz) | Z 均值 | Z 斜率 (/s) | 窗末 Z | gK/gL 均值 | I_E | Z·局部抑制 |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for x in rows:
        lines.append(f"| {x['k100']:g} | {x['tau_K_s']:g} | {x['K_gain']:g} | {x['window_s'][0]:g}–{x['window_s'][1]:g} | {x['entry_s'] if x['entry_s'] is not None else '-'} | {x['brief_rate_per_s']:.2f} | {x['brief_duration_ms_median'] and round(x['brief_duration_ms_median'])} | {x['brief_peak_Hz_median'] and round(x['brief_peak_Hz_median'])} | {x['mean_E_rate_Hz']:.1f} | {x['Z_mean']:.3f} | {x['Z_slope_per_s']:+.4f} | {x['Z_end']:.3f} | {x['gK_mean']:.3f} | {x['I_E_mean']:.1f} | {x['Z_local_I_mean']:.1f} |")
    (OUT/'analysis'/'interictal_regime_table.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['analyze','plot','table','branches','compare_return','regime']);ap.add_argument('--names',nargs='*');ap.add_argument('--label');ap.add_argument('--name');ap.add_argument('--folder')
    a=ap.parse_args()
    if a.action=='analyze':print(json.dumps(old.safe(analyze(a.name,a.folder)[0]),indent=1)[:4000])
    elif a.action=='plot':plot_mechanism(a.name,a.folder)
    elif a.action=='branches':plot_branches(a.names,a.label)
    elif a.action=='compare_return':compare_return(a.name,a.folder)
    elif a.action=='regime':interictal_regime_table()
    else:summary_table()
