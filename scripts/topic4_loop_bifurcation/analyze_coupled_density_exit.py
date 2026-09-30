#!/usr/bin/env python3
"""Absolute-time native/coupled-density comparison; no retrospective alignment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time,pickle
import numpy as np
from campaign import ROOT,read,write,sha
from reconstruct_loop_future_drive import OUT,SOURCE
from coupled_density_exit import ADAPTED


def stream(kind,keys):
    acc={k:[] for k in keys}
    for path in sorted((SOURCE/kind).glob('*.npz')):
        a,b=map(int,path.stem.split('_'))
        if b<=100000 or a>=200000:continue
        with np.load(path) as z:
            for k in keys:acc[k].append(z[k])
    return {k:np.concatenate(v) for k,v in acc.items()}


def native_data(geo):
    chunks=stream('chunks',['time_ms','spikes_1ms','regions_1ms','field_time_ms','field_5ms'])
    mechanism=stream('mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
    regional=stream('regional_chunks',['time_ms','values'])
    sizes=geo['group_size'];pop=geo['population'];regions=geo['group_region']
    weights=np.array([sizes[(pop==0)&(regions==q)].sum() for q in range(3)])
    counts=np.bincount(geo['group_cell'][pop==0],weights=sizes[pop==0],minlength=400)
    rates=np.c_[chunks['spikes_1ms'][:,0]/32000,chunks['regions_1ms'][:,:3]/weights]*1000
    assert rates.shape==(10000,4) and len(mechanism['time_ms'])==10000
    fields=chunks['field_5ms']/counts/.005
    with (SOURCE/'states/t20s.pkl').open('rb') as f:terminal=pickle.load(f)['engine']
    masks=[np.arange(32000)]+[np.flatnonzero(geo['group_region'][geo['cell_group'][:32000]]==q) for q in range(3)]
    endZ=np.array([terminal['slow']['z'][ix].mean() for ix in masks]);endK=np.array([terminal['termination_mechanism']['sahp_g'][ix].mean() for ix in masks])
    vals=regional['values'];allZ=(vals[:,:,8]*weights).sum(1)/32000;allK=(vals[:,:,0]*weights).sum(1)/32000
    return dict(name='Native',rate_time_ms=chunks['time_ms'],rates=rates,global_time_ms=mechanism['time_ms'],
        R=mechanism['global_E_rate_Hz'],Graw=mechanism['global_raw_conductance_ratio'],
        slow_time_ms=regional['time_ms'],Z=np.c_[allZ,vals[:,:,8]],K=np.c_[allK,vals[:,:,0]],
        field_time_ms=chunks['field_time_ms'],fields=fields,final_Z=endZ,final_K=endK,cell_counts=counts)


def candidate(seed,geo):
    with np.load(OUT/f'num{seed}/trajectory.npz') as z:
        E=geo['population']==0;sizes=geo['group_size'];reg=geo['group_region']
        masks=[E]+[E&(reg==q) for q in range(3)];out={}
        for key,target in [('group_rate_Hz','rates'),('group_Z','Z'),('group_K','K')]:
            x=z[key].astype(float);out[target]=np.stack([np.average(x[:,mask],weights=sizes[mask],axis=1) for mask in masks],axis=1)
        out.update(name=f'Density {seed}',rate_time_ms=z['time_ms']-.5,slow_time_ms=z['time_ms'],
            global_time_ms=z['time_ms'],R=z['global_R_Hz'],Graw=30*z['global_s'],
            fields=z['field_E_Hz'].reshape(2000,5,400).mean(1),field_time_ms=10000+np.arange(2000)*5+2.5,
            final_Z=out['Z'][-1],final_K=out['K'][-1],cell_counts=z['cell_counts'])
    return out


def first_sustained(t,mask,n):
    good=np.flatnonzero(np.convolve(mask.astype(int),np.ones(n,dtype=int),mode='valid')==n)
    return float(t[good[0]]/1000) if len(good) else None


def metrics(d,reference):
    rates10=d['rates'].reshape(1000,10,4).mean(1);t10=d['rate_time_ms'].reshape(1000,10).mean(1)
    windows=[]
    for lo,hi in [(10,12),(12,14),(14,16),(16,18),(18,20)]:
        keep=(d['rate_time_ms']>=lo*1000)&(d['rate_time_ms']<hi*1000)
        slow=(d['slow_time_ms']>=lo*1000)&(d['slow_time_ms']<hi*1000)
        glob=(d['global_time_ms']>=lo*1000)&(d['global_time_ms']<hi*1000)
        f=(d['field_time_ms']>=lo*1000)&(d['field_time_ms']<hi*1000)
        ref=(reference['field_time_ms']>=lo*1000)&(reference['field_time_ms']<hi*1000)
        delta=d['fields'][f].mean(0)-reference['fields'][ref].mean(0)
        windows.append(dict(window_s=[lo,hi],mean_rate_allE_A_B_other=d['rates'][keep].mean(0).tolist(),
            mean_Z_allE_A_B_other=d['Z'][slow].mean(0).tolist(),mean_K_allE_A_B_other=d['K'][slow].mean(0).tolist(),
            mean_Graw=float(d['Graw'][glob].mean()),mean_field_weighted_RMS_Hz=float(np.sqrt(np.average(delta**2,weights=d['cell_counts'])))))
    # Physical K retention uses causal global R; joint activity is only a readout.
    rlow=first_sustained(d['global_time_ms'],d['R']<=5,100)
    quiet=first_sustained(t10,(rates10[:,:3]<=5).all(1),10)
    zero=np.flatnonzero((d['global_time_ms']>=((rlow or 20)*1000))&(d['Graw']<95.19851312666987/(18+17.662847938268442)))
    released=float(d['global_time_ms'][zero[0]]/1000) if len(zero) and rlow is not None else None
    return dict(name=d['name'],first_causal_R_le5_for100ms_s=rlow,first_joint_allE_A_B_le5_for100ms_s=quiet,
        first_Graw_below_allcell_resource_block_after_Rlow_s=released,final_Z_allE_A_B_other=d['final_Z'].tolist(),
        final_K_allE_A_B_other=d['final_K'].tolist(),windows=windows,
        diagnostic_definitions='CausalRreadoutat1ms; jointallE/coreA/coreB means10ms bins for100ms. Theseare reporteddiagnostics,not a new controller. Absencebefore20s is censored.',
        first_recovery_or_exit_is_not_sufficient_restoration=True)


def main(wait=False):
    geo=dict(np.load(ADAPTED/'geometry.npz'));native=native_data(geo);rows=[metrics(native,native)];data=[native];done=[]
    while len(done)<2:
        for seed in [927671,927672]:
            path=OUT/f'num{seed}/result.json'
            if seed in done or not path.exists():continue
            assert read(path)['status']=='COMPLETE'
            d=candidate(seed,geo);m=metrics(d,native);rows.append(m);data.append(d);done.append(seed)
            write(OUT/f'num{seed}/comparison.json',m)
            print('COUPLED COMPARISON',seed,m['first_causal_R_le5_for100ms_s'],m['first_joint_allE_A_B_le5_for100ms_s'],m['final_Z_allE_A_B_other'],flush=True)
        result=dict(status='COMPLETE' if len(done)==2 else 'PARTIAL',completed=done,rows=rows,
            statistical_unit='One native source trajectory and two numericalparticle streams with same exactexpected exogenousinput and same nativeinitialstate. Not twoindependentnative orclinicalreplicates.',
            native_correspondence_certified=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
        write(OUT/'comparison.json',result)
        if len(done)==2:break
        if not wait:return
        write(OUT/'analysis_progress.json',dict(status='WAITING_COUPLED_RUN',pid=os.getpid(),completed=done,updated_epoch=time.time()));time.sleep(30)
    plot(data,geo);write(OUT/'analysis_progress.json',dict(status='COMPLETE',completed=done,updated_epoch=time.time()))


def plot(data,geo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(13.4,9.5),layout='constrained');gs=fig.add_gridspec(4,4,width_ratios=[1.4,1.4,1.,1.])
    timeaxes=[fig.add_subplot(gs[i,:2]) for i in range(4)];colors=['black','#267f8e','#cf7b2b']
    for d,color in zip(data,colors):
        t=d['global_time_ms']/1000;s=d['slow_time_ms']/1000
        for ax,tx,y in zip(timeaxes,[t,t,s,s],[d['R'],d['Graw'],d['K'][:,0],d['Z'][:,0]]):
            ax.plot(tx,y,color=color,lw=1.3,label=d['name'])
    for ax,label in zip(timeaxes,['Causal E rate (Hz)','Raw global G / gL','Mean K / gL','Mean Z']):ax.set_ylabel(label);ax.set_xlim(10,20)
    timeaxes[0].axhline(5,color='.65',lw=.8,ls=':');timeaxes[0].legend(frameon=False,ncol=3,fontsize=8)
    timeaxes[1].axhline(95.19851312666987/(18+17.662847938268442),color='.65',lw=.8,ls=':')
    timeaxes[-1].set_xlabel('Native time (s)')
    for row,(lo,hi) in enumerate([(12,12.1),(16.6,16.7),(17,17.1)]):
        for col,d in enumerate(data[:2]):
            ax=fig.add_subplot(gs[row,2+col]);keep=(d['field_time_ms']>=lo*1000)&(d['field_time_ms']<hi*1000)
            value=d['fields'][keep].mean(0).reshape(20,20)
            im=ax.imshow(value,origin='lower',extent=[0,20,0,20],vmin=0,vmax=500,cmap='magma',interpolation='nearest')
            for center in geo['centers_mm']:ax.add_patch(Circle(center,1.5,fill=False,color='#00bec7',lw=.8))
            ax.set_title(f'{d["name"]}\n{lo:g}–{hi:g} s',fontsize=9);ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if col==0:ax.set_ylabel('y (mm)')
            if row==2:ax.set_xlabel('x (mm)')
    cax=fig.add_subplot(gs[3,2:]);cax.axis('off');fig.colorbar(im,ax=cax,orientation='horizontal',fraction=.35,label='E population rate (Hz)')
    cax.text(.5,.9,'Spatial example fixed to numerical stream 927671.\nSame absolute windows; no phase alignment.',ha='center',va='top',transform=cax.transAxes,fontsize=9)
    fig.suptitle('Coupled density continuation from the native 10-s state',weight='bold')
    fig.text(.5,-.012,'Only external mean-rate input is prescribed. Recurrent firing, G/R and Z/M/K evolve freely. Candidate validation; formal stability unverified.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'coupled_density_exit.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
