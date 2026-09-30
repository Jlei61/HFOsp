#!/usr/bin/env python3
"""Selected physical-history return, not a full stability calculation."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import shutil
import time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from coupled_density_exit import ADAPTED
from physical_history_return import OUT,SOURCES
from analyze_mean_exit_interval import projection


def main(wait):
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    while not all((OUT/'runs'/n/'result.json').exists() for n in SOURCES):
        failed=[n for n in SOURCES if (OUT/'runs'/n/'progress.json').exists() and read(OUT/'runs'/n/'progress.json')['status']=='FAILED']
        if failed:write(dest/'progress.json',dict(status='STOPPED_ON_WORKER_FAILURE',names=failed));return
        write(dest/'progress.json',dict(status='WAITING_FOUR_REGISTERED_RETURNS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    write(dest/'progress.json',dict(status='ANALYZING',pid=os.getpid(),updated_epoch=time.time()))
    sizes,masks,counts,proj=projection();geo=dict(np.load(ADAPTED/'geometry.npz'));I=geo['population']==1
    icounts=np.bincount(geo['group_cell'][I],weights=sizes[I],minlength=400);assert (icounts>0).all()
    iproj=sparse.coo_matrix((sizes[I]/icounts[geo['group_cell'][I]],(geo['group_cell'][I],np.flatnonzero(I))),shape=(400,len(I))).tocsr()
    data={};rng_reference=None
    for name in SOURCES:
        folder=OUT/'runs'/name;assert read(folder/'implementation_qa.json')['status']=='PASS'
        chunks=sorted((folder/'chunks').glob('*.npz'));assert len(chunks)==50;blocks=[]
        for j,p in enumerate(chunks):
            assert tuple(map(int,p.stem.split('_')))==(j*100,(j+1)*100)
            with np.load(p) as z:
                v=z['group_output'].astype(float)
                rates=np.stack([np.average(v[:,0,m],weights=sizes[m],axis=1) for m in masks],axis=1)
                drift=np.stack([np.average((v[:,8,m]-v[:,1,m])/5,weights=sizes[m],axis=1) for m in masks],axis=1)
                ef=np.asarray((proj@v[:,0].T).T);iff=np.asarray((iproj@v[:,0].T).T);mf=np.asarray((proj@v[:,2].T).T)
                assert np.allclose(ef@counts/counts.sum(),rates[:,0],atol=1e-5)
                blocks.append(dict(rate=rates,drift=drift,field=ef,I_field=iff,M_field=mf,R=z['global_R_Hz'],G=30*z['global_s']))
        d={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};d['time']=np.arange(1,5001)/1000;data[name]=d
        np.savez_compressed(dest/f'{name}_readouts.npz',**d)
        with np.load(folder/'final_state.npz') as z:
            rng={k:z[k] for k in ['rng','external_rng','clock']};assert int(z['clock'][0])==250000
            if rng_reference is None:rng_reference=rng
            else:
                for k in rng:assert np.array_equal(rng[k],rng_reference[k]),(name,k)
            with np.load(OUT/'held_fields.npz') as f:
                assert np.array_equal(z['state'][:32000,:,6],np.broadcast_to(f['Z'][:,None],(32000,64)))
                assert np.array_equal(z['state'][:32000,:,7],np.broadcast_to(f['K'][:,None],(32000,64)))
    ref=data['same_K_control'];rows=[]
    def rms(x,w):return float(np.sqrt(np.average(x*x,weights=w)))
    for name,d in data.items():
        rows1=[]
        for second in range(5):
            s=slice(second*1000,(second+1)*1000)
            rows1.append(dict(interval_s=[second,second+1],mean_rate_Hz=d['rate'][s].mean(0).tolist(),
                core_Zdot_if_released=d['drift'][s,1:3].mean(0).tolist(),
                E_field_RMS_to_control_Hz=rms(d['field'][s].mean(0)-ref['field'][s].mean(0),counts),
                I_field_RMS_to_control_Hz=rms(d['I_field'][s].mean(0)-ref['I_field'][s].mean(0),icounts),
                M_field_RMS_to_control_counts=rms(d['M_field'][s].mean(0)-ref['M_field'][s].mean(0),counts)))
        tail=rows1[-1];rate_diff=d['rate'][-1000:].mean(0)-ref['rate'][-1000:].mean(0)
        stationarity=rms(d['field'][-1000:].mean(0)-d['field'][-2000:-1000].mean(0),counts)
        quiet=float((d['rate'][-1000:].reshape(-1,10,4).mean(1)[:,:3]<5).all(1).mean())
        guards=dict(E_field=tail['E_field_RMS_to_control_Hz']<=1,I_field=tail['I_field_RMS_to_control_Hz']<=1,
            E_M_field=tail['M_field_RMS_to_control_counts']<=1,core_rate=bool(abs(rate_diff[1:3]).max()<=1),
            allE_rate=bool(abs(rate_diff[0])<=.2),last_second_E_field_stationary=stationarity<=1)
        rows.append(dict(name=name,one_second_windows=rows1,final_rate_difference_Hz=rate_diff.tolist(),
            final_E_field_change_from_previous_second_Hz=stationarity,final_jointquiet_fraction=quiet,
            selected_high_return_guards=guards if name!='from_quiet' else None,
            returned_to_high_control=all(guards.values()) if name!='from_quiet' else None))
    high_return=all(row['returned_to_high_control'] for row in rows if row['name']!='from_quiet')
    quiet=next(row for row in rows if row['name']=='from_quiet')['final_jointquiet_fraction']>=.95
    result=dict(status='COMPLETE_FOUR_SELECTED_HISTORY_RETURNS',rows=rows,both_neighbor_high_histories_return=high_return,
        quiet_history_remains_quiet=bool(quiet),paired_future_RNGs_and_clock_bitwise=True,held_fields_bitwise=True,
        interpretation='Only the selected physically generated history perturbations are tested. Return of E/I/M spatial means is a finite macro-observable property; it does not certify convergence of all microscopic phases, all perturbationdirections, stable/unstablebranches or an infinite-replica bifurcation. Persistent distinct states must not be forced onto one branch.',
        formal_bifurcation_allowed=False,agent_visual='PENDING',human_visual='PENDING',producer_sha256=sha(__file__))
    write(dest/'result.json',result);plot(data,counts);shutil.copy2(__file__,dest/'producer.py')
    write(dest/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


def plot(data,counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#344a80','#d47b25','#94588f','#428f81'];labels=['From K 9.3875','Same K control','From K 9.4625','From quiet history']
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained');ref=data['same_K_control']
    for (name,d),color,label in zip(data.items(),colors,labels):
        ax[0].plot(d['time'],d['R'],color=color,label=label,lw=1)
        if name!='from_quiet':
            delta=d['field'].reshape(50,100,400).mean(1)-ref['field'].reshape(50,100,400).mean(1)
            err=np.sqrt(np.average(delta**2,weights=counts,axis=1))
            ax[1].plot((np.arange(50)+.5)/10,err,color=color,lw=1,label=label)
    ax[0].set(ylabel='Causal E rate (Hz)',ylim=(0,205));ax[0].spines['bottom'].set_position(('outward',4));ax[0].legend(frameon=False,fontsize=8)
    ax[1].set(ylabel='E spatial RMS from control (Hz)',ylim=(0,None));ax[1].set_yscale('symlog',linthresh=.1)
    for axis in ax:axis.set(xlabel='Time at common K = 9.425 (s)',xlim=(0,5))
    fig.suptitle('Physical history return: selected directions, shared future input',weight='bold')
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/physical_history_return.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(12,3.5),layout='constrained');centers=np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for axis,(name,d),label in zip(axes,data.items(),labels):
        im=axis.imshow(d['field'][-1000:].mean(0).reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=500,interpolation='nearest')
        for xy in centers:axis.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=.9))
        axis.set(title=label,xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])
    fig.colorbar(im,ax=axes.tolist(),shrink=.75,label='E rate, 4–5 s (Hz)');fig.suptitle('Same held Z/K and future input: final spatial responses',weight='bold')
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/physical_history_return_spatial.{ext}',dpi=180)
    plt.close(fig)
    with (ROOT/'figures/README.md').open('a') as f:
        for stem in ['physical_history_return','physical_history_return_spatial']:
            f.write(f'\n\n### {stem}.png / {stem}.svg\n三个相邻参数产生的完整高史及一个静默史，在同一K9.425、相同Z场和未来随机流下续演五秒。图示实际宏观返回及最后一秒的空间形态；完整数值另含I空间率、M及核心资源收支。\n**关注点**：这是所选物理历史方向的宏观恢复检验，不能代替全部相位状态、线性谱或正式稳定性认证；人工待审。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
