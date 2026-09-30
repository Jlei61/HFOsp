#!/usr/bin/env python3
"""Two bounded full-spatial frequency probes using the established sine assay.

This retains original actual-K9 mean inputs, raw physical-G perturbation, full
and half amplitudes, target/channel independent noise, and every weak component.
It does not extrapolate frequency responses from representative cells.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,subprocess,time
import numpy as np
import cupy as cp
from campaign import ROOT,PYTHON,read,write,sha
import measure_all_target_dc as dc
from measure_target_direct_response import run

OUT=ROOT/'all_target_frequency_direct'
FREQUENCIES=[1.,5.]


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(ROOT/'direct_exit_small_K_step/result.json')['candidate_near_equilibrium']
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_FULL_SPATIAL_DYNAMIC_PROBE',created_epoch=time.time(),
        question='How does the spatial feedback operator near the actual corresponding high state change when the measured cell response, original delays, synaptic filters and causalG/M dynamics are retained?',
        design=dict(frequencies_Hz=FREQUENCIES,partitions=[[0,20000],[20000,40000]],devices=[0,1],
            replicas=256,record_ms=4000,burn_ms=1000,base_seeds=[929011,929021],target_batch=128,
            channels=['effective_mean','varianceE','varianceI','physical_applied_G_E_only'],
            amplitudes='Exactly the all-target DC amplitudes: mean .02*(theta-11), varianceE5%, varianceI10%, physicalG .002*(1+g), repeated at half amplitude'),
        physics='Exactly the all-target DC reference inputs and original density single-increment colored-current/membrane/reset/refractory ordering. Independent target/channel numerical noise; paired +/- and full/half paths. These are fixedinput local probes, not new native seeds.',
        method='Existing independently checked one-frequency sinusoidal kernel; no new multisine shortcut. Store complex gain, paired amplitude error, raw outputs and eight independent replica-block gains.',
        assessment='Retain every failed-amplitude and weak component. Per-component tolerance max(10%abs(DC),2pairedSEM,1e-7); individual DC estimability from original fullDC SNR>=10. Spatial operator/action uncertainty, exact update order and frequency coverage must be assessed separately before stability claims.',
        limits='Only 1Hz and5Hz are initial dynamic probes. They do not cover the temporal spectrum or certify stable/unstable branches or Hopf. Inputs remain near the K9 corresponding state, not an exact selfconsistent root or every nearby branch point. The failed multisine pilot remains failed under its registered gate.',
        producer_sha256=sha(__file__),kernel_sha256=sha(dc.__file__),input_sha256=sha(dc.SOURCE/'inputs.npz'),
        formal_bifurcation_allowed=False))
    print('Registered 1Hz and5Hz full-spatial direct probes',flush=True)


def worker(index,part,device):
    contract=read(OUT/'contract.json');assert contract['producer_sha256']==sha(__file__) and contract['kernel_sha256']==sha(dc.__file__)
    folder=OUT/f'frequency_{index:02d}'/f'part_{part}';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'progress.json').exists(),'No silent restart'
    cp.cuda.Device(device).use();started=time.time();frequency=FREQUENCIES[index]
    write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    with np.load(dc.SOURCE/'inputs.npz') as z:base=z['pars'];physical=z['physical'];g=z['g']
    with np.load(ROOT/'all_target_dc_direct/measured_dc.npz') as z:DC=z['gain'][:,:,0];dcsem=z['SEM'][:,:,0]
    k=dc.kernel();start,stop=contract['design']['partitions'][part]
    for lo in range(start,stop,128):
        hi=min(lo+128,stop);pars=[];cells=[];channels=[];amplitudes=[]
        for i in range(lo,hi):
            steps=[.02*(base[i,1]-11),.05*physical[i,1],.1*physical[i,2],.002*(1+g[i])]
            for channel in range(4 if i<32000 else 3):
                assert steps[channel]>0
                for factor in [1.,.5]:
                    p=base[i].copy();p[20]=channel;p[22]=g[i];p[23]=20. if i<32000 else 10.
                    amp=steps[channel]*factor;p[4]=amp/(physical[i,channel] if channel in [1,2] else 1.)
                    p[5]=2*np.pi*frequency/1000*.1
                    pars.append(p);cells.append(i);channels.append(channel);amplitudes.append(amp)
        amp=np.array(amplitudes);cells=np.array(cells);channels=np.array(channels)
        observed=run(k,np.array(pars),256,4000,1000,contract['design']['base_seeds'][index]+100000*lo)
        gain=(observed[:,:,0]+1j*observed[:,:,1])/(4*amp[:,None]);mean=gain.mean(1)
        sem=np.sqrt(np.mean(abs(gain-mean[:,None])**2,axis=1)/256)
        delta=gain[1::2]-gain[::2];dmean=delta.mean(1);dsem=np.sqrt(np.mean(abs(delta-dmean[:,None])**2,axis=1)/256)
        tolerance=np.maximum.reduce([.1*abs(DC[cells[::2],channels[::2]]),2*dsem,np.full(len(dmean),1e-7)])
        np.savez_compressed(folder/f'batch_{lo:05d}.npz',cell=cells,channel=channels,amplitude=amp,observed=observed,
            gain=mean,complex_SEM=sem,replicate_block_gain=gain.reshape(len(gain),8,32).mean(2),
            paired_amplitude_difference=dmean,paired_amplitude_SEM=dsem,amplitude_tolerance=tolerance,
            amplitude_pass=abs(dmean)<=tolerance,DC_estimable=abs(DC[cells,channels])/np.maximum(dcsem[cells,channels],1e-15)>=10,
            mean_rate_Hz=observed[:,:,2:].mean((1,2))/4)
        write(folder/'progress.json',dict(status='MEASURING',pid=os.getpid(),device=device,frequency_Hz=frequency,
            completed_targets=hi-start,total_targets=stop-start,elapsed_s=time.time()-started,updated_epoch=time.time()))
        if (lo-start)//128%10==0:print('DIRECT FREQUENCY',frequency,part,hi-start,stop-start,flush=True)
    result=dict(status='COMPLETE',frequency_Hz=frequency,part=part,elapsed_s=time.time()-started,formal_bifurcation_allowed=False)
    write(folder/'result.json',result);write(folder/'progress.json',result)


def collect(index):
    folder=OUT/f'frequency_{index:02d}';gain=np.full((40000,4,2),np.nan+0j);sem=np.full(gain.shape,np.nan)
    block=np.full((40000,4,2,8),np.nan+0j);passed=np.zeros((40000,4),bool);coverage=np.zeros(gain.shape,int)
    for part in range(2):
        assert read(folder/f'part_{part}/result.json')['status']=='COMPLETE'
        for path in sorted((folder/f'part_{part}').glob('batch_*.npz')):
            with np.load(path) as z:
                cell=z['cell'];ch=z['channel'];a=np.arange(len(cell))%2
                assert not coverage[cell,ch,a].any();coverage[cell,ch,a]+=1
                gain[cell,ch,a]=z['gain'];sem[cell,ch,a]=z['complex_SEM'];block[cell,ch,a]=z['replicate_block_gain']
                passed[cell[::2],ch[::2]]=z['amplitude_pass']
    expected=np.ones(gain.shape,int);expected[32000:,3]=0;assert np.array_equal(coverage,expected)
    assert np.isfinite(gain[expected>0]).all()
    gain[32000:,3]=0;sem[32000:,3]=0;block[32000:,3]=0
    np.savez_compressed(folder/'measured_response.npz',gain=gain,complex_SEM=sem,replicate_block_gain=block,amplitude_pass=passed,coverage=coverage)
    with np.load(ROOT/'all_target_dc_direct/measured_dc.npz') as z:estimable=abs(z['gain'][:,:,0])/np.maximum(z['SEM'][:,:,0],1e-15)>=10
    summary=[]
    for pop,sl in [('E',slice(0,32000)),('I',slice(32000,None))]:
        for ch in range(4 if pop=='E' else 3):
            mask=estimable[sl,ch];p=passed[sl,ch]
            summary.append(dict(population=pop,channel=ch,targets=len(mask),DC_estimable=int(mask.sum()),
                amplitude_pass_among_estimable=int(p[mask].sum()),amplitude_fail_among_estimable=int((~p[mask]).sum())))
    result=dict(status='COMPLETE_MEASURED_FREQUENCY',frequency_Hz=FREQUENCIES[index],summary=summary,
        dynamic_stability_established=False,formal_bifurcation_allowed=False)
    write(folder/'analysis.json',result);return result


def supervise():
    assert not (OUT/'supervisor.json').exists(),'No silent restart'
    write(OUT/'supervisor.json',dict(pid=os.getpid(),started_epoch=time.time()));results=[]
    # Explicit dependency avoids competing with the bounded fresh-root checks.
    while not (ROOT/'direct_exit_K_pair/result.json').exists():
        write(OUT/'progress.json',dict(status='WAITING_DIRECT_K_PAIR',pid=os.getpid(),updated_epoch=time.time()));time.sleep(10)
    for index in range(2):
        folder=OUT/f'frequency_{index:02d}';folder.mkdir(exist_ok=True);children=[]
        for part in range(2):
            with (folder/f'part_{part}.log').open('x') as log:
                p=subprocess.Popen([PYTHON,__file__,'worker','--index',str(index),'--part',str(part),'--device',str(part)],stdout=log,stderr=subprocess.STDOUT)
            children.append(p)
        while any(p.poll() is None for p in children):
            write(OUT/'progress.json',dict(status='MEASURING_FREQUENCY',frequency_Hz=FREQUENCIES[index],pid=os.getpid(),
                workers=[dict(pid=p.pid,status=p.poll()) for p in children],updated_epoch=time.time()));time.sleep(10)
        if any(p.returncode!=0 for p in children):
            write(OUT/'result.json',dict(status='FAILED_WORKER_NO_AUTORESTART',index=index,returncodes=[p.returncode for p in children]));return
        results.append(collect(index))
    result=dict(status='TWO_DIRECT_FREQUENCY_PROBES_COMPLETE',rows=results,dynamic_stability_established=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','supervise','collect'])
    for name in ['index','part','device']:p.add_argument('--'+name,type=int)
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.index,a.part,a.device) if a.command=='worker' else collect(a.index) if a.command=='collect' else supervise()
