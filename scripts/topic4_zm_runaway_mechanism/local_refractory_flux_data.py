"""Record population spike flux and refractory availability on TRAIN inputs.

This is local LIF calibration, never a particle replacement for the spatial
rate network. Every original per-path phase count must be bitwise unchanged.
"""
from common import OUT,np,read,write,log
from native_cycle_waveform_response import CODE
from datetime import datetime
from pathlib import Path
import argparse
import hashlib

DEST=OUT/'local_refractory_flux'
PARENT=OUT/'nonlinear_rate_response'


def register():
    DEST.mkdir(exist_ok=True);path=DEST/'contract.json';assert not path.exists()
    rows=read(PARENT/'profiles.json')['rows'];ids=[r['id'] for r in rows if r['split']=='train']
    assert ids==list(range(224))
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),
        question='Can a local response be calibrated as spike flux from the currently available population, retaining actual reset/refractory history rather than imposing the stationary rate ceiling instantaneously?',
        scope='Calibration recording only. No new spatial model, fitted hazard, onset result or bifurcation is claimed.',
        source=str(PARENT),source_code_sha256=hashlib.sha256(CODE.encode()).hexdigest(),
        profile_ids=ids,validation_profiles_excluded=True,replicates=4096,dt_ms=.1,seed=920043,
        recordings='Per-path original128phase counts; aggregate spike counts and eligible-neuron counts at every native step, including burn.',
        identity='At stepk before firing, available_count=R-sum(spike_counts[j],j=k-nref+1,...,k-1), with spikes before initialization zero. Native decrement-before-membrane convention retained.',
        interpretation='Counts/available_count is an empirical conditional firing probability over0.1ms, not automatically a continuous-time hazard. Any continuous response must separately specify and verify its time discretization, stationary limit and derivatives.',
        preserved='Same original input samples, interpolation, membrane/synaptic/reset/refractory update and noise stream. No native future network spikes used for prediction. Actual conditional flux labels only enter local training.',
        acceptance='Original per-path phase counts bitwise equal; aggregate availability identity exact in integers;0<=spikecount<=availablecount<=R; independent time/phase/count readout.',
        budget='One replay of224existingTRAINING profiles only. No validation targets or new input-domain tuning. Only completed artifacts may be explicitly resumed; no automatic rerun.',
        stop='This data script does not launch fitting or network simulations. A refractory-aware candidate must have its own explicit equation, scope and validation before use.'))


def acquire(device):
    import cupy as cp
    c=read(DEST/'contract.json');assert hashlib.sha256(CODE.encode()).hexdigest()==c['source_code_sha256']
    assert not (DEST/'jobs.json').exists()
    source=np.load(PARENT/'prepared.npz');profiles=read(PARENT/'profiles.json')['rows']
    R=c['replicates'];B=128;dt=c['dt_ms'];assert R%128==0
    code=CODE.replace('const double* wave,unsigned int* counts,',
        'const double* wave,unsigned int* counts,unsigned int* stepcounts,unsigned int* eligiblecounts,')
    old='double cur=mu+ia-ig;bool fired=false;ref=max(0,ref-1);'
    assert code.count(old)==1
    code=code.replace(old,'double cur=mu+ia-ig;bool eligible=(ref<=1);bool fired=false;ref=max(0,ref-1);')
    old='if(t>=0&&fired){int bin=min((int)(phase*B),B-1);counts[id*B+bin]++;}'
    assert code.count(old)==1
    code=code.replace(old,old+'''
     unsigned int hit=__ballot_sync(0xffffffff,fired);
     unsigned int free=__ballot_sync(0xffffffff,eligible);
     if((threadIdx.x&31)==0){
       long long index=(long long)g*(steps+burn)+t+burn;
       if(hit)atomicAdd(stepcounts+index,(unsigned int)__popc(hit));
       if(free)atomicAdd(eligiblecounts+index,(unsigned int)__popc(free));
     }
    ''')
    cp.cuda.Device(device).use();kernel=cp.RawKernel(code,'waveform',options=('--fmad=false',))
    dest=DEST/'training_data';dest.mkdir(exist_ok=True)
    progress=dict(status='RUNNING',expected=224,completed=[],pid=__import__('os').getpid());write(DEST/'jobs.json',progress)
    for index in c['profile_ids']:
        row=profiles[index];assert row['split']=='train' and row['replicates']==R
        wave=source['wave'][index:index+1];pars=source['pars'][index:index+1]
        steps,burn=row['record_steps'],row['burn_steps'];T=row['period_ms'];W=wave.shape[-1]
        path=dest/f'profile{index:03d}.npz';assert not path.exists()
        counts=cp.zeros((1,R,B),dtype=cp.uint32);spikes=cp.zeros((1,steps+burn),dtype=cp.uint32);eligible=cp.zeros_like(spikes)
        kernel((R//128,),(128,),(cp.asarray(pars),cp.asarray(wave),counts,spikes,eligible,
            np.int32(1),np.int32(R),np.int32(W),np.int32(B),np.int32(steps),np.int32(burn),float(dt),float(T),np.uint64(c['seed'])))
        count=counts.get()[0];fired=spikes.get()[0];available=eligible.get()[0]
        target=np.load(PARENT/f'local_data/profile{index:03d}.npz')['counts']
        assert np.array_equal(count,target),f'Original spike replay failed at profile{index}'
        nref=int(pars[0,19]);cs=np.r_[np.int64(0),np.cumsum(fired,dtype=np.int64)]
        k=np.arange(len(fired));past=cs[k]-cs[np.maximum(k-nref+1,0)]
        expected=R-past
        assert np.array_equal(expected,available),(index,int(abs(expected-available).max()))
        assert np.all(fired<=available) and np.all(available<=R)
        assert fired[burn:].sum(dtype=np.uint64)==count.sum(dtype=np.uint64)
        np.savez_compressed(path,spike_counts=fired,available_counts=available,phase_counts=count,
            dt_ms=dt,period_ms=T,burn_steps=burn,record_steps=steps,replicates=R,nref_steps=nref,
            profile_id=index,pop=row['pop'],threshold_mv=pars[0,1])
        progress['completed'].append(index);write(DEST/'jobs.json',progress)
        if (index+1)%16==0:log('REFRACTORY FLUX TRAIN DATA',len(progress['completed']),224)
    progress['status']='COMPLETE';write(DEST/'jobs.json',progress)
    write(DEST/'replay_qa.json',dict(status='PASS',profiles=224,per_path_phase_counts_bitwise=True,
        exact_integer_availability_identity=True,all_probabilities_valid=True,validation_targets_used=False,
        model_promoted=False,scope=c['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','acquire']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    register() if a.command=='register' else acquire(a.device)
