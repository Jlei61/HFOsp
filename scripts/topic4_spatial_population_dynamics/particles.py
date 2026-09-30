"""Finite-population LIF dynamics with an autonomous delayed block graph.

Shared linear recurrent synapses are factored exactly within a block; membrane,
threshold, reset/refractory and private core afferents stay neuron-resolved.
No native firing history drives this model.
"""
from shared import *
from scipy import sparse
from numba import njit
import argparse,time
from inputs import generate

@njit(cache=True)
def scatter(ring,counts,ptr,columns,weights,absolute_step,P,M):
    for b in range(P):
        if counts[b]==0:continue
        for k in range(ptr[b],ptr[b+1]):
            d=columns[k]//P;a=columns[k]%P
            ring[(absolute_step+d)%M,a]+=weights[k]*counts[b]

@njit(cache=True)
def evolve(ext,start,ptr,pop_region,threshold,ext_index,field_index,contact_weights,raster_index,
           ampa,gaba,params,state,gains=None,deterministic=None):
    dt,arE,adE,arI,adI,decE,decI,refE,refI,incrementE,incrementI,signal,reset=params
    V,ref,gext,cext,gE,cE,gI,cI,ringE,ringI=state
    steps=len(ext);P=len(ptr)-1;M=ringE.shape[0];N=len(V)
    frames=steps//20;raw_contacts=np.zeros((frames,15));counts=np.zeros((frames,P),np.uint32)
    field=np.zeros((frames,400),np.uint32)
    raster=np.zeros((steps,np.max(raster_index)+1),np.bool_)
    moments=np.zeros((steps//100,P,6)) # mean V, refractory fraction, E mean/var, I mean, EI covariance
    for k in range(steps):
        t=start+k;slot=t%M;frame=k//20;spikes=np.zeros(P,np.int64)
        for a in range(P):
            reg=pop_region[a];isE=reg<3
            additional=0. if reg<2 else signal*dt*(incrementE if isE else incrementI)
            gE[a]=gE[a]*arE+ringE[slot,a]+additional;gI[a]=gI[a]*arI+ringI[slot,a]
            ringE[slot,a]=0.;ringI[slot,a]=0.
            cE[a]=gE[a]+(cE[a]-gE[a])*adE;cI[a]=gI[a]+(cI[a]-gI[a])*adI
            decay=decE if isE else decI;refractory_steps=int(refE if isE else refI)
            sumv=0.;sumref=0.;sume=0.;sume2=0.;sumi=0.;sumei=0.
            for i in range(ptr[a],ptr[a+1]):
                drive=cE[a];inhibition=cI[a]
                if gains is not None:
                    base=0. if reg<2 else deterministic[k,0 if isE else 1]
                    drive=base+gains[i,0]*(drive-base);inhibition*=gains[i,1]
                if ext_index[i]>=0:
                    gext[i]=gext[i]*arE+ext[k,ext_index[i]]*incrementE
                    cext[i]=gext[i]+(cext[i]-gext[i])*adE;drive+=cext[i]
                ref[i]=max(0,ref[i]-1)
                if ref[i]==0:
                    net=drive-inhibition;V[i]=net+(V[i]-net)*decay
                    if V[i]>=threshold[i]:
                        V[i]=reset;ref[i]=refractory_steps;spikes[a]+=1
                        if isE:
                            field[frame,field_index[i]]+=1
                            for j in range(15):raw_contacts[frame,j]+=contact_weights[i,j]
                        if raster_index[i]>=0:raster[k,raster_index[i]]=True
                else:V[i]=reset
                if (k+1)%100==0:
                    sumv+=V[i];sumref+=ref[i]>0;sume+=drive;sume2+=drive*drive;sumi+=inhibition;sumei+=drive*inhibition
            counts[frame,a]+=spikes[a]
            if (k+1)%100==0:
                n=ptr[a+1]-ptr[a];moments[k//100,a,0]=sumv/n;moments[k//100,a,1]=sumref/n
                moments[k//100,a,2]=sume/n;moments[k//100,a,3]=max(0.,sume2/n-(sume/n)**2)
                moments[k//100,a,4]=sumi/n;moments[k//100,a,5]=sumei/n-(sume/n)*(sumi/n)
        scatter(ringE,spikes,ampa[0],ampa[1],ampa[2],t,P,M)
        scatter(ringI,spikes,gaba[0],gaba[1],gaba[2],t,P,M)
    return counts,raw_contacts,field,raster,moments

def parameters():
    cfg=read(PRIOR/'model_config.json');p=cfg['params'];dt=p['dt']
    return np.array([dt,np.exp(-dt/p['tau_r_AMPA']),np.exp(-dt/p['tau_d_AMPA']),
        np.exp(-dt/p['tau_r_GABA']),np.exp(-dt/p['tau_d_GABA']),np.exp(-dt/p['tau_m_E']),np.exp(-dt/p['tau_m_I']),
        round(p['tau_ref_E']/dt),round(p['tau_ref_I']/dt),p['tau_m_E']/p['tau_r_AMPA']*p['J_ext_E'],
        p['tau_m_I']/p['tau_r_AMPA']*p['J_ext_I'],cfg['signal_per_ms'],p['V_reset']])

def operators(Jvalue=J,partition='adaptive'):
    suffix='' if partition=='adaptive' else '_'+partition
    model=np.load(OUT/f'model{suffix}.npz');reg=model['region'][model['order'][model['ptr'][:-1]]];P=len(reg);ops=[]
    for kind in ('ampa','gaba'):
        mat=sparse.load_npz(OUT/f'{kind}_mean{suffix}.npz');data=mat.data.copy()
        if kind=='ampa':
            src=np.repeat(np.arange(P),np.diff(mat.indptr));dst=mat.indices%P
            scale=(reg[src]<2)&(reg[src]==reg[dst]);data[scale]*=Jvalue/J
        ops.append((mat.indptr.astype(np.int64),mat.indices.astype(np.int64),data))
    return ops

def deterministic_trace(steps):
    p=parameters();dt,ae,be=p[:3];g=np.zeros(2);c=g.copy();trace=np.empty((steps,2))
    for k in range(steps):
        g=g*ae+p[11]*dt*p[9:11];c=g+(c-g)*be;trace[k]=c
    return trace

def main(seed,Jvalue,duration,partition='adaptive',receiver_gains=False):
    suffix='' if partition=='adaptive' else '_'+partition
    kind='gain' if receiver_gains else 'mean'
    started=time.time();label=f'{kind}{suffix}_J{Jvalue:g}_s{seed}';folder=OUT/('runs' if duration==12000. else 'canary')/label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    z=np.load(OUT/f'model{suffix}.npz');P=len(z['count']);N=len(z['order']);D=read(OUT/f'prepared{suffix}.json')['delay_slots']
    ex=np.load(generate(seed));arrivals=ex['core_arrivals'];steps=round(duration/.1);ops=operators(Jvalue,partition);pars=parameters()
    gains=np.load(OUT/f'node_gain{suffix}.npz')['gain_sorted'] if receiver_gains else None
    if receiver_gains:assert Jvalue==J,'Node gains must be re-derived before changing J'
    det=deterministic_trace(steps) if receiver_gains else None
    state=(np.full(N,pars[-1]),np.zeros(N,np.int32),np.zeros(N),np.zeros(N),
        np.zeros(P),np.zeros(P),np.zeros(P),np.zeros(P),np.zeros((D,P)),np.zeros((D,P)))
    reg=z['region'][z['order'][z['ptr'][:-1]]];buffers=[[],[],[],[],[]]
    for start in range(0,steps,10000):
        num=min(10000,steps-start)
        result=evolve(arrivals[start:start+num],start,z['ptr'],reg,z['vtheta'],z['ext_index'],z['field_sorted'],
            z['weights_sorted'],z['raster_index'],ops[0],ops[1],pars,state,gains,None if det is None else det[start:start+num])
        assert all(np.isfinite(x).all() for x in state)
        for out,part in zip(buffers,result):out.append(part)
        write(folder/'progress.json',dict(status='SIMULATING',simulated_ms=(start+num)*.1,seconds=time.time()-started))
        print(label,(start+num)*.1,'ms',time.time()-started,flush=True)
    counts,raw,field,raster,moments=[np.concatenate(parts) for parts in buffers]
    six=np.stack([counts[:,reg==r].sum(1) for r in range(6)],axis=1)
    env=smooth_contacts(raw);group_env=smooth_contacts((counts/z['count'])@z['contact_weights'].T)
    times,ids=np.nonzero(raster)
    np.savez_compressed(folder/'trajectory.npz',six_counts=six,group_counts=counts,
        contact_envelope=env,group_contact_envelope=group_env,field_counts=field.reshape(-1,20,20),
        raster_times_ms=times*.1,raster_neuron_ids=z['raster_neuron_ids'][ids],raster_regions=z['raster_regions'][ids],
        raster_sample_ids=z['raster_neuron_ids'],population_moments=moments.astype(np.float32),
        nu_core=ex['nu_core'][:steps].astype(np.float32))
    np.savez_compressed(folder/'final_state.npz',**{name:value for name,value in zip(['V','ref','gext','cext','gE','cE','gI','cI','ringE','ringI'],state)})
    sizes=np.bincount(z['region'],minlength=6)
    write(folder/'result.json',dict(status='COMPLETE',model='delayed spatial finite-particle LIF',receiver_gains=receiver_gains,J=Jvalue,seed=seed,duration_ms=duration,
        analysis_ms=[2000,duration],seconds=time.time()-started,mean_six_hz=six[1000:].mean(0)/sizes/.002 if duration>2000 else None,
        cells=N,populations=P,partition=partition,original_thresholds=True,original_private_afferent_draws=True,
        limitations='Only group input means, optionally measured static receiver E/I gains; source/delay-specific heterogeneity and private recurrent fluctuations absent; not a low-dimensional rate model.'))
    write(folder/'progress.json',dict(status='COMPLETE',simulated_ms=duration,seconds=time.time()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=848101);p.add_argument('--J',type=float,default=J)
    p.add_argument('--duration',type=float,default=12000.);p.add_argument('--partition',default='adaptive')
    p.add_argument('--receiver-gains',action='store_true')
    a=p.parse_args();main(a.seed,a.J,a.duration,a.partition,a.receiver_gains)
