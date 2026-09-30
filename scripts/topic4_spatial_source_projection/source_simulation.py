"""Autonomous LIF particles with source-only projection and exact target profiles."""
from shared_source import *
from particles import parameters
from inputs import generate
from numba import njit
import argparse,time

@njit(cache=True)
def scatter(ring,counts,ptr,target,delay,weight,t):
    slots=(np.arange(ring.shape[0])+t)%ring.shape[0]
    for b in range(len(counts)):
        if counts[b]==0:continue
        for j in range(ptr[b],ptr[b+1]):ring[slots[delay[j]],target[j]]+=weight[j]*counts[b]

@njit(cache=True)
def evolve(ext,start,ptr,pop_region,threshold,ext_index,field_index,contact_weights,raster_index,
           ampa,gaba,params,state):
    dt,arE,adE,arI,adI,decE,decI,refE,refI,incrementE,incrementI,signal,reset=params
    V,ref,gE,cE,gI,cI,ringE,ringI=state
    steps=len(ext);P=len(ptr)-1;M=ringE.shape[0]
    frames=steps//20;raw=np.zeros((frames,15));counts=np.zeros((frames,P),np.uint32)
    field=np.zeros((frames,400),np.uint32);raster=np.zeros((steps,np.max(raster_index)+1),np.bool_)
    moments=np.zeros((steps//100,P,7))
    for k in range(steps):
        t=start+k;slot=t%M;frame=k//20;spikes=np.zeros(P,np.int64)
        for a in range(P):
            reg=pop_region[a];isE=reg<3;decay=decE if isE else decI;refractory=int(refE if isE else refI)
            increment=incrementE if isE else incrementI
            sumv=0.;sumv2=0.;sumref=0.;sume=0.;sume2=0.;sumi=0.;sumei=0.
            for i in range(ptr[a],ptr[a+1]):
                gE[i]=gE[i]*arE+ringE[slot,i];gI[i]=gI[i]*arI+ringI[slot,i]
                ringE[slot,i]=0.;ringI[slot,i]=0.
                external=ext[k,ext_index[i]] if ext_index[i]>=0 else signal*dt
                gE[i]+=external*increment
                cE[i]=gE[i]+(cE[i]-gE[i])*adE;cI[i]=gI[i]+(cI[i]-gI[i])*adI
                ref[i]=max(0,ref[i]-1)
                if ref[i]==0:
                    net=cE[i]-cI[i];V[i]=net+(V[i]-net)*decay
                    if V[i]>=threshold[i]:
                        V[i]=reset;ref[i]=refractory;spikes[a]+=1
                        if isE:
                            field[frame,field_index[i]]+=1
                            for j in range(15):raw[frame,j]+=contact_weights[i,j]
                        if raster_index[i]>=0:raster[k,raster_index[i]]=True
                else:V[i]=reset
                if (k+1)%100==0:
                    sumv+=V[i];sumv2+=V[i]*V[i];sumref+=ref[i]>0;sume+=cE[i];sume2+=cE[i]*cE[i];sumi+=cI[i];sumei+=cE[i]*cI[i]
            counts[frame,a]+=spikes[a]
            if (k+1)%100==0:
                n=ptr[a+1]-ptr[a];moments[k//100,a,0]=sumv/n;moments[k//100,a,1]=sumref/n
                moments[k//100,a,2]=sume/n;moments[k//100,a,3]=max(0.,sume2/n-(sume/n)**2)
                moments[k//100,a,4]=sumi/n;moments[k//100,a,5]=sumei/n-(sume/n)*(sumi/n)
                moments[k//100,a,6]=max(0.,sumv2/n-(sumv/n)**2)
        scatter(ringE,spikes,ampa[0],ampa[1],ampa[2],ampa[3],t)
        scatter(ringI,spikes,gaba[0],gaba[1],gaba[2],gaba[3],t)
    return counts,raw,field,raster,moments

STATE_NAMES=['V','ref','gE','cE','gI','cI','ringE','ringI']

def initial(N,D,p):
    return (np.full(N,p[-1]),np.zeros(N,np.int32),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros((D,N)),np.zeros((D,N)))

def operators(partition='adaptive1'):
    out=[]
    for kind in ('ampa','gaba'):
        z=np.load(operator_directory(partition)/f'{kind}_source_operator.npz');out.append(tuple(z[k] for k in ['ptr','target','delay','weight']))
    return out

def main(seed,duration=12000.,gpu=False,partition='adaptive1'):
    prefix='source_only'+('_half' if partition=='half' else '')
    begin=time.time();label=f'{prefix}{"_cuda" if gpu else ""}_J{J:g}_s{seed}';folder=OUT/('runs' if duration==12000. else f'canary_{duration:g}ms')/label
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    z=model(partition);P=len(z['count']);N=len(z['order']);D=read(operator_directory(partition)/'prepared.json')['delay_slots'];pars=parameters()
    ops=operators(partition);inputs=np.load(generate(seed));ext=inputs['core_arrivals'];steps=round(duration/.1);assert len(ext)>=steps
    reg=z['region'][z['order'][z['ptr'][:-1]]];state=initial(N,D,pars);next_step=0
    checkpoint=folder/'checkpoint.npz'
    if checkpoint.exists():
        ck=np.load(checkpoint);state=tuple(ck[k] for k in STATE_NAMES);next_step=int(ck['next_step'])
    native=np.load(V10/'native/a/trajectory.npz');sample=native['raster_sample_ids'];sample_reg=z['region'][sample]
    inverse=np.empty(N,np.int64);inverse[z['order']]=np.arange(N);raster_index=np.full(N,-1,np.int64);raster_index[inverse[sample]]=np.arange(len(sample))
    if gpu:
        assert read(OUT/'numerical_validation_gpu.json')['status']=='PASS_IMPLEMENTATION_ONLY'
        from source_gpu import GPU
        engine=GPU(z['ptr'],reg,z['vtheta'],z['ext_index'],z['field_sorted'],z['weights_sorted'],raster_index,ops[0],ops[1],pars,state)
    chunk=5000
    for start in range(next_step,steps,chunk):
        num=min(chunk,steps-start)
        assert num%100==0
        if gpu:result=engine.evolve(ext[start:start+num],start)
        else:
            result=evolve(ext[start:start+num],start,z['ptr'],reg,z['vtheta'],z['ext_index'],z['field_sorted'],
                z['weights_sorted'],raster_index,ops[0],ops[1],pars,state)
        assert all(np.isfinite(v).all() for v in result)
        reference=OUT/'runs'/f'{prefix}_J{J:g}_s{seed}'/f'chunk_{start:06d}.npz'
        if gpu and reference.exists():
            cpu=np.load(reference);errors={}
            for j,key in enumerate(['counts','raw','field','raster','moments']):
                if key in ('counts','field','raster'):assert np.array_equal(cpu[key],result[j]),('CPU/GPU',key,start)
                else:
                    errors[key]=float(np.max(abs(cpu[key]-result[j])))
                    assert np.allclose(cpu[key],result[j],rtol=0,atol=2e-9),('CPU/GPU',key,start,errors[key])
            write(folder/f'cpu_parity_{start:06d}.json',dict(status='PASS',start_step=start,end_step=start+num,integer_outputs_identical=True,float_max_errors=errors))
        np.savez_compressed(folder/f'chunk_{start:06d}.npz',**dict(zip(['counts','raw','field','raster','moments'],result)))
        if (start+num)%30000==0 or start+num==steps:
            if gpu:state=engine.host_state()
            assert all(np.isfinite(v).all() for v in state)
            tmp=checkpoint.with_name('checkpoint.tmp.npz')
            np.savez(tmp,next_step=start+num,**dict(zip(STATE_NAMES,state)));tmp.replace(checkpoint)
        write(folder/'progress.json',dict(status='SIMULATING',simulated_ms=(start+num)*.1,seconds=time.time()-begin,resumed_from_step=next_step))
        print(label,(start+num)*.1,'ms',time.time()-begin,flush=True)
    chunks=[np.load(folder/f'chunk_{s:06d}.npz') for s in range(0,steps,chunk)]
    counts,raw,field,raster,moments=[np.concatenate([c[k] for c in chunks]) for k in ['counts','raw','field','raster','moments']]
    six=np.stack([counts[:,reg==r].sum(1) for r in range(6)],axis=1);times,ids=np.nonzero(raster)
    extra={}
    if partition=='half':
        parent=model();pc=np.stack([counts[:,z['parent_group']==a].sum(1) for a in range(len(parent['count']))],axis=1)
        assert np.array_equal(pc.sum(1),counts.sum(1))
        extra['common_group_contact_envelope']=smooth_contacts((pc/parent['count'])@parent['contact_weights'].T)
    np.savez_compressed(folder/'trajectory.npz',six_counts=six,group_counts=counts,contact_envelope=smooth_contacts(raw),
        group_contact_envelope=smooth_contacts((counts/z['count'])@z['contact_weights'].T),field_counts=field.reshape(-1,20,20),
        raster_times_ms=times*.1,raster_neuron_ids=sample[ids],raster_regions=sample_reg[ids],raster_sample_ids=sample,
        population_moments=moments.astype(np.float32),nu_core=inputs['nu_core'][:steps].astype(np.float32),**extra)
    sizes=np.bincount(z['region'],minlength=6)
    write(folder/'result.json',dict(status='COMPLETE',model='source-only projected spatial LIF particles',J=J,seed=seed,duration_ms=duration,
        analysis_ms=[2000,duration],seconds_this_process=time.time()-begin,resumed_from_step=next_step,
        mean_six_hz=six[1000:].mean(0)/sizes/.002 if duration>2000 else None,cells=N,source_populations=P,partition=partition,
        target_space_and_delay_profiles='original, retained separately for each receiving cell',
        original_thresholds=True,original_private_afferent_draws=True,fitted_parameters=0,backend='CUDA ordered target ownership' if gpu else 'CPU ordered source scatter',
        limitations='Source spikes are averaged within each spatial group; individual source correlations are not preserved. All neuron states retained; not a low-dimensional rate model.'))
    write(folder/'progress.json',dict(status='COMPLETE',simulated_ms=duration,seconds=time.time()-begin))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=848101);p.add_argument('--duration',type=float,default=12000.)
    p.add_argument('--gpu',action='store_true');p.add_argument('--partition',choices=['adaptive1','half'],default='adaptive1')
    a=p.parse_args();main(a.seed,a.duration,a.gpu,a.partition)
