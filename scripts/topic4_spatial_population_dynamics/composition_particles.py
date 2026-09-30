"""Spatial population particles retaining measured six-source composition."""
from shared import *
from particles import parameters,operators,deterministic_trace
from inputs import generate
from numba import njit
import argparse,time

@njit(cache=True)
def scatter_composition(ring,counts,ptr,columns,weights,source_region,t,P,M):
    for b in range(P):
        if counts[b]==0:continue
        q=source_region[b]
        for k in range(ptr[b],ptr[b+1]):
            d=columns[k]//P;a=columns[k]%P
            ring[(t+d)%M,a,q]+=weights[k]*counts[b]

@njit(cache=True)
def evolve(ext,start,ptr,pop_region,threshold,ext_index,field_index,contact_weights,raster_index,
           ampa,gaba,params,state,gains,deterministic):
    dt,arE,adE,arI,adI,decE,decI,refE,refI,incrementE,incrementI,signal,reset=params
    V,ref,gext,cext,g,c,ring=state
    steps=len(ext);P=len(ptr)-1;M=ring.shape[0]
    frames=steps//20;raw=np.zeros((frames,15));counts=np.zeros((frames,P),np.uint32)
    field=np.zeros((frames,400),np.uint32);raster=np.zeros((steps,np.max(raster_index)+1),np.bool_)
    moments=np.zeros((steps//100,P,6))
    for k in range(steps):
        t=start+k;slot=t%M;frame=k//20;spikes=np.zeros(P,np.int64)
        for a in range(P):
            reg=pop_region[a];isE=reg<3
            for q in range(6):
                ar=arE if q<3 else arI;ad=adE if q<3 else adI
                g[a,q]=g[a,q]*ar+ring[slot,a,q];ring[slot,a,q]=0.
                c[a,q]=g[a,q]+(c[a,q]-g[a,q])*ad
            decay=decE if isE else decI;refractory=int(refE if isE else refI)
            base=0. if reg<2 else deterministic[k,0 if isE else 1]
            sumv=0.;sumref=0.;sume=0.;sume2=0.;sumi=0.;sumei=0.
            for i in range(ptr[a],ptr[a+1]):
                drive=base;inhibition=0.
                for q in range(3):
                    drive+=gains[i,q]*c[a,q];inhibition+=gains[i,q+3]*c[a,q+3]
                if ext_index[i]>=0:
                    gext[i]=gext[i]*arE+ext[k,ext_index[i]]*incrementE
                    cext[i]=gext[i]+(cext[i]-gext[i])*adE;drive+=cext[i]
                ref[i]=max(0,ref[i]-1)
                if ref[i]==0:
                    net=drive-inhibition;V[i]=net+(V[i]-net)*decay
                    if V[i]>=threshold[i]:
                        V[i]=reset;ref[i]=refractory;spikes[a]+=1
                        if isE:
                            field[frame,field_index[i]]+=1
                            for j in range(15):raw[frame,j]+=contact_weights[i,j]
                        if raster_index[i]>=0:raster[k,raster_index[i]]=True
                else:V[i]=reset
                if (k+1)%100==0:
                    sumv+=V[i];sumref+=ref[i]>0;sume+=drive;sume2+=drive*drive;sumi+=inhibition;sumei+=drive*inhibition
            counts[frame,a]+=spikes[a]
            if (k+1)%100==0:
                n=ptr[a+1]-ptr[a];moments[k//100,a,0]=sumv/n;moments[k//100,a,1]=sumref/n
                moments[k//100,a,2]=sume/n;moments[k//100,a,3]=max(0.,sume2/n-(sume/n)**2)
                moments[k//100,a,4]=sumi/n;moments[k//100,a,5]=sumei/n-(sume/n)*(sumi/n)
        scatter_composition(ring,spikes,ampa[0],ampa[1],ampa[2],pop_region,t,P,M)
        scatter_composition(ring,spikes,gaba[0],gaba[1],gaba[2],pop_region,t,P,M)
    return counts,raw,field,raster,moments

def initial(N,P,D,p):
    return (np.full(N,p[-1]),np.zeros(N,np.int32),np.zeros(N),np.zeros(N),np.zeros((P,6)),np.zeros((P,6)),np.zeros((D,P,6)))

def main(seed,duration=12000.):
    start_time=time.time();label=f'composition_adaptive1_J{J:g}_s{seed}'
    folder=OUT/('runs' if duration==12000. else 'canary')/label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    z=np.load(OUT/'model_adaptive1.npz');P=len(z['count']);N=len(z['order']);D=read(OUT/'prepared_adaptive1.json')['delay_slots']
    ex=np.load(generate(seed));arrivals=ex['core_arrivals'];steps=round(duration/.1);ops=operators(J,'adaptive1');p=parameters()
    gains=np.load(OUT/'source_composition_adaptive1.npz')['gain_sorted'];det=deterministic_trace(steps);state=initial(N,P,D,p)
    reg=z['region'][z['order'][z['ptr'][:-1]]];buffers=[[],[],[],[],[]]
    for start in range(0,steps,10000):
        num=min(10000,steps-start)
        result=evolve(arrivals[start:start+num],start,z['ptr'],reg,z['vtheta'],z['ext_index'],z['field_sorted'],
            z['weights_sorted'],z['raster_index'],ops[0],ops[1],p,state,gains,det[start:start+num])
        assert all(np.isfinite(x).all() for x in state)
        for buf,value in zip(buffers,result):buf.append(value)
        write(folder/'progress.json',dict(status='SIMULATING',simulated_ms=(start+num)*.1,seconds=time.time()-start_time))
        print(label,(start+num)*.1,'ms',time.time()-start_time,flush=True)
    counts,raw,field,raster,moments=[np.concatenate(b) for b in buffers]
    six=np.stack([counts[:,reg==r].sum(1) for r in range(6)],axis=1);times,ids=np.nonzero(raster)
    np.savez_compressed(folder/'trajectory.npz',six_counts=six,group_counts=counts,contact_envelope=smooth_contacts(raw),
        group_contact_envelope=smooth_contacts((counts/z['count'])@z['contact_weights'].T),field_counts=field.reshape(-1,20,20),
        raster_times_ms=times*.1,raster_neuron_ids=z['raster_neuron_ids'][ids],raster_regions=z['raster_regions'][ids],
        raster_sample_ids=z['raster_neuron_ids'],population_moments=moments.astype(np.float32),nu_core=ex['nu_core'][:steps].astype(np.float32))
    np.savez_compressed(folder/'final_state.npz',**dict(zip(['V','ref','gext','cext','g','c','ring'],state)))
    sizes=np.bincount(z['region'],minlength=6)
    write(folder/'result.json',dict(status='COMPLETE',model='spatial finite-particle LIF with measured source-region composition',J=J,seed=seed,duration_ms=duration,
        analysis_ms=[2000,duration],seconds=time.time()-start_time,mean_six_hz=six[1000:].mean(0)/sizes/.002,
        cells=N,populations=P,partition='adaptive1',original_thresholds=True,original_private_afferent_draws=True,fitted_parameters=0,
        limitations='Still averages within-source-region target-specific spatial/delay profiles and private recurrent fluctuations. Not a low-dimensional rate model.'))
    write(folder/'progress.json',dict(status='COMPLETE',simulated_ms=duration,seconds=time.time()-start_time))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=848101);p.add_argument('--duration',type=float,default=12000.)
    a=p.parse_args();main(a.seed,a.duration)
