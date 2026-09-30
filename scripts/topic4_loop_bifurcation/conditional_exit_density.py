#!/usr/bin/env python3
"""Actual G/K exit input test, with native joint states and pending arrivals.

All future group counts and global R/G are prescribed observations. This is
an independent conductance-domain diagnostic, not closed-loop validation.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from numba import njit
from campaign import ROOT,read,write,sha
import density_spatial as physical
from conditional_density_inputs import ARRIVAL_CODE,OPS
from observe_native_exit_inputs import OUT as SOURCE

OUT=ROOT/'conditional_exit_density'
CODE=r'''
extern "C" __global__ void exit_steps(double* state,int* refs,unsigned char* memory,
 const double* pars,const double* constants,const double* arr,const double* drive,
 const double* global,const double* pending,const int* members,int* counts,
 int G,int R,int start,int steps,int depth,int cells){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=G*R)return;
 int g=id/R,ref=refs[id],member=members[id];double x[8];
 for(int j=0;j<8;j++)x[j]=state[id*8+j];
 curandStatePhilox4_32_10_t* all=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=all[id];
 for(int k=start;k<start+steps;k++){
  if(k<depth){x[1]+=pending[(long long)k*cells+member]/constants[0];
              x[3]+=pending[((long long)depth+k)*cells+member]/constants[2];}
  float4 n=curand_normal4(&rng);double a[4];
  for(int j=0;j<4;j++)a[j]=arr[((long long)k*4+j)*G+g];
  bool sp=native_cell(x,ref,pars+6*g,constants,a,drive[(long long)k*G+g],global+2*k,30.,n.x,n.z);
  if(sp)counts[(long long)id*10+(k%100)/10]++;
 }
 refs[id]=ref;all[id]=rng;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
'''


@njit
def mean_path(arr,drive,pars,c,initial,pending,observed,substitute):
    out=arr.copy();means=np.empty((len(arr),2,len(pars)))
    syn=np.empty((2,len(pars)));cur=np.empty((2,len(pars)))
    for g in range(len(pars)):
        syn[0,g]=initial[g,1];syn[1,g]=initial[g,3]
        cur[0,g]=initial[g,2];cur[1,g]=initial[g,4]
    for k in range(len(arr)):
        for g in range(len(pars)):
            for j in range(2):
                pulse=pending[k,j,g] if k<len(pending) else 0.
                ext=pars[g,4]*drive[k,g] if j==0 else 0.
                if substitute:
                    target=(observed[k,j,g]-c[2*j+1]*cur[j,g])/(1-c[2*j+1])
                    out[k,j,g]=(target-c[2*j]*syn[j,g]-pulse)*c[4+j]/pars[g,0]/.1-ext
                syn[j,g]=c[2*j]*syn[j,g]+pulse+pars[g,0]/c[4+j]*(out[k,j,g]+ext)*.1
                cur[j,g]=syn[j,g]+(cur[j,g]-syn[j,g])*c[2*j+1]
                means[k,j,g]=cur[j,g]
    return out,means


def prepare(cp):
    z=dict(np.load(SOURCE/'inputs.npz'));initial=dict(np.load(SOURCE/'initial_local_state.npz'))
    geo=dict(np.load(OPS/'geometry.npz'));selected=z['selected_groups'];G=len(selected);P=len(geo['group_size']);T=len(z['time_ms'])
    assert T==33000 and np.array_equal(selected,initial['selected_groups'])
    matrices=[sparse.load_npz(OPS/f'{name}.npz')[selected].tocsr() for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
    spikes=cp.asarray(z['spikes']);sizes=cp.asarray(geo['group_size'],dtype='f8');output=cp.zeros((T,6,G))
    kernel=cp.RawKernel(ARRIVAL_CODE,'apply',options=('--fmad=false',))
    for j,a in enumerate(matrices):
        parts=(cp.asarray(a.indptr,dtype='i4'),cp.asarray(a.indices,dtype='i4'),cp.asarray(a.data))
        kernel((G,T),(128,),(*parts,spikes,sizes,output,np.int32(P),np.int32(G),np.int32(j)))
        cp.cuda.get_current_stream().synchronize()
    arr=output.get()[:,:4].copy();checks=[]
    for k in [0,1,357,358,999,32999]:
        slots=k-np.arange(1,matrices[0].shape[1]//P+1);history=np.zeros((len(slots),P));ok=slots>=0
        history[ok]=z['spikes'][slots[ok]]/geo['group_size']/.1
        oracle=np.array([a@history.ravel() for a in matrices]);err=float(abs(oracle-arr[k]).max());assert err<1e-8
        checks.append(dict(step=k,error=err))
    del spikes,sizes,output
    p=read(OPS/'prepared.json')['params'];E=geo['population'][selected]==0
    pars=np.c_[np.where(E,p['tau_m_E'],p['tau_m_I']),np.where(E,p['tau_ref_E'],p['tau_ref_I'])/.1,
        geo['threshold_mv'][selected],E,np.where(E,p['J_ext_E'],p['J_ext_I']),geo['group_size'][selected]]
    c=np.array([np.exp(-.1/p[n]) for n in ['tau_r_AMPA','tau_d_AMPA','tau_r_GABA','tau_d_GABA']]+[p['tau_r_AMPA'],p['tau_r_GABA'],p['V_reset']])
    ids=initial['selected_group_index'];size=np.bincount(ids,minlength=G);depth=len(initial['ring_sE'])
    order=(int(initial['step'])+np.arange(depth))%depth
    pending=np.stack([initial['ring_sE'][order],initial['ring_sI'][order]])
    initial_means=np.array([initial['state'][ids==g].mean(0) for g in range(G)])
    pending_means=np.array([[np.bincount(ids,weights=pending[j,k],minlength=G)/size for j in range(2)] for k in range(depth)])
    names=z['moment_names'].tolist();actual=z['moments'][:,[names.index('IE'),names.index('II')]]
    arr,projected=mean_path(arr,z['external_rate_per_ms'],pars,c,initial_means,pending_means,actual,False)
    matched,measured=mean_path(arr,z['external_rate_per_ms'],pars,c,initial_means,pending_means,actual,True)
    err=float(abs(measured-actual).max());assert err<1e-9
    # Numeric quadrature copies complete native joint states, including the
    # same cell's pending past spikes. It never independently shuffles Z or K.
    members=np.array([np.flatnonzero(ids==g)[np.arange(8192)%size[g]] for g in range(G)],dtype='i4')
    write(OUT/'input_qa.json',dict(status='PASS',operator_checks=checks,measured_mean_inversion_error=err,
        pending_history_ms=depth*.1,pending='Originalpercellqueuedarrivalspastt0 usedexactly once, pairedwithsamecellinitialstate; newGaussianarrivalsonlyfromobservedpostt0counts.',
        quadrature='Balancedempiricalcopies at8192pergroup; groupaveragethresholds remaincurrentcandidate approximation.',
        total_initial_cells=len(ids)))
    np.savez_compressed(OUT/'input_summary.npz',selected_groups=selected,pars=pars,projected_current_means=projected,
        native_counts=z['spikes'][:,selected],native_moments=z['moments'],moment_names=z['moment_names'],
        time_ms=z['time_ms'],global_R_and_s=z['global_R_and_s'],pending_means=pending_means,initial_means=initial_means)
    return z,initial,pars,c,members,pending,{'projected_full':arr,'measured_full':matched}


class LocalExit:
    def __init__(self,cp,module,arr,z,initial,pars,c,members,pending,seed):
        self.G,self.R=members.shape;self.cp=cp;self.module=module
        self.depth=pending.shape[1];self.cells=pending.shape[2]
        self.state=cp.asarray(initial['state'][members]);self.ref=cp.asarray(initial['ref'][members],dtype='i4')
        self.members=cp.asarray(members);self.pending=cp.asarray(pending)
        self.pars,self.c,self.arr,self.drive,self.glob=[cp.asarray(x) for x in [pars,c,arr,z['external_rate_per_ms'],z['global_R_and_s']]]
        self.counts=cp.zeros((self.G,self.R,10),dtype='i4')
        size=cp.zeros(1,dtype='i4');module.get_function('rng_bytes')((1,),(1,),(size,))
        self.rng=cp.empty(self.G*self.R*int(size.get()[0]),dtype='u1')
        module.get_function('init_rng')(((self.G*self.R+127)//128,),(128,),
            (self.rng,np.int32(self.G*self.R),np.uint64(seed),np.int32(self.R)))
    def advance(self,start,steps):
        self.module.get_function('exit_steps')(((self.G*self.R+127)//128,),(128,),
            (self.state,self.ref,self.rng,self.pars,self.c,self.arr,self.drive,self.glob,self.pending,self.members,self.counts,
             np.int32(self.G),np.int32(self.R),np.int32(start),np.int32(steps),np.int32(self.depth),np.int32(self.cells)))


def main(device,wait):
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_SOURCE_COMPLETE_AND_SCORING',created_epoch=time.time(),
        question='Does the current joint-density local process reproduce firing suppression and Z/K recovery under the actual native exit G/R and incoming activity?',
        source=str(SOURCE),interval_s=[16.7,20.],same16geometric_targets=True,replicas=8192,numerical_seeds=[927651,927652],
        design='Two currentmeanarms(projectedvsmeasured) xtwonumericalstreams. Originalfullsquaredweightvariance,newGaussianpostt0arrivals,actualmembermeanexternaldrive at0.1ms. NativeR/Gprescribed;localV,current,ref,Z,M,K evolveindependently withunchangednative_cell.',
        initial='Originalselectednativecellsfulljointstateandrefractory; pendingqueuedrecurrentarrivalspairedwithsamecells. No covariance reconstructionorzeropast.',
        evaluation='Fixed20ms firingbins; windows16.7-16.9(suppression),16.9-17.6(Gtail),17.6-20(recovery). Allrecordsretained. CompareZ/Kmean anddistribution andrawcurrentmean; no fittedparametersoracceptancetoleranceinvented.',
        limitation='TeacherforcedlocalG/Kdomainvalidation only. No autonomousGfeedback,spatialpropagation,linearresponse,stabilityorformalbranchcertification.',
        bounded='Exactlyfourlocalconditions aftersourcefullengineandobservationsbitwisegate; noothernetworklaunch.',
        engine_sha256=sha(physical.__file__),producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    while not (SOURCE/'replay_audit.json').exists():
        if not wait:raise RuntimeError('Native exit observer gate not complete')
        write(OUT/'progress.json',dict(status='WAITING_NATIVE_REPLAY_GATE',pid=os.getpid(),updated_epoch=time.time()))
        time.sleep(20)
    assert read(SOURCE/'replay_audit.json')['status']=='PASS'
    import cupy as cp
    cp.cuda.Device(device).use();start=time.time()
    z,initial,pars,c,members,pending,inputs=prepare(cp)
    module=cp.RawModule(code=physical.CODE+CODE,options=('--fmad=false',),name_expressions=['rng_bytes','init_rng','exit_steps'])
    # Exact partitioning check crosses an initial pending-arrival boundary.
    a=LocalExit(cp,module,inputs['projected_full'],z,initial,pars,c,members[:,:64],pending,927659)
    b=LocalExit(cp,module,inputs['projected_full'],z,initial,pars,c,members[:,:64],pending,927659)
    a.advance(0,400);b.advance(0,357);b.advance(357,43)
    equal={key:np.array_equal(getattr(a,key).get(),getattr(b,key).get()) for key in ['state','ref','rng','counts']}
    assert all(equal.values());write(OUT/'chunk_qa.json',dict(status='PASS',equal=equal));del a,b
    completed=[]
    for seed in [927651,927652]:
        for name,arr in inputs.items():
            job=f'{name}_num{seed}';e=LocalExit(cp,module,arr,z,initial,pars,c,members,pending,seed);rates=[];mom=[]
            for tick in range(0,33000,100):
                e.counts.fill(0);e.advance(tick,100);rates.append(e.counts.sum(1).get().T/e.R*1000.)
                x=e.state;mom.append(cp.stack([x[:,:,6].mean(1),x[:,:,7].mean(1),x[:,:,5].mean(1),x[:,:,2].mean(1),
                    x[:,:,4].mean(1),x[:,:,6].std(1),x[:,:,7].std(1)],axis=1).get())
                if (tick+100)%5000==0:write(OUT/'progress.json',dict(status='RUNNING',job=job,pid=os.getpid(),time_ms=16700+(tick+100)*.1,completed=completed))
            np.savez_compressed(OUT/f'{job}.npz',rate_Hz=np.concatenate(rates),moments=np.array(mom),
                moment_names=['Z','K','M','IE','II','Zstd','Kstd'],time_ms=16700+np.arange(10,3301,10.),
                final_state=e.state.get(),final_ref=e.ref.get(),selected_groups=z['selected_groups'])
            completed.append(job);print('EXIT CONDITIONAL DENSITY COMPLETE',job,flush=True)
            del e;cp.get_default_memory_pool().free_all_blocks()
    assert sha(physical.__file__)==read(OUT/'contract.json')['engine_sha256']
    result=dict(status='COMPLETE',completed=completed,elapsed_s=time.time()-start,engine_unchanged=True,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);p.add_argument('--wait',action='store_true')
    a=p.parse_args();main(a.device,a.wait)
