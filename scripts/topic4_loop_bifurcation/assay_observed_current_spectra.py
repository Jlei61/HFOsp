#!/usr/bin/env python3
"""Six-target prescribed-spectrum test; not an autonomous noise closure."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from observe_source_time_structure import OUT as TEMPORAL
from observe_source_aggregation import OUT as SOURCE,INITIAL
from conditional_density_inputs import OPS
from native_target_thresholds import load
import run_topic4_loop_zk_conditional as native

OUT=TEMPORAL/'prescribed_spectrum_response'
CELLS=[5148,38408,20590,17487,28942,27474]
CODE=r'''
extern "C" __global__ void replay(const double* current,const double* p,const int* extra,
 int* counts,unsigned char* flags,int R,int N,int steps,int burn,int emit){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=R)return;
 double v=p[4];int ref=(int)p[5],count=0;
 int warm=burn+extra[k];
 for(int t=-warm;t<steps;t++){
  int row=(t+warm)%N;double inf=current[(long long)row*R+k];
  ref=max(0,ref-1);bool fired=false;
  if(ref==0){v=inf+(v-inf)*p[0];if(v>=p[1]){v=p[2];ref=(int)p[3];fired=true;}}
  else v=p[2];
  if(t>=0){count+=fired;if(emit)flags[(long long)t*R+k]=fired;}
 }
 counts[k]=count;
}
'''


def main():
    import cupy as cp
    cp.cuda.Device(1).use();OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(TEMPORAL/'selected_current_replay_exact_thresholds/result.json')['exact_selected_spikes']
    write(OUT/'contract.json',dict(status='FROZEN_BEFORE_PRESCRIBED_SPECTRUM_RESPONSE',created_epoch=time.time(),
        question='Does preserving the measured netcurrent temporal spectrum resolve the remaining scalarvariance Gaussian response error, or is spectral information alone insufficient?',
        selection='Six developmenttargets fixed before new counts: remainingcoreoutlier5148, Imeaninputcontrol38408, and four previouslyassayed recruitmentedge targets20590/17487/28942/27474.',
        design='Actual native individualthreshold/Z/K, fixedobservedmeanM/G. Prescribe full2s netcurrent autospectrum including measured E/Icrosscovariance. Generate1024 independentGaussian Fourier realizations per target,2speriodic,1s+uniform0-1s burn,16srecord (eight repeats, not eight independent samples). Seed929491. Compare to earlier samevariance low-passGaussian andnative2scount.',
        limits='Data-prescribed2sspectrum andfiniteperiodic input, not a selfconsistent autonomous closure, native-network trial or dynamicstability validation. No fitted frequency/time constant; conditional explanatory test only.',
        unit='Independent Gaussianrealizations estimate MCprecision; six selectedtargets from one nativeconditionaltrajectory are not native seeds.',
        physical_kernel='Original membrane/refractory arithmetic, independently checked against every supplied-current native spike before stochasticassay.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    tr=dict(np.load(TEMPORAL/'target_traces.npz'));allcells=tr['cells'];ix=[int(np.flatnonzero(allcells==c)[0]) for c in CELLS]
    current=tr['IE_II_V_M'][:,:,ix];theta,meta=load();state=native.read_pickle(INITIAL)['engine'];p=read(OPS/'prepared.json')['params']
    Z=state['slow']['z'][CELLS];K=np.r_[state['termination_mechanism']['sahp_g'],np.zeros(8000)][CELLS]
    E=np.array(CELLS)<32000;G=30*tr['causal_R_G'][:,1]
    assert np.array_equal(K[None,:]+E[None,:]*Z[None,:]*G[:,None],np.broadcast_to(K,(20000,6)))
    tm=np.where(E,p['tau_m_E'],p['tau_m_I']);decay=np.exp(-.1/tm)**(1+K)
    refs=np.rint(np.where(E,p['tau_ref_E'],p['tau_ref_I'])/.1).astype(int)
    original=sparse.load_npz(TEMPORAL/'source_spikes_0p1ms.npz').tocsr()[CELLS].toarray().T.astype('u1')
    kernel=cp.RawKernel(CODE,'replay',options=('--fmad=false',))
    def run(inf,pars,burn,extra,steps,emit=False):
        inf=np.ascontiguousarray(inf);R=inf.shape[1]
        count=cp.zeros(R,dtype='i4');flags=cp.zeros((steps,R) if emit else (1,1),dtype='u1')
        kernel(((R+127)//128,),(128,),(cp.asarray(inf),cp.asarray(pars),cp.asarray(extra,dtype='i4'),count,flags,
            np.int32(R),np.int32(len(inf)),np.int32(steps),np.int32(burn),np.int32(emit)))
        return count.get(),flags.get() if emit else None
    qa=[]
    for j,c in enumerate(CELLS):
        value=current[:,0,j]-Z[j]*current[:,1,j]-.0005*current[:,3,j]
        if E[j]:value+=K[j]*(-30+17.662847938268442)
        inf=(value+K[j]*(-17.662847938268442))/(1+K[j])
        pars=np.array([decay[j],theta[c],p['V_reset'],refs[j],state['V'][c],state['ref'][c]])
        count,flags=run(inf[:,None],pars,0,np.zeros(1,int),20000,True)
        assert np.array_equal(flags[:,0],original[:,j]),c
        qa.append(dict(cell=c,every_native_spike_exact=True))
    write(OUT/'implementation_qa.json',dict(status='PASS',rows=qa,threshold_source=meta))
    old={r['cell']:r for r in read(SOURCE/'local_response_factorial_exact_thresholds/result.json')['effects']}
    rng=np.random.default_rng(929491);rows=[];started=time.time()
    for j,c in enumerate(CELLS):
        h=1+K[j];raw=current[:,0,j]-Z[j]*current[:,1,j]
        fluct=(raw-raw.mean())/h;amplitude=abs(np.fft.rfft(fluct));amplitude[0]=0.;N=len(fluct)
        mu=(raw.mean()-.0005*current[:,3,j].mean()-30*K[j])/h
        pars=np.array([decay[j],theta[c],p['V_reset'],refs[j],p['V_reset'],0.])
        samples=[];variances=[]
        for batch in range(8):
            real=rng.standard_normal((128,len(amplitude)));imag=rng.standard_normal(real.shape)
            spec=(real+1j*imag)*amplitude/np.sqrt(2);spec[:,0]=0.;spec[:,-1]=real[:,-1]*amplitude[-1]
            x=np.fft.irfft(spec,n=N,axis=1);variances.extend(x.var(1).tolist())
            extra=rng.integers(0,10001,size=128,dtype='i4')
            count,_=run((x+mu).T,pars,10000,extra,160000)
            samples.extend((count/16.).tolist())
        samples=np.array(samples);variances=np.array(variances);expected=float(fluct.var())
        vsem=float(variances.std(ddof=1)/np.sqrt(len(variances)))
        assert abs(variances.mean()-expected)<6*vsem+1e-12
        row=dict(cell=c,native_rate_Hz=float(original[:,j].sum()/2),
            measured_variance_lowpass_Gaussian_rate_Hz=old[c]['rates_Hz'][3],
            measured_spectrum_Gaussian_rate_Hz=float(samples.mean()),MC_SEM_Hz=float(samples.std(ddof=1)/32),
            expected_effective_current_variance=expected,generated_variance_mean=float(variances.mean()),generated_variance_SEM=vsem)
        rows.append(row);np.savez_compressed(OUT/f'target_{c}.npz',rate_samples_Hz=samples,variance_samples=variances)
        write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),completed=len(rows),total=len(CELLS),elapsed_s=time.time()-started));print(row,flush=True)
    result=dict(status='COMPLETE_PRESCRIBED_SPECTRUM_RESPONSE',rows=rows,
        data_prescribed_not_autonomous=True,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);write(OUT/'progress.json',dict(status='COMPLETE',elapsed_s=time.time()-started))


if __name__=='__main__':main()
