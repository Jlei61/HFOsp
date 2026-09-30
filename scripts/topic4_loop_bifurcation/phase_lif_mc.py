"""Stationary count observer with randomized starting phase, same cell physics.

Local diagnostic only. ``phase_ms=0, dc_during_burn=False`` reproduces the
existing count assay. No native or coupled density engine is modified.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import cupy as cp
from audit_target_root_response import density_condition
import lif_mc


def kernel():
    code=lif_mc.CODE
    replacements={
        'void assay(':'void phase_assay(',
        'unsigned long long seed,int crn){':'unsigned long long seed,int crn,int phase_steps,int warm_dc){',
        'curandStatePhilox4_32_10_t rng;curand_init(seed,crn?(unsigned long long)k:(unsigned long long)id,0,&rng);':
        '''unsigned long long stream=crn==2?(((unsigned long long)(g/2)<<32)+(unsigned long long)k):(crn==1?(unsigned long long)k:(unsigned long long)id);
 curandStatePhilox4_32_10_t rng;curand_init(seed,stream,0,&rng);
 unsigned long long mixed=(seed^stream)+0x9e3779b97f4a7c15ULL;
 mixed=(mixed^(mixed>>30))*0xbf58476d1ce4e5b9ULL;
 mixed=(mixed^(mixed>>27))*0x94d049bb133111ebULL;mixed^=mixed>>31;
 int extra_burn=phase_steps>0?(int)(mixed%(unsigned long long)(phase_steps+1)):0;''',
        'for(int t=-burn;t<steps;t++){':'for(int t=-burn-extra_burn;t<steps;t++){',
        'double osc=(t>=0&&modulated)?':'double osc=((t>=0||warm_dc)&&modulated)?',
        'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else gf=sqrt(1+o);':
        'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else if(channel==2)gf=sqrt(1+o);',
        'if(ref[side]==0){v[side]=p[18]*v[side]+(1-p[18])*cur;':
        'double decay=p[18];if(channel==3 && o!=0.){double h=1.+p[22];cur=(h*cur-17.662847938268442*o)/(h+o);decay=exp(-.1*(h+o)/p[23]);}\n   if(ref[side]==0){v[side]=decay*v[side]+(1-decay)*cur;'
    }
    for old,new in replacements.items():
        assert code.count(old)==1,old
        code=code.replace(old,new)
    return cp.RawKernel(code,'phase_assay',options=('--fmad=false',))


def run(pars,replicas,duration_ms,burn_ms,seed,device=0,phase_ms=1000.,dc_during_burn=False,stream_mode=0):
    cp.cuda.Device(device).use();pars=np.asarray(pars,dtype=np.float64)
    assert np.isfinite(pars).all() and not pars[:,5].any(),'Static response only; dynamic modulation unchanged elsewhere.'
    assert phase_ms>=0 and stream_mode in [0,1,2]
    n=len(pars)*replicas;out=cp.zeros((n,4))
    kernel()(((n+127)//128,),(128,),(cp.asarray(pars),out,np.int32(replicas),np.int32(len(pars)),
        np.int32(round(duration_ms/.1)),np.int32(round(burn_ms/.1)),np.uint64(seed),np.int32(stream_mode),
        np.int32(round(phase_ms/.1)),np.int32(dc_during_burn)))
    return out.get().reshape(len(pars),replicas,4)
