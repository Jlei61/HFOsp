"""Paired multi-frequency response of the unchanged conditional cell update.

No network fit. Simultaneous tones require independent single-tone validation
and amplitude checks before using the measured susceptibility in a network.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import cupy as cp
from audit_target_root_response import density_condition
import lif_mc


def wave_table(frequencies,steps,burn):
    """Match the original sine assay phase and zero modulation during burn."""
    f=np.asarray(frequencies,dtype=float);nf=len(f)
    co=np.ones(nf);si=np.zeros(nf);cd=np.cos(2*np.pi*f*.0001);sd=np.sin(2*np.pi*f*.0001)
    out=np.zeros((steps,1+2*nf))
    for t in range(-burn,steps):
        cc=co*cd-si*sd;si=si*cd+co*sd;co=cc
        if t>=0:
            sine=np.where(f==0,1.,si)
            out[t,0]=sine.sum()/np.sqrt(nf)
            out[t,1:1+nf]=np.where(f==0,.5,si)
            out[t,1+nf:]=np.where(f==0,0.,co)
    return out


def kernel(nf):
    assert 1<=nf<=16
    code=lif_mc.CODE
    replacements={
        'void assay(const double* pars,double* output,':'void multisine(const double* pars,const double* waves,double* output,',
        'curand_init(seed,crn?(unsigned long long)k:(unsigned long long)id,0,&rng);':
            'curand_init(seed,((unsigned long long)(g/2)<<32)+(unsigned long long)k,0,&rng);',
        'double sp=0,sm=0,ns[2]={0,0},co=1,si=0,cd=cos(p[5]),sd=sin(p[5]);':
            f'double re[{nf}]={{0}},im[{nf}]={{0}},ns[2]={{0,0}};',
        'double cc=co*cd-si*sd;si=si*cd+co*sd;co=cc;':'',
        'double osc=(t>=0&&modulated)?p[4]*(p[5]==0.?1.:si):0.;':
            f'double osc=(t>=0&&modulated)?p[4]*waves[t*{1+2*nf}]:0.;',
        'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else gf=sqrt(1+o);':
            'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else if(channel==2)gf=sqrt(1+o);',
        'if(ref[side]==0){v[side]=p[18]*v[side]+(1-p[18])*cur;':
            'double decay=p[18];if(channel==3 && o!=0.){double h=1.+p[22];cur=(h*cur-17.662847938268442*o)/(h+o);decay=exp(-.1*(h+o)/p[23]);}\n   if(ref[side]==0){v[side]=decay*v[side]+(1-decay)*cur;',
        'if(modulated){double d=(double)fired[0]-(double)fired[1];sp+=d*(p[5]==0.?.5:si);sm+=d*(p[5]==0.?0.:co);}':
            f'if(modulated){{double d=(double)fired[0]-(double)fired[1];if(d!=0.)for(int j=0;j<{nf};j++){{re[j]+=d*waves[t*{1+2*nf}+1+j];im[j]+=d*waves[t*{1+2*nf}+1+{nf}+j];}}}}',
        'output[id*4]=sp;output[id*4+1]=sm;output[id*4+2]=ns[0];output[id*4+3]=ns[1];':
            f'for(int j=0;j<{nf};j++){{output[id*{2*nf+2}+j]=re[j];output[id*{2*nf+2}+{nf}+j]=im[j];}}output[id*{2*nf+2}+{2*nf}]=ns[0];output[id*{2*nf+2}+{2*nf+1}]=ns[1];'
    }
    for before,after in replacements.items():
        assert code.count(before)==1,before;code=code.replace(before,after)
    return cp.RawKernel(code,'multisine',options=('--fmad=false',))


def run(pars,replicas,duration,burn,seed,frequencies,k=None,waves=None):
    nf=len(frequencies);steps=round(duration/.1);burn_steps=round(burn/.1)
    if waves is None:waves=cp.asarray(wave_table(frequencies,steps,burn_steps))
    if k is None:k=kernel(nf)
    n=len(pars)*replicas;out=cp.zeros((n,2*nf+2))
    k(((n+127)//128,),(128,),(cp.asarray(pars),waves,out,np.int32(replicas),np.int32(len(pars)),
        np.int32(steps),np.int32(burn_steps),np.uint64(seed),np.int32(1)))
    return out.get().reshape(len(pars),replicas,2*nf+2)
