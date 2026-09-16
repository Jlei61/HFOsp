#!/usr/bin/env python3
"""Source-equation execution port of LAS-Model Exp1 (100 x 100 rate field).

Original MIT copyright (c) 2019 Jyun-you Liou; see reference source_license.txt.
Only observation storage and plotting differ from the archived experiment.
"""
import json
from dataclasses import replace,asdict
import time
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
import run_topic4_liou_original_reference as ref

class Projection2D:
    def __init__(self,n=100):
        self.n=n
        self.ke=ref.gaussian_kernel(n,.02)
        self.ki=ref.gaussian_kernel(n,.03)
    def conv(self,x,k):
        a=x.reshape(self.n,self.n)
        return convolve1d(convolve1d(a,k,axis=0,mode='constant',cval=0.),k,axis=1,mode='constant',cval=0.)
    def __call__(self,x):
        # Original global denominator is prod(O.n), including inactive outside-disk cells.
        return 100.*self.conv(x,self.ke).ravel(),250.*self.conv(x,self.ki).ravel()+50.*x.mean(),50.*x.mean()

def qa():
    p=Projection2D(100);rng=np.random.RandomState(401);x=rng.rand(100,100)
    errs=[]
    for k in [p.ke,p.ki]:
        exact=convolve2d(x,np.outer(k,k),mode='same',boundary='fill')
        err=float(np.max(np.abs(exact-p.conv(x.ravel(),k))));assert err<1e-12;errs.append(err)
    ref.write_json(ref.OUT/'reference_2d_qa.json',{'status':'PASS','separable_vs_author_full_2d_convolution_max_abs_error':errs,'gaussian_halfwidths':[len(p.ke)//2,len(p.ki)//2],'global_denominator':10000})

def run():
    qa()
    n=100
    p=replace(ref.protocol('exp2a'),experiment='exp1',n=n*n,e_l=-58.,duration_ms=99990)
    folder=ref.OUT/'reference_runs/exp1_2d'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise RuntimeError('Completed result exists')
    meta=asdict(p);meta.update({'shape':[n,n],'no_core':True,'stimulus_center':[.5,.1],'stimulus_radius':.05,'boundary':'Zero exterior plus source circular output mask','global_denominator':n*n})
    ref.write_json(folder/'protocol.json',meta)
    ij=np.indices((n,n))
    radius=np.sqrt(((ij[0]-n/2+.5)/n)**2+((ij[1]-n/2+.5)/n)**2)
    mask=(radius<.5).ravel()
    stimulus=200.*(np.sqrt((ij[0]/(n-1)-.5)**2+(ij[1]/(n-1)-.1)**2)<.05).ravel()
    zeros=np.zeros(n*n)
    s=ref.initial_state(p);proj=Projection2D(n)
    pe,pi=zeros.copy(),zeros.copy();pg=0.;gf=0.
    nt=p.duration_ms//10+1
    state=np.lib.format.open_memmap(folder/'state.npy',mode='w+',dtype='float32',shape=(nt,6,n*n))
    field=np.lib.format.open_memmap(folder/'field_Hz.npy',mode='w+',dtype='float32',shape=(nt,n,n))
    trace=np.zeros((p.duration_ms,7),np.float32)
    start=time.time()
    for k in range(p.duration_ms):
        if k%10==0:
            state[k//10]=s[:6];field[k//10]=s[6].reshape(n,n)*1000.
        ref.update(s,pe,pi,stimulus if 2000<k<5000 else zeros,zeros,p)
        s[6]*=mask
        gf=(gf+pg/15.)*np.exp(-1/15.)
        trace[k]=[s[6,mask].mean()*1000,s[6].mean()*1000,s[2,mask].mean(),s[3,mask].mean()/.2,s[1,mask].mean(),s[5,mask].mean()/.2,gf/.2]
        pe,pi,pg=proj(s[6])
        if (k+1)%1000==0:
            assert np.all(np.isfinite(s)) and np.all(s[2]>0)
            ref.write_json(folder/'progress.json',{'status':'RUNNING','time_s':(k+1)/1000,'target_s':p.duration_ms/1000,'elapsed_wall_s':time.time()-start,'disk_mean_rate_Hz':float(trace[k,0]),'disk_Cl_mean_mM':float(trace[k,2])})
    state[-1]=s[:6];field[-1]=s[6].reshape(n,n)*1000;state.flush();field.flush()
    np.savez_compressed(folder/'traces.npz',trace=trace,time_ms=np.arange(1,p.duration_ms+1),field_time_ms=np.arange(nt)*10,mask=mask.reshape(n,n),stimulus_pA=stimulus.reshape(n,n),kernel_E=proj.ke,kernel_I=proj.ki,trace_names=np.array(['disk_E_Hz','square_E_Hz','disk_Cl_mM','disk_gK_nS','disk_phi_mV','disk_gI_nS','global_gI_nS']))
    result={'status':'COMPLETE','observed_s':p.duration_ms/1000,'elapsed_wall_s':time.time()-start,'source_experiment':'Exp1.m','source_commit':'95ca7bdf71edbc46b3b26f6a5b73c43c5aa90ca7','tail5_disk_E_Hz':float(trace[-5000:,0].mean()),'final_Cl_mM':float(trace[-1,2]),'active_disk_cells':int(mask.sum()),'square_cells':n*n,'human_visual_review':'PENDING'}
    ref.write_json(folder/'result.json',result);ref.write_json(folder/'progress.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
