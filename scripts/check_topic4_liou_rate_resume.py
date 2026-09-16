#!/usr/bin/env python3
"""Source-checkpoint continuation to distinguish early rounding from code drift."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat,savemat
from scipy.signal import convolve
import run_topic4_liou_original_reference as ref

r=ref.OUT;q=r/'octave_source_validation'
p=ref.protocol('exp2a');proj=ref.Projection(p);zeros=np.zeros(p.n)
cp=loadmat(q/'exp2a_source_epsilon_probe_all.mat',variable_names=['checkpoint'])['checkpoint']
s=ref.initial_state(p);pe=zeros.copy();pi=zeros.copy()
stim=((np.arange(1,501)>50)&(np.arange(1,501)<75))*200.
for k in range(16000):
    ref.update(s,pe,pi,stim if 2000<k<5000 else zeros,zeros,p)
    pe,pi,_=proj(s[6])
out={'unrounded_16s_state_max_error':np.max(abs(s[:7]-cp),axis=1).tolist()}
savemat(q/'port_checkpoint16.mat',{'port_checkpoint':s[:7],'source_checkpoint':cp})
s=np.vstack([cp,np.zeros((1,p.n))])
field=np.zeros((8400,500));idx=0
for k in range(16000,99999):
    if k%10==0:field[idx]=s[6]*1000;idx+=1
    pe,pi,_=proj(s[6]);ref.update(s,pe,pi,zeros,zeros,p)
source=loadmat(q/'exp2a_full_source.mat',variable_names=['source_field'])['source_field'].T[1600:]
err=np.max(abs(source-field),axis=1)
out.update(resume_max_error_Hz=float(err.max()),first_error_over1Hz_s=float(16+np.flatnonzero(err>1)[0]*.01) if np.any(err>1) else None)
np.savez(r/'solver_validation/source_checkpoint_resume.npz',time_s=np.arange(8400)*.01+16,field=field)
(r/'solver_validation/source_checkpoint_resume.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
