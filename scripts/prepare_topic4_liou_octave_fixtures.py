#!/usr/bin/env python3
"""Numerical fixtures for actual source-method execution in Octave."""
from pathlib import Path
import json
import numpy as np
from scipy.io import savemat,loadmat
import run_topic4_liou_original_reference as ref

def prepare():
    folder=ref.OUT/'octave_source_validation';folder.mkdir(exist_ok=True)
    rng=np.random.RandomState(7301)
    for name in ['exp2a','exp4a','exp4b']:
        p=ref.protocol(name);s=ref.initial_state(p);s[0]+=rng.normal(0,4,p.n);s[1]+=rng.rand(p.n)*10;s[2]+=rng.rand(p.n)*15;s[3:6]=rng.rand(3,p.n)*5
        s[6]=(rng.rand(p.n)<.1) if p.spiking else rng.rand(p.n)*.2;s[7]=rng.randint(0,15,p.n)
        first=s.copy();u=rng.rand(400,p.n);proj=ref.Projection(p);spikes=[];checkpoints=[]
        stimulus=np.full(p.n,80.)
        for k in range(400):
            pe,pi,pg=proj(s[6])
            ref.update(s,pe,pi,stimulus,u[k],p)
            spikes.append(s[6].copy())
            if (k+1)%100==0:checkpoints.append(s.copy())
        savemat(folder/f'{name}_fixture.mat',{'initial_state':first,'uniforms':u.T,'stimulus':stimulus[:,None],'expected_final':s,'expected_output':np.array(spikes).T,'expected_checkpoints':np.array(checkpoints),'is_spiking':int(p.spiking),'wlocal':p.w_local_i,'wglobal':p.w_global_i,'expected_ke':proj.ke[:,None],'expected_ki':proj.ki[:,None]})

def check():
    folder=ref.OUT/'octave_source_validation';rows=[]
    for name in ['exp2a','exp4a','exp4b']:
        data=loadmat(folder/f'{name}_source_result.mat');expected=loadmat(folder/f'{name}_fixture.mat')
        err=float(np.max(abs(data['actual_final']-expected['expected_final'])))
        output=float(np.max(abs(data['actual_output']-expected['expected_output'])))
        assert err<1e-8,(name,err)
        assert output<1e-8,(name,output)
        rows.append({'experiment':name,'400_steps_final_max_abs_error':err,'all_outputs_max_abs_error':output,'all_spikes_identical':bool(output==0) if name.startswith('exp4') else None})
    ref.write_json(ref.OUT/'octave_source_method_qa.json',{'status':'PASS','runtime':'GNU Octave10.3.0','rows':rows,'boundary':'Author method bodies with struct-return wrappers and supplied uniform draws. Not MATLAB GUI or random-stream reproduction.'})
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    import sys
    check() if '--check' in sys.argv else prepare()
