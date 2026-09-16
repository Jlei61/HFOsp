import sys,time,json
from pathlib import Path
sys.path.insert(0,'/home/honglab/leijiaxin/HFOsp')
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from scripts.patient_state_v1.gpu_two_timescale import dataset,kernel
from scripts.patient_state_v1.common import RUN,write_json
cuda.select_device(1);d=dataset(True);theta=np.load(RUN/'two_scale_particle_posterior_v1_15/checkpoint.npz')['initial_center'];arrays=[cuda.to_device(np.ascontiguousarray(d[k])) for k in ['dt','reset','slow_dt','slow_reset','y']];h=cuda.to_device(d['x'][:,1].copy());rows=[]
for n in [16,32,64,128]:
 t=time.time();rng=create_xoroshiro128p_states(n*1024,seed=510000+n);cuda.synchronize();a=time.time();out=cuda.device_array(n);pars=cuda.to_device(np.tile(theta,(n,1)));kernel[n,1024](*arrays,h,pars,1,rng,out,cuda.device_array((0,0,0)));out.copy_to_host();b=time.time();row=dict(filters=n,rng_init_seconds=a-t,kernel_and_compile_seconds=b-a);rows.append(row);print(row,flush=True)
write_json(RUN/'two_scale_particle_posterior_v1_15/runtime_probe.json',dict(rows=rows,scope='Measured under concurrent GPU work; first kernel may include JIT compilation'))
