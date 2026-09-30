"""Original global and spatial OU input laws for finite-population controls."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.topic4_spatial_ou_drive import SpatialOUDrive,SpatialOUConfig


class NativeSharedDrive:
    def __init__(self,positions,params,spatial_spec,nu,seed,dt=.1):
        spec=dict(spatial_spec);spec.pop('role',None);offset=spec.pop('seed_offset',500000)
        spec['seed']=seed+offset
        self.spatial=SpatialOUDrive(positions,20.,dt,SpatialOUConfig(**spec))
        self.rng=np.random.default_rng(seed+700000)
        self.a=np.exp(-dt/params['tau_n'])
        self.b=params['sigma_n']*1e-3*np.sqrt(params['tau_n']/2)*np.sqrt(1-self.a**2)
        self.xi=0.;self.nu=nu;self.dt=dt;self.step_index=0
        self.global_trace=[]

    def block(self,steps,scale=1):
        rates=np.empty((steps,40000))
        for k in range(steps):
            self.xi=self.a*self.xi+self.b*self.rng.standard_normal()
            nu=max(self.nu+self.xi,0.)
            rates[k].fill(nu)
            rates[k,:32000]=np.maximum(nu+self.spatial.step(self.step_index*self.dt),0.)
            self.global_trace.append(nu);self.step_index+=1
        return np.tile(rates,(1,scale)) if scale!=1 else rates

    def arrays(self):
        return dict(global_rate_per_ms=np.asarray(self.global_trace),**self.spatial.trace_arrays())
