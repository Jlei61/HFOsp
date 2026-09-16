"""Finite parameter steps from a fully specified periodic history."""
from common import *
from explore import orbit_history,simulate,recurrence
import numpy as np,argparse

def run(name,source,g,dt=.05):
    s=System();state=orbit_history(s,source,dt);r,rf,hf,hist=simulate(s,g,state,dt,8000.)
    tail=r[-round(3000/dt):];row=dict(name=name,source=str(source),g=g,dt_ms=dt,duration_ms=8000.,mean_hz=(tail.mean(0)*1000).tolist(),minimum_hz=(tail.min(0)*1000).tolist(),maximum_hz=(tail.max(0)*1000).tolist(),recurrence=recurrence(r,dt))
    dest=OUT/'transitions';dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'{name}_dt{dt:g}.npz',r=r[::10],dt=dt*10,g=g,final_r=rf,final_h=hf,final_history=hist)
    write(f'transitions/{name}_dt{dt:g}.json',row);print(json.dumps(row),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--source',required=True);ap.add_argument('--g',type=float,required=True);ap.add_argument('--dt',type=float,default=.05);a=ap.parse_args();run(a.name,a.source,a.g,a.dt)
