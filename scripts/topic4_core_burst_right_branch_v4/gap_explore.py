"""A bounded carried-history check between the two solved periodic families."""
from common import *
from explore import orbit_history,simulate,recurrence
import numpy as np

def main():
    s=System();dt=.1;duration=8000.
    initial=OUT/'periodic/burst_end/g1.36000000_N2048.npz'
    state=orbit_history(s,initial,dt);dest=OUT/'gap_trajectories';dest.mkdir(exist_ok=True);rows=[]
    for g in (1.365,1.37,1.375):
        path=dest/f'g{g:.8f}.npz'
        if path.exists():
            z=np.load(path);r=z['r'];state=(z['final_r'],z['final_h'],z['final_history'])
        else:
            r,rf,hf,hist=simulate(s,g,state,dt,duration);state=(rf,hf,hist)
            np.savez_compressed(path,r=r,g=g,dt=dt,final_r=rf,final_h=hf,final_history=hist)
        tail=r[-40000:];row=dict(g=g,source=str(path),initialization=str(initial),duration_ms=duration,dt_ms=dt,
            recurrence=recurrence(r,dt),mean_hz=(tail.mean(0)*1000).tolist(),minimum_hz=(tail.min(0)*1000).tolist(),maximum_hz=(tail.max(0)*1000).tolist())
        rows.append(row);write('gap_exploration.json',rows);print(json.dumps(row),flush=True);initial=path

if __name__=='__main__':main()
