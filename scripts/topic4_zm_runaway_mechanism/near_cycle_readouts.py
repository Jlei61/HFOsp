"""Original event rules and whole-field recurrence near the actual rate onset."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.signal import find_peaks
from canonical_readouts import assess,OUT


def main():
    rows=[]
    folders=sorted(list((OUT/'runs').glob('rate_critical_*'))+list((OUT/'runs').glob('rate_near_cycle_*'))+list((OUT/'runs').glob('rate_tail_*')))
    for p in folders:
        if not (p/'result.json').exists():continue
        z=np.load(p/'trajectory.npz');f=z['field_E_hz'][-4000:];w=z['cell_counts']/z['cell_counts'].sum();g=f@w
        global_error=np.array([np.linalg.norm(g[lag:]-g[:-lag])/np.linalg.norm(g[lag:]) for lag in range(180,1901)])
        candidates,_=find_peaks(-global_error);best=sorted(candidates,key=lambda i:global_error[i])[:12]
        returns=[]
        for i in best:
            lag=int(i+180);delta=f[lag:]-f[:-lag]
            returns.append(dict(lag_ms=lag,relative_field_error=float(np.sqrt(np.sum(delta*delta*w)/np.sum(f[lag:]**2*w))),
                                relative_global_error=float(global_error[i])))
        returns.sort(key=lambda q:q['relative_field_error'])
        pk,_=find_peaks(g,height=50,distance=120)
        qa=assess(z['field_E_hz'],z['cell_counts'],p.name)
        q=dict(label=p.name,source=str(p/'trajectory.npz'),D=json.loads((p/'result.json').read_text())['D_initial'],
               category=qa['category'],tail=qa['tail'],returns=returns,peaks_ms=pk.tolist(),peak_rates_hz=g[pk].tolist(),IEI_ms=np.diff(pk).tolist())
        rows.append(q)
        print(p.name,q['category'],q['tail']['mean_rate_hz'],returns[:1],flush=True)
        (OUT/'actual_rate_near_recurrence.json').write_text(json.dumps(dict(status='RUNNING',rows=rows),indent=2))
    (OUT/'actual_rate_near_recurrence.json').write_text(json.dumps(dict(status='COMPLETE',rows=rows,
        claim='Finite-window full-field recurrence screen and original event definitions; not a bifurcation classification'),indent=2))


if __name__=='__main__':main()
