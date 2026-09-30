"""Compare both near-critical fields from exactly the same complete history.

The lower-D trajectory is a 12-s continuation; the higher-D continuation is
stored as 8+12 seconds. Match their first 12 seconds rather than conflating
their different absolute warmup clocks or total recorded lengths.
"""
from canonical_readouts import assess,OUT
from pathlib import Path
import numpy as np
import json
from scipy.signal import find_peaks


def read(path):return json.loads(path.read_text())


def main():
    root=OUT/'runs'
    low=root/'rate_tail_D14497_D0.1449700_dt0.05'
    high=root/'rate_critical_finer_restart_D0.1449750_dt0.05'
    ext=root/'rate_tail_D144975_D0.1449750_dt0.05'
    lc,hc,ec=[read(x/'contract.json') for x in [low,high,ext]]
    assert Path(lc['initial']).resolve()==Path(hc['initial']).resolve()
    assert Path(ec['initial']).resolve()==(high/'trajectory.npz').resolve()
    assert all(q['dt_ms']==.05 and q['Z']=='held' and q['M']=='dynamic' for q in [lc,hc,ec])
    l,h,e=[np.load(x/'trajectory.npz') for x in [low,high,ext]]
    assert np.array_equal(l['cell_counts'],h['cell_counts']) and np.array_equal(l['cell_counts'],e['cell_counts'])
    upper=np.concatenate([h['field_E_hz'],e['field_E_hz']])[:12000]
    rows=[]
    for field,folder,contract in [(l['field_E_hz'],low,lc),(upper,high,hc)]:
        assert len(field)==12000
        q=assess(field,l['cell_counts'],folder.name)
        f=field[-4000:];w=l['cell_counts']/l['cell_counts'].sum();g=f@w
        err=np.array([np.linalg.norm(g[lag:]-g[:-lag])/np.linalg.norm(g[lag:]) for lag in range(180,601)])
        candidates,_=find_peaks(-err);best=sorted(candidates,key=lambda k:err[k])[:8]
        returns=[]
        for k in best:
            lag=int(k+180);delta=f[lag:]-f[:-lag]
            returns.append(dict(lag_ms=lag,relative_field_error=float(np.sqrt(np.sum(delta**2*w)/np.sum(f[lag:]**2*w)))))
        returns.sort(key=lambda x:x['relative_field_error'])
        rows.append(dict(D=contract['D_initial'],category=q['category'],tail=q['tail'],returns=returns))
    answer=dict(status='MATCHED_COMPLETE_HISTORY_12S_COMPARISON',shared_initial=lc['initial'],
        duration_ms=12000,dt_ms=.05,Z='held spatial field',M='dynamic',rows=rows,
        high_sources=[str(high/'trajectory.npz'),str(ext/'trajectory.npz')],
        high_segment_lengths_ms=[8000,4000],low_source=str(low/'trajectory.npz'),
        claim='Finite-time consequence of changing only spatial Z from a shared complete fast/M/delay state; not a certified bifurcation point.')
    (OUT/'matched_critical_history_audit.json').write_text(json.dumps(answer,indent=2))
    print(json.dumps(answer,indent=2))


if __name__=='__main__':main()
