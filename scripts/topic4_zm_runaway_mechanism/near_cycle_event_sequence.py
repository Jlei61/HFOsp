"""Read the deterministic pre-escape event sequence without assuming its bifurcation.

Use the original complete-event definition. Integrated event activity is less
affected by 1-ms peak sampling than peak height. Events remain within-trajectory
observations, not independent statistical replicates or a closed return map.
"""
from canonical_readouts import assess, OUT
import numpy as np
import json
from scipy.signal import find_peaks


def main():
    rows=[]
    for folder in sorted((OUT/'runs').glob('rate_critical_*')):
        if not (folder/'trajectory.npz').exists():continue
        sources=[folder/'trajectory.npz']
        name=folder.name
        tail=None
        if name=='rate_critical_near_restart_D0.1449700_dt0.05':
            tail=OUT/'runs/rate_tail_D14497_D0.1449700_dt0.05/trajectory.npz'
        elif name=='rate_critical_finer_restart_D0.1449750_dt0.05':
            tail=OUT/'runs/rate_tail_D144975_D0.1449750_dt0.05/trajectory.npz'
        z=np.load(sources[0]);f=z['field_E_hz'];counts=z['cell_counts']
        if tail and tail.exists():
            extra=np.load(tail)
            assert np.array_equal(extra['cell_counts'],counts)
            f=np.concatenate([f,extra['field_E_hz']]);sources.append(tail)
        g=f@(counts/counts.sum());q=assess(f,counts,name);events=[]
        for event in q['events']:
            lo=int(event['start_ms']);hi=int(event['end_ms']);peak=lo+int(np.argmax(g[lo:hi]))
            events.append(dict(start_ms=lo,end_ms=hi,duration_ms=hi-lo,
                peak_ms=peak,peak_1ms_hz=float(g[peak]),
                integrated_spikes_per_E_neuron=float(g[lo:hi].sum()/1000)))
        for left,right in zip(events[:-1],events[1:]):
            left['next_start_interval_ms']=right['start_ms']-left['start_ms']
            left['next_integrated_activity_difference']=right['integrated_spikes_per_E_neuron']-left['integrated_spikes_per_E_neuron']
        row=dict(label=name,D=json.loads((folder/'result.json').read_text())['D_initial'],
                 sources=list(map(str,sources)),category=q['category'],events=events)
        rows.append(row)
        print(name,'events',len(events),'first40',[(e['start_ms'],round(e['peak_1ms_hz'],2),round(e['integrated_spikes_per_E_neuron'],5)) for e in events[:40]],flush=True)
    (OUT/'near_cycle_event_sequence.json').write_text(json.dumps(dict(status='COMPLETE',rows=rows,
        claim='Descriptive deterministic event sequence; no scalar-map closure or bifurcation type inferred.'),indent=2))


if __name__=='__main__':main()
