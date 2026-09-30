"""Report the prespecified 1--4 s grouping control without branch claims."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'


def safe(x):
    if isinstance(x, dict): return {k:safe(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [safe(v) for v in x]
    if isinstance(x, np.ndarray): return safe(x.tolist())
    if isinstance(x, np.generic): return safe(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def stretches(x):
    d = np.diff(np.r_[False, x, False].astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def summarize(folder,start_ms=1000,end_ms=4000):
    config=json.load(open(folder/'config.json'))
    initial_ms=config.get('initial_ms',0.)
    left=round((start_ms-initial_ms)/10);right=round((end_ms-initial_ms)/10)
    with np.load(folder/'trajectory.npz') as z:
        rates = z['rate_1ms'].reshape(-1,10,4).mean(1)
        raw = z['field_1ms']; counts = z['count_e']
        cell = np.arange(1600).reshape(40,40)
        # Original Fig.5 observation cells are 1 mm, while dynamics use 0.5 mm.
        spatial = (raw*counts[None,:]).reshape(-1,20,2,20,2).sum((2,4)).reshape(-1,400)
        observed_counts = counts.reshape(20,2,20,2).sum((1,3)).reshape(400)
        spatial /= np.maximum(observed_counts,1.)
        assert np.allclose(spatial@observed_counts/32000., z['rate_1ms'][:,0])
    assert 0<=left<right<=len(rates),(folder,left,right,len(rates))
    x = rates[left:right,0]
    quiet = [(a,b) for a,b in stretches(x<5) if b-a>=2]
    events = [(a,b) for (_,a),(b,_) in zip(quiet[:-1],quiet[1:])
              if b-a>=2 and x[a:b].max()>=20]
    if events: category = 'self_limited_events'
    elif not quiet and np.mean(x>=5)>=.95: category = 'persistent_activity'
    elif np.mean(x<5)>=.95: category = 'low_activity'
    else: category = 'unresolved'
    event_rows = []
    for a,b in events:
        lo,hi = a+left,b+left
        peak_ms = (lo+np.argmax(rates[lo:hi,0]))*10+5
        image = spatial[peak_ms-25:peak_ms+25].mean(0)
        event_rows.append(dict(start_ms=lo*10+initial_ms,end_ms=hi*10+initial_ms,duration_ms=(b-a)*10,
            core_max_hz=rates[lo:hi,1:3].max(0),
            occupied_E_fraction_at_peak=np.average(image>=50,weights=observed_counts)))
    stats = dict(category=category, window_ms=[start_ms,end_ms], mean_E_hz=x.mean(),
        q10_E_hz=np.quantile(x,.1), q90_E_hz=np.quantile(x,.9), quiet_fraction=np.mean(x<5),
        finite_events=len(events), quiet_intervals=len(quiet),
        mean_core_hz=rates[left:right,1:3].mean(0),
        event_duration_median_ms=np.median([e['duration_ms'] for e in event_rows]) if event_rows else None,
        peak_recruitment_median=np.median([e['occupied_E_fraction_at_peak'] for e in event_rows]) if event_rows else None,
        events=event_rows)
    return stats


def main():
    rows = []
    base = OUT/'particle_controls/selected_g40'
    for folder in sorted(base.glob('*_4000ms*')):
        s = folder/'status.json'
        if not s.exists() or json.load(open(s))['status'] != 'COMPLETE': continue
        cfg = json.load(open(folder/'config.json'))
        rows.append(dict(folder=str(folder), D=cfg['D'], seed=cfg['seed'],
            model='individual' if cfg['thresholds']=='original individual values' else 'grouped',
            statistics=summarize(folder)))
    pairs = []
    for D in [0., .225, .25]:
        for seed in [1901,1902]:
            available = {r['model']:r for r in rows if r['D']==D and r['seed']==seed}
            if len(available)!=2: continue
            a,b = [available[k]['statistics'] for k in ('grouped','individual')]
            pairs.append(dict(D=D,seed=seed,grouped=a,individual=b,
                category_agrees=a['category']==b['category'],
                mean_rate_difference_hz=a['mean_E_hz']-b['mean_E_hz'],
                quiet_fraction_difference=a['quiet_fraction']-b['quiet_fraction']))
    result = dict(status='COMPLETE_DIAGNOSTIC' if len(rows)==12 else 'INCOMPLETE_DIAGNOSTIC',
        completed_trajectories=len(rows),expected_trajectories=12,rows=rows,pairs=pairs,
        inference='Paired selected-g40 grouping control; two seeds are experimental repetitions. No equilibrium, periodic-orbit, eigenvalue or native-SNN acceptance claim.',
        common_noise_equivalence='NOT_TESTED',bifurcation_started=False)
    (OUT/'grouping_control_comparison.json').write_text(json.dumps(safe(result),indent=2)+'\n')
    for p in pairs:
        print(p['D'],p['seed'],p['grouped']['category'],p['individual']['category'],
              'mean',round(p['grouped']['mean_E_hz'],2),round(p['individual']['mean_E_hz'],2),
              'events',p['grouped']['finite_events'],p['individual']['finite_events'])


if __name__ == '__main__': main()
