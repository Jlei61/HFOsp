"""Separate integration and initial-history effects at one fixed physical field."""
from common import np, read, write, model
from onset_state_continuation import regional_weights
from audit_core_a_natural_entry_step import intervals
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse


def main(destination):
    out = Path(destination).resolve(); contract = read(out/'contract.json')
    s = model(40); W = regional_weights(s)
    folders = {k: Path(v).resolve() for k,v in contract['existing_controls'].items()}
    folders.update({r['label']: out/r['label'] for r in contract['new_arms']})
    source = Path(contract['source']).resolve()
    rows = []; rates = {}; smooth = {}
    duration = contract['duration_ms']
    for label,folder in folders.items():
        assert read(folder/'jobs.json')['status'] == 'COMPLETE'
        c = read(folder/'contract.json'); assert Path(c['source']).resolve() == source
        with np.load(folder/'trajectory.npz') as z:
            r = z['group_rate_hz'].astype(float); regional = r @ W.T
            bound = abs(np.spacing(z['group_rate_hz']).astype(float)) @ W.T
            assert np.all(abs(regional-z['regional_rate_hz']) <= bound+1e-10)
            assert np.isfinite(r).all() and r.min() >= 0
            field = z['Z'].copy()
        with np.load(source) as initial, np.load(folder/'final_state.npz') as final:
            assert np.array_equal(field, initial['syn'][5])
            assert np.array_equal(field, final['syn'][5])
            assert np.all(final['parameters'][19] == 0) and np.all(final['parameters'][20] == 1)
            assert int(final['clock'][0]) == round((int(initial['clock'][0])*c['source_dt_ms']+len(r))/c['dt_ms'])
        original = read(folder/'result.json')
        sm_full = uniform_filter1d(regional,10,axis=0,mode='nearest')
        for j in range(4):
            quiet = [(a,b) for a,b in intervals(sm_full[:,j] < 5) if b-a >= 20]
            assert quiet == [tuple(q) for q in original['rows'][j]['quiet_intervals_ms']]
        # Apply identical observation window; recompute censoring at 5 s.
        regional = regional[:duration]; sm = uniform_filter1d(regional,10,axis=0,mode='nearest')
        regions = []
        for j,name in enumerate(['Global E','Core A','Core B','Surround']):
            quiet = [(a,b) for a,b in intervals(sm[:,j]<5) if b-a >= 20]
            edges = [(0,0),*quiet,(duration,duration)]
            activity = [dict(start_ms=b,end_ms=c,duration_ms=c-b,left_censored=b==0,right_censored=c==duration)
                        for (_,b),(c,_) in zip(edges[:-1],edges[1:]) if c>b]
            complete = [a for a in activity if not a['left_censored'] and not a['right_censored']]
            regions.append(dict(region=name, mean_hz=float(regional[:,j].mean()),
                quiet_fraction=sum(b-a for a,b in quiet)/duration,
                max_complete_ms=max((a['duration_ms'] for a in complete),default=None),
                complete_at_least_1s=[a for a in complete if a['duration_ms']>=1000],activities=activity))
        rows.append(dict(label=label,source=str(folder),dt_ms=c['dt_ms'],regions=regions))
        rates[label] = regional; smooth[label] = sm
    pairs = [('old_fine_point_interpolated','old_fine_conservative'),
             ('old_coarse','old_fine_conservative'),('midpoint_coarse','midpoint_fine')]
    comparisons = []
    for a,b in pairs:
        blocks = []
        for lo,hi in [(0,120),(120,500),(500,1000),(1000,2000),(2000,5000)]:
            delta = smooth[a][lo:hi]-smooth[b][lo:hi]
            blocks.append(dict(window_ms=[lo,hi],regional_rms_hz=np.sqrt((delta**2).mean(0)).tolist()))
        comparisons.append(dict(labels=[a,b],smoothed_rate_difference=blocks))
    result = dict(status='INDEPENDENT_READOUT_PASS',rows=rows,comparisons=comparisons,
        same_physical_field_and_original_source=True,
        interpretation='Same deterministic field/history controls. Finite-window event agreement is not trajectory convergence, a periodic-orbit certificate, or a bifurcation type. Conservative versus interpolated history isolates an initialization contribution.',
        target_entry_type='NOT_ESTABLISHED',model_promoted=False)
    write(out/'independent_audit.json',result)
    for r in rows:
        q=r['regions'][1]
        print(r['label'],'A max complete ms',q['max_complete_ms'],'long',q['complete_at_least_1s'],flush=True)
    print('comparisons',comparisons,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');main(p.parse_args().destination)
