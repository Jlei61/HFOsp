"""Reconstruct regional activity and quiet intervals of a numerical cycle."""
from common import np, read, write, model
from onset_state_continuation import regional_weights
from audit_core_a_natural_entry_step import intervals
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse


def main(parent):
    parent=Path(parent).resolve();out=parent/'actual_orbit_profile'
    assert read(parent/'result.json')['status']=='NUMERICAL_PERIODIC_ROOT'
    assert read(out/'jobs.json')['status']=='COMPLETE'
    s=model(40);W=regional_weights(s)
    with np.load(out/'profile.npz') as z:
        stored=z['group_rate_hz'];r=stored.astype(float);dt=float(z['dt_ms']);T=float(z['period_ms'])
        regional=r@W.T;bound=abs(np.spacing(stored).astype(float))@W.T
        assert np.all(abs(regional-z['regional_rate_hz'])<=bound+1e-10)
        assert np.all(np.isfinite(r)) and r.min()>=0
        Z=z['Z'].copy()
    sm=uniform_filter1d(regional,round(10/dt),axis=0,mode='wrap')
    rows=[]
    for j,name in enumerate(['Global E','Core A','Core B','Surround']):
        quiet=[(float(a*dt),float(b*dt)) for a,b in intervals(sm[:,j]<5) if (b-a)*dt>=20]
        rows.append(dict(region=name,minimum_10ms_hz=float(sm[:,j].min()),maximum_10ms_hz=float(sm[:,j].max()),
            quiet_intervals_in_recorded_period_ms=quiet,maximum_recorded_quiet_ms=max((b-a for a,b in quiet),default=0.)))
    result=dict(status='INDEPENDENT_CYCLE_ACTIVITY_RECONSTRUCTION_PASS',period_ms=T,dt_ms=dt,
        Z_A=float(Z@W[1]),rows=rows,
        both_cores_have_nontrivial_burst_and_quiet=all(q['maximum_recorded_quiet_ms']>=20 and q['maximum_10ms_hz']>50 for q in rows[1:3]),
        definition='Same10ms smoothing and below5Hz for at least20ms quiet definition as uninterrupted controls. Wrap smoothing at the period edge; interval endpoints are reported in the recorded period without merging across that edge. Edge-bin rounding is at most one dt.',
        scope='Phenotype and nontrivial amplitude of this closed numerical orbit only. No stability, bifurcation or SNN correspondence certificate.',model_promoted=False)
    write(out/'independent_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');main(p.parse_args().parent)
