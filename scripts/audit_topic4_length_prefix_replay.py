"""Compare identical physics/seeds across 20s and 60s without another simulation."""
from pathlib import Path
import json
import numpy as np
from scripts import run_topic4_propagation_recovery_night as night
rt=night.rt;run=night.old


def main():
    short_id='recovery_upper_wide_recurrence';long_id='refine_near_EE075'
    short=rt.read(run.OUT/'candidates'/f'{short_id}.json');long=rt.read(run.OUT/'candidates'/f'{long_id}.json')
    for k in ['parameters','centers_mm','radii_mm']:assert short[k]==long[k]
    assert long['core_mean_rate_scale']==1.
    results=[];pending=[]
    for seed in [847101,847102]:
      a_path=run.result_path(short['stage'],short_id,2511,seed);b_path=run.result_path(long['stage'],long_id,2511,seed)
      if not run.complete(b_path):pending.append(seed);continue
      a_meta=rt.read(a_path);b_meta=rt.read(b_path)
      if not (a_meta['actual_duration_ms']==20000 and b_meta['actual_duration_ms']==60000):raise ValueError('full trajectories required')
      checks=[]
      with np.load(a_path.with_suffix('.npz')) as a,np.load(b_path.with_suffix('.npz')) as b:
        for key in ['contact_names','contact_xy_mm','positions_E','positions_I','h','vtheta','core_index_E','core_index_I','input_is_stochastic']:
            aa=a[key];bb=b[key];equal=np.array_equal(aa,bb,equal_nan=True) if aa.dtype.kind in 'fc' else np.array_equal(aa,bb)
            checks.append(dict(array=key,kind='static',identical=bool(equal)))
        for key in ['sheet_activity_counts','rate_E','rate_I']+[k for k in a.files if k.startswith('trace_')]+['input_rate_audit_values']:
            aa=a[key];bb=b[key][:len(aa)];equal=np.array_equal(aa,bb)
            checks.append(dict(array=key,kind='20s dynamic prefix',identical=bool(equal),maximum_absolute_delta=float(np.max(abs(aa.astype(float)-bb.astype(float))))))
        # Envelope convolution uses zero padding at the end of the short record.
        # Discard only its true 3-sigma Gaussian end boundary (8x2ms), not events.
        dt=float(a['contact_envelope_dt_ms']);half=int(np.ceil(3*5./dt));stop=a['contact_envelope'].shape[1]-half
        aa=a['contact_envelope'][:,:stop];bb=b['contact_envelope'][:,:stop]
        checks.append(dict(array='contact_envelope',kind='prefix excluding short-record convolution boundary',identical=bool(np.array_equal(aa,bb)),excluded_tail_ms=half*dt,maximum_absolute_delta=float(np.max(abs(aa-bb)))))
      results.append(dict(seed=seed,topology=2511,checks=checks,identical=all(c['identical'] for c in checks),short=str(a_path),long=str(b_path),
          short_arrays_sha256=a_meta['arrays_sha256'],long_arrays_sha256=b_meta['arrays_sha256']))
    status='PENDING' if pending else 'PASS' if all(r['identical'] for r in results) else 'DIFFERENT_PREFIX_REQUIRES_EXPLANATION'
    output=dict(status=status,results=results,pending_seeds=pending,interpretation='Checks only this repeated physical condition and its 20s prefix; does not establish bitwise repeatability of every candidate or historical physics.',producer=__file__,producer_sha256=rt.sha(__file__))
    rt.write(night.OUT/'length_prefix_replay_audit.json',output);print(json.dumps(dict(status=status,complete=len(results),pending=pending)))


if __name__=='__main__':main()
