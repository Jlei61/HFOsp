"""Read-only native regional resource evidence for the Core A question.

Event resource differences are net changes, including recovery. They do not
identify gross depletion or a bifurcation. The statistical unit is one complete
self-limited event in the single original Fig.5 trajectory.
"""
from common import OUT, BASE, model, np, write, log
from onset_state_continuation import regional_weights
from native_readouts import readouts, NATIVE
import csv

DEST = OUT/'core_a_resource_bifurcation_20260923'


def main():
    s = model(40); W = regional_weights(s)
    p = np.load(OUT/'transient_native_Z_path_20260923/native_Z_path.npz')
    # Keep only real observation times, discarding the intermediate 5-ms grid.
    times = p['source_time_ms']; idx = np.searchsorted(p['time_ms'], times)
    assert np.array_equal(p['time_ms'][idx], times)
    z = p['Z'][idx] @ W.T
    src = BASE/'native_reference/seed9108401_readouts.npz'
    native = np.load(src); t = native['t']; field = native['rate_cells']
    coarse = model(20)
    count = np.bincount(coarse.geo['group_cell'][coarse.E],
                        weights=coarse.sizes[coarse.E], minlength=400)
    assert np.array_equal(count,np.load(NATIVE.parent.parent/'geometry.npz')['cell_e_counts'])
    events, _, whole, sm = readouts(t, field, count, 'native9108401')
    # The saved cell field is float32; the original allE was computed before
    # that cast. Bound its roundoff explicitly instead of claiming bitwise parity.
    rounding_bound = (np.spacing(field).astype(float)/2) @ (count/count.sum())
    error = abs(whole-native['allE'])
    assert np.all(error <= rounding_bound+1e-10)
    assert np.array_equal(sm<5,native['smoothed']<5)
    rows = []
    for e in events:
        a = int(np.searchsorted(t,e['start_ms'])); b = a+int(e['duration_ms'])
        if not (t[a]>=500 and b<len(t) and t[b]<=9420 and a>=20 and b+20<=len(sm)):
            continue
        if not ((sm[a-20:a]<5).all() and (sm[b:b+20]<5).all()):
            continue
        start, end = float(t[a]), float(t[b])
        row = dict(start_ms=start,end_ms=end,duration_ms=e['duration_ms'],
                   peak_global_hz=e['peak_hz'])
        for j,name in enumerate(['global','coreA','coreB','surround']):
            za,zb = np.interp([start,end], times,z[:,j])
            row.update({f'Z_{name}_start':float(za),f'Z_{name}_end':float(zb),
                        f'net_depletion_{name}':float(za-zb)})
        rows.append(row)
    with (DEST/'native_complete_event_resource.csv').open('w') as f:
        out = csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    depletion = np.array([[r[f'net_depletion_{n}'] for n in ['coreA','coreB','surround']] for r in rows])
    result = dict(status='DESCRIPTIVE_AUDIT_PASS',native_seed=9108401,
        source=str(src),resource_source=str(OUT/'transient_native_Z_path_20260923/native_Z_path.npz'),
        saved_float32_field_reconstruction_max_error_hz=float(error.max()),
        original_quiet_classification_identical=True,
        sample='Complete global self-limited events within500–9420ms, one original native trajectory; not independent seeds.',
        event_definition='Original10ms smoothed all-E rate, active>=5Hz for>=20ms, peak>=20Hz;20ms quiet on both sides.',
        resource_definition='Cell-count-weighted regional mean Z at event endpoints; linear interpolation of original5/10ms samples. Positive Z_start-Z_end is NET depletion, not gross depletion without recovery.',
        n_complete_events=len(rows),regions=['Core A','Core B','Surround'],
        median_net_depletion=np.median(depletion,axis=0).tolist(),
        range_net_depletion=[depletion.min(0).tolist(),depletion.max(0).tolist()],
        events_A_net_depletion_greater_than_surround=int((depletion[:,0]>depletion[:,2]).sum()),
        events_A_depletes_surround_net_recovers=int(((depletion[:,0]>0)&(depletion[:,2]<=0)).sum()),
        events_surround_net_depletes=int((depletion[:,2]>0).sum()),
        scope='Descriptive native support for regional resource asymmetry. No isolated-Core-A causality or bifurcation type inferred.')
    write(DEST/'native_resource_evidence.json',result)
    np.savez_compressed(DEST/'native_regional_Z.npz',time_ms=times,Z_global_A_B_surround=z)
    log('NATIVE CORE A RESOURCE',result)


if __name__=='__main__':main()
