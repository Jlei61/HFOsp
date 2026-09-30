"""Audit earlier fine-grid fields; do not mix Z detail or time steps into a bracket."""
from native_spatial_refinement import DEST
from native_spatial_refinement_audit import canonical,broad
from common import *


def main():
    c=read(OUT/'native_spatial_early_fields_contract.json')
    assert read(DEST/'early_fields_batch.json')['status']=='COMPLETE'
    fields=np.load(DEST/'native_fields_g40.npz');s=model(40);rows=[]
    for t,D in zip(c['times_ms'],c['D']):
        folder=OUT/'runs'/f'native_g40_nativeZ_t{t}_dt{c["dt_ms"]}'
        run=read(folder/'contract.json');z=np.load(folder/'trajectory.npz')
        j=np.flatnonzero(fields['times_ms']==t).item()
        assert np.array_equal(z['Z_source'],fields['fields'][j])
        assert np.all(z['Z_every50ms']==z['Z_source'])
        assert np.array_equal(z['final_state'][11],z['Z_source'])
        assert run['Z']=='held' and run['M']=='dynamic'
        assert np.ptp(z['M_every50ms'],axis=0).max()>1e-10
        assert run['dt_ms']==c['dt_ms'] and run['duration_ms']==c['duration_ms']
        assert run['rate_history_scheme']=='instantaneous rate at the labelled endpoint'
        assert Path(run['initial']).resolve()==(DEST/f'initial_g40_dt{c["dt_ms"]}.npz').resolve()
        assert np.array_equal(z['time_ms'],np.arange(12000)+1)
        assert abs(1-z['Z_source'][s.E]@s.mean_weights-D)<2e-14
        q=canonical(folder)
        rows.append(dict(native_field_time_ms=int(t),D=D,Z_mean=1-D,source=str(folder/'trajectory.npz'),
            canonical=q,broad_start_ms=broad(z['field_E_hz'],z['cell_counts'])))
        log('EARLY FINE FIELD AUDIT',t,q['category'],q['tail']['mean_rate_hz'])
    selflimited=[r for r in rows if r['canonical']['category']=='SELF_LIMITED']
    upper=OUT/'runs'/f'native_g40_nativeZ_D0.2190000_dt{c["dt_ms"]}'
    q=dict(status='SELF_LIMITED_SIDE_FOUND' if selflimited else 'SELF_LIMITED_SIDE_NOT_SHOWN',rows=rows,
        matched_upper_available=(upper/'result.json').exists(),matched_upper_source=str(upper),
        scope='Two deterministic finite-window continuations. No critical point or attractor classification. A same-detail/same-step upper arm is required before assigning a state bracket.')
    write(DEST/'early_fields_result.json',q)


if __name__=='__main__':main()
