"""Verify the explicit native periodic branch bridge and independent endpoint."""
from native_path import *
from native_joint_orbit_check import phase_distance,sanity


def main():
    sanity();contract=read(OUT/'native_cycle_long_connection_contract.json')
    folder=OUT/'periodic/native_long_arm_connection_G8505_M65536'
    run=read(folder/'result.json');assert run['status']=='SEGMENT_COMPLETE'
    previous=(OUT/contract['source']).resolve();s=model();attach_native_path(s)
    rows=[]
    for row in run['rows']:
        path=Path(row['path']).resolve()
        assert Path(row['source_orbit']).resolve()==previous
        z=np.load(path);source=np.load(previous)
        assert len(z['r'])==contract['N'] and row['nonlinear_samples']==contract['M']
        assert float(z['residual'])<contract['criteria']['accepted_residual_hz']
        assert row['residual']==float(z['residual'])
        s.set_D(float(z['D']));assert np.max(abs(s.Z-z['Z']))<1e-12
        rows.append(dict(source=str(previous),target=str(path),D=float(z['D']),T_ms=float(z['T']),
                         residual_hz=float(z['residual']),**phase_distance(source['r'],z['r'])))
        previous=path
    for target in contract['period_targets_ms']:
        assert any(abs(row['T_ms']-target)<1e-8 for row in rows)
    ref=OUT/contract['independent_endpoint'];a=np.load(previous);b=np.load(ref)
    endpoint=dict(D_difference=abs(float(a['D'])-float(b['D'])),
        period_difference_ms=abs(float(a['T'])-float(b['T'])),
        max_Z_difference=float(np.max(abs(a['Z']-b['Z']))),**phase_distance(a['r'],b['r']))
    gates=dict(endpoint_D=endpoint['D_difference']<contract['criteria']['endpoint_D_difference'],
        endpoint_waveform=endpoint['phase_aligned_relative_L2']<contract['criteria']['endpoint_phase_aligned_rate_L2'],
        endpoint_period=endpoint['period_difference_ms']<1e-8)
    q=dict(status='CONNECTION_PASS' if all(gates.values()) else 'INDEPENDENT_ENDPOINT_NOT_MATCHED',
        contract=str(OUT/'native_cycle_long_connection_contract.json'),rows=rows,
        independently_solved_endpoint=str(ref),endpoint=endpoint,gates=gates,
        scope='Explicit branch connection only. No stability interpolation, onset type, or SNN correspondence inherited.')
    write(folder/'connection_audit.json',q);log('NATIVE LONG ARM CONNECTION',q)


if __name__=='__main__':main()
