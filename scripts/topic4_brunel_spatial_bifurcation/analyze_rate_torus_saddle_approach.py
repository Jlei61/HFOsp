"""Compare each resolved torus with a periodic saddle at exactly the same J.

This tests a proposed saddle-cycle approach. Distances and period scaling do
not by themselves certify a homoclinic connection or the torus stability.
"""
from compare_rate_torus_periodic_targets import *
from rate_periodic_accuracy import defect


def main(a):
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    original=PERIODIC_OUT/'orbits/TR2_endpoint_middle_N128.npz'
    paths=[PERIODIC_OUT/'tori/endpoint_pod12_mesh_TR2_a0.042183Hz_N64x64.npz']
    paths += [PERIODIC_OUT/f'tori/endpoint_further_TR2_a{amp:.6f}Hz_N64x64.npz' for amp in [.04,.038]]
    rows=[]
    for path in paths:
        z=np.load(path);J=float(z['J']);tag=f'TR2_saddle_matched_{path.stem}_N128'
        target=PERIODIC_OUT/'orbits'/f'{tag}.npz'
        if not target.exists():
            t=np.load(original);o=Periodic(s,128,a.device);o.low_memory=True
            r,T,JJ,e,h=o.solve(t['r'],float(t['T']),J,tol=2e-12)
            assert e<2e-12 and JJ==J
            save_orbit(s,r,T,JJ,e,h,tag);del o
        t=np.load(target);check=defect(s,target,a.device)
        assert float(t['J'])==J and check['maximum_group_defect_Hz']<1e-8
        x=resample(resample(z['r']*1000,256,axis=0),256,axis=1)
        c=resample(t['r']*1000,256,axis=0);d,phase=distances(x,c,weights)
        cc=resample(np.load(original)['r']*1000,256,axis=0)
        correction,_=distances(c[:,None,:],cc,weights)
        row=dict(torus=str(path),target=str(target),J_EE_core=J,T_ms=float(z['T']),
            slow_period_s=float(2*np.pi/z['nu']/1000),
            distance_range_Hz=[float(d.min()),float(d.max())],
            original_target_correction_Hz=float(correction[0]),distances_Hz=d,
            fast_phase_shifts_cycles=phase,periodic_target_accuracy=check)
        rows.append(row)
        print('SAME J SADDLE DISTANCE',row['slow_period_s'],row['distance_range_Hz'],
              'target correction',row['original_target_correction_Hz'],flush=True)
    spec=read(PERIODIC_OUT/'poincare_floquet/TR2_endpoint_middle_N128_dt0.05.json')
    mu=np.array([complex(*v) for v in spec['multipliers']]);exponents=np.log(abs(mu))/(spec['T_ms']/1000)
    unstable=exponents[exponents>0];stable=exponents[exponents<0]
    assert len(unstable)==1
    saddle_log_coefficient=1/unstable[0]+1/abs(max(stable))
    times=np.array([r['slow_period_s'] for r in rows]);ds=np.array([r['distance_range_Hz'][0] for r in rows])
    out=dict(status='SADDLE_CYCLE_APPROACH_DIAGNOSTIC',rows=rows,
        adjacent_period_over_log_distance_slopes_s=np.diff(times)/np.log(ds[:-1]/ds[1:]),
        saddle_exponents_per_s=[float(unstable[0]),float(max(stable))],
        reference_log_scaling_coefficient_s=float(saddle_log_coefficient),
        eigenvalue_source=str(PERIODIC_OUT/'poincare_floquet/TR2_endpoint_middle_N128_dt0.05.json'),
        scope='Same-parameter periodic targets remove parameter-offset contamination. Log scaling is a diagnostic based on the leading stable/unstable saddle exponents; rate-profile distance is not a certified full-state manifold distance. Global connection type and finite torus stability remain unclassified.')
    write(PERIODIC_OUT/'TR2_same_parameter_saddle_approach.json',out)
    print('SADDLE SCALING',out['adjacent_period_over_log_distance_slopes_s'],saddle_log_coefficient,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args())
