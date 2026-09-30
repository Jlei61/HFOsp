"""Measure PD2 departure with one physical amplitude across temporal meshes.

The branch-switch coordinate normalizes the mode maximum on each mesh. Its
numerical amplitude therefore differs slightly across meshes. Use the
neuron-weighted RMS of the T-antiperiodic rate component for comparison.
"""
from rate_periodic import *


def main():
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();rows=[]
    for source in ['PDupperchild_branch_N4096.json','PDupperchildMesh_branch_N2048.json']:
        path=PERIODIC_OUT/source
        if not path.exists():continue
        for q in read(path):
            z=np.load(q['orbit']);r=z['r']*1000;half=len(r)//2
            odd=(r[:half]-r[half:])/2
            amplitude=float(np.sqrt(np.mean(odd**2@weights)))
            rows.append(dict(source=str(path),orbit=q['orbit'],N=len(r),
                switch_coordinate=q['amplitude_hz'],J_EE_core=q['J_EE_core'],
                own_mesh_parent_J=q['J_EE_core']-q['J_shift'],
                J_shift=q['J_shift'],T_ms=q['T_ms'],
                odd_rate_RMS_Hz=amplitude,
                J_shift_over_odd_rate_RMS_squared=q['J_shift']/amplitude**2,
                minimum_collocation_rate_Hz=float(r.min()),
                half_period_relative_mismatch=q['half_period_relative_mismatch']))
    fine=sorted([q for q in rows if q['N']==4096],key=lambda q:q['odd_rate_RMS_Hz'])
    coarse=sorted([q for q in rows if q['N']==2048],key=lambda q:q['odd_rate_RMS_Hz'])
    comparison=None
    if fine and coarse:
        aa,bb=fine[0],coarse[0]
        comparison=dict(fine_source=aa['orbit'],coarse_source=bb['orbit'],
            coefficient_relative_difference=abs(aa['J_shift_over_odd_rate_RMS_squared']-
                bb['J_shift_over_odd_rate_RMS_squared'])/abs(aa['J_shift_over_odd_rate_RMS_squared']),
            scope='Departure coefficient relative to each mesh critical point, using physical odd-component amplitude. Coarse orbit negativity is retained and excludes it as a display trajectory.')
    result=dict(status='NONZERO_CHILD_DEPARTURE_QUANTIFIED_STABILITY_PENDING',rows=rows,
        common_amplitude_definition='sqrt(mean_t sum_g (n_g/N_total) * [(r_g(t)-r_g(t+T))/2]^2), Hz; r is the full 2T child and T is its half period.',
        mesh_comparison=comparison,
        fine_coefficient_relative_spread=(max(q['J_shift_over_odd_rate_RMS_squared'] for q in fine)/
            min(q['J_shift_over_odd_rate_RMS_squared'] for q in fine)-1) if fine else None,
        all_fine_shifts_positive=all(q['J_shift']>0 for q in fine) if fine else None,
        criticality='Independent child Floquet and parent crossing checks remain required for final classification.')
    write(PERIODIC_OUT/'PD_upper_child_departure.json',result)
    print('PD2 DEPARTURE',len(rows),'orbits',comparison,flush=True)


if __name__=='__main__':main()
