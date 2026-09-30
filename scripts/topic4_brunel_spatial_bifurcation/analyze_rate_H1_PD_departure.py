"""Physical amplitude and paired-mesh evidence for the PD3 doubled branch.

The parent already has unstable directions. A child on its newly unstable
side is not by itself a stable attractor; keep criticality unclassified here.
"""
from rate_periodic import *


def main():
    s=RateField();weight=s.geo['group_size']/s.geo['group_size'].sum();rows=[]
    for source in ['PDreturnchild_branch_N2048.json','PDreturnchildMesh_branch_N1024.json']:
        for q in read(PERIODIC_OUT/source):
            z=np.load(q['orbit']);r=z['r']*1000;half=len(r)//2
            odd=(r[:half]-r[half:])/2;amp=float(np.sqrt(np.mean(odd**2@weight)))
            rows.append(dict(source=source,orbit=q['orbit'],N=len(r),
                switch_coordinate=q['amplitude_hz'],J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],
                J_shift=q['J_shift'],odd_rate_RMS_Hz=amp,
                departure_coefficient=q['J_shift']/amp**2,
                minimum_rate_Hz=float(r.min()),half_period_relative_mismatch=q['half_period_relative_mismatch']))
    fine=[q for q in rows if q['N']==2048];coarse=[q for q in rows if q['N']==1024]
    pair=[]
    for c in coarse:
        f=min(fine,key=lambda q:abs(q['odd_rate_RMS_Hz']-c['odd_rate_RMS_Hz']))
        pair.append(dict(fine=f['orbit'],coarse=c['orbit'],
            coefficient_relative_difference=abs(f['departure_coefficient']-c['departure_coefficient'])/abs(f['departure_coefficient']),
            absolute_J_difference=abs(f['J_EE_core']-c['J_EE_core'])))
    accurate=read(PERIODIC_OUT/'PDreturnchild_accuracy.json')
    assert accurate['status']=='PASS'
    passed=(len(fine)>=3 and bool(pair) and all(p['coefficient_relative_difference']<.01 for p in pair)
        and all(q['J_shift']<0 and q['minimum_rate_Hz']>0 and q['half_period_relative_mismatch']>1e-5 for q in fine))
    result=dict(status='MESH_CHECKED_DOUBLED_BRANCH' if passed else 'VALIDATION_INCOMPLETE',
        rows=rows,mesh_pairs=pair,continuous_accuracy_source=str(PERIODIC_OUT/'PDreturnchild_accuracy.json'),
        amplitude_definition='Neuron-weighted RMS Hz of (r(t)-r(t+T/2))/2 over the full 2T child.',
        branch_side='J < J_PD3',parent_stability='ALREADY_UNSTABLE',criticality='NOT_COMPUTED',
        scope='Nonzero full-space 2T branch with converged departure coefficient across temporal meshes. This establishes local branch geometry, not its critical multiplier, full Floquet stability, or a global connection.')
    write(PERIODIC_OUT/'PD_return_child_departure.json',result)
    print('PD3 DEPARTURE',result,flush=True)


if __name__=='__main__':main()
