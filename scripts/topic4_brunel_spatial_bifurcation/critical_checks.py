"""Scientific consistency checks for the critical-case figure revision."""
from common import *
from model import SpatialBrunel
BASE=OUT/'critical_revision'

def main():
    b=np.load(BASE/'branch.npz');q=read(BASE/'branch.json');sp=read(BASE/'stability/result.json')
    assert sp['status']=='COMPLETE' and not sp['unresolved_positive_search']
    assert len(b['J'])==q['points'] and len(b['reversal_indices'])==12
    assert max(row['residual'] for row in q['rows'])<1e-9
    spectral_error=max(x['residual'] for r in sp['rows'] for x in r['roots']);assert spectral_error<1e-8
    jumps=np.linalg.norm(np.diff(np.c_[b['rates']/.01,b['J']/.05],axis=0),axis=1)
    assert max(x['rate_norm_difference'] for x in q['join_checks'])<.0081
    ny=read(BASE/'nyquist_refined/result.json');assert [r['unstable_root_count_candidate'] for r in ny['rows']]==[0,2,4]
    assert max(r['maximum_phase_increment'] for r in ny['rows'])<.25
    critical=[]
    for J,signs in [(0.948,[-1,-1]),(.955,[1,-1]),(.963,[1,1])]:
        r=min(sp['rows'],key=lambda q:abs(q['J_EE_core']-J));assert abs(r['J_EE_core']-J)<1e-10
        modes=sorted(r['roots'],key=lambda v:np.argmax(v['regional_energy']));growth=[m['lambda_per_ms'][0]*1000 for m in modes]
        assert np.sign(growth).tolist()==signs
        critical.append(dict(J_EE_core=J,growth_per_s=growth,frequency_hz=[m['frequency_hz'] for m in modes]))
    folds=read(BASE/'fold_audit.json');assert len(folds['rows'])==12
    assert max(x['quadrature96_fixedpoint_residual'] for x in folds['rows'])<1e-10
    assert max(x['quadrature96_zero_mode_residual'] for x in folds['rows'])<1e-8
    fixed=read(BASE/'fixed_J/result.json');assert fixed['distinct_roots']==fixed['occupied_level_pairs']==9
    assert all(x['unstable_certificate'] and x['root_residual']<1e-8 for x in fixed['rows'])
    geo=np.load(OUT/'operators/g40/geometry.npz');e=geo['population']==0;size=geo['group_size'];reg=geo['group_region']
    cells=np.bincount(geo['group_cell'][e],weights=size[e],minlength=1600);regions=np.array([size[e&(reg==k)].sum() for k in range(3)])
    native=[]
    for row in read(BASE/'readouts/result.json')['rows']:
        rec=read(ROOT/row['source']);z=np.load(ROOT/rec['trajectory']);x=np.load(ROOT/rec['exact_readout'])
        error=float(abs(z['field_E_hz'].astype(float)@cells/1000-z['regional_rates_hz'][:,:3]@regions/1000).max());assert error<.002
        assert x['lfp_raw'].shape==(20000,15) and x['contact_names'].tolist()==rec['contact_names']
        native.append(dict(J_EE_core=row['J_EE_core'],maximum_field_count_error=error,burst_CV=[d['IEI_CV'] for d in row['dynamics']]))
    write(BASE/'delivery_checks.json',dict(status='PASS',branch_points=len(b['J']),spectral_points=sp['spectral_points'],
        maximum_stored_eigenpair_residual=spectral_error,maximum_scaled_arc_step=float(jumps.max()),critical_case_modes=critical,
        full_determinant_counts=[r['unstable_root_count_candidate'] for r in ny['rows']],stationary_folds=12,same_J_unstable_equilibria=9,
        native_cases=native,meaning='Numerical consistency of this spatial closure and native readout identity. Not acceptance of full nonlinear rate/SNN equivalence or mesh-converged fold multiplicity.',human_visual_acceptance=False))
    print('CRITICAL CHECKS PASS',len(b['J']),sp['spectral_points'],flush=True)

if __name__=='__main__':main()
