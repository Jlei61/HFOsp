"""Promote the antiperiodic root only after mesh and monodromy checks."""
from rate_periodic import *
from audit_rate_filter_states import filter_state_minima


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',default='PD_double_low')
    p.add_argument('--display-label',default='PD1')
    p.add_argument('--monodromy-label',default='PD')
    p.add_argument('--child-classification',default='PD_child_classification.json')
    a=p.parse_args()
    if a.label!='PD_double_low':
        assert a.display_label!='PD1' and a.monodromy_label!='PD', 'Distinct roots require distinct output labels'
        assert a.child_classification!='PD_child_classification.json', 'Do not reuse PD1 child classification'
    roots=sorted([read(f) for f in PERIODIC_OUT.glob(a.label+'_N*.json')],key=lambda x:x['N'])
    assert len(roots)>=2
    lo,q=roots[-2:];dj=abs(q['J_EE_core']-lo['J_EE_core'])
    assert q['N']>=2048 and dj<1e-7,(q['N'],dj)
    assert q['antiperiodic_relative_residual']<1e-7 and abs(q['dborder_dJ'])>1e-6
    checks=[]
    for f in (PERIODIC_OUT/'floquet').glob(a.label+'_eval*_dt*.json'):
        z=read(f)
        if abs(z['J_EE_core']-q['J_EE_core'])>max(1e-7,10*dj):continue
        vals=np.array([complex(*x) for x in z['multipliers']]);i=int(np.argmin(abs(vals+1)))
        if abs(vals[i]+1)>.01 or z['identified_neutral_index'] is None:continue
        assert max(z['residuals'])<1e-6
        checks.append(dict(source=str(f),J_EE_core=z['J_EE_core'],dt_ms=z['dt_ms'],
                           multiplier=vals[i],distance_from_minus_one=abs(vals[i]+1),
                           phase_multiplier_error=z['phase_multiplier_error']))
    assert checks and min(v['dt_ms'] for v in checks)<=.05001,checks
    s=RateField();u=np.load(PERIODIC_OUT/f'{a.label}_mode_N{q["N"]}.npz')['u']
    energy=np.mean(abs(u)**2,axis=0)*s.geo['group_size']*s.E
    by=[float(energy[s.geo['group_region']==k].sum()/energy.sum()) for k in range(3)]
    mass=s.geo['group_size']*s.E
    baseline=[float(mass[s.geo['group_region']==k].sum()/mass.sum()) for k in range(3)]
    row=dict(status='VALIDATED_PD',label=a.display_label,internal_label=a.label,
             J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],N=q['N'],mesh_J_difference=dj,
             lower_mesh_N=lo['N'],antiperiodic_relative_residual=q['antiperiodic_relative_residual'],
             crossing_border_slope=q['dborder_dJ'],monodromy_checks=checks,
             E_rate_mode_energy_A_B_surround=by,criticality='NOT_COMPUTED',
             E_population_fraction_A_B_surround=baseline,
             per_E_cell_mean_squared_mode_relative_to_network=[x/y for x,y in zip(by,baseline)],
             meaning='Simple antiperiodic mode and independently checked multiplier -1 on the two-burst cycle; child-branch side and stability require separate calculation')
    direct=[read(f) for f in PERIODIC_OUT.glob(f'{a.monodromy_label}_monodromy_check_N{q["N"]}_dt*.json')]
    if direct:
        row['direct_monodromy_checks']=sorted(direct,key=lambda v:v['dt_ms'])
        if len(direct)>1:
            fine,coarse=row['direct_monodromy_checks'][0],row['direct_monodromy_checks'][-1]
            row['minus_one_defect_coarse_to_fine_ratio']=coarse['minus_one_relative_defect']/fine['minus_one_relative_defect']
    continuous=PERIODIC_OUT/f'{a.label}_continuous_defect_N{q["N"]}.json'
    assert continuous.exists(), 'Every PD parent needs an oversampled continuous-orbit check'
    if continuous.exists():
        defect=read(continuous)
        assert Path(defect['orbit']).resolve()==Path(q['orbit']).resolve()
        assert abs(defect['J_EE_core']-q['J_EE_core'])<1e-12
        assert defect['maximum_group_defect_Hz']<.1 and max(defect['regional_defect_Hz'])<.001
        assert defect['minimum_rate_Hz']>=-1e-9
        profile=np.load(q['orbit'])
        defect['filter_state_check']=filter_state_minima(s,profile['r'],float(profile['T']))
        row['continuous_orbit_check']=defect
        row['eigenmode_validation_status']='VALIDATED_PD'
        row['full_physical_profile_status']=('PASS' if defect['filter_state_check']['positive']
                                             else 'TEMPORAL_REFINEMENT_REQUIRED')
        row['full_acceptance'] = defect['filter_state_check']['positive']
    followup=PERIODIC_OUT/(a.label+'_filter_state_followup.json')
    if followup.exists():
        f=read(followup)
        # A corrected waveform alone cannot transfer the critical label.
        # Accept only a same-root fine parent with its own null residual
        # and independently converging full-state monodromy checks.
        same_root=(f.get('source_root_N')==q['N'] and
            Path(f.get('source_root_orbit','')).resolve()==Path(q['orbit']).resolve() and
            abs(f.get('J_EE_core',float('inf'))-q['J_EE_core'])<1e-12)
        if f.get('status')=='FILTER_AND_CRITICAL_MODE_RECHECKED' and same_root:
            assert f['N']>=q['N'] and f['antiperiodic_relative_residual']<1e-7
            fine=f['continuous_check'];direct=f['monodromy_checks']
            assert fine['filter_state_check']['positive'] and fine['minimum_rate_Hz']>=-1e-9
            assert fine['maximum_group_defect_Hz']<.1 and max(fine['regional_defect_Hz'])<.001
            assert len(direct)>=3 and all(Path(v['orbit']).resolve()==Path(f['orbit']).resolve() for v in direct)
            errors=np.array([v['minus_one_relative_defect'] for v in sorted(direct,key=lambda v:-v['dt_ms'])])
            assert errors[-1]<1e-4 and np.all(errors[:-1]/errors[1:]>3)
            row.update(filter_state_followup=dict(source=str(followup),**f),accepted_parent_orbit=f['orbit'],
                full_physical_profile_status='PASS',full_acceptance=True,
                fine_parent_antiperiodic_relative_residual=f['antiperiodic_relative_residual'])
            row['direct_monodromy_checks']=sorted(direct,key=lambda v:v['dt_ms'])
            row['minus_one_defect_coarse_to_fine_ratio']=errors[0]/errors[-1]
    if a.label=='PD_double_upper':
        direct=row.get('direct_monodromy_checks',[])
        assert continuous.exists() and len(direct)>=2
        assert direct[0]['dt_ms']<=.025001
        assert direct[0]['minus_one_relative_defect']<1e-4
        assert row['minus_one_defect_coarse_to_fine_ratio']>3
    child=PERIODIC_OUT/a.child_classification
    physical_PD1=PERIODIC_OUT/'PD_double_low_child_validation.json'
    if a.label=='PD_double_low' and physical_PD1.exists():
        physical=read(physical_PD1)
        if physical.get('canonical_criticality_promoted',False):
            assert physical['status']=='SUBCRITICAL_PD' and physical['full_physical_child_checks']
            assert abs(physical['parent_J_EE_core']-row['J_EE_core'])<1e-12
            assert Path(physical['parent_orbit']).resolve()==Path(row['accepted_parent_orbit']).resolve()
            identity=read(physical['radial_history_identity_source'])
            assert identity['status']=='RADIAL_HISTORY_IDENTITY_PASS'
            assert Path(identity['child_orbit']).resolve()==Path(physical['child_orbit']).resolve()
            child=physical_PD1
    if child.exists():
        cc=read(child)
        # A nonnegative mixture can conceal negative constituent filters.
        # Historical child classifications lacking this physical-state gate
        # retain their evidence but cannot supply an accepted criticality.
        if cc['status'] in ['SUBCRITICAL_PD','SUPERCRITICAL_PD'] and cc.get('full_physical_child_checks',False):
            row.update(criticality=cc['status'],child_multiplier=cc['child_mu'],child_side=cc['child_side'],child_classification_source=str(child))
            if cc.get('canonical_criticality_promoted',False):
                row.update(child_stability=cc['child_stability'],
                    child_validation_status='PHYSICAL_CHILD_AND_RADIAL_MODE_CHECKED')
            row['meaning']='Antiperiodic root independently checked at multiplier -1; the nonlinear child side and sampled stability are documented by child_classification_source. Global continuation remains separate.'
            mu=complex(*cc['child_mu'])
            if abs(mu)>1:row['child_unstable_multiplier']=cc['child_mu']
        else:
            row['child_validation_status']=cc['status']
            if cc['status'] in ['SUBCRITICAL_PD','SUPERCRITICAL_PD']:
                row['child_validation_status']='PHYSICAL_CHILD_RECHECK_REQUIRED'
                row['prior_child_criticality']=cc['status']
                row['child_classification_source']=str(child)
    if not row.get('full_acceptance',False):
        row['meaning']+=' Critical-mode evidence is retained, but full physical parent-waveform acceptance is withheld until its constituent filters are resolved.'
    write(PERIODIC_OUT/(a.label+'_validation.json'),row);print('PD VALIDATION',row,flush=True)


if __name__=='__main__':main()
