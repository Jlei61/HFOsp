"""Numerical evidence index for periodic continuation; no scientific report prose."""
from plot_rate_periodic_completion import *
import csv


def main():
 fs=families();rows=[]
 for i,q in enumerate(read(RATE_OUT/'hopfs.json')['rows']):
  nf=read(PERIODIC_OUT/f'normal_form_{"AB"[i]}.json')
  rows.append(dict(label=f'H{i+1}',internal_label=f'H{"AB"[i]}',type='Hopf',J_EE_core=q['J_EE_core'],T_ms=1000/q['frequency_hz'] if 'frequency_hz' in q else None,mesh_N=None,periodic_residual_hz=None,derivative=None,curvature=None,source=str(RATE_OUT/f'hopf_{"AB"[i]}.npz')))
 for q in additional_hopfs():
  label=q['label'].split('_')[0]
  rows.append(dict(label=label,internal_label=label,type='Hopf on unstable equilibrium',J_EE_core=q['J_EE_core'],T_ms=1000/q['frequency_hz'],mesh_N=None,periodic_residual_hz=None,derivative=q['transversality']['real_exponent_derivative_per_ms_per_J'],curvature=None,source=q['validation']))
 for q in critical():
  rows.append(dict(label=CRITICAL_LABELS[q['label']],internal_label=q['label'],type=q['type'] if 'type' in q else 'torus',J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],mesh_N=q['N'],periodic_residual_hz=q.get('residual_hz',q.get('orbit_residual_hz')),derivative=q.get('dJ_dcoordinate',q.get('dJ_dlogT')),curvature=q.get('d2J_dcoordinate2',q.get('d2J_dlogT2')),source=q['orbit']))
 for i,q in enumerate(read(OUT/'critical_revision/fold_audit.json')['rows']):
  rows.append(dict(label=f'LP{i+1}',internal_label=q['label'],type='stationary fold',J_EE_core=q['J_EE_core'],T_ms=None,mesh_N=None,periodic_residual_hz=None,derivative=None,curvature=q['quadratic_coefficient'],source=q['source']))
 with (PERIODIC_OUT/'critical_points.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 meshes=[]
 for name in CRITICAL_NAMES+['TR_A_B','TR_A_return','PD_double_low','PD_double_upper','PD_A_return']:
  versions=sorted([read(f) for f in PERIODIC_OUT.glob(name+'_N*.json')],key=lambda q:q['N'])
  for lo,hi in zip(versions[:-1],versions[1:]):meshes.append(dict(label=name,N_low=lo['N'],N_high=hi['N'],J_absolute_difference=abs(lo['J_EE_core']-hi['J_EE_core']),T_ms_absolute_difference=abs(lo['T_ms']-hi['T_ms'])))
 group={family:dict(count=len(rr),J_min=min(q['J_EE_core'] for q in rr),J_max=max(q['J_EE_core'] for q in rr),maximum_periodic_residual_hz=max(q['residual_hz'] for q in rr)) for family,rr in fs.items()}
 status=dict(status='PARTIAL_BRANCH_COMPLETION',spatial_cells=400,population_groups=935,local_continuous_states=8415,full_delay_history_retained=True,
    periodic_families=group,critical_points=rows,mesh_checks=meshes,
    confirmed_core_claims=['The first two Hopf bifurcations are locally supercritical; additional unstable-branch Hopfs are indexed separately','Distinct small and large periodic solutions coexist at J=.942; earlier stable multiplier estimates and exact physical-parent rechecks are separate evidence','First torus locally subcritical from a mesh-validated two-angle branch','Second torus locally supercritical; noncritical parent spectrum checked separately','Located folds of large A-leading, B-leading and alternating cycles have individual physical-profile and mode validation statuses','A/B leading order alternates inside one solved full-network period; its full-state stability evidence is indexed separately'],
    stationary_spectrum_scope=dict(selected_eigenpair_sites=len(read(RATE_OUT/'branch_spectrum.json')['rows']),
        determinant_count_sites=[q['J_EE_core'] for q in read(RATE_OUT/'nyquist.json')['rows']],
        complete=False,meaning='A positive characteristic root certifies instability, not the full unstable root count or absence of additional Hopf crossings.'),
    not_established=['Stable irregular-burst attractor','Global nonlinear fate beyond the local TR1 torus branch','Exhaustive Floquet crossing inventory on all traced segments','Exhaustive characteristic-root crossings on the unstable stationary branches','Global connection between every small-oscillation and large-burst family','Native SNN dynamic equivalence across the whole parameter range'],
    first_torus=read(PERIODIC_OUT/'TR_A_B_validation.json') if (PERIODIC_OUT/'TR_A_B_validation.json').exists() else None,
    additional_equilibrium_hopfs=additional_hopfs(),
    second_torus=read(PERIODIC_OUT/'TR_A_return_nonlinear_validation.json') if (PERIODIC_OUT/'TR_A_return_nonlinear_validation.json').exists() else None,
    full_stationary_root_counts=read(PERIODIC_OUT/'stationary_root_counts/summary.json') if (PERIODIC_OUT/'stationary_root_counts/summary.json').exists() else None,
    no_inference_from=['Time-simulation extrema','Failed Newton steps','Plot-projection intersections','IEI CV alone'],
    period_doubling=read(PERIODIC_OUT/'PD_double_low_validation.json') if (PERIODIC_OUT/'PD_double_low_validation.json').exists() else None,
    additional_period_doubling=read(PERIODIC_OUT/'PD_double_upper_validation.json') if (PERIODIC_OUT/'PD_double_upper_validation.json').exists() else None,
    H1_return_period_doubling=read(PERIODIC_OUT/'PD_A_return_validation.json') if (PERIODIC_OUT/'PD_A_return_validation.json').exists() else None,
    observer_source=str(PERIODIC_OUT/'periodic_contact_observations.json'),figure_acceptance='Agent visual checks only; pending user review')
 review=PERIODIC_OUT/'rate_filter_state_positivity_audit.json'
 if review.exists():
  check=read(review)
  status['constituent_filter_state_review']=dict(source=str(review),status=check['status'],
      checked_profiles=len(check['rows']),profiles_requiring_refinement=sum(not q['positive'] for q in check['rows']),
      scope='Critical-mode evidence and full physical-state waveform acceptance are separate.')
 # Rebuilding the numerical index cannot certify that the rendered figure
 # has been rebuilt from its new accepted orbit sources.
 status['rendered_composite_pending_refresh']=True
 write(PERIODIC_OUT/'analysis_status.json',status)
 print('counts',group,'critical',len(rows),flush=True)

if __name__=='__main__':main()
