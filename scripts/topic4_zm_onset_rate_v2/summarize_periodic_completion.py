"""Persist evidence and explicit remaining scientific work; no simulations."""
from periodic_stability_zm import refresh
from periodic_zm import *
import hashlib


def main():
    spectra=refresh();best={}
    for q in spectra:
        key=Path(q['source']).stem
        if key not in best or (q['status']!='UNRESOLVED' and (best[key]['status']=='UNRESOLVED' or q['dt_ms']<best[key]['dt_ms'])):
            best[key]=q
    write(PERIODIC_OUT/'best_stability_by_orbit.json',dict(rows=list(best.values()),rule='Keep classified evidence; among classified runs use smaller time step; all attempts remain in stability_by_orbit.json'))
    lpc=read(PERIODIC_OUT/'LPC_burst_N1024.json');coarse=read(PERIODIC_OUT/'LPC_burst_N512.json')
    hopf=read(PERIODIC_OUT/'hopf_high.json');nf=read(PERIODIC_OUT/'normal_form_high.json')
    orbits=[read(p) for p in (PERIODIC_OUT/'orbits').glob('*.json')]
    eq=[read(p) for p in (PERIODIC_OUT/'equilibrium_classification').glob('*.json')]
    eqrows=[q for bundle in eq for q in bundle.get('rows',[])]
    rootcounts=[dict(file=str(p),**read(p)) for p in (PERIODIC_OUT/'equilibrium_counts').glob('*.json')]
    remaining=['1 mm to 0.5 mm spatial convergence, especially the localized surround LPC mode',
        'Link the conditional Z power path to the actual autonomous Z/M entry field',
        'Native SNN dynamics and 2-D propagation correspondence; depleted local variance-response mismatch remains',
        'Classify the remaining equilibrium portions and extend the high-rate periodic family if required; sampled stability does not exclude narrow windows']
    summary=dict(model='Fixed 1 mm / 935-group spatial rate DDE',J_EE_core=1.,D_domain=[0,1],Z='held spatial power path',M='dynamic',
        converged_orbits=sum(q['status']=='CONVERGED' for q in orbits),
        best_Floquet_counts={k:sum(q['status']==k for q in best.values()) for k in ['STABLE','UNSTABLE','UNRESOLVED']},
        equilibrium_sample_counts={k:sum(q['status']==k for q in eqrows) for k in ['STABLE','UNSTABLE','UNRESOLVED']},
        separate_equilibrium_contour_counts=rootcounts,
        LPC=lpc,LPC_time_mesh_difference=dict(D=abs(lpc['D']-coarse['D']),T_ms=abs(lpc['T_ms']-coarse['T_ms'])),
        high_Hopf=dict(D=hopf['D'],global_E_hz=hopf['global_E_hz'],frequency_hz=hopf['frequency_hz'],criticality=nf['criticality']),
        crossing_stepsize_check=read(PERIODIC_OUT/'crossing_stepsize_check.json'),
        claim='Conditional loss of a self-limited cycle and appearance of persistent spatial activity; not established native SNN global onset',
        remaining=remaining,human_visual_acceptance='PENDING')
    write(PERIODIC_OUT/'analysis_summary.json',summary)
    old=read(DEST/'analysis_summary.json');old['periodic_completion']=summary;write(DEST/'analysis_summary.json',old)
    running=[]
    names={'floquet_zm.py','periodic_stability_zm.py','cycle_fold_zm.py','continue_periodic_zm.py','equilibrium_root_count.py','continue_from_cycle.py'}
    for folder in Path('/proc').glob('[0-9]*'):
        try:args=(folder/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if len(args)>1 and b'topic4_zm_onset_rate_v2/' in args[1] and Path(args[1].decode()).name in names:
            running.append(dict(pid=int(folder.name),script=Path(args[1].decode()).name))
    state=read(DEST/'status.json');state.update(stage='PERIODIC_BIFURCATION_CANDIDATE_DELIVERED',task_complete=False,
        periodic_continuation='SELF_LIMITED_BRANCH_TO_D0_AND_THROUGH_LPC; HIGH_HOPF_SMALL_CYCLES',
        Floquet=summary['best_Floquet_counts'],own_jobs_running=bool(running),own_jobs=running,
        onset_bifurcation_type='CONDITIONAL_SELF_LIMITED_EXIT_LPC; NATIVE_GLOBAL_ONSET_NOT_ESTABLISHED',
        high_rate_Hopf='SUPERCRITICAL',spatial_grid_convergence='NOT_TESTED',remaining=remaining)
    write(DEST/'status.json',state)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.rglob('*.py')}
    write(PERIODIC_OUT/'source_snapshot_hashes.json',files)
    print(clean(summary),flush=True)


if __name__=='__main__':main()
