"""Keep a later PD4-child negative-multiplier candidate separate from roots.

The upper endpoint initially has only one time mesh and a large phase defect.
This records a discriminating follow-up interval, never a validated crossing.
"""
from audit_rate_PD4_child_spectral_change import real_span
from complete_rate_positive_stability import DEST, PERIODIC_OUT, read, write, values, paired_modes, np
from pathlib import Path
import time


def main():
    folder=DEST/'H2_local_PD'
    profiles=read(folder/'physical_children.json')['rows']
    sources=[];rows=[];bases=[]
    for index in [1,2]:
        profile=profiles[index]
        path=PERIODIC_OUT/f'poincare_floquet/PD4_child_index{index}_full_k6_20260920_dt0.05.json'
        q=read(path);mu=values(q)
        assert Path(q['orbit']).resolve()==Path(profile['orbit']).resolve()
        assert profile['physical_check']['filter_state_check']['positive']
        assert profile['physical_check']['maximum_group_defect_Hz']<1e-6
        negative=np.flatnonzero((abs(mu.imag)<1e-8)&(mu.real<0))
        assert len(negative)==2
        oscillatory=np.flatnonzero((abs(mu.imag)>1e-6)&(abs(mu)>1))
        assert len(oscillatory)==2
        weak=negative[np.argmin(abs(mu[negative]))]
        selected=np.r_[negative,oscillatory]
        assert max(np.asarray(q['residuals'])[selected]/np.maximum(1,abs(mu[selected])))<1e-6
        with np.load(path.with_suffix('.npz')) as z:
            vectors=np.r_[z['local_vectors'][:,selected],z['history_vectors'][:,selected]]
        bases.append(dict(real=real_span(vectors[:,:2]),four=real_span(vectors)))
        paired_path=path.with_name(path.name.replace('dt0.05','dt0.025'))
        pair=None
        if paired_path.exists():
            fine=read(paired_path)
            assert Path(fine['orbit']).resolve()==Path(profile['orbit']).resolve()
            pair=paired_modes(q,fine)
        sources.append(str(path))
        rows.append(dict(child_index=index,amplitude_Hz=profile['amplitude_hz'],
            J_EE_core=profile['J_EE_core'],orbit=profile['orbit'],spectrum_source=str(path),
            negative_real_multipliers=mu[negative].real,
            negative_real_block_at_minus_one=float(np.prod(-1-mu[negative].real)),
            selected_weak_negative_multiplier=float(mu[weak].real),
            selected_weak_residual=float(q['residuals'][weak]),
            selected_weak_phase_overlap=float(q['phase_overlap'][weak]),
            phase_tangent_relative_defect=q['phase_tangent_relative_defect'],
            paired_classification=pair,paired_source=str(paired_path) if pair else None))
    comparisons={}
    for key in ['real','four']:
        left,right=bases[0][key],bases[1][key]
        assert left.shape==right.shape
        cs=np.linalg.svd(left.T@right,compute_uv=False)
        comparisons[key]=np.degrees(np.arccos(np.clip(cs,0,1)))
    assert rows[0]['selected_weak_negative_multiplier']<-1<rows[1]['selected_weak_negative_multiplier']<0
    result=dict(status='LATER_NEGATIVE_MULTIPLIER_CROSSING_CANDIDATE',timestamp=time.time(),
        rows=rows,reference_phase_subspace_angles_degrees=comparisons,
        amplitude_bracket_Hz=[q['amplitude_Hz'] for q in rows],
        J_EE_core_bracket=[q['J_EE_core'] for q in rows],
        unit_circle_crossing_located=False,mode_continuation_completed=False,
        new_period_doubling_established=False,global_branch_completeness=False,
        relation_to_earlier_sign_change='This .04-.08 amplitude interval follows the .03-.04 positive-to-negative growing real-pair interval. These are different questions; neither endpoint sign pattern alone locates a bifurcation.',
        next_required_evidence='Complete upper-endpoint paired-step/phase checks; follow the negative spectral block and independently solve the antiperiodic boundary-value zero on the physical doubled branch before assigning a new PD.',
        scope='Endpoint eigenpairs and descriptive full-state/history subspace comparisons only. The larger-amplitude first mesh has a substantial autonomous-phase defect, so this is a follow-up candidate, not a newly confirmed root or unstable-dimension change.')
    write(folder/'later_negative_multiplier_candidate.json',result)
    print('LATER PD CANDIDATE',[(q['amplitude_Hz'],q['selected_weak_negative_multiplier'],
        q['phase_tangent_relative_defect'],q['paired_classification'] is not None) for q in rows],flush=True)
    print('SUBSPACE ANGLES',comparisons,flush=True)


if __name__=='__main__':main()
