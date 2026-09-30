"""Audit the accepted gap-filling states and separate them from failed probes."""
from zm_model import *
import hashlib


def main():
    s=ZMRate();base=DEST/'g20'
    names=['D_gap_lower_guarded','D_gap_middle_down','D_gap_middle_up','D_gap_middle_to_low','D_gap_rate_slices']
    branches=[];rates=[];proven=0
    for name in names:
        parent=base/name;info=read(parent/'result.json');rows=[r for r in info['rows'] if r.get('converged',True)]
        maximum=0.;bounds=True;Z_error=0.
        for row in rows:
            with np.load(parent/f'point{row["index"]:04d}.npz') as z:r=z['r'];D=float(z['D'])
            maximum=max(maximum,float(abs(s.residual(r,D)).max()*1000))
            bounds &=bool(0<=D<=1 and r.min()>=0 and np.all(r<1/s.ref) and s.Z.min()>=0 and s.Z.max()<=1)
            Z_error=max(Z_error,abs(s.Z[s.E]@s.mean_weights-(1-D)))
        rootfile=parent/'temporal_gap_modes/result.json'
        roots=read(rootfile)['rows'] if rootfile.exists() else []
        verified=[r for r in roots if r['equilibrium_stability']=='UNSTABLE'];proven+=len(verified)
        quality=[r['incoming_step_quality'] for r in rows if r.get('incoming_step_quality')]
        branch=dict(branch=name,status=info['status'],accepted_points=len(rows),
            D_range=[min(r['D'] for r in rows),max(r['D'] for r in rows)],
            global_E_hz_range=[min(r['global_E_hz'] for r in rows),max(r['global_E_hz'] for r in rows)],
            recomputed_max_residual_hz=maximum,physical_bounds_pass=bounds,max_Z_mean_error=Z_error,
            verified_unstable_samples=len(verified),
            min_step_tangent_cosine=min(q['tangent_cosine'] for q in quality) if quality else None,
            max_predictor_correction_fraction=max(q['correction_fraction'] for q in quality) if quality else None)
        assert maximum<2.1e-8 and bounds and Z_error<1e-10,branch
        branches.append(branch);rates.extend(r['global_E_hz'] for r in rows)
    fixed=read(DEST/'model_identity.json')['files']
    identity=all(hashlib.sha256(Path(row['snapshot']).read_bytes()).hexdigest()==row['sha256'] for row in fixed)
    assert identity
    y=np.sort(np.array([v for v in rates if 100<=v<=210]))
    fold=read(base/'D_gap_lower_guarded/verified_outer_fold/result.json')['rows'][0]
    result=dict(status='ORIGINAL_VISUAL_GAP_POPULATED_WITH_VERIFIED_EQUILIBRIA',branches=branches,
        frozen_mathematical_source_unchanged=identity,M='dynamic',Z='fixed along the original prescribed D path',
        verified_unstable_new_samples=proven,
        maximum_spacing_in_100_to_210_hz=float(np.max(np.diff(y))),
        middle_seed=read(base/'D_gap_fixedpoint_homotopy_0p205/result.json')['rows'][0],
        additional_verified_fold=fold,
        all_solver_homotopy_intermediates_excluded=True,
        original_upper_state_connection=read(DEST/'gap_original_upper_connection.json'),
        upper_extension_state_connection=read(DEST/'gap_upper_branch_connection.json'),
        original_lower_state_connection=read(DEST/'gap_original_lower_connection.json'),
        claim='Additional spatial equilibrium branches fill the plotted rate gap; this is not proof that all branches form one continuous path or that the SN is seizure onset',
        periodic_orbits='NOT_COMPUTED',human_visual_acceptance='PENDING')
    write(DEST/'gap_completion.json',result);print(clean(result))


if __name__=='__main__':main()
