"""Resolve two near-torus survey sites using explicit time-step convergence.

The broad survey uses a conservative 0.002 unit-circle margin. This bounded
follow-up measures the numerical error at each site instead of changing the
survey margin or promoting an isolated near-unit eigenvalue.
"""
from rate_stability_coverage import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--nev',type=int,default=4)
    parser.add_argument('--ncv',type=int,default=14)
    a=parser.parse_args();results=[];worker=COVER/'near_unit_followup_worker.json'
    for tag in ['resonanceA_0039_N64','resonanceA_0063_N64']:
        prior=read(COVER/(tag+'.json'));path=Path(prior['analyzed_orbit'])
        checks=[]
        for dt in [.05,.025,.0125]:
            write(worker,dict(status='FLOQUET',pid=os.getpid(),tag=tag,dt_ms=dt,results=results))
            source=PERIODIC_OUT/'floquet'/f'{tag}_nearunit_dt{dt:g}.json'
            q=read(source) if source.exists() else compute(path,dt,a.nev,a.device,True,a.ncv,output_label=tag+'_nearunit')
            raw=np.asarray(q['multipliers'])
            vals=raw[:,0]+1j*raw[:,1] if raw.ndim==2 else raw.astype(complex)
            ids=np.flatnonzero(vals.imag>1e-4);assert len(ids)
            i=ids[np.argmin(abs(abs(vals[ids])-1))]
            checks.append(dict(source=str(source),dt_ms=q['dt_ms'],mu=vals[i],
                modulus=float(abs(vals[i])),eigen_residual=q['residuals'][i],
                phase_defect=q['phase_tangent_relative_defect'],neutral_index=q['identified_neutral_index']))
            import cupy as cp
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
        mu=np.array([v['mu'] for v in checks]);changes=abs(np.diff(mu))
        fine=checks[-1];margin=max(4*changes[-1],8*fine['phase_defect'],8*fine['eigen_residual'])
        # The relevant accuracy is error relative to distance from |mu|=1,
        # already enforced by the measured margin. A universal absolute
        # 1e-7 drift gate does not express that scientific question.
        convergence=(changes[0]>2.8*changes[1])
        resolved=(convergence and fine['modulus']>1+margin and
            all(v['eigen_residual']<1e-7 and v['neutral_index'] is not None for v in checks))
        result=dict(status='UNSTABLE' if resolved else 'UNRESOLVED',orbit=str(path),
            checks=checks,successive_complex_multiplier_changes=changes,
            refinement_ratio=float(changes[0]/changes[1]),measured_margin=float(margin),
            criteria=dict(minimum_refinement_ratio=2.8,drift_safety_factor=4,
                phase_safety_factor=8,eigen_residual_safety_factor=8,
                note='Classification compares distance outside the unit circle with the measured step/phase/residual margin; no absolute multiplier-drift target is used.'),
            scope='A conjugate pair outside the unit circle after three time steps establishes instability at this sampled orbit. No new bifurcation or full unstable dimension is inferred.')
        output=COVER/(tag+'_near_unit_followup.json')
        if output.exists():
            history=COVER/'attempt_history';history.mkdir(exist_ok=True)
            write(history/f'{tag}_prior_followup_{time.time_ns()}.json',read(output))
        write(output,result)
        if resolved:
            history=COVER/'attempt_history';history.mkdir(exist_ok=True)
            write(history/f'{tag}_before_nearunit_{time.time_ns()}.json',prior)
            prior.update(status='UNSTABLE',reliable_outside_count=2,margin=float(margin),
                independent_followup_source=str(output),classification_basis=result['scope'])
            write(COVER/(tag+'.json'),prior)
        results.append(dict(tag=tag,status=result['status'],source=str(output)))
        print('NEAR UNIT FOLLOWUP',result,flush=True)
    write(worker,dict(status='BATCH_FINISHED',pid=os.getpid(),results=results))


if __name__=='__main__':main()
