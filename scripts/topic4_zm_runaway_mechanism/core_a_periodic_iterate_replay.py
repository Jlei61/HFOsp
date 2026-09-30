"""Check original-flow reproducibility at a saved nonconverged cycle iterate.

Large nonlinear remainders under shrinking Newton updates motivate this
check. No derivative, projection, parameter change or refit is used here.
"""
from common import np, read, write, log
from onset_state_continuation import build
from onset_poincare_corrector import SectionReturn
from onset_segment_flow import fixed_time, fixed_times
from onset_period_return import errors, dynamical_state
from pathlib import Path
import argparse, os


def main(a):
    source=Path(a.source).resolve(); out=Path(a.destination).resolve()
    out.mkdir(parents=True,exist_ok=True); assert not (out/'jobs.json').exists()
    contract=read(source.parent/'contract.json'); dt=contract['dt_ms']
    meta=read(source/'accepted_period.json'); T=meta['period_ms']
    durations=np.array(meta['durations_ms']); assert abs(durations.sum()-T)<1e-8
    base=dict(np.load(source/'accepted_node00.npz'))
    write(out/'contract.json',dict(source=str(source),dt_ms=dt,period_ms=T,device=a.device,
        question='Do repeated original full-period flows and the segmented-prefix readout agree at this saved nonconverged iterate?',
        method='Two independent fixed_time evaluations and one fixed_times evaluation with the actual saved shooting partition. Each starts from the identical full state. No Newton update or physical state clipping. Full dynamical states compared in original physical blocks.',
        acceptance='Repeated fixed_time full canonical endpoints bitwise equal; shared-prefix versus independent endpoint relativeL2<1e-10. This tests reproducibility, not periodicity, stability or a bifurcation.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid()); write(out/'jobs.json',jobs)
    try:
        e=build(a.device,dt); A=SectionReturn(base,e,T); x=A.xref; rows=[]; endpoints=[]
        w=e.s.sizes/e.s.sizes.sum()
        for method in ['fixed_time_first','fixed_time_repeat','shared_prefix']:
            y,_=fixed_times(A,x,np.cumsum(durations))[-1] if method=='shared_prefix' else fixed_time(A,x,T)
            state=A.state(y); closure=errors(dynamical_state(base),dynamical_state(state),w)
            row=dict(method=method,whole_closure=closure['combined_relative_rms'],blocks=closure['blocks'])
            if endpoints:
                row.update(first_endpoint_bitwise=bool(np.array_equal(y,endpoints[0])),
                    first_endpoint_relative_L2=float(np.linalg.norm(y-endpoints[0])/np.linalg.norm(endpoints[0])))
            endpoints.append(y); rows.append(row); write(out/'progress.json',rows); log('CYCLE ITERATE REPLAY',row)
        passed=rows[1]['first_endpoint_bitwise'] and rows[2]['first_endpoint_relative_L2']<1e-10
        np.savez_compressed(out/'canonical_endpoints.npz',first=endpoints[0],repeat=endpoints[1],shared=endpoints[2],
            coordinate_scale=A.c.scale,coordinate_weight=A.c.weight)
        result=dict(status='PASS' if passed else 'ORIGINAL_FLOW_REPLAY_MISMATCH',rows=rows,
                    periodicity='NOT_ESTABLISHED',bifurcation_type='NOT_ESTABLISHED',model_promoted=False)
        write(out/'result.json',result); jobs.update(status='COMPLETE',scientific_status=result['status'])
        write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc)); write(out/'jobs.json',jobs); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--destination',required=True)
    p.add_argument('--device',type=int,required=True); main(p.parse_args())
