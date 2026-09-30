"""Identify whether a near-root residual is an interpolation positivity floor."""
from common import OUT,np,read,write,log
from core_a_multiple_shooting import setup
from onset_state_continuation import build
from onset_cubic_section import weights
from fine_rate_frozen_Z_fields import restore,capture
from onset_period_return import dynamical_state,errors
import os,argparse
from pathlib import Path


def main(a):
    base=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/actual_D0270/recurrence'
    out=Path(a.destination) if a.destination else base/'dense_output_floor_diagnostic'
    out.mkdir(exist_ok=True);assert not (out/'jobs.json').exists()
    source=Path(a.resume) if a.resume else base/'multiple_nine_burst_refine/iteration00'
    seed=Path(a.source) if a.source else base/'nine_burst_numerical_replay'
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    write(out/'contract.json',dict(source=str(source),segments=a.segments,
        question='Does the near-root residual, now almost entirely in rate history, come from negative cubic interpolation values at quiet shooting nodes rather than physical lack of periodicity?',
        method='Retain the same original full-step flow and four actual adjacent states. Compare its cubic interpolation with their convex linear interpolation only as a diagnostic. Quantify the minimum mismatch contributed by negative values in physically nonnegative coordinates. No clipping of accepted flow, no solver-gate change and no bifurcation label.',model_promoted=False))
    e=build(a.device);parts,T=setup(e,seed,source,common_scale='cycle_rms',segments=6)
    rows=[]
    for j in a.segments:
        A=parts[j];h=T/6;n=int(np.floor(h/e.dt));a=h/e.dt-n;restore(e,A.base)
        whole,tail=divmod(n-1,round(10/e.dt))
        for _ in range(whole):e.chunk()
        for _ in range(tail):e.step()
        e.cp.cuda.get_current_stream().synchronize();points=[A.c.pack(capture(e))]
        for _ in range(3):
            e.step();e.cp.cuda.get_current_stream().synchronize();points.append(A.c.pack(capture(e)))
        cubic=sum(w*q for w,q in zip(weights(a),points));linear=(1-a)*points[1]+a*points[2]
        target=dynamical_state(A.state(parts[(j+1)%6].xref));current=dynamical_state(A.state(cubic))
        floor={k:v.copy() for k,v in target.items()};negative=[]
        for key,sl in [('syn',slice(None)),('local',slice(0,6)),('history',slice(None))]:
            values=current[key][sl];neg=np.minimum(values,0.);floor[key][sl]+=neg
            negative.append(dict(block=key,negative_count=int((values<0).sum()),minimum=float(values.min())))
        w=e.s.sizes/e.s.sizes.sum();actual=errors(target,current,w);lower=errors(target,floor,w)
        row=dict(segment=j,period_ms=T,segment_duration_ms=h,fractional_step=a,
            actual_matching=actual,positivity_only_lower_bound=lower,negative=negative,
            cubic_admissible=bool(A.admissible(cubic)),linear_admissible=bool(A.admissible(linear)),
            negative_floor_fraction_of_squared_error=float((lower['combined_relative_rms']/actual['combined_relative_rms'])**2))
        rows.append(row);np.savez_compressed(out/f'segment{j}_samples.npz',points=np.array(points),cubic=cubic,linear=linear,
            scale=A.c.scale,weight=A.c.weight,target=parts[(j+1)%6].xref)
        write(out/'progress.json',rows);log('DENSE OUTPUT FLOOR',row)
    write(out/'result.json',dict(status='DIAGNOSTIC_COMPLETE',rows=rows,model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--resume');p.add_argument('--destination')
    p.add_argument('--device',type=int,default=0);p.add_argument('--segments',type=int,nargs='+',default=[2])
    main(p.parse_args())
