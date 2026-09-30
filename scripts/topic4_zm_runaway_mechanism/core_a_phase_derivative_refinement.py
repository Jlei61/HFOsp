"""Check whether the failed neutral-mode test is a phase-derivative error.

Keep the original numerical orbit, physical flow, mesh and acceptance gate.
Only approximate dX/dt at its phase with successively higher-order one-sided
stencils of exact original forward steps. A positive result cannot replace
independent orbit phase/mesh checks or identify a critical crossing.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn
from onset_segment_flow import SegmentDerivative
from fine_rate_frozen_Z_fields import restore,capture
from pathlib import Path
import argparse,os,time,math


def main(parent,device,destination=None,uncached=False):
    parent=Path(parent).resolve();r=read(parent/'result.json')
    assert r['status']=='NUMERICAL_PERIODIC_ROOT'
    dest=Path(destination).resolve() if destination else parent/'neutral_derivative_refinement'
    dest.mkdir(exist_ok=True)
    assert not(dest/'jobs.json').exists()
    dt=read(parent/'contract.json')['dt_ms']
    T=r.get('period_ms') or r.get('rows',r.get('iterations'))[-1]['period_ms']
    if uncached:
        # Use the original full variational equations when the refined
        # trajectory would exceed GPU cache capacity. Its nonlinear FD
        # check must already have passed in the source root correction.
        checks=r.get('derivative_checks',[])
        assert checks and checks[-1]['relative_error']<1e-3
    else:
        qa=OUT/'core_a_bifurcation_type_20260924/numerical_checks/exact_cached_tangent/result.json'
        assert dt==.05 and read(qa)['status']=='PASS'
    write(dest/'contract.json',dict(source=str(parent),dt_ms=dt,orders=[2,4,6,8],cached_derivative=not uncached,
        question='Is the previously failed autonomous neutral-mode test limited by its second-order estimate of the phase velocity?',
        unchanged='Same complete closed numerical root, same fixed-time actual variational flow, all Z held, all M dynamic, same1e-3 neutral residual gate. No physical equation, waveform, period or mesh changes.',
        method='Forward derivative coefficients w_j=(-1)^(j+1)binomial(n,j)/j applied to actual state differences X(j*dt)-X(0), divided bydt. Full-coordinate original monodromy applied independently to each resulting direction.',
        scope='Numerical diagnostic of phase differentiation only. Convergence of this check cannot replace independent shifted-orbit/mesh validation or certify a Floquet crossing.',model_promoted=False))
    write(dest/'jobs.json',dict(status='RUNNING',pid=os.getpid()));start=time.time()
    e=build(device,dt);base=dict(np.load(parent/('root_state.npz' if(parent/'root_state.npz').exists() else 'latest_state.npz')))
    A=CubicSectionReturn(base,e,T,5.);x=A.xref;restore(e,base);diff=[]
    for _ in range(8):
        e.step();e.cp.cuda.get_current_stream().synchronize();diff.append(A.c.pack(capture(e))-x)
    if not uncached:
        free,_=e.cp.cuda.runtime.memGetInfo();required=(int(np.floor(T/dt))+2)*44*e.s.P*8
        assert required<.8*free,(required,free)
    F=SegmentDerivative(A,x,T,cached=not uncached);rows=[];last=None
    for order in [2,4,6,8]:
        flow=sum(((-1)**(j+1)*math.comb(order,j)/j)*diff[j-1] for j in range(1,order+1))/dt
        product=F(flow);error=float(np.linalg.norm(product-flow)/np.linalg.norm(flow))
        row=dict(order=order,neutral_mode_relative_residual=error,
                 derivative_relative_change=None if last is None else float(np.linalg.norm(flow-last)/np.linalg.norm(flow)),
                 original_neutral_gate_passed=bool(error<1e-3))
        rows.append(row);last=flow;write(dest/'progress.json',rows);log('PHASE DERIVATIVE REFINEMENT',row)
    write(dest/'result.json',dict(status='PHASE_DERIVATIVE_DIAGNOSTIC_COMPLETE',rows=rows,
        seconds=time.time()-start,physical_Floquet='NOT_PROMOTED',model_promoted=False))
    write(dest/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=0)
    p.add_argument('--destination')
    p.add_argument('--uncached',action='store_true')
    a=p.parse_args();main(a.parent,a.device,a.destination,a.uncached)
