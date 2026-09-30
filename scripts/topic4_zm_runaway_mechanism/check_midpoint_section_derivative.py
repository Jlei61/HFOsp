"""Validate composition and implicit crossing of the matching derivative.

The section is placed at a known 40 ms trajectory endpoint. This provides a
real transverse crossing without presuming that this trajectory is periodic.
"""
from common import OUT,np,read,write,log
from onset_exponential_midpoint import ExponentialMidpointEngine
from onset_midpoint_tangent import MidpointTangent
from onset_midpoint_cached_tangent import MidpointCachedTangent
from onset_cubic_section import CubicSectionReturn
from onset_segmented_poincare import SegmentedPoincareDerivative
from fine_rate_frozen_Z_fields import capture,restore
import argparse,os,gc

BASE=OUT/'core_a_bifurcation_type_20260924'
DEST=BASE/'numerical_checks/exponential_midpoint/section_variational_check'


def main(device,off_grid=False):
    global DEST
    if off_grid:DEST=DEST.with_name('section_variational_check_offgrid')
    assert read(DEST.parent/'cached_variational_check/result.json')['status']=='PASS'
    DEST.mkdir(exist_ok=True);assert not(DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(duration_ms=40,meshes_ms=[.05,.025],segments=2,off_grid=off_grid,
        crossing_time=('40+0.37*dt ms, away from a cubic interpolation knot. The initial exact40ms test is preserved separately; its centered derivative straddles two interpolation polynomials.' if off_grid else 'exact40ms grid knot'),
        section='Fixed hyperplane through an actual40ms trajectory endpoint, normal to its centered fullstate velocity. This is a known crossing, explicitly not a periodic-return claim.',
        gates='Segmented versus independently integrated uncached full derivative<1e-10 including implicit time. Actual centered nonlinear section-return differences<1e-3 and converge with epsilon; physical perturbations admissible.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(DEST/'jobs.json',jobs);rows=[]
    try:
        for dt,label in [(.05,'matched_natural_short_entry_step'),(.025,'matched_natural_short_entry_step_fine')]:
            e=ExponentialMidpointEngine(dt=dt,device=device);e.graph()
            path=BASE/'reference_stability_gap'/label/'held_short_field/checkpoint1000.npz'
            base=dict(np.load(path));A=CubicSectionReturn(base,e,40,1.)
            x=A.xref.copy();restore(e,base)
            for _ in range(3):e.chunk()
            for _ in range(round(10/dt)-1):e.step()
            e.cp.cuda.get_current_stream().synchronize();left=A.c.pack(capture(e))
            e.step();e.cp.cuda.get_current_stream().synchronize();middle=A.c.pack(capture(e))
            e.step();e.cp.cuda.get_current_stream().synchronize();right=A.c.pack(capture(e))
            target=40.
            if off_grid:
                from onset_segment_flow import fixed_time
                target=40.+.37*dt;middle,normal=fixed_time(A,x,target)
            else:normal=(right-left)/(2*dt)
            A.normal=normal/np.linalg.norm(normal);A.xref=middle.copy()
            y,meta=A(x);slope=A.last_time_slope.copy();assert abs(meta['period_ms']-target)<1e-7
            J=SegmentedPoincareDerivative(A,x,meta['period_ms'],slope,2,
                tangent_class=MidpointTangent,cached_tangent_class=MidpointCachedTangent)
            rng=np.random.default_rng(9250825);v=x*rng.uniform(-1,1,x.shape);exact=J(v);tests=[]
            for eps in [2e-5,1e-5,5e-6]:
                answers=[]
                for sign in [-1,1]:
                    xx=x+sign*eps*v;assert A.admissible(xx)
                    answers.append(A(xx)[0])
                fd=(answers[1]-answers[0])/(2*eps)
                error=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-12))
                tests.append(dict(epsilon=eps,relative_error=error));log('MIDPOINT SECTION FD',dt,tests[-1])
            assert tests[-1]['relative_error']<1e-3
            if off_grid:
                assert tests[-1]['relative_error']<max(1e-7,tests[0]['relative_error']/3),tests
            rows.append(dict(dt_ms=dt,partition=J.partition_check,tests=tests))
            write(DEST/'progress.json',rows)
            cp=e.cp;del J,A,e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        write(DEST/'result.json',dict(status='PASS',rows=rows,not_a_periodic_orbit=True,model_promoted=False))
        jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--off-grid',action='store_true');a=p.parse_args();main(a.device,a.off_grid)
