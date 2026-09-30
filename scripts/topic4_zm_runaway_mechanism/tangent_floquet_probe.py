"""Test a branch tangent against the full delayed variational flow.

This is a targeted eigenvector probe, not a substitute for the leading spectrum.
Near a genuine cycle fold the phase-quotiented branch tangent must approach a
unit multiplier. A large defect rejects that interpretation at the given mesh.
"""
from phase_audit import *
from chunk_monodromy import ChunkEndpointMonodromy
from streaming_periodic import StreamPeriodic
from scipy.interpolate import CubicSpline
from scipy.signal import resample
from orbit_reconstruction import orbit_states
import chunk_monodromy
chunk_monodromy.orbit_states=orbit_states


def branch_vector(o,z,m,h):
    r=z['r'];v=z['tangent'];n=r.size
    vectors=[]
    for sign in [-1,1]:
        sol=dict(r=r+sign*h*v[:n].reshape(r.shape)*.001,
                 T=float(z['T'])*np.exp(sign*h*v[-2]),
                 D=float(z['D'])+sign*h*v[-1]*.001)
        full,_=orbit_states(o,sol,len(r))
        state=full[0].copy();state[11]=0.;del full
        # Evaluate histories at fixed physical times, not fixed orbit phases.
        nr=8*len(r);rr=resample(sol['r'],nr,axis=0)
        rr=np.concatenate([rr,rr[:1]],axis=0)
        grid=np.arange(nr+1)*sol['T']/nr
        interp=CubicSpline(grid,rr,axis=0,bc_type='periodic')
        hist=interp((-np.arange(1,m.Dd+1)*m.dt)%sol['T'])
        vectors.append(np.r_[state.ravel(),hist.ravel()])
    return (vectors[1]-vectors[0])/(2*h)


def main(a):
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    z=np.load(a.orbit);sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    StreamPeriodic.harmonic_block=a.harmonic_block
    o=StreamPeriodic(s,len(z['r']),a.device);o.cache_mean_operators=False
    m=ChunkEndpointMonodromy(s,o,sol,dtmax=a.dt,device=a.device)
    full,_=orbit_states(o,sol,m.n);phase=m.phase_vector(full);del full
    mp=m.matvec(phase);phase_defect=np.linalg.norm(mp-phase)/np.linalg.norm(phase)
    phase_projection=mp@phase/(phase@phase)
    log('PHASE',phase_defect,phase_projection)
    def project(x):return x-phase*(phase@x)/(phase@phase)
    v=branch_vector(o,z,m,1e-6);v2=branch_vector(o,z,m,5e-7)
    fd_error=float(np.linalg.norm(v-v2)/np.linalg.norm(v2))
    assert fd_error<1e-4,fd_error
    s.set_D(sol['D']);o.cache=None;o.cache_key=None;o.cp.get_default_memory_pool().free_all_blocks()
    v=project(v2);v/=np.linalg.norm(v);rows=[]
    dest=OUT/'floquet'/a.label;dest.mkdir(parents=True,exist_ok=True)
    for k in range(a.steps):
        w=project(m.matvec(v));mu=float(v@w);growth=float(np.linalg.norm(w))
        residual=float(np.linalg.norm(w-mu*v));unit_defect=float(np.linalg.norm(w-v))
        q=dict(iteration=k,multiplier_rayleigh=mu,growth=growth,
               eigenvector_defect=residual,unit_multiplier_defect=unit_defect)
        rows.append(q);log('TANGENT PROBE',q)
        v=w/growth
        write(dest/'progress.json',dict(status='RUNNING',rows=rows))
        if k>=2 and residual<1e-5:break
    np.savez_compressed(dest/'direction.npz',vector=v,phase=phase)
    write(dest/'result.json',dict(status='TARGETED_PROBE_COMPLETE',orbit=a.orbit,
          D=sol['D'],T_ms=sol['T'],dt_ms=m.dt,phase_defect=float(phase_defect),
          phase_projection=float(phase_projection),branch_vector_fd_error=fd_error,
          phase_valid=bool(phase_defect<.005 and abs(phase_projection-1)<.003),
          rows=rows,claim='Targeted multiplier probe; leading spectrum and step refinement still required'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--label',required=True)
    p.add_argument('--family',choices=['native','rate'],default='native')
    p.add_argument('--dt',type=float,default=.025);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--harmonic-block',type=int,default=33)
    p.add_argument('--device',type=int,default=0);main(p.parse_args())
