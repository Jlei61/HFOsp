"""Candidate temporal eigenvalues of a verified current-model equilibrium.

Use the actual conditioned response and physical private-variance operators.
No old-v3 spectral gain/bound is imported. A search for some eigenvalues can
prove instability, but cannot certify stability or completeness.
"""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from scipy import sparse
from scipy.sparse.linalg import eigs,spsolve
from pathlib import Path
import argparse,os,time


def refine(s,r,op,lam,dt,v=None):
    def matrix(z):return s.characteristic(r,z,dt=dt,operating=op)
    if v is None:
        ev,vv=eigs(matrix(lam),k=3,sigma=0,tol=1e-9)
        v=vv[:,np.argmin(abs(ev))]
    pivot=int(np.argmax(abs(v)));v=v/v[pivot]
    c=sparse.csr_matrix(([1.],([0],[pivot])),shape=(1,s.P));trace=[]
    for it in range(25):
        C=matrix(lam);f=C@v;error=float(np.linalg.norm(f)/np.linalg.norm(v))
        trace.append(dict(iteration=it,lambda_per_ms=[float(lam.real),float(lam.imag)],relative_residual=error))
        if error<1e-9:return lam,v/np.linalg.norm(v),trace
        h=1e-6;derivative=(matrix(lam+h)-matrix(lam-h))/(2*h)
        B=sparse.bmat([[C,sparse.csr_matrix((derivative@v)[:,None])],[c,sparse.csr_matrix((1,1))]],format='csc')
        step=spsolve(B,np.r_[-f,0j]);accepted=False
        for alpha in 2.**-np.arange(14):
            ll=lam+alpha*step[-1];vv=v+alpha*step[:-1]
            if abs(ll)>2:continue
            err=np.linalg.norm(matrix(ll)@vv)/np.linalg.norm(vv)
            if err<error:lam,v=ll,vv;accepted=True;break
        if not accepted:break
    return None,None,trace


def main(a):
    source=Path(a.source);data=np.load(source);s=PhysicalDelayConditionalDrift();s.set_Z(data['Z']);r=data['r']
    error=float(abs(s.residual(r)).max());assert error<1e-11,('Not a root',error)
    dest=source.parent/'temporal_spectrum';dest.mkdir(exist_ok=True);assert not (dest/'result.json').exists()
    write(dest/'contract.json',dict(source=str(source),static_residual_per_ms=error,
        model='Actual current conditioned39 response and corrected physical private variance; transient correction first derivative vanishes at stationarity. Full spatial delayed network and dynamic M.',
        method='Bordered Newton solve C(lambda)v=0 with normalized eigenvector; independent residual at each candidate, step refinement from .05 to .025 and continuous-time formula.',
        scope='Candidate eigenvalues only. A verified positive-real-part root proves instability; absence in this finite search does not prove stability or a complete spectrum. No bifurcation without parameter crossing and nondegeneracy.',model_promoted=False))
    op=s.local_operating(*s.moments(r));rows=[]
    for index,guess in enumerate([.001+0j,.005+.015j,.005+.06j,.005+.18j,.005+.35j]):
        try:lam,v,trace=refine(s,r,op,guess,.05)
        except Exception as exc:
            write(dest/f'trial{index}.json',dict(status='FAILED',error=repr(exc)));continue
        write(dest/f'trial{index}.json',dict(status='ROOT_PASS' if lam is not None else 'NO_ROOT',trace=trace))
        if lam is None or any(abs(lam-complex(*row['coarse_lambda_per_ms']))<1e-7 for row in rows):continue
        checks=[];fine_v=v
        for dt in [.05,.025,None]:
            ll,vv,tr=refine(s,r,op,lam,dt,v.copy())
            if ll is None:checks.append(dict(dt_ms=dt,status='NO_ROOT',trace=tr));continue
            C=s.characteristic(r,ll,dt=dt) # Independently recompute operating state.
            err=float(np.linalg.norm(C@vv)/np.linalg.norm(vv))
            energy=s.sizes*abs(vv)**2;energy/=energy.sum()
            row=dict(dt_ms=dt,status='ROOT_PASS',lambda_per_ms=[float(ll.real),float(ll.imag)],
                growth_per_s=float(ll.real*1000),frequency_hz=float(abs(ll.imag)*1000/(2*np.pi)),independent_residual=err,
                mode_energy_A_B_surround=[float(energy[s.geo['group_region']==j].sum()) for j in range(3)])
            checks.append(row);np.savez_compressed(dest/f'mode{len(rows)}_dt{dt}.npz',lambda_per_ms=ll,v=vv,r=r,Z=s.Z)
            log('CORE A EQUILIBRIUM EIGENVALUE',row)
        rows.append(dict(coarse_lambda_per_ms=[float(lam.real),float(lam.imag)],checks=checks))
        write(dest/'progress.json',dict(status='RUNNING',pid=os.getpid(),rows=rows))
    write(dest/'result.json',dict(status='COMPLETE',source=str(source),rows=rows,
        spectrum_complete=False,bifurcation_type='NOT_ESTABLISHED',model_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');main(p.parse_args())
