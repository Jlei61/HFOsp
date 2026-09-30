"""Track distinct characteristic roots with adaptive intermediate corrections.

Interpolated states are a numerical homotopy only. All saved roots and crossing
brackets belong to exact stored equilibria. Duplicate convergences are rejected.
"""
from rate_stationary_contour_modes import *


def advance(s,r0,J0,r1,J1,modes):
    for steps in [1,2,4,8,16,32,64]:
        current=modes.copy();failed=False
        for j in range(1,steps+1):
            f=j/steps;r=r0*(1-f)+r1*f;J=J0*(1-f)+J1*f;found={}
            for k,(lam,v) in current.items():
                q=refine(s,r,J,lam,v,tol=2e-11)
                if q is None and abs(lam.imag)<1e-8:
                    # At a real-root collision a real initial guess cannot
                    # leave the real axis, even when the continued roots are
                    # now a complex pair. Probe both conjugate directions;
                    # retain the same overlap/distinctness/residual checks.
                    for offset in [1e-3j,-1e-3j,3e-3j,-3e-3j]:
                        candidate=refine(s,r,J,lam+offset,v,tol=2e-11)
                        if candidate is None:continue
                        l,u,e=candidate
                        if abs(np.vdot(v,u))/(np.linalg.norm(v)*np.linalg.norm(u))<.85:continue
                        if any(abs(l-other[0])<1e-7 for other in found.values()):continue
                        q=candidate
                        print('REAL COLLISION RESEED',k,lam,l,flush=True)
                        break
                if q is None:failed=True;break
                l,u,e=q
                overlap=abs(np.vdot(v,u))/(np.linalg.norm(v)*np.linalg.norm(u))
                if overlap<.85 or any(abs(l-other[0])<1e-7 for other in found.values()):
                    failed=True;break
                found[k]=(l,u)
            if failed:break
            current=found
        if not failed:return current,steps
        print('MODE SUBDIVISION',steps,'retry',flush=True)
    return None,64


def main(a):
    s=RateField();branch=np.load(RATE_OUT/'equilibrium_branch.npz');seed=np.load(a.seed)
    ids=np.flatnonzero(seed['roots'].imag>1e-5)
    modes={int(k):(seed['roots'][k],seed['vectors'][k]) for k in ids}
    r0=seed['rates'];J0=float(seed['J']);rows=[];previous={};crossings=[]
    step=(1 if a.stop>a.start else -1)*a.stride
    dest=DEST/f'{a.namespace}new_modes_{a.start}_to_{a.stop}.json'
    indices=list(range(a.start,a.stop,step))+[a.stop]
    if getattr(a,'resume',False) and dest.exists():
        prior=read(dest);rows=prior['rows'];crossings=prior['crossing_brackets']
        if rows:
            last=rows[-1]['branch_index'];modes={};previous={}
            for item in rows[-1]['roots']:
                k=item['mode'];z=np.load(DEST/f'{a.namespace}tracked_mode{k}_branch{last:04d}.npz')
                modes[k]=(complex(z['lam']),z['vector'])
                previous[k]=dict(index=last,J=float(z['J']),lam=complex(z['lam']))
            r0=branch['rates'][last];J0=float(branch['J'][last]);indices=indices[indices.index(last)+1:]
    for i in indices:
        r=branch['rates'][i];J=float(branch['J'][i]);found,n=advance(s,r0,J0,r,J,modes)
        if found is None:
            write(dest,dict(status='TRACKING_UNRESOLVED',source=a.seed,rows=rows,
                failed_branch_index=i,crossing_brackets=crossings,namespace=a.namespace))
            raise RuntimeError(f'Cannot retain distinct modes at branch {i}')
        current=[]
        for k,(l,u) in found.items():
            overlap=abs(np.vdot(modes[k][1],u))/(np.linalg.norm(modes[k][1])*np.linalg.norm(u))
            residual=float(np.linalg.norm(s.characteristic(r,J,l)@u)/np.linalg.norm(u))
            current.append(dict(mode=k,lambda_per_ms=l,residual=residual,step_eigenfunction_overlap=overlap))
            if k in previous and previous[k]['lam'].real*l.real<0 and l.imag>1e-5:
                crossings.append(dict(mode=k,indices=[i,previous[k]['index']],J=[J,previous[k]['J']],
                    lambdas=[l,previous[k]['lam']],status='UNREFINED_HOPF_BRACKET'))
            previous[k]=dict(index=i,J=J,lam=l)
            np.savez_compressed(DEST/f'{a.namespace}tracked_mode{k}_branch{i:04d}.npz',lam=l,vector=u,rates=r,J=J)
        rows.append(dict(branch_index=i,J_EE_core=J,roots=current,homotopy_substeps=n))
        write(dest,dict(status='RUNNING',pid=os.getpid(),source=a.seed,rows=rows,
            crossing_brackets=crossings,namespace=a.namespace,
            scope='Distinct root tracking between exact equilibria; not a complete root inventory between sites. Interpolated corrections are solver homotopy, not physical states.'))
        print('DISTINCT MODE TRACK',i,'substeps',n,'crossings',len(crossings),flush=True)
        modes=found;r0=r;J0=J
    q=read(dest);q['status']='BATCH_COMPLETE';write(dest,q)
    print('DISTINCT CROSSINGS',crossings,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed');p.add_argument('--start',type=int,required=True)
    p.add_argument('--stop',type=int,required=True);p.add_argument('--namespace',required=True)
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--resume',action='store_true',help='Resume after the last saved exact equilibrium, preserving mode identities and brackets')
    main(p.parse_args())
