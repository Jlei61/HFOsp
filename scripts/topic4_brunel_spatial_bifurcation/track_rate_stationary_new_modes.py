"""Track independently extracted eigenmodes along the stored stationary arc."""
from rate_stationary_contour_modes import *


def main(a):
    s=RateField();branch=np.load(RATE_OUT/'equilibrium_branch.npz');seed=np.load(a.seed)
    ids=np.flatnonzero(seed['roots'].imag>1e-5);modes={int(k):(seed['roots'][k],seed['vectors'][k]) for k in ids}
    rows=[];previous={};crossings=[];dest=DEST/f'{a.namespace}new_modes_{a.start}_to_{a.stop}.json';started=time.time()
    step=1 if a.stop>a.start else -1
    for i in range(a.start,a.stop+step,step):
        r=branch['rates'][i];J=float(branch['J'][i]);current=[]
        for k,(lam,v) in modes.items():
            q=refine(s,r,J,lam,v)
            if q is None:
                current.append(dict(mode=k,status='REFINEMENT_FAILED'));continue
            l,u,e=q;overlap=abs(np.vdot(v,u))/(np.linalg.norm(v)*np.linalg.norm(u))
            mass=s.geo['group_size']*abs(u)**2*s.E;mass/=mass.sum()
            row=dict(mode=k,status='CONVERGED',lambda_per_ms=l,residual=e,step_eigenfunction_overlap=overlap,
                E_rate_mode_energy_by_region=[mass[s.geo['group_region']==j].sum() for j in range(3)])
            current.append(row)
            if k in previous and previous[k]['lam'].real*l.real<0 and abs(l.imag)>1e-5:
                crossings.append(dict(mode=k,indices=[i,previous[k]['index']],J=[J,previous[k]['J']],
                    lambdas=[l,previous[k]['lam']],status='UNREFINED_HOPF_BRACKET'))
            previous[k]=dict(index=i,J=J,lam=l);modes[k]=(l,u)
            np.savez_compressed(DEST/f'{a.namespace}tracked_mode{k}_branch{i:04d}.npz',lam=l,vector=u,rates=r,J=J)
        rows.append(dict(branch_index=i,J_EE_core=J,roots=current))
        write(dest,dict(status='RUNNING',pid=os.getpid(),source=a.seed,rows=rows,crossing_brackets=crossings,
            interpretation='Tracked eigenpairs, not complete intermediate spectra. Brackets require a coupled equilibrium/imaginary-root solve and transversality validation.'))
        print('NEW RATE MODE TRACK',i,J,[(q['mode'],q.get('lambda_per_ms'),q.get('step_eigenfunction_overlap')) for q in current],flush=True)
    q=read(dest);q['status']='BATCH_COMPLETE';q['seconds']=time.time()-started;write(dest,q)
    print('NEW CROSSING BRACKETS',crossings,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed');p.add_argument('--start',type=int,default=252);p.add_argument('--stop',type=int,default=230)
    p.add_argument('--namespace',default='',help='Distinct mode ordering/source namespace for independent contour seeds')
    main(p.parse_args())
