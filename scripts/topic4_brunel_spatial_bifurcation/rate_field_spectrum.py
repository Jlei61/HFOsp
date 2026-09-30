"""Temporal eigenvalues of the SAME DDE used in run_rate_field.py."""
from rate_field import *
from scipy.sparse.linalg import eigs,spsolve
from scipy.optimize import root
from functools import lru_cache
import argparse


def refine(s,r,J,lam,v,tol=1e-9):
    pivot=np.argmax(abs(v));v=v/v[pivot];constraint=sparse.csr_matrix((np.ones(1),([0],[pivot])),shape=(1,s.P))
    for it in range(24):
        m=s.characteristic(r,J,lam);res=m@v;err=np.linalg.norm(res)/np.linalg.norm(v)
        if err<tol:return lam,v/np.linalg.norm(v),float(err)
        h=1e-6;dm=(s.characteristic(r,J,lam+h)-s.characteristic(r,J,lam-h))/(2*h)
        mat=sparse.bmat([[m,sparse.csr_matrix((dm@v)[:,None])],[constraint,sparse.csr_matrix((1,1))]],format='csc')
        d=spsolve(mat,np.r_[-res,0j]);v+=d[:-1];lam+=d[-1]
        if abs(lam)>.6:return None
    return None


def mode(s,J,lam,core,rbase):
    r,ok,_=s.solve(J,rbase);assert ok
    ev,v=eigs(s.characteristic(r,J,lam),k=6,sigma=0,tol=1e-10)
    energy=s.geo['group_size'][:,None]*abs(v)**2;mask=s.E&(s.geo['group_region']==core)
    frac=energy[mask].sum(0)/energy.sum(0);ids=np.flatnonzero(frac>.1)
    k=ids[np.argmin(abs(ev[ids]))] if len(ids) else np.argmax(frac)
    return ev[k],v[:,k],r


def hopfs(s):
    r0,ok,_=s.solve(.94);assert ok
    rows=[]
    for core in [0,1]:
        def fun(x):
            J,omega=x;ev,v,r=mode(s,J,1j*omega,core,r0)
            return [ev.real,ev.imag]
        fit=root(fun,[.95,.037],tol=1e-10);J,omega=fit.x;ev,v,r=mode(s,J,1j*omega,core,r0)
        assert abs(ev)<1e-8,(fit.message,fit.x,ev)
        v/=np.linalg.norm(v);energy=s.geo['group_size']*abs(v)**2;energy/=energy.sum()
        q=dict(core='AB'[core],J_EE_core=J,frequency_hz=omega*1000/(2*np.pi),rates_hz=s.regional_rates(r),
            equilibrium_residual=float(abs(s.residual(r,J)).max()),characteristic_residual=float(np.linalg.norm(s.characteristic(r,J,1j*omega)@v)),
            regional_energy=[energy[s.E&(s.geo['group_region']==k)].sum() for k in range(3)],criticality='NOT_COMPUTED')
        np.savez_compressed(RATE_OUT/f'hopf_{"AB"[core]}.npz',rates=r,J=J,omega=omega,vector=v)
        rows.append(q);write(RATE_OUT/'hopfs.json',dict(rows=rows,model='Same autonomous spatial rate DDE as all time-domain panels'))
        print('HOPF',q,flush=True)
    return rows


def track(s):
    b=np.load(OUT/'critical_revision/branch.npz');seeds=[];rows=[]
    for core in ['A','B']:
        z=np.load(RATE_OUT/f'hopf_{core}.npz');seeds.append((complex(0,float(z['omega'])),z['vector']))
    ids=sorted(set(list(range(54,113,2))+list(range(113,len(b['J']),16))+list(b['reversal_indices'])+[len(b['J'])-1]))
    for index in ids:
        J=float(b['J'][index]);r=b['rates'][index];roots=[]
        for lam,v in seeds:
            found=refine(s,r,J,lam,v)
            if found is not None and not any(abs(found[0]-q[0])<1e-6 for q in roots):roots.append(found)
        if roots:seeds=[q[:2] for q in roots]
        row=dict(index=index,J_EE_core=J,roots=[dict(lambda_per_ms=l,residual=e) for l,v,e in roots],positive=any(l.real>1e-7 for l,v,e in roots))
        rows.append(row)
        if len(rows)%10==0:print('spectrum',index,J,[(q[0].real,q[0].imag) for q in roots],flush=True)
        write(RATE_OUT/'branch_spectrum.json',dict(rows=rows,spectrum_complete=False,meaning='Selected temporal eigenpairs of the new rate DDE. A positive root certifies instability; a negative tracked root alone does not certify stability.'))
    np.savez_compressed(RATE_OUT/'equilibrium_branch.npz',**{k:b[k] for k in b.files})


def fill_positive(s):
    b=np.load(RATE_OUT/'equilibrium_branch.npz');data=read(RATE_OUT/'branch_spectrum.json');z=np.load(RATE_OUT/'upper_positive_seed.npz')
    seed=(complex(z['lam']),z['vector']);last=seed
    for row in data['rows'][::-1]:
        if row['positive']:continue
        index=row['index'];r=b['rates'][index];J=float(b['J'][index]);found=None
        for lam,v in [last,seed]:
            q=refine(s,r,J,lam,v)
            if q is not None and q[0].real>1e-7:found=q;break
        if found is None:
            for freq in [6.,15.,30.]:
                lam=.012+2j*np.pi*freq/1000;ev,vec=eigs(s.characteristic(r,J,lam),k=3,sigma=0,tol=1e-9)
                q=refine(s,r,J,lam,vec[:,np.argmin(abs(ev))])
                if q is not None and q[0].real>1e-7:found=q;break
        if found is not None:
            lam,v,err=found;last=(lam,v);row['positive']=True
            row['roots'].append(dict(lambda_per_ms=lam,residual=err,origin='Independent positive-mode search for the same rate DDE'))
        print('fill',index,J,'positive',row['positive'],flush=True)
        write(RATE_OUT/'branch_spectrum.json',data)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',action='store_true');p.add_argument('--fill-positive',action='store_true');a=p.parse_args();s=RateField()
    if not (RATE_OUT/'hopfs.json').exists():hopfs(s)
    if a.track:track(s)
    if a.fill_positive:fill_positive(s)
