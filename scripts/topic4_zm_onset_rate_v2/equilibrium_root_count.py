"""Count right-half-plane roots of the same rate DDE by the argument principle.

The contour uses I-H(lambda)K(lambda), with no poles in Re(lambda)>=0.
An absolute-gain bound excludes roots outside the enclosing semicircle.
Adaptive midpoint determinant phases are checked by independent doubling.
"""
from model_zm import *
from scipy.sparse.linalg import eigs, splu
import argparse


def parity(p):
    seen=np.zeros(len(p),bool);cycles=0
    for i in range(len(p)):
        if seen[i]:continue
        cycles+=1
        while not seen[i]:seen[i]=True;i=p[i]
    return (len(p)-cycles)%2


def outer_bound(s,r,D,R):
    s.set_D(D);gm,ge,gi=s.gains(s.moments(r));a,b,qa,qb=s.matrices(1.)
    ha=1/np.sqrt((1+(R*s.rise[0])**2)*(1+(R*s.decay[0])**2))
    hg=1/np.sqrt((1+(R*s.rise[1])**2)*(1+(R*s.decay[1])**2))
    va=1/np.sqrt(1+(R*s.tau[0]/2)**2);vg=1/np.sqrt(1+(R*s.tau[1]/2)**2)
    H=s.alpha/np.sqrt(1+(R*s.tf)**2)+(1-s.alpha)/np.sqrt(1+(R*s.ts)**2)
    K=sparse.diags(abs(gm)*s.tm)@(s.area[0]*ha*abs(a)+sparse.diags(s.Z*s.area[1]*hg)@abs(b))
    K+=sparse.diags(abs(ge)*s.tm*s.area[0]**2*va)@abs(qa)
    K+=sparse.diags(abs(gi)*s.tm*(s.Z*s.area[1])**2*vg)@abs(qb)
    K+=sparse.diags(.5*s.E*abs(gm)/np.sqrt(1+(R*1000)**2))
    B=sparse.diags(H)@K
    return float(max(abs(eigs(B,k=1,which='LM',return_eigenvectors=False,tol=1e-10))))


def count(s,r,D,N=128,mesh='uniform'):
    R=.5
    while outer_bound(s,r,D,R)>.7:R*=2
    phases={};start=time.time()
    def detphase(lam):
        lam=complex(round(lam.real,14),round(lam.imag,14))
        if lam.imag<0:return detphase(lam.conjugate()).conjugate()
        if lam in phases:return phases[lam]
        C=sparse.diags(s.filter_response(lam))@s.characteristic(r,D,lam)
        lu=splu(C.tocsc());u=lu.U.diagonal()
        angle=np.sum(np.angle(u))+np.pi*(parity(lu.perm_r)+parity(lu.perm_c))
        phases[lam]=np.exp(1j*angle);return phases[lam]
    # Counterclockwise boundary of Re lambda > 0: down imaginary axis,
    # then lower -> upper semicircle. No shift hides near-zero positive roots.
    results=[]
    for nn in [N,2*N,4*N,8*N,16*N]:
        axis=np.linspace(1.,-1.,nn+1)
        if mesh=='cubic':axis=axis**3
        contour=np.r_[1j*R*axis,R*np.exp(1j*np.linspace(-np.pi/2,np.pi/2,nn+1))[1:]]
        values=np.array([detphase(complex(x)) for x in contour])
        changes=np.angle(np.r_[values[1:],values[:1]]/values)
        winding=float(changes.sum()/(2*np.pi))
        row=dict(N=nn,winding=winding,max_phase_step=float(abs(changes).max()),count=int(round(winding)))
        results.append(row);print('COUNT',D,row,'s',round(time.time()-start,1),flush=True)
        if len(results)>=2 and results[-1]['count']==results[-2]['count'] and row['max_phase_step']<.7:break
    ok=len(results)>=2 and results[-1]['count']==results[-2]['count'] and results[-1]['max_phase_step']<.7 and results[-1]['count']>=0
    return dict(D=D,global_E_hz=s.global_rate(r),unstable_roots=results[-1]['count'] if ok else None,
                radius_per_ms=R,outer_gain_spectral_radius=outer_bound(s,r,D,R),refinements=results,
                status='RESOLVED' if ok else 'NEEDS_REFINEMENT',mesh=mesh,seconds=time.time()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--D',type=float,nargs='+',default=[1.,.8,.6,.4]);p.add_argument('--N',type=int,default=128)
    p.add_argument('--source');p.add_argument('--label',default='source');p.add_argument('--mesh',choices=['uniform','cubic'],default='uniform')
    args=p.parse_args();s=ZMSpatialRate();r=np.full(s.P,.45);out=DEST/'periodic_completion/equilibrium_counts';out.mkdir(exist_ok=True)
    if args.source:
        z=np.load(args.source);r=z['r'];D=float(z['D']);assert max(abs(s.residual(r,D)))*1000<1e-6
        q=count(s,r,D,args.N,args.mesh);q['source']=args.source;write(out/f'{args.label}.json',q)
        np.savez_compressed(out/f'{args.label}.npz',r=r,D=D)
        sys.exit(0)
    for D in args.D:
        r,ok,_=s.solve_D(D,r);assert ok
        q=count(s,r,D,args.N,args.mesh);write(out/f'upper_D{D:.6f}.json',q);np.savez_compressed(out/f'upper_D{D:.6f}.npz',r=r,D=D)
