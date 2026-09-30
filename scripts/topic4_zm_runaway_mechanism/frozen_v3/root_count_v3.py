"""Right-half-plane characteristic root count of the v3 equilibrium linearisation (argument principle).

Contour: imaginary axis from +iR to -iR, then the right semicircle of radius R (counter-clockwise
boundary of Re lambda > 0). R is enlarged until an absolute-gain bound excludes roots outside.
det M(lambda) is evaluated via sparse LU (phase of U diagonal + permutation parities). The count is
accepted only when two successive contour refinements agree and the largest phase step is < 0.7 rad.
Zero-frequency determinant sign: negative => at least one positive real root (det -> +1 at +infinity).
"""
from dynamics_v3 import *
from scipy.sparse.linalg import splu,eigs
import argparse

def parity(p):
    seen=np.zeros(len(p),bool);cycles=0
    for i in range(len(p)):
        if seen[i]:continue
        cycles+=1
        while not seen[i]:seen[i]=True;i=p[i]
    return (len(p)-cycles)%2

def outer_bound(s,r,R,dynamic_z):
    mu,ve,vi=s.moments(r);g=s.phi(mu,ve,vi);a,b,qa,qb=s.matrices()
    ha=1/np.sqrt((1+(R*s.rise[0])**2)*(1+(R*s.decay[0])**2));hg=1/np.sqrt((1+(R*s.rise[1])**2)*(1+(R*s.decay[1])**2))
    va=1/np.sqrt(1+(R*s.tau[0]/2)**2);vg=1/np.sqrt(1+(R*s.tau[1]/2)**2)
    w,_=s.resp.weights(s,mu,ve,vi);al,aE,aI,eE,eI=w;tf,ts,tE,tI,tvE,tvI=s.poles
    Hm=al+(1-al)/np.sqrt(1+(R*ts)**2);HE=abs(aE)+abs(1-aE)/np.sqrt(1+(R*tvE)**2);HI=abs(aI)+abs(1-aI)/np.sqrt(1+(R*tvI)**2)
    gm=abs(g['d_mu'])*Hm;gE=abs(g['d_ve'])*HE+abs(g['d_mu'])*abs(eE);gI=abs(g['d_vi'])*HI+abs(g['d_mu'])*abs(eI)
    K=sparse.diags(gm*s.tm)@(s.area[0]*ha*abs(a)+sparse.diags(s.Z*s.area[1]*hg)@abs(b))
    K=K+sparse.diags(gE*s.tm*s.area[0]**2*va)@abs(qa)+sparse.diags(gI*s.Z**2*s.tm*s.area[1]**2*vg)@abs(qb)
    K=K+sparse.diags(.5*s.E*gm/np.sqrt(1+(R*TAU_M)**2))
    if dynamic_z:
        ig=s.tm*s.area[1]*(b@r);vgv=s.tm*s.area[1]**2*(qb@r);sd=np.sqrt(np.maximum(s.tm*vgv/(2*s.tau[1]),1e-20));u=(THRESHOLD_Z-ig)/sd;pdf=norm.pdf(u)
        hz=s.E/np.sqrt(1+(R*TAU_Z)**2);Zop=sparse.diags(hz*abs(pdf/sd)*s.tm*s.area[1]*hg)@abs(b)+sparse.diags(hz*abs(pdf*u/(2*np.maximum(vgv,1e-20)))*s.tm*s.area[1]**2*vg)@abs(qb)
        K=K+sparse.diags(gm*ig+gI*2*s.Z*vgv)@Zop
    return float(max(abs(eigs(K,k=1,which='LM',return_eigenvectors=False,tol=1e-10))))

def det_sign_zero(s,r,dynamic_z=False):
    C=s.characteristic(r,0.,dynamic_z);lu=splu(C.tocsc());d=lu.U.diagonal()
    return float(np.prod(np.sign(d.real))*(-1)**(parity(lu.perm_r)+parity(lu.perm_c)))

def count(s,r,N=128,dynamic_z=False,verbose=True):
    R=.5
    while outer_bound(s,r,R,dynamic_z)>.7:R*=2
    phases={};start=time.time()
    def detphase(lam):
        lam=complex(round(lam.real,14),round(lam.imag,14))
        if lam.imag<0:return detphase(lam.conjugate()).conjugate()
        if lam in phases:return phases[lam]
        lu=splu(s.characteristic(r,lam,dynamic_z));u=lu.U.diagonal()
        angle=np.sum(np.angle(u))+np.pi*(parity(lu.perm_r)+parity(lu.perm_c));phases[lam]=np.exp(1j*angle);return phases[lam]
    results=[]
    for nn in [N,2*N,4*N,8*N,16*N,32*N]:
        axis=np.linspace(1.,-1.,nn+1)
        contour=np.r_[1j*R*axis,R*np.exp(1j*np.linspace(-np.pi/2,np.pi/2,nn+1))[1:]]
        values=np.array([detphase(complex(x)) for x in contour]);changes=np.angle(np.r_[values[1:],values[:1]]/values)
        winding=float(changes.sum()/(2*np.pi));row=dict(N=nn,winding=winding,max_phase_step=float(abs(changes).max()),count=int(round(winding)))
        results.append(row)
        if verbose:log('COUNT',row,'s',round(time.time()-start,1))
        if len(results)>=2 and results[-1]['count']==results[-2]['count'] and row['max_phase_step']<.7:break
    ok=len(results)>=2 and results[-1]['count']==results[-2]['count'] and results[-1]['max_phase_step']<.7 and results[-1]['count']>=0
    return dict(unstable_roots=results[-1]['count'] if ok else None,radius_per_ms=R,refinements=results,status='RESOLVED' if ok else 'NEEDS_REFINEMENT',
                det_zero_sign=det_sign_zero(s,r,dynamic_z),dynamic_z=dynamic_z,seconds=time.time()-start)

def refine_root(s,r,lam,v=None,dynamic_z=False,tol=1e-9):
    """Newton refinement of a characteristic root near lam (bordered system)."""
    if v is None:
        ev,vec=eigs(s.characteristic(r,lam,dynamic_z),k=4,sigma=0,tol=1e-10);v=vec[:,np.argmin(abs(ev))]
    pivot=np.argmax(abs(v));v=v/v[pivot];c=sparse.csr_matrix((np.ones(1),([0],[pivot])),shape=(1,s.P))
    for it in range(30):
        A=s.characteristic(r,lam,dynamic_z);f=A@v;err=np.linalg.norm(f)/np.linalg.norm(v)
        if err<tol:return lam,v/np.linalg.norm(v),float(err)
        h=1e-6;d=(s.characteristic(r,lam+h,dynamic_z)-s.characteristic(r,lam-h,dynamic_z))/(2*h)
        B=sparse.bmat([[A,sparse.csr_matrix((d@v)[:,None])],[c,sparse.csr_matrix((1,1))]],format='csc')
        ch=spsolve(B,np.r_[-f,0j])
        for step in 2.**-np.arange(12):
            nl=lam+step*ch[-1];nv=v+step*ch[:-1]
            if np.linalg.norm(s.characteristic(r,nl,dynamic_z)@nv)/np.linalg.norm(nv)<err:lam,v=nl,nv;break
        else:return None
    return None

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('points',nargs='+');p.add_argument('--dynamic-z',action='store_true');p.add_argument('--out',default='equilibrium_stability');a=p.parse_args()
    resp=ResponseParams(DEST/'response_closure/closure.json') if (DEST/'response_closure/closure.json').exists() else ResponseParams()
    s=DynamicModel(resp=resp,quiet=True);out=DEST/a.out;out.mkdir(exist_ok=True,parents=True)
    for path in a.points:
        z=np.load(path);r=z['r'];s.set_D(float(z['D']));assert abs(s.residual(r)).max()<1e-8
        q=count(s,r,dynamic_z=a.dynamic_z);q.update(source=path,D=float(z['D']),global_E_hz=s.global_rate(r),response=getattr(resp,'source','PLACEHOLDER'))
        write(out/(Path(path).stem+('_dynZ' if a.dynamic_z else '')+'.json'),q);log(path,q['status'],q['unstable_roots'],'det0 sign',q['det_zero_sign'])
