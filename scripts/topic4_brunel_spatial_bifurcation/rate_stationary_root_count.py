"""Numerical argument-principle count for the actual autonomous rate DDE.

The full 935-group characteristic determinant is normalized to I-L(lambda).
Positive filter constants and nonnegative delays give an analytic row-norm
bound outside a finite right-half-plane box. This removes the old unsupported
inference from one high-frequency sample. Contour sampling is still numerical,
not an interval-arithmetic certificate or an inventory between sampled states.
"""
from rate_field import *
from nyquist import parity
from scipy.sparse.linalg import splu
import argparse

DEST=RATE_OUT/'periodic_completion/stationary_root_counts'


class Characteristic:
    def __init__(self,s,r,J):
        assert J>=0 and max(abs(s.residual(r,J)))<1e-8
        assert all(np.all(raw[3].data>=0) for raw in s.raw)
        self.s=s;self.r=r;self.J=J;self.gm,self.ge,self.gi=s.gains(s.moments(r,J))
        self.rows=[np.asarray(abs(m).sum(1)).ravel() for m in s.matrices(J)]

    def matrix(self,lam):
        s=self.s;a,b,qa,qb=s.matrices(self.J,lam)
        ha=1/((1+lam*s.rise[0])*(1+lam*s.decay[0]))
        hg=1/((1+lam*s.rise[1])*(1+lam*s.decay[1]))
        K=sparse.diags(self.gm*s.tm)@(s.area[0]*ha*a-sparse.diags(s.Z*s.area[1]*hg)@b)
        K+=sparse.diags(self.ge*s.tm*s.area[0]**2/(1+lam*s.tau[0]/2))@qa
        K+=sparse.diags(self.gi*s.tm*(s.Z*s.area[1])**2/(1+lam*s.tau[1]/2))@qb
        K-=sparse.diags(.5*s.E*self.gm/(1+1000*lam))
        return sparse.eye(s.P,format='csc')-sparse.diags(s.filter_response(lam))@K

    def bound(self,x=0.,y=0.):
        """Valid for Re(lambda)>=x and |Im(lambda)|>=y, x,y>=0."""
        assert x>=0 and y>=0
        s=self.s;a,b,qa,qb=self.rows
        def h(t):return 1/np.sqrt((1+x*t)**2+(y*t)**2)
        H=s.alpha*h(s.tf)+(1-s.alpha)*h(s.ts)
        K=abs(self.gm)*s.tm*(s.area[0]*h(s.rise[0])*h(s.decay[0])*a+
            s.Z*s.area[1]*h(s.rise[1])*h(s.decay[1])*b)
        K+=abs(self.ge)*s.tm*s.area[0]**2*h(s.tau[0]/2)*qa
        K+=abs(self.gi)*s.tm*(s.Z*s.area[1])**2*h(s.tau[1]/2)*qb
        K+=.5*s.E*abs(self.gm)*h(1000.)
        return float(max(H*K))


def count(s,r,J,offset=1e-8):
    c=Characteristic(s,r,J);R=W=.1
    while c.bound(x=R)>.4:R*=1.5
    while c.bound(y=W)>.4:W*=1.5
    # Traverse the upper half of a counterclockwise rectangle, from R to
    # R+iW to offset+iW to offset. Its conjugate reflection closes the loop;
    # the total winding is the half-path phase change divided by pi.
    freq=np.unique(np.r_[0.,np.geomspace(1e-10,.002,45),np.linspace(.002,.12,237),np.geomspace(.12,W,65)])
    nodes=np.r_[R+1j*np.linspace(0,W,25),np.linspace(R,offset,25)[1:]+1j*W,
        offset+1j*freq[::-1][1:]]
    cache={};traces=[];started=time.time();last=None;stable=0
    def evaluate(z):
        z=complex(z)
        if z not in cache:
            lu=splu(c.matrix(z).tocsc());d=lu.U.diagonal()
            p=np.angle(np.exp(1j*(np.angle(d).sum()+np.pi*(parity(lu.perm_r)+parity(lu.perm_c)))))
            cache[z]=(float(p),float(np.log(abs(d)).sum()))
        return cache[z]
    for level in range(12):
        for z in nodes:evaluate(z)
        vals=np.array([evaluate(z) for z in nodes]);phase=np.unwrap(vals[:,0]);logmag=vals[:,1]
        winding=(phase[-1]-phase[0])/np.pi;n=int(round(winding))
        jump=abs(np.diff(phase));logs=abs(np.diff(logmag))
        stable=stable+1 if n==last else 0;last=n
        traces.append(dict(refinement=level,nodes=len(nodes),count=n,raw_winding=winding,
            maximum_phase_increment=float(max(jump)),maximum_log_magnitude_increment=float(max(logs))))
        print('RATE ROOT COUNT',J,offset,traces[-1],flush=True)
        if level>=2 and stable>=2 and max(jump)<.2 and max(logs)<1.:
            break
        ids=np.arange(len(nodes)-1) if level<2 else np.flatnonzero((jump>=.15)|(logs>=.8))
        if not len(ids):ids=np.arange(len(nodes)-1)
        chosen=set(ids.tolist());new=[]
        for k,z in enumerate(nodes[:-1]):
            new.append(z)
            if k in chosen:new.append((z+nodes[k+1])/2)
        nodes=np.r_[new,nodes[-1]]
    valid=level<11 and stable>=2 and max(jump)<.2 and max(logs)<1. and abs(winding-n)<1e-6
    return dict(status='NUMERICALLY_RESOLVED' if valid else 'UNRESOLVED',J_EE_core=J,
        positive_root_count=n if valid else None,real_axis_offset_per_ms=offset,
        box_right_per_ms=R,box_imaginary_per_ms=W,
        right_half_plane_tail_row_bound=c.bound(x=R),high_frequency_tail_row_bound=c.bound(y=W),
        equilibrium_residual=float(max(abs(s.residual(r,J)))),
        count_domain='Re(lambda)>offset; all nine local states and physical delay distribution represented by Schur elimination. Eliminated filters have no right-half-plane poles.',
        tail_bound='Triangle inequality, nonnegative physical delays, nonnegative edge magnitudes, positive filter constants; row norm bound <1 excludes roots outside the box throughout Re(lambda)>=0.',
        contour_nodes=nodes,unwrapped_phase=phase,log_absolute_determinant=logmag,refinements=traces,
        elapsed_seconds=time.time()-started,
        limitation='Numerical contour resolution, not a rigorous enclosure. Does not exclude roots in 0<=Re(lambda)<=offset or crossings between parameter samples.')


def checks():
    s=RateField();r,ok,_=s.solve(.942);assert ok;c=Characteristic(s,r,.942)
    errors=[];bounds=[]
    for lam in [.02+.03j,.1+.2j,.01+1.j,1+.2j]:
        a=c.matrix(lam);b=sparse.diags(s.filter_response(lam))@s.characteristic(r,.942,lam)
        errors.append(float(max(abs((a-b).data))))
        actual=float(np.asarray(abs(sparse.eye(s.P)-a).sum(1)).max())
        bound=c.bound(x=lam.real,y=abs(lam.imag));assert actual<=bound*(1+1e-12)
        bounds.append(dict(lambda_per_ms=lam,actual_row_norm=actual,bound=bound))
    rng=np.random.default_rng(39871);deterrors=[]
    for _ in range(6):
        a=rng.normal(size=(9,9))+1j*rng.normal(size=(9,9));lu=splu(sparse.csc_matrix(a))
        sign=np.exp(1j*(np.angle(lu.U.diagonal()).sum()+np.pi*(parity(lu.perm_r)+parity(lu.perm_c))))
        exact=np.linalg.slogdet(a)[0];deterrors.append(abs(sign-exact))
    assert max(errors)<1e-12 and max(deterrors)<1e-12
    q=dict(status='PASS',normalized_matrix_max_errors=errors,analytic_bound_checks=bounds,
        LU_determinant_phase_errors=deterrors,scope='Operator identity and bound implementation, not contour completeness.')
    write(DEST/'operator_checks.json',q);print('ROOT COUNT CHECKS',q,flush=True)


def main(a):
    DEST.mkdir(exist_ok=True,parents=True)
    if a.check:checks();return
    s=RateField();branch=np.load(RATE_OUT/'equilibrium_branch.npz');tasks=[]
    for J in a.values:
        r,ok,_=s.solve(J);assert ok;tasks.append((f'J{J:.8f}',J,r,None))
    for i in a.indices:
        tasks.append((f'branch{i:04d}',float(branch['J'][i]),branch['rates'][i],i))
    for tag,J,r,index in tasks:
        for offset in a.offsets:
            dest=DEST/f'{tag}_offset{offset:g}.json'
            if dest.exists() and read(dest)['status']=='NUMERICALLY_RESOLVED':continue
            row=count(s,r,J,offset);row['branch_index']=index;write(dest,row)
            print('SAVED ROOT COUNT',tag,row['status'],row['positive_root_count'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--values',type=float,nargs='*',default=[]);p.add_argument('--indices',type=int,nargs='*',default=[])
    p.add_argument('--offsets',type=float,nargs='+',default=[1e-8,1e-6]);main(p.parse_args())
