"""Temporal roots of the fixed Z/M characteristic, with mode continuation.

One certified root in the right half-plane proves instability. Absence of
such a root in this selected-mode search never certifies stability.
"""
from zm_model import *
from scipy.sparse.linalg import eigs,spsolve
from response_transport import white_transport
import argparse

class Linearization:
    def __init__(self,s,r,D):
        s.set_D(D);self.s=s;self.r=r;self.D=D
        self.moments=s.moments(r);self.details=s.phi(*self.moments,details=True)
        self.gains=s.gains(self.moments)
        f,lo,hi,sigma,teff=self.details
        self.dc=white_transport(1e-8+0j,lo,hi,sigma,s.tm,f).real

    def matrix(self,lam):
        s=self.s;f,lo,hi,sigma,teff=self.details
        norm=np.ones((2,s.P),complex) if abs(lam)<1e-10 else white_transport(complex(lam),lo,hi,sigma,s.tm,f)/self.dc
        gm=self.gains[0]*norm[0];ge=self.gains[1]*norm[1];gi=self.gains[2]*norm[1]
        a,b,qa,qb=s.matrices(1.,complex(lam))
        ha=1/((1+lam*s.rise[0])*(1+lam*s.decay[0]));hg=1/((1+lam*s.rise[1])*(1+lam*s.decay[1]))
        hqa=2/(2+lam*s.tau[0]);hqg=2/(2+lam*s.tau[1])
        K=sparse.diags(gm*s.tm)@(s.area[0]*ha*a-sparse.diags(s.Z*s.area[1]*hg)@b)
        K+=sparse.diags(ge*s.tm*s.area[0]**2*hqa)@qa+sparse.diags(gi*s.tm*(s.Z*s.area[1])**2*hqg)@qb
        diagonal=np.ones(s.P,complex)
        if s.mode=='dynamic_M':diagonal+=.5*s.E*gm/(1+1000*lam)
        return sparse.diags(diagonal)-K

def refine(L,lam,v=None):
    if v is None:
        ev,V=eigs(L.matrix(lam),k=3,sigma=0,tol=1e-10);v=V[:,np.argmin(abs(ev))]
    pivot=np.argmax(abs(v));v=v/v[pivot]
    c=sparse.csr_matrix((np.ones(1),([0],[pivot])),shape=(1,len(v)))
    for k in range(18):
        A=L.matrix(lam);f=A@v;error=np.linalg.norm(f)/np.linalg.norm(v)
        if error<1e-9:return lam,v/np.linalg.norm(v),float(error),k
        h=1e-6;d=(L.matrix(lam+h)-L.matrix(lam-h))/(2*h)
        B=sparse.bmat([[A,sparse.csr_matrix((d@v)[:,None])],[c,sparse.csr_matrix((1,1))]],format='csc')
        change=spsolve(B,np.r_[-f,0j]);step=1.
        for j in range(10):
            nl=lam+step*change[-1];nv=v+step*change[:-1]
            if abs(nl)<.5 and abs(nl+.001)>1e-7:
                nf=L.matrix(nl)@nv
                if np.linalg.norm(nf)/np.linalg.norm(nv)<error:lam=nl;v=nv;break
            step*=.5
        else:return None
    return None

def main(a):
    s=ZMRate(a.grid,mode=a.mode,m_current=a.m_current)
    parent=DEST/f'g{a.grid}'/a.branch;dest=parent/a.label;dest.mkdir(parents=True,exist_ok=True)
    files=sorted(parent.glob(a.pattern))[::a.stride]
    if a.max_D is not None:files=[f for f in files if float(np.load(f)['D'])<=a.max_D]
    if not files:raise RuntimeError('No equilibrium continuation points')
    lam=complex(a.real,a.imag);v=None;rows=[]
    for file in files:
        d=np.load(file);r=d['r'];D=float(d['D']);L=Linearization(s,r,D)
        if not rows:
            C=s.characteristic(r,D,lam);error=abs((C-L.matrix(lam)).data).max() if (C-L.matrix(lam)).nnz else 0.
            # Old Taylor evaluation loses precision for moderately negative
            # bounds; the transport is independently checked against 60 digits.
            assert error<1e-8,error
        found=refine(L,lam,v)
        row=dict(point=file.name,D=D,global_E_hz=s.global_rate(r))
        if found is None:
            row['status']='ROOT_NOT_RESOLVED';v=None
        else:
            lam,v,res,it=found;energy=s.sizes*abs(v)**2;energy/=energy.sum()
            row.update(status='VERIFIED_TEMPORAL_ROOT',lambda_per_ms=lam,frequency_hz=abs(lam.imag)*1000/(2*np.pi),
                characteristic_residual=res,iterations=it,
                E_mode_energy_A_B_surround=[float(energy[s.E&(s.geo['group_region']==i)].sum()) for i in range(3)],
                inhibitory_energy=float(energy[~s.E].sum()),
                equilibrium_stability='UNSTABLE' if lam.real>1e-7 else 'NOT_DETERMINED_BY_THIS_ROOT')
            np.savez_compressed(dest/file.name,r=r,D=D,lam=lam,vector=v)
        rows.append(row);write(dest/'result.json',dict(status='COMPLETE' if file==files[-1] else 'RUNNING',rows=rows,
            response='Frozen shifted-bound response; no new fit or biological parameter change',
            spectrum_complete=False,scope='Selected temporal eigenmode; a positive real part proves instability only'))
        print(row,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--branch',default='D_arclength_lower')
    p.add_argument('--label',default='temporal_core_A');p.add_argument('--stride',type=int,default=1)
    p.add_argument('--pattern',default='point*.npz')
    p.add_argument('--max-D',type=float)
    p.add_argument('--real',type=float,default=.01072);p.add_argument('--imag',type=float,default=.02846)
    p.add_argument('--mode',choices=['dynamic_M','frozen_M'],default='dynamic_M');p.add_argument('--m-current',type=float,default=0)
    main(p.parse_args())
