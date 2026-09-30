"""Dealiased Fourier-Galerkin check of the same continuous spatial DDE.

Odd numbers of retained samples give complete sine/cosine pairs. Nonlinear
responses are evaluated on a finer temporal mesh and projected back onto the
retained harmonics, reducing spurious phase locking to collocation points.
No network parameter, response closure or slow-variable equation is changed.
"""
from native_path import *
from periodic_v3 import PeriodicV3,RS
from native_cycles import save
from scipy.signal import resample
import argparse


class Galerkin(PeriodicV3):
    def __init__(self,s,N,M,device=1):
        assert N%2==1 and M>=2*N
        super().__init__(s,N,device);self.M=M

    def interpolate(self,x,num,axis=0):
        cp=self.cp;old=x.shape[axis];f=cp.fft.rfft(x,axis=axis)
        shape=list(f.shape);shape[axis]=num//2+1
        g=cp.zeros(shape,dtype=f.dtype);sl=[slice(None)]*x.ndim;sl[axis]=slice(0,min(f.shape[axis],g.shape[axis]))
        g[tuple(sl)]=f[tuple(sl)]
        return cp.fft.irfft(g,n=num,axis=axis)*(num/old)

    def inputs(self,r,T,Z):
        return self.interpolate(super().inputs(r,T,Z),self.M,axis=1)

    def phi(self,inp):
        cp=self.cp;n=self.M*self.s.P;out=cp.empty((25,n));args=[cp.ascontiguousarray(x.ravel()) for x in inp]
        self.phik(((n+127)//128,),(128,),(*args,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,out,np.int32(n)))
        return out.reshape(25,self.M,self.s.P)

    def residual(self,r,T,Z,details=False):
        inp=self.inputs(r,T,Z);ph=self.phi(inp);F=(r-self.interpolate(ph[0],self.N))/RS
        return (F,inp,ph) if details else F

    def evaluate(self,y,reference,phase,D,amplitude=None,arc=None,derivative=False):
        from cupyx.scipy.sparse.linalg import LinearOperator
        cp=self.cp;s=self.s;n=self.N*s.P;extra=2 if amplitude is not None or arc is not None else 1
        r=y[:n].reshape(self.N,s.P)*RS;T=float(cp.exp(y[-extra]));Dv=float(y[-1]*1e-3) if extra==2 else D
        s.set_D(Dv);Z=s.Z.copy();F,inp,ph=self.residual(r,T,Z,True)
        constraints=[cp.sum((r-reference)*phase)/RS]
        if amplitude is not None:
            q,target=amplitude;projection=cp.vdot(q,cp.fft.rfft(r,axis=0)[1]/self.N)/cp.vdot(q,q);constraints.append(projection.real-target)
        if arc is not None:
            yp,tan,weight=arc;constraints.append(cp.sum((y-yp)*tan*weight**2))
        res=cp.concatenate([F.ravel(),cp.asarray(constraints)])
        if not derivative:return res
        h=1e-6;colT=(self.residual(r,T*np.exp(h),Z)-self.residual(r,T*np.exp(-h),Z)).ravel()/(2*h)
        colD=None
        if extra==2:
            hD=1e-6;s.set_D(Dv+hD);Zp=s.Z.copy();s.set_D(Dv-hD);Zm=s.Z.copy();s.set_D(Dv)
            colD=(self.residual(r,T,Zp)-self.residual(r,T,Zm)).ravel()/(2*hD)*1e-3
        pmu,pvE,pvI=ph[1],ph[2],ph[3];w=ph[5:10];gr=ph[10:25].reshape(5,3,self.M,s.P)
        mu,vE,vI,mus,vEf,vIf,vEv,vIv=inp
        def matvec(dy):
            dr=dy[:n].reshape(self.N,s.P)*RS;di=self.inputs(dr,T,Z)
            for i in [0,3]:di[i]-=self.gp[3]
            for i in [1,4,6]:di[i]-=self.gp[4]
            dmu,dvE,dvI,dmus,dvEf,dvIf,dvEv,dvIv=di;dm=cp.stack([dmu,dvE,dvI]);dw=cp.einsum('pcnj,cnj->pnj',gr,dm)
            dmueff=w[0]*dmu+(1-w[0])*dmus+(mu-mus)*dw[0]+w[3]*(dvE-dvEf)+(vE-vEf)*dw[3]+w[4]*(dvI-dvIf)+(vI-vIf)*dw[4]
            dvEe=w[1]*dvE+(1-w[1])*dvEv+(vE-vEv)*dw[1];dvIe=w[2]*dvI+(1-w[2])*dvIv+(vI-vIv)*dw[2]
            dphi=self.interpolate(pmu*dmueff+pvE*dvEe+pvI*dvIe,self.N)
            out=((dr-dphi)/RS).ravel()+colT*dy[-extra]
            if extra==2:out+=colD*dy[-1]
            cc=[cp.sum(dr*phase)/RS]
            if amplitude is not None:cc.append((cp.vdot(q,cp.fft.rfft(dr,axis=0)[1]/self.N)/cp.vdot(q,q)).real)
            if arc is not None:cc.append(cp.sum(dy*tan*weight**2))
            return cp.concatenate([out,cp.asarray(cc)])
        return res,LinearOperator((len(y),len(y)),matvec=matvec,dtype=np.float64),dict(inputs=inp,phi=ph)


def main(a):
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    if a.stream:
        from streaming_periodic import StreamGalerkin
        o=StreamGalerkin(s,a.N,a.M,a.device)
    else:o=Galerkin(s,a.N,a.M,a.device)
    if a.no_operator_cache:o.cache_mean_operators=False
    cp=o.cp;z=np.load(a.orbit)
    r=resample(z['r'],a.N,axis=0);T=float(z['T']);D=float(z['D']) if a.D is None else a.D
    if a.check:
        rr=cp.asarray(r);der=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(rr,axis=0),n=o.N,axis=0)
        ph=der/cp.sum(der*der)*RS;y=cp.r_[(rr/RS).ravel(),cp.asarray([np.log(T)])]
        f,A,_=o.evaluate(y,rr,ph,D,derivative=True)
        rng=np.random.default_rng(447);v=cp.asarray(rng.normal(size=len(y)));v[-1]*=.001
        rows=[]
        for h in [1e-4,3e-5,1e-5]:
            fd=(o.evaluate(y+h*v,rr,ph,D)-o.evaluate(y-h*v,rr,ph,D))/(2*h);av=A@v
            rows.append(dict(h=h,relative=float(cp.linalg.norm(fd-av)/cp.linalg.norm(fd))))
        write(OUT/f'galerkin_jvp_{a.N}_{a.M}.json',rows);log('JVP',rows)
        assert min(x['relative'] for x in rows)<1e-5
    sol=o.solve(r,T,D,maxiter=20,tol=2e-8,restart=40)
    q=save(s,sol,OUT/'periodic',a.label)
    rr=cp.asarray(sol['r']);s.set_D(sol['D']);inp=o.inputs(rr,sol['T'],s.Z);ph=o.phi(inp)
    defect=(o.interpolate(rr,a.M)-ph[0])*1000
    q.update(method='dealiased Fourier-Galerkin',nonlinear_samples=a.M,
        offgrid_rms_hz=float(cp.sqrt(cp.mean(defect**2))),offgrid_max_hz=float(cp.max(cp.abs(defect))))
    write(OUT/'periodic'/f'{a.label}.json',q);log('GALERKIN',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--N',type=int,default=513)
    p.add_argument('--M',type=int,default=2048);p.add_argument('--device',type=int,default=1)
    p.add_argument('--D',type=float);p.add_argument('--family',choices=['native','rate'],default='native')
    p.add_argument('--label',required=True);p.add_argument('--check',action='store_true')
    p.add_argument('--stream',action='store_true')
    p.add_argument('--no-operator-cache',action='store_true');main(p.parse_args())
