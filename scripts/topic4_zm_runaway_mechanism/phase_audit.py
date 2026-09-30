"""Audit Floquet phase invariance and delayed-rate time stamps.

The continuous DDE requires rate history evaluated at its labelled endpoint.
The historical solver stored the average of the two Heun stage rates at that
endpoint, introducing a half-step lag. Both schemes must converge as dt -> 0.
This file compares them without changing the frozen model or legacy results.
"""
from native_path import *
from periodic_v3 import PeriodicV3
from floquet_v3 import Monodromy, orbit_states, NS
import argparse,gc


class EndpointMonodromy(Monodromy):
    phase_method='chain rule for the rate history'

    def phase_vector(self,full):
        # A DDE history of rates must be differentiated through r=Phi(Y).
        # Differentiating a truncated Fourier interpolant of r amplifies its
        # small high-frequency truncation error, especially during sharp bursts.
        # The state derivative retains the spectral derivative of the LTI states.
        cp=self.cp;n=self.n;Y=full[:-1]
        lam=2j*np.pi*np.fft.rfftfreq(n,d=self.T/n)
        dY=np.fft.irfft(np.fft.rfft(Y,axis=0)*lam[:,None,None],n=n,axis=0)
        dY[:,11]=0.;hist=cp.empty((self.Dd,self.s.P));work=cp.empty_like(self.y)
        for j in range(1,self.Dd+1):
            idx=(-j)%n
            self.k['tangent_rhs'](((self.s.P+127)//128,),(128,),
                (self.orbit[idx],cp.asarray(dY[idx]),self.arr,self.pars,self.consts,
                 self.SE,self.SI,self.WE,self.WI,work,hist[j-1]))
        return np.r_[dY[0].ravel(),hist.get().ravel()]

    def step(self,i):
        super().step(i)
        p=self.s.P;n=(p+127)//128
        # The updated state is at t_(i+1); its instantaneous rate is the history value.
        self.k['tangent_rhs']((n,),(128,),(self.orbit[i+1],self.y,self.arr,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.f2,self.dr2))
        self.cp.copyto(self.hist[(i+1)%self.depth],self.dr2)

    def matvec(self,x):
        cp=self.cp;p=self.s.P
        with self.stream:
            x=cp.asarray(x);self.y[:]=x[:NS*p].reshape(NS,p);self.hist[self.order]=x[NS*p:].reshape(self.Dd,p)
            if not self.dynamic_z:self.y[11]=0.
            self.k['tangent_rhs'](((p+127)//128,),(128,),
                (self.orbit[0],self.y,self.arr,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.f,self.dr))
            cp.copyto(self.hist[0],self.dr)
            self.graph.launch(stream=self.stream)
            result=cp.concatenate([self.y.ravel(),self.hist[self.finalorder].ravel()])
        self.stream.synchronize();self.calls+=1
        if self.calls%10==0:log('ENDPOINT MONODROMY',self.calls,'sec',round(time.time()-self.start,1))
        return result.get()


def main(a):
    s=model();attach_native_path(s);z=np.load(a.orbit);sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    if a.stream:
        from streaming_periodic import StreamPeriodic
        o=StreamPeriodic(s,len(sol['r']),a.device)
        o.cache_mean_operators=False
    else:o=PeriodicV3(s,len(sol['r']),a.device)
    out=OUT/'phase_audit';out.mkdir(parents=True,exist_ok=True);rows=[]
    for scheme in a.scheme:
        for dt in a.dt:
            cls=EndpointMonodromy if scheme=='endpoint' else Monodromy
            if a.chunk and scheme=='endpoint':
                from chunk_monodromy import ChunkEndpointMonodromy
                cls=ChunkEndpointMonodromy
            m=cls(s,o,sol,dtmax=dt,device=a.device);full,_=orbit_states(o,sol,m.n)
            vectors={}
            if a.phase_method in ['spectral','both']:vectors['spectral']=Monodromy.phase_vector(m,full)
            if a.phase_method in ['chain','both']:vectors['chain']=EndpointMonodromy.phase_vector(m,full)
            del full;o.cache_key=None;o.cache=None;o.cp.get_default_memory_pool().free_all_blocks()
            for phase_method,ph in vectors.items():
                mp=m.matvec(ph)
                row=dict(scheme=scheme,phase_method=phase_method,dt=m.dt,N=len(sol['r']),T=sol['T'],D=sol['D'],
                         phase_defect=float(np.linalg.norm(mp-ph)/np.linalg.norm(ph)),
                         phase_projection=float(mp@ph/(ph@ph)),orbit=a.orbit)
                log('PHASE',row);rows.append(row);write(out/f'{Path(a.orbit).stem}_{a.phase_method}_dt{a.dt}.json',rows)
            del m,ph,mp;gc.collect();o.cp.get_default_memory_pool().free_all_blocks()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--dt',type=float,nargs='+',default=[.1,.05,.025])
    p.add_argument('--scheme',nargs='+',default=['historical','endpoint']);p.add_argument('--device',type=int,default=1)
    p.add_argument('--stream',action='store_true')
    p.add_argument('--chunk',action='store_true')
    p.add_argument('--phase-method',choices=['spectral','chain','both'],default='chain')
    main(p.parse_args())
