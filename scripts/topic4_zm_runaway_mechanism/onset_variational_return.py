"""Monodromy diagnostics for a current onset-side periodic shooting seed.

Uses the actual GPU variational equations, including dynamic M and complete
rate history. An approximate return and a finite Krylov probe do not certify
a bifurcation or replace time-step and eigen-residual checks.
"""
from common import np,read,write,log
from onset_state_continuation import DEST,build
from fine_rate_frozen_Z_fields import capture,restore
from onset_tangent_cuda import Tangent
from datetime import datetime
import argparse,os,time


class Coordinates:
    def __init__(self,base,s):
        self.P=s.P;self.depth=len(base['history']);self.tick=int(base['clock'][0])
        self.weight=np.sqrt(s.sizes/s.sizes.sum())[None,:]
        a=self.raw(base)
        rms=np.sqrt(((a*a)*(self.weight**2)).sum(1))
        floors=np.r_[np.full(4,1.),.001,np.full(6,1.),np.full(36,.001),np.full(self.depth,1e-5)]
        self.floor=floors[:,None]
        self.scale=np.maximum(rms,floors)[:,None]
        self.size=a.size;self.E=s.E

    def raw(self,state):
        tick=int(state['clock'][0]);h=state['history']
        return np.concatenate([state['syn'][:5],state['local'],h[(tick-np.arange(len(h)))%len(h)]])

    def pack(self,state):return (self.raw(state)*self.weight/self.scale).ravel()

    def tangent(self,t):
        tick=int(t.e.local.clock.get()[0]);h=t.history.get()
        a=np.concatenate([t.syn.get(),t.local.get(),h[(tick-np.arange(len(h)))%len(h)]])
        return (a*self.weight/self.scale).ravel()

    def set_tangent(self,t,v):
        a=v.reshape(-1,self.P)*self.scale/self.weight
        a=a.copy();a[4,~self.E]=0
        h=np.empty((self.depth,self.P));h[(self.tick-np.arange(self.depth))%self.depth]=a[47:]
        t.reset();t.syn[:]=t.e.cp.asarray(a[:5]);t.local[:]=t.e.cp.asarray(a[5:47]);t.history[:]=t.e.cp.asarray(h)
        t.e.cp.cuda.get_current_stream().synchronize()


class Return:
    def __init__(self,label,device):
        self.e=e=build(device,dt=read(DEST/label/'jobs.json')['condition'].get('dt_ms',.05))
        self.base={k:v for k,v in np.load(DEST/label/'final_state.npz').items()}
        restore(e,self.base);self.t=t=Tangent(e);t.graph()
        self.c=Coordinates(self.base,e.s)
        ret=read(DEST/label/'period_return/result.json');self.T=ret['interpolated_period_ms']
        self.steps=int(np.floor(self.T/e.dt));self.alpha=self.T/e.dt-self.steps
        self.nchunk=self.steps//round(10/e.dt);remainder=self.steps%round(10/e.dt)
        with t.stream:
            t.stream.begin_capture()
            for _ in range(remainder):t.step()
            self.tail=t.stream.end_capture()
        restore(e,self.base)
        x=[self.c.pack(self.base)]
        for _ in range(2):
            e.step();e.cp.cuda.get_current_stream().synchronize();x.append(self.c.pack(capture(e)))
        self.phase=(-3*x[0]+4*x[1]-x[2])/(2*e.dt)
        self.phase/=np.linalg.norm(self.phase)
        self.calls=0

    def __call__(self,v):
        e=self.e;t=self.t;c=self.c
        restore(e,self.base);c.set_tangent(t,v)
        for _ in range(self.nchunk):t.chunk()
        self.tail.launch(t.stream);t.stream.synchronize()
        y0=c.tangent(t)
        t.step();e.cp.cuda.get_current_stream().synchronize();y1=c.tangent(t)
        self.calls+=1
        return (1-self.alpha)*y0+self.alpha*y1


def probe(label,device,krylov):
    qa=DEST/'tangent_implementation'
    assert read(qa/'result.json')['status']=='PASS'
    assert read(qa/'full_state_check.json')['status']=='PASS'
    folder=DEST/label/'variational_return';folder.mkdir(exist_ok=True)
    assert not (folder/'result.json').exists()
    write(folder/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        method='Variationalreturnofsameconditioned39+transientcorrectedphysical-delayconditionaldrift. FixedfullZ,Mdynamic. Linearinterpolationat the measuredperiod; allsynaptic/covariance/memory/delaystates included.',
        phase_gate='Relative ||Aphase-phase||<1e-3 before any transverseKrylovdiagnostic. This approximate-orbit gate does not replace refinedorbit/stepchecks.',
        transverse='Project the autonomous phase direction, then at most the specified Arnoldidimension. Ritzvalues andArnoldiresidualestimates are candidatesonly; no stablelabel orcriticalsymbol from a finite unrefined probe.',
        krylov_dimension=krylov,budget=f'Onephasevariationalreturn andatmost{krylov}transversereturns; no adaptiverepeat inthisprobe.',
        model_promoted=False))
    write(folder/'jobs.json',dict(status='RUNNING',pid=os.getpid(),stage='setup',completed_returns=0))
    start=time.time();A=Return(label,device);phase=A.phase;y=A(phase)
    projection=float(phase@y);defect=float(np.linalg.norm(y-phase))
    gate=dict(period_ms=A.T,dt_ms=A.e.dt,phase_projection=projection,phase_relative_defect=defect,
              phase_gate_passed=bool(defect<1e-3))
    write(folder/'phase_check.json',gate);log('ONSET PHASE RETURN',label,gate)
    if not gate['phase_gate_passed']:
        write(folder/'result.json',dict(status='PHASE_GATE_FAILED_NO_SPECTRUM',**gate,model_promoted=False))
        write(folder/'jobs.json',dict(status='COMPLETE_NEGATIVE',pid=os.getpid(),completed_returns=A.calls))
        return
    if krylov:
        rng=np.random.default_rng(92318);v=rng.normal(size=A.c.size)
        v.reshape(-1,A.c.P)[4,~A.e.s.E]=0
        v-=phase*(phase@v);v/=np.linalg.norm(v)
        vectors=[v];H=np.zeros((krylov+1,krylov));rows=[]
        for k in range(krylov):
            v=A(vectors[k]);v-=phase*(phase@v)
            # Two MGS passes control orthogonality in the full delay space.
            for _ in range(2):
                for j in range(k+1):
                    h=float(vectors[j]@v);H[j,k]+=h;v-=h*vectors[j]
            H[k+1,k]=np.linalg.norm(v)
            values,small=np.linalg.eig(H[:k+1,:k+1])
            order=np.argsort(-abs(values))[:min(6,len(values))]
            candidates=[dict(real=float(values[j].real),imag=float(values[j].imag),
                modulus=float(abs(values[j])),arnoldi_residual_estimate=float(abs(H[k+1,k]*small[-1,j]))) for j in order]
            row=dict(dimension=k+1,candidates=candidates);rows.append(row)
            write(folder/'krylov_progress.json',rows);log('ONSET RETURN RITZ',label,row)
            write(folder/'jobs.json',dict(status='RUNNING',pid=os.getpid(),stage='Arnoldi',completed_returns=A.calls))
            if H[k+1,k]<1e-14:break
            if k+1<krylov:vectors.append(v/H[k+1,k])
        np.savez_compressed(folder/'hessenberg.npz',H=H,period_ms=A.T)
    else:rows=[]
    write(folder/'result.json',dict(status='DIAGNOSTIC_COMPLETE_NOT_CERTIFIED',**gate,
        krylov=rows,seconds=time.time()-start,returns=A.calls,
        scope='Currentstep,approximateperiodicseed. No refinedorbit,time-stepconvergence,independentleadingeigenresidualorcriticalcrossing certificate.',model_promoted=False))
    write(folder/'jobs.json',dict(status='COMPLETE',pid=os.getpid(),completed_returns=A.calls))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--device',type=int,default=1)
    p.add_argument('--krylov',type=int,default=0);a=p.parse_args();assert 0<=a.krylov<=16;probe(a.label,a.device,a.krylov)
