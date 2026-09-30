"""Arnoldi spectrum of the actual full-state linearized stationary flow.

The rate-field characteristic validates candidate growth rates independently.
This finite-dimensional search proves instability when it finds a validated
growing mode; it does not use an old-model high-frequency stability bound.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from fine_rate_frozen_Z_fields import capture,restore
from onset_tangent_cuda import Tangent
from onset_variational_return import Coordinates
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from refractory_rate_response import covariance_matrices
from nonlinear_rate_response import normalized_input,SCALE
from core_a_static_spectrum import refine
from pathlib import Path
import argparse,os,time


def seed(e,s,r):
    a,b,qa,qb=s.matrices();F=s.tm*s.area[0]*(a@r)+s.private_mu;G=s.tm*s.area[1]*(b@r)
    ve=s.tm*s.area[0]**2*(qa@r)+s.private_ve;vi=s.tm*s.area[1]**2*(qb@r)
    syn=np.array([F,F,G,G,.5*s.E*r,s.Z]);x=np.column_stack([F-s.Z*G-syn[4],ve,s.Z**2*vi]);u=normalized_input(x,s.theta)/SCALE
    state=np.zeros((42,s.P))
    for pop,mask in [('E',s.E),('I',~s.E)]:
        A,B,C,_,_=covariance_matrices(pop,e.dt)
        for ch,value in enumerate([ve,vi]):state[3*ch:3*ch+3,mask]=np.linalg.solve(np.eye(3)-A[ch],B[ch])[:,None]*value[mask]
    for ch in range(3):state[6+12*ch:6+12*(ch+1)]=u[:,ch]
    e.syn[:]=e.cp.asarray(syn);e.local.state[:]=e.cp.asarray(state);e.local.history[:]=e.cp.asarray(r)
    e.local.rate[:]=e.cp.asarray(r);e.emitted[:]=e.cp.asarray(r);e.local.physical[:]=e.cp.asarray(np.array([x[:,0],ve,vi]));e.local.clock.fill(0)
    e.transport.pars[19].fill(0);e.transport.pars[20].fill(1);e.cp.cuda.get_current_stream().synchronize()


def main(a):
    source=Path(a.source);data=np.load(source);s=PhysicalDelayConditionalDrift();s.set_Z(data['Z']);r=data['r'];assert abs(s.residual(r)).max()<1e-11
    dest=source.parent/'stationary_flow_spectrum';dest.mkdir(exist_ok=True);assert not (dest/'jobs.json').exists()
    write(dest/'contract.json',dict(source=str(source),method='Actual full-state variational map over20ms, Arnoldi with two orthogonalization passes. Includes synapses, dynamic M, covariance, input histories and full delays. Resolve logarithm aliases against the independent physical rate characteristic.',
        dt_ms=.05,krylov_dimension=a.dimension,scope='Verified positive-real-part eigenvalues prove stationary instability; no complete stable-spectrum certificate from a finite Krylov search.',model_promoted=False))
    write(dest/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    e=build(a.device);seed(e,s,r);base=capture(e);t=Tangent(e);t.graph();c=Coordinates(base,e.s);T=20.;calls=0;start=time.time()
    def flow(v):
        nonlocal calls
        restore(e,base);c.set_tangent(t,v)
        for _ in range(2):t.chunk()
        calls+=1;return c.tangent(t)
    rng=np.random.default_rng(92403);v=rng.normal(size=c.size);v.reshape(-1,c.P)[4,~s.E]=0;v/=np.linalg.norm(v)
    vectors=[v];H=np.zeros((a.dimension+1,a.dimension));progress=[]
    for k in range(a.dimension):
        w=flow(vectors[k])
        for _ in range(2):
            for j in range(k+1):
                h=float(vectors[j]@w);H[j,k]+=h;w-=h*vectors[j]
        H[k+1,k]=np.linalg.norm(w);values,small=np.linalg.eig(H[:k+1,:k+1]);order=np.argsort(-abs(values))[:6]
        candidates=[dict(real=float(values[j].real),imag=float(values[j].imag),modulus=float(abs(values[j])),arnoldi_residual=float(abs(H[k+1,k]*small[-1,j]))) for j in order]
        row=dict(dimension=k+1,seconds=time.time()-start,candidates=candidates);progress.append(row);write(dest/'progress.json',progress)
        if (k+1)%4==0:log('CORE A STATIONARY ARNOLDI',row)
        if k+1<a.dimension:
            assert H[k+1,k]>1e-14;vectors.append(w/H[k+1,k])
    np.savez_compressed(dest/'hessenberg.npz',H=H,dt_ms=e.dt,flow_time_ms=T)
    op=s.local_operating(*s.moments(r));rows=[]
    for j in np.argsort(-abs(values))[:8]:
        rho=complex(values[j])
        if rho.imag<-1e-8:continue
        coeff=small[:,j];vec=sum(coeff[k]*vectors[k] for k in range(len(vectors)));vec/=np.linalg.norm(vec)
        actual=flow(vec.real)+1j*flow(vec.imag) if rho.imag else flow(vec.real)
        residual=float(np.linalg.norm(actual-rho*vec));raw=vec.reshape(-1,c.P)*c.scale/c.weight;vr=raw[47];vr/=np.linalg.norm(vr)
        lam0=np.log(rho)/T;aliases=[]
        for n in range(-4,5):
            lam=lam0+2j*np.pi*n/T;C=s.characteristic(r,lam,dt=e.dt,operating=op)
            aliases.append((float(np.linalg.norm(C@vr)),lam))
        aliases.sort(key=lambda x:x[0]);best,guess=aliases[0];ll,vv,tr=refine(s,r,op,guess,e.dt,vr.copy())
        row=dict(multiplier=[rho.real,rho.imag],direct_flow_residual=residual,initial_characteristic_residual=best,selected_alias_lambda_per_ms=[guess.real,guess.imag],
            status='TEMPORAL_ROOT_PASS' if ll is not None else 'NO_TEMPORAL_ROOT')
        if ll is not None:
            C=s.characteristic(r,ll,dt=e.dt);en=s.sizes*abs(vv)**2;en/=en.sum()
            row.update(lambda_per_ms=[float(ll.real),float(ll.imag)],growth_per_s=float(ll.real*1000),frequency_hz=float(abs(ll.imag)*1000/(2*np.pi)),independent_characteristic_residual=float(np.linalg.norm(C@vv)),
                mode_energy_E_A_B_surround_I=[float(en[s.E&(s.geo['group_region']==i)].sum()) for i in range(3)]+[float(en[~s.E].sum())])
            np.savez_compressed(dest/f'rate_mode{len(rows)}.npz',v=vv,lambda_per_ms=ll,r=r,Z=s.Z)
        rows.append(row);write(dest/'eigenvalue_progress.json',rows);log('CORE A STATIONARY MODE',row)
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,flow_calls=calls,seconds=time.time()-start,spectrum_complete=False,model_promoted=False))
    write(dest/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--device',type=int,default=0);p.add_argument('--dimension',type=int,default=32);a=p.parse_args();main(a)
