"""Spatial delay-rate candidate. Every closure assumption remains under test."""
from common import *
import argparse,time,math
from scipy import sparse,special,integrate
from numba import njit,prange,set_num_threads
set_num_threads(8)

@njit(cache=True)
def antiderivative(x,table,lo,step):
    if x<lo:
        return (-math.log(-x)-1/(4*x*x)+3/(16*x**4)+math.log(-lo)+1/(4*lo*lo)-3/(16*lo**4))/math.sqrt(math.pi)
    j=min(int((x-lo)/step),len(table)-2);a=(x-lo)/step-j
    return table[j]*(1-a)+table[j+1]*a

@njit(cache=True)
def phi(mu,ve,vi,tm,ref,th,tw,table,lo,step,reset):
    ans=np.zeros(len(mu))
    for i in range(len(mu)):
        sig=math.sqrt(max(ve[i]+vi[i],0.))
        m=mu[i]-1.0325*math.sqrt(max((ve[i]*4.2+vi[i]*19.)/tm[i],0.))
        for q in range(th.shape[1]):
            if tw[i,q]==0:continue
            if sig<1e-7:
                if m>th[i,q]:ans[i]+=tw[i,q]/(ref[i]+tm[i]*math.log((m-reset)/(m-th[i,q])))
                continue
            a=(reset-m)/sig;b=(th[i,q]-m)/sig
            if b>=12.:continue
            val=antiderivative(b,table,lo,step)-antiderivative(a,table,lo,step)
            ans[i]+=tw[i,q]/(ref[i]+tm[i]*math.sqrt(math.pi)*max(val,0.))
    return ans

@njit(cache=True)
def sparse_delay(ptr,col,weight,hist,head,P,D):
    y=np.zeros(P)
    for i in range(P):
        for k in range(ptr[i],ptr[i+1]):
            lag=col[k]//P;source=col[k]%P
            y[i]+=weight[k]*hist[(head-lag)%D,source]
    return y

@njit(cache=True)
def sparse_mv(ptr,col,val,x):
    y=np.zeros(len(ptr)-1)
    for i in range(len(y)):
        for k in range(ptr[i],ptr[i+1]):y[i]+=val[k]*x[col[k]]
    return y

@njit(cache=True,parallel=True)
def sparse_window(ptr,col,val,hist,offset):
    y=np.zeros(len(ptr)-1)
    for i in prange(len(y)):
        for k in range(ptr[i],ptr[i+1]):y[i]+=val[k]*hist[offset+col[k]]
    return y

@njit(cache=True)
def evolve(steps,dt,reg,tm,ref,tr,jext,th,tw,ops,D,nu,table,lo,step,reset,initial):
    P=len(reg);r,gE,gI,cE,cI,hist,head=initial
    arE=math.exp(-dt/.7);arI=math.exp(-dt/1.);adE=math.exp(-dt/3.5);adI=math.exp(-dt/18.)
    at=np.exp(-dt/tr);frames=np.zeros((steps//20,P));acc=np.zeros(P)
    for k in range(steps):
        offset=(head[0]+1)*P
        e=sparse_window(ops[0][0],ops[0][1],ops[0][2],hist,offset)
        inh=sparse_window(ops[1][0],ops[1][1],ops[1][2],hist,offset)
        qe=sparse_mv(ops[2][0],ops[2][1],ops[2][2],r)
        qi=sparse_mv(ops[3][0],ops[3][1],ops[3][2],r)
        ve=tm*qe;vi=tm*qi
        for i in range(P):
            ex=nu[k,reg[i]]
            e[i]+=tm[i]/.7*jext[i]*ex
            if reg[i]<2:ve[i]+=tm[i]*jext[i]**2*ex
        gE=gE*arE+dt*e;gI=gI*arI+dt*inh
        cE=gE+(cE-gE)*adE;cI=gI+(cI-gI)*adI
        f=phi(cE-cI,ve,vi,tm,ref,th,tw,table,lo,step,reset)
        r=at*r+(1-at)*f
        head[0]=(head[0]+1)%D
        for i in range(P):
            hist[head[0]*P+i]=r[i];hist[(head[0]+D)*P+i]=r[i]
        acc+=r*dt
        if (k+1)%20==0:
            frames[k//20]=acc;acc[:]=0.
    return frames,(r,gE,gI,cE,cI,hist,head)

def table_build():
    lo=-50.;step=.0002;x=np.arange(lo,12.+step/2,step)
    return integrate.cumulative_trapezoid(special.erfcx(-x),x,initial=0),lo,step

def main(grid,seed,duration):
    start=time.time();folder=OUT/f'grid{grid}';cfg=read(OUT/'model_config.json');p=cfg['params'];dt=p['dt']
    z=np.load(folder/'model.npz');reg=z['region'];P=len(reg);D=read(folder/'prepared.json')['delay_bins']
    tm=np.where(reg<3,p['tau_m_E'],p['tau_m_I']);ref=np.where(reg<3,p['tau_ref_E'],p['tau_ref_I'])
    tr=np.where(reg<3,5.,2.5);jext=np.where(reg<3,p['J_ext_E'],p['J_ext_I'])
    ops=[]
    for name in ('ampa_delay','gaba_delay','ampa_variance','gaba_variance'):
        m=sparse.load_npz(folder/f'{name}.npz')
        if name.endswith('delay'):
            coo=m.tocoo();col=(D-1-coo.col//P)*P+coo.col%P
            m=sparse.coo_matrix((coo.data,(coo.row,col)),shape=m.shape).tocsr();m.sort_indices()
        ops.append((m.indptr.astype(np.int32),m.indices.astype(np.int32),m.data))
    table,lo,step=table_build();steps=round(duration/dt)
    drive=runtime.CoreOUMixture(np.array([0,1]),cfg['signal_per_ms'],.95,0.,dt,p['tau_n'],p['sigma_n'],seed)
    nu=np.full((steps,6),cfg['signal_per_ms'])
    for k in range(steps):nu[k,:2]=np.maximum(0.,cfg['signal_per_ms']+drive.step(k*dt))
    state=(np.zeros(P),np.zeros(P),np.zeros(P),np.zeros(P),np.zeros(P),np.zeros(2*D*P),np.array([D-1],np.int64))
    result=[];stem=f'grid{grid}_seed{seed}';dest=OUT/('rate' if duration==DURATION else 'canary');dest.mkdir(exist_ok=True)
    for startstep in range(0,steps,10000):
        num=min(10000,steps-startstep)
        out,state=evolve(num,dt,reg,tm,ref,tr,jext,z['threshold'],z['threshold_weight'],tuple(ops),D,
                         nu[startstep:startstep+num],table,lo,step,p['V_reset'],state)
        assert np.isfinite(out).all() and out.min()>=0 and np.all(out/2 <= 1/ref+1e-12)
        result.append(out)
        write(dest/f'{stem}_progress.json',dict(status='RUNNING',simulated_ms=(startstep+num)*dt,seconds=time.time()-start))
    counts=np.concatenate(result);env=smooth_contacts(counts@z['contact_weights'].T)
    six=np.stack([counts[:,reg==j]@z['count'][reg==j] for j in range(6)],axis=1)
    field=np.zeros((len(counts),grid*grid))
    for i in np.flatnonzero(reg<3):field[:,z['cell'][i]]+=counts[:,i]*z['count'][i]
    np.savez_compressed(dest/f'{stem}.npz',group_expected_counts=counts.astype(np.float32),six_counts=six,
        contact_envelope=env,field_counts=field.reshape(-1,grid,grid).astype(np.float32),nu_core=nu[:,:2].astype(np.float32))
    write(dest/f'{stem}.json',dict(status='COMPLETE',grid=grid,seed=seed,J=J,duration_ms=duration,seconds=time.time()-start,
        candidate='spatial colored-Siegert cascade with current core OU; unvalidated dynamic closure',
        mean_six_hz=[float(six[1000:,j].mean()/z['count'][reg==j].sum()/.002) if len(six)>1000 else None for j in range(6)],
        formula='native two-stage synapse expectation; instant independent variance; empirical thresholds; E5/I2.5 response',
        calibration='none on current network',noise='exact core OU; private Poisson retained only through moments'))
    write(dest/f'{stem}_progress.json',dict(status='COMPLETE',simulated_ms=duration,seconds=time.time()-start))
    print(stem,'COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,required=True);p.add_argument('--seed',type=int,default=848101)
    p.add_argument('--duration',type=float,default=DURATION);a=p.parse_args();main(a.grid,a.seed,a.duration)
