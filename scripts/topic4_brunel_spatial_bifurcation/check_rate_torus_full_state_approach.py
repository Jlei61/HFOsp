"""Full nine-state and continuous-delay norm near the candidate saddle cycle.

Reconstruct every linear state at k*omega+l*nu, including the rate history at
physical delay times. No spatial projection is introduced. Fast-phase averaged
Parseval norms avoid storing large sampled state/history fields.
"""
from rate_periodic import *


def coefficients(r):
    n,m,P=r.shape
    assert n%2==0 and (m==1 or m%2==0)
    c=np.fft.fftshift(np.fft.fft2(r,axes=(0,1)),axes=(0,1))/(n*m)
    c=np.concatenate([c,c[:1]],axis=0);c[[0,-1]]*=.5
    if m>1:
        c=np.concatenate([c,c[:,:1]],axis=1);c[:,[0,-1]]*=.5
    return c,np.arange(-n//2,n//2+1),np.arange(-m//2,m//2+1) if m>1 else np.array([0])


def states(s,rf,lam,J,device,batch=32):
    K,P=rf.shape;o=Periodic(s,2*(batch-1),device);cp=o.cp
    rf=cp.asarray(rf);ll=cp.asarray(lam)[:,None];arr=[]
    for k,(d,mask,index,ptr) in enumerate(o.raw):
        scale=cp.where(mask,J**(1 if k==0 else 2),1.) if k in [0,2] else 1.
        out=cp.empty_like(rf);edges=d.shape[0]
        for start in range(0,K,batch):
            stop=min(start+batch,K);n=stop-start
            phase=cp.exp(-cp.asarray(s.delays)[:,None]*ll[start:stop,0])
            data=(d@phase).T.copy()*scale
            op=o.cs.csr_matrix((data.ravel(),index[:n*edges],ptr[:n*P+1]),shape=(n*P,n*P))
            out[start:stop]=(op@rf[start:stop].ravel()).reshape(n,P)
        arr.append(out)
    a,b,qa,qb=arr;tm,ref,th,alpha,tf,ts,E=o.gpars
    H=alpha/(1+ll*tf)+(1-alpha)/(1+ll*ts);target=rf/H
    qa_state=tm*s.area[0]*a/(1+ll*s.rise[0]);qg_state=tm*s.area[1]*b/(1+ll*s.rise[1])
    local=[target/(1+ll*tf),target/(1+ll*ts),qa_state,qa_state/(1+ll*s.decay[0]),
        qg_state,qg_state/(1+ll*s.decay[1]),tm*s.area[0]**2*qa/(1+ll*s.tau[0]/2),
        tm*s.area[1]**2*qb/(1+ll*s.tau[1]/2),.5*E*rf/(1+ll*1000)]
    return cp,local,arr


def rhs_check(s,local,arr,kt,kp,omega,nu,cp):
    k=np.repeat(kt,len(kp));l=np.tile(kp,len(kt));lam=1j*(k*omega+l*nu)
    scales=np.array([1000.,1000.,1.,1.,1.,1.,.1,.1,1.])[:,None];rows=[]
    for theta,psi in [(0.,.21),(1.17,2.08),(3.43,4.79)]:
        phase=cp.asarray(np.exp(1j*(k*theta+l*psi)))[:,None]
        y=np.array([cp.sum(a*phase,axis=0).real.get() for a in local])
        dy=np.array([cp.sum(a*phase*cp.asarray(lam)[:,None],axis=0).real.get() for a in local])
        arrivals=np.array([cp.sum(a*phase,axis=0).real.get() for a in arr])
        error=(s.rhs(y,arrivals)-dy)*scales
        rows.append(dict(fast_angle=theta,slow_angle=psi,maximum_scaled_RHS_residual_per_ms=float(abs(error).max())))
    assert max(r['maximum_scaled_RHS_residual_per_ms'] for r in rows)<1e-7
    return rows


def check(row,s,device):
    z=np.load(row['torus']);target=np.load(row['target']);r=z['r'];J=float(z['J'])
    assert J==float(target['J'])
    c,kt,kp=coefficients(r);d,ks,_=coefficients(target['r'][:,None,:])
    omega=2*np.pi/float(z['T']);omega_s=2*np.pi/float(target['T']);nu=float(z['nu'])
    cp,local,arr=states(s,c.reshape(-1,s.P),1j*(kt[:,None]*omega+kp[None,:]*nu).ravel(),J,device)
    torus_rhs=rhs_check(s,local,arr,kt,kp,omega,nu,cp);del arr
    _,saddle,arr=states(s,d[:,0],1j*ks*omega_s,J,device)
    saddle_rhs=rhs_check(s,saddle,arr,ks,np.array([0]),omega_s,0.,cp);del arr
    sizes=cp.asarray(s.geo['group_size']/s.geo['group_size'].sum());nk=len(kt);nl=len(kp)
    center=int(np.argmin(row['distances_Hz']));M=len(row['distances_Hz'])
    # Include both adjacent slow samples; do not force the rate-optimal slow
    # phase to be the minimum in the full-state norm.
    ids=np.array([(center+i)%M for i in [-1,0,1]])
    psi=cp.asarray(2*np.pi*ids/M);shifts=cp.asarray(np.array(row['fast_phase_shifts_cycles'])[ids])
    k=cp.asarray(kt);l=cp.asarray(kp);phase=cp.exp(-2j*np.pi*k[:,None]*shifts)
    loc=cp.zeros(3);den=cp.array(0.)
    scales=[1000.,1000.,1.,1.,1.,1.,.1,.1,1.]
    select=kt-ks[0];outside=np.ones(len(ks),bool);outside[select]=False
    def weighted(v):return cp.sum(cp.abs(v)**2*sizes,axis=(0,2))
    slow=cp.exp(1j*l[:,None]*psi)
    for a,b,scale in zip(local,saddle,scales):
        x=cp.einsum('klp,lj->kjp',a.reshape(nk,nl,s.P),slow)
        diff=x-b[select,None,:]*phase[:,:,None]
        tail=cp.sum(cp.abs(b[outside])**2*sizes)
        loc+=(weighted(diff)+tail)*scale**2
        den+=cp.sum(cp.abs(b)**2*sizes)*scale**2
    rf=cp.asarray(c);sf=cp.asarray(d[:,0]);history_rows=[]
    tau=float(max(s.delays));rate_tail=cp.sum(cp.abs(sf[outside])**2*sizes)*1e6
    saddle_history_norm=cp.sum(cp.abs(sf)**2*sizes)*1e6
    for quadrature in [8,16]:
        nodes,weights=np.polynomial.legendre.leggauss(quadrature);hist=cp.zeros(3)
        for time_,weight in zip((nodes+1)*tau/2,weights/2):
            slow=cp.exp(1j*l[:,None]*(psi-nu*time_))
            x=cp.einsum('klp,lj->kjp',rf,slow)*cp.exp(-1j*k*omega*time_)[:,None,None]
            ref=sf[select,None,:]*phase[:,:,None]*cp.exp(-1j*k*omega_s*time_)[:,None,None]
            hist+=weight*(weighted(x-ref)*1e6+rate_tail)
        full=cp.sqrt((loc+hist)/(den+saddle_history_norm))
        history_rows.append(dict(quadrature_nodes=quadrature,relative_full_state_history_distances=full.get(),
            relative_local_distances=cp.sqrt(loc/den).get(),history_RMS_Hz=cp.sqrt(hist).get()))
    # The independently reconstructed rate norm must reproduce the existing
    # sampled-field diagnostic, including all high-frequency saddle tails.
    xx=cp.einsum('klp,lj->kjp',rf,cp.exp(1j*l[:,None]*psi))
    rate=cp.sqrt(weighted(xx-sf[select,None,:]*phase[:,:,None])*1e6+rate_tail).get()
    expected=np.array(row['distances_Hz'])[ids]
    agreement=float(max(abs(rate-expected)))
    assert agreement<1e-10
    a=np.array(history_rows[0]['relative_full_state_history_distances']);b=np.array(history_rows[1]['relative_full_state_history_distances'])
    convergence=float(max(abs(a-b))/max(b));assert convergence<1e-6
    out=dict(torus=row['torus'],target=row['target'],J_EE_core=J,slow_period_s=row['slow_period_s'],
        slow_indices=ids,closest_rate_index=center,rate_RMS_Hz=rate,rate_reconstruction_difference_Hz=agreement,
        delay_horizon_ms=tau,history_quadrature_checks=history_rows,history_quadrature_relative_change=convergence,
        minimum_checked_full_state_history_distance=float(min(b)),
        torus_full_RHS_checks=torus_rhs,saddle_full_RHS_checks=saddle_rhs,
        spatial_groups=s.P,local_states_per_group=9)
    print('FULL STATE SADDLE APPROACH',out['slow_period_s'],out['minimum_checked_full_state_history_distance'],convergence,flush=True)
    return out


def main(a):
    s=RateField();source=read(PERIODIC_OUT/'TR2_same_parameter_saddle_approach.json');rows=[]
    import gc,cupy as cp
    for row in source['rows']:
        rows.append(check(row,s,a.device));gc.collect();cp.get_default_memory_pool().free_all_blocks()
    t=np.array([q['slow_period_s'] for q in rows]);d=np.array([q['minimum_checked_full_state_history_distance'] for q in rows])
    out=dict(status='FULL_STATE_APPROACH_SUPPORTED',rows=rows,
        adjacent_log_distance_period_slopes_s=np.diff(t)/np.log(d[:-1]/d[1:]),
        norm='Neuron-weighted fast-phase average of all nine local state differences with scales [1000,1000,1,1,1,1,.1,.1,1], plus the delay-interval average of squared rate differences in Hz. Divide by the corresponding saddle squared norm. Same phase alignment for every state/history variable.',
        scope='Full-state proximity diagnostic to same-J saddles. It does not solve the connecting invariant manifolds or establish finite torus stability; global bifurcation remains a candidate.')
    write(PERIODIC_OUT/'TR2_full_state_saddle_approach.json',out);print('FULL STATE SLOPES',out['adjacent_log_distance_period_slopes_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args())
