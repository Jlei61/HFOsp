"""Independent CPU derivative of the full nine-state periodic residual.

This does not propagate a neutral mode through strongly growing transverse
directions. It checks the same generalized periodic variational equation by
differentiating the full physical RHS and time derivative at fixed phase.
It supplements, rather than silently overrides, monodromy validation.
"""
from rate_periodic import *


def physical_fields(s,r,T,J,M):
    N=len(r);K=N//2+1;rf=np.fft.rfft(r,axis=0)/N
    cf=np.empty((9,K,s.P),complex);af=np.empty((4,K,s.P),complex)
    for k,v in enumerate(rf):
        lam=2j*np.pi*k/T
        af[:,k]=[a@v for a in s.matrices(J,lam)]
        a,b,aa,bb=af[:,k];drive=v/s.filter_response(lam)
        qa=s.tm*s.area[0]*a/(1+lam*s.rise[0]);ia=qa/(1+lam*s.decay[0])
        qg=s.tm*s.area[1]*b/(1+lam*s.rise[1]);ig=qg/(1+lam*s.decay[1])
        cf[:,k]=[drive/(1+lam*s.tf),drive/(1+lam*s.ts),qa,ia,qg,ig,
            s.tm*s.area[0]**2*aa/(1+lam*s.tau[0]/2),
            s.tm*s.area[1]**2*bb/(1+lam*s.tau[1]/2),.5*s.E*v/(1+lam*1000)]
    def unpack(c):
        pad=np.zeros((M//2+1,s.P),complex);pad[:K]=c*M;pad[K-1]*=.5
        return np.fft.irfft(pad,n=M,axis=0)
    state=np.array([unpack(c) for c in cf]);arrival=np.array([unpack(c) for c in af])
    lam=2j*np.pi*np.arange(K)[:,None]/T
    velocity=np.array([unpack(c*lam) for c in cf]);rhs=np.empty_like(state)
    for i in range(M):rhs[:,i]=s.rhs(state[:,i],arrival[:,i])
    return state,velocity,rhs


def check(label,factor=2,epsilon_scale=1.):
    versions=sorted([read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')],key=lambda q:q['N'])
    q=versions[-1];z=np.load(q['orbit']);r=z['r'];T=float(z['T']);J=float(z['J'])
    tangent=np.load(PERIODIC_OUT/f'{label}_tangent_N{len(r)}.npz')['tangent']
    dr=tangent[:-2].reshape(r.shape)/1000;dT=T*tangent[-2];dJ=tangent[-1]/1000
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();M=factor*len(r)
    scale=max(np.linalg.norm(dr)/np.linalg.norm(r),abs(dT/T),abs(dJ/J),1e-3)
    assert 0<epsilon_scale<=1
    eps=min(.02,2e-4/scale)*epsilon_scale;rows=[]
    def norm(a):return np.sqrt(np.mean(a*a@weights,axis=-1))
    def evaluate(h,direction,kind):
        yp,vp,fp=physical_fields(s,r+h*direction,T+h*dT,J+h*dJ,M)
        ym,vm,fm=physical_fields(s,r-h*direction,T-h*dT,J-h*dJ,M)
        dy=(yp-ym)/(2*h);dv=(vp-vm)/(2*h);df=(fp-fm)/(2*h)
        error=norm(dv-df)/np.maximum(norm(dv)+norm(df),1e-14)
        out=dict(kind=kind,epsilon=float(h),relative_residual_by_state=error,
            maximum_relative_residual=float(max(error)),
            maximum_rate_rhs_tangent_defect_Hz_per_ms=float(np.max(abs(dv[:2]-df[:2]))*1000),
            tangent_RMS_by_state=norm(dy))
        print('FULL RHS TANGENT',label,out,flush=True);return out
    for h in [eps,eps/2,eps/4]:rows.append(evaluate(h,dr,'fold_tangent'))
    bad=evaluate(eps/4,np.roll(dr,1,axis=1),'permuted_group_negative_control')
    result=dict(label=label,orbit=q['orbit'],N=len(r),check_N=M,J_EE_core=J,T_ms=T,
        checks=rows,negative_control=bad,
        status='INDEPENDENT_VARIATIONAL_BVP_CHECK_ONLY',
        interpretation='Centered derivative of y_dot(theta;s)-F(y(theta;s), delayed y(theta;s),J(s)) with r_s=r+s*dr, T_s=T+s*dT, J_s=J+s*dJ. All linear filters and physical delays reconstructed separately on CPU; no GPU BVP Jacobian is used.',
        state_order=['fast_rate','slow_rate','AMPA_rise','AMPA_current','GABA_rise','GABA_current','E_variance','I_variance','M'],
        scope='Independent tangent-equation diagnostic with epsilon refinement and a non-null control. Does not automatically promote a fold, establish adjacent stability, or replace the recorded monodromy evidence.')
    result['epsilon_scale']=epsilon_scale
    suffix='' if epsilon_scale==1 else f'_e{epsilon_scale:g}'
    write(PERIODIC_OUT/(label+'_independent_variational_BVP'+suffix+'.json'),result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('labels',nargs='+');p.add_argument('--factor',type=int,default=2)
    p.add_argument('--epsilon-scale',type=float,default=1.)
    a=p.parse_args()
    for label in a.labels:check(label,a.factor,a.epsilon_scale)
