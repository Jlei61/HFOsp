"""One bounded numerical homotopy excluding the known high-both root.

The auxiliary continuation coordinate is not a physical network parameter.
Only a distinct root of the original equation at auxiliary value one counts.
"""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from core_a_deflated_stationary import factor
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import logit
import os,time

DEST=OUT/'core_a_bifurcation_type_20260924/asymmetric_stationary_homotopy'


def main():
    DEST.mkdir(exist_ok=True);assert not(DEST/'jobs.json').exists();start=time.time()
    old=np.load(DEST.parent/'numerical_root_homotopy/physical_candidate.npz');known=old['q'];s=PhysicalDelayConditionalDrift();s.set_Z(old['Z'])
    initial=np.load(DEST.parent/'asymmetric_stationary_deflation/mean_lowB.npz')['initial']
    q0=logit(np.clip(initial*s.ref,1e-10,1-1e-10));P=s.P;eye=sparse.eye(P,format='csc');wa=4.
    write(DEST/'contract.json',dict(question='Can an asymmetric original equilibrium be reached when numerical continuation explicitly avoids the known high-both stationary root?',
        equations='Exactly the existing native Core-A-only endpoint field, original stationary response, graph and dynamic-M equilibrium law. No physical parameter is varied.',
        method='H(q,a)=(1-a)(q0-q)+a*G(q), G=(1+1/RMS(q-q_known)^2)*F(q). Pseudo-arclength with exact sparse-plus-rank-one Jacobian via Sherman-Morrison. Onlya=1 may represent an original equilibrium. Auxiliary turns are never network SN points.',
        initial='Original actual spatial mean with a100mHz Core-B numerical guess, not the failed deflated minimizer.',
        reference='https://arxiv.org/abs/1410.5620',budget='One numerical path, at most120accepted points or180trials, with original residual<1e-11/ms and separation from known root>1e-3 required. No physical continuation of irrelevant high-both roots.',model_promoted=False))
    def evaluate(q,a,jac=True):
        data=value_and_jacobian(s,q,jac);F=data[1];m,g,rho=factor(q,known);G=m*F
        H=(1-a)*(q0-q)+a*G
        if not jac:return H
        return H,(a*m*data[3]-(1-a)*eye).tocsc(),G-(q0-q),a*G,g
    def bordered(q,a,row,rhs):
        H,J,col,u,g=evaluate(q,a);B=sparse.vstack([sparse.hstack([J,sparse.csc_matrix(col[:,None])]),sparse.csc_matrix(row[None,:])],format='csc')
        lu=splu(B);rhs_u=np.r_[u,0.];v=np.r_[g,0.];w=lu.solve(rhs);z=lu.solve(rhs_u);den=float(1+v@z)
        if abs(den)<1e-12:raise np.linalg.LinAlgError('Numerical bordered deflation singularity')
        answer=w-z*float(v@w)/den
        residual=float(np.linalg.norm(B@answer+rhs_u*(v@answer)-rhs)/max(np.linalg.norm(rhs),1.))
        assert residual<1e-6,('Bordered solve residual',residual)
        return answer
    def normalize(v):return v/np.sqrt(v[:-1]@v[:-1]/P+(wa*v[-1])**2)
    y=np.r_[q0,0.];v=normalize(np.r_[evaluate(q0,0.)[2],1.]);ds=.08;rows=[];trials=[];status='NO_DISTINCT_ORIGINAL_ROOT_ESTABLISHED';result_extra={}
    for trial in range(180):
        pred=y+ds*v;x=pred.copy();row=np.r_[v[:-1]/P,wa*wa*v[-1]];success=False;trace=[]
        for it in range(14):
            H=evaluate(x[:-1],x[-1],False);arc=float(row@(x-pred));norm=float(np.sqrt(H@H/P+arc*arc));trace.append(norm)
            if max(abs(H).max(),abs(arc))<2e-9:success=True;break
            try:delta=bordered(x[:-1],x[-1],row,-np.r_[H,arc])
            except (np.linalg.LinAlgError,AssertionError,RuntimeError):break
            alpha=min(1.,4./max(abs(delta[:-1]).max(),1e-12))
            for _ in range(20):
                xx=x+alpha*delta
                if abs(xx[:-1]).max()<80 and np.linalg.norm(xx[:-1]-known)/np.sqrt(P)>1e-9:
                    hh=evaluate(xx[:-1],xx[-1],False);aa=float(row@(xx-pred))
                    if np.sqrt(hh@hh/P+aa*aa)<norm:x=xx;break
                alpha*=.5
            else:break
        trials.append(dict(trial=trial,accepted=success,auxiliary_a=float(x[-1]),step=ds,errors=trace));write(DEST/'trials.json',trials)
        if not success:
            ds*=.5
            if ds<1e-4:break
            continue
        previous=y.copy();y=x
        v=normalize(bordered(y[:-1],y[-1],row,np.r_[np.zeros(P),1.]))
        entry=dict(index=len(rows),auxiliary_a=float(y[-1]),errors=trace,seconds=time.time()-start);rows.append(entry)
        write(DEST/'accepted.json',rows);write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid(),latest=entry))
        np.savez_compressed(DEST/'latest_auxiliary.npz',q=y[:-1],a=y[-1],Z=s.Z);log('DEFLATED NUMERICAL HOMOTOPY',entry)
        if (previous[-1]-1)*(y[-1]-1)<=0:
            w=(1-previous[-1])/(y[-1]-previous[-1]);q=(1-w)*previous[:-1]+w*y[:-1]
            for _ in range(20):
                r,F,op,J=value_and_jacobian(s,q);m,g,rho=factor(q,known)
                if np.max(abs(s.residual(r)))<1e-11:break
                base=splu(J.tocsc()).solve(-F);delta=base/(1-g@base);merit=m*np.linalg.norm(F)
                for alpha in 2.**-np.arange(20):
                    qq=q+alpha*delta
                    if np.linalg.norm(qq-known)/np.sqrt(P)<1e-9:continue
                    mm=factor(qq,known)[0];ff=value_and_jacobian(s,qq,False)[1]
                    if mm*np.linalg.norm(ff)<merit:q=qq;break
                else:break
            r,F,op=value_and_jacobian(s,q,False);err=float(np.max(abs(s.residual(r))));rho=factor(q,known)[2]
            if err<1e-11 and rho>1e-3:
                status='DISTINCT_ORIGINAL_EQUILIBRIUM_ROOT';np.savez_compressed(DEST/'physical_candidate.npz',q=q,r=r,Z=s.Z)
                result_extra=dict(original_rate_residual_per_ms=err,distance_from_known=float(rho),regional_rates_hz=s.regional_rates(r),stability='NOT_COMPUTED',bifurcation_type='NOT_ESTABLISHED');break
        if len(rows)>=120:break
        ds=min(.35,ds*(1.4 if len(trace)<=4 else (1.1 if len(trace)<=7 else .7)))
    write(DEST/'result.json',dict(status=status,accepted_points=len(rows),trials=len(trials),seconds=time.time()-start,auxiliary_endpoint=float(y[-1]),**result_extra,model_promoted=False))
    write(DEST/'jobs.json',dict(status='COMPLETE',pid=os.getpid(),scientific_status=status))


if __name__=='__main__':main()
