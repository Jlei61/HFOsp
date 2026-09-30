"""Refine and classify a physical local-Z equilibrium turning point.

A generic equilibrium saddle-node certificate does not by itself establish
that this equilibrium fold causes the observed aperiodic onset transition.
"""
from common import OUT,np,read,write,log
from core_a_equilibrium_branch import Family,DEST as BRANCH
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from scipy import sparse
from scipy.sparse.linalg import spsolve,eigs
from scipy.optimize import brentq
import os,time

DEST=OUT/'core_a_bifurcation_type_20260924/fold_certificate'


def main():
    candidate=read(BRANCH/'fold_candidate.json');i,j=candidate['points']
    lo=np.load(BRANCH/f'point{i:03d}.npz');hi=np.load(BRANCH/f'point{j:03d}.npz')
    s=PhysicalDelayConditionalDrift();family=Family(s);P=s.P;weight=50.
    y0=np.r_[lo['q'],lo['D_A']];y1=np.r_[hi['q'],hi['D_A']];chord=y1-y0
    length=np.sqrt(chord[:-1]@chord[:-1]/P+(weight*chord[-1])**2);u=chord/length
    arcrow=np.r_[u[:-1]/P,weight*weight*u[-1]];cache={};trace=[];start=time.time()
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(question='Is the actual local-resource equilibrium turn a nondegenerate saddle-node, and where is its spatial zero mode?',
        source_points=[str(BRANCH/f'point{k:03d}.npz') for k in [i,j]],
        method='Root of D_A tangent along a pseudo-arclength chart of F(q,D_A)=0. Check original physical residual, left/right zero modes, single zero eigenvalue, nonzero parameter transversality and quadratic coefficient under finite-difference refinement, and nonzero temporal characteristic derivative at lambda=0 for both time steps and continuous limit.',
        interpretation='Certifies an equilibrium bifurcation of this complete fixed-Z/dynamic-M system only. Attractor relevance and native onset attribution require separate evidence; a critical symbol is not promoted to the onset diagram solely by this certificate.',model_promoted=False))
    def correct(sigma):
        key=float(sigma)
        if key in cache:return cache[key]
        pred=y0+sigma*u
        if cache:
            nearest=min(cache,key=lambda z:abs(z-sigma));x=cache[nearest]['y']+(sigma-nearest)*u
        else:x=pred.copy()
        for it in range(15):
            F,J,col=family.evaluate(x[:-1],x[-1]);arc=float(arcrow@(x-pred));norm=np.sqrt(F@F/P+arc*arc)
            B=sparse.vstack([sparse.hstack([J,sparse.csc_matrix(col[:,None])]),sparse.csc_matrix(arcrow[None,:])],format='csc')
            if abs(F).max()<3e-11 and abs(arc)<1e-12:break
            delta=spsolve(B,-np.r_[F,arc]);accepted=False
            for alpha in 2.**-np.arange(18):
                nxt=x+alpha*delta;ff=family.evaluate(nxt[:-1],nxt[-1],False);aa=float(arcrow@(nxt-pred))
                if np.sqrt(ff@ff/P+aa*aa)<norm:x=nxt;accepted=True;break
            assert accepted,('arc correction stalled',sigma,norm)
        else:raise RuntimeError('arc correction exhausted')
        F,J,col=family.evaluate(x[:-1],x[-1]);B=sparse.vstack([sparse.hstack([J,sparse.csc_matrix(col[:,None])]),sparse.csc_matrix(arcrow[None,:])],format='csc')
        v=spsolve(B,np.r_[np.zeros(P),1.]);v/=np.sqrt(v[:-1]@v[:-1]/P+(weight*v[-1])**2)
        result=dict(y=x,tangent=v);cache[key]=result
        row=dict(sigma=float(sigma),D_A=float(x[-1]),D_tangent=float(v[-1]),F_infinity=float(abs(F).max()),seconds=time.time()-start)
        trace.append(row);write(DEST/'progress.json',dict(status='RUNNING',pid=os.getpid(),trace=trace));log('CORE A FOLD REFINE',row)
        return result
    write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    sigma=brentq(lambda a:correct(a)['tangent'][-1],0.,length,xtol=1e-12,rtol=1e-13)
    critical=correct(sigma);q=critical['y'][:-1];D=float(critical['y'][-1]);tm=family.set(D)
    r,F,op,J=value_and_jacobian(s,q);v=critical['tangent'][:-1];v/=np.linalg.norm(v)
    ev,wv=eigs(J.T,k=4,sigma=1e-7,tol=1e-11);k=int(np.argmin(abs(ev)));w=wv[:,k].real;w/=w@v
    static=dict(original_rate_residual_per_ms=float(abs(s.residual(r)).max()),logit_residual=float(abs(F).max()),
        right_null_relative=float(np.linalg.norm(J@v)),left_null_relative=float(np.linalg.norm(J.T@w)/np.linalg.norm(w)),
        closest_static_eigenvalues=[[float(z.real),float(z.imag)] for z in ev],
        left_right_product=float(w@v))
    coefficients=[]
    for h in [1e-3,5e-4,2.5e-4]:
        jp=value_and_jacobian(s,q+h*v)[3];jm=value_and_jacobian(s,q-h*v)[3]
        b=float(.5*w@((jp@v-jm@v)/(2*h)))
        coefficients.append(dict(q_step=h,quadratic_coefficient=b))
    transversality=[]
    for h in [1e-6,5e-7,2.5e-7]:
        family.set(D+h);plus=value_and_jacobian(s,q,False)[1];family.set(D-h);minus=value_and_jacobian(s,q,False)[1]
        a=float(w@((plus-minus)/(2*h)));transversality.append(dict(D_step=h,parameter_coefficient=a))
    family.set(D);vr=r*(1-r*s.ref)*v;vr/=np.linalg.norm(vr)
    energy=s.sizes*abs(vr)**2;energy/=energy.sum()
    temporal=[]
    for dt in [.05,.025,None]:
        C=s.characteristic(r,0.,dt=dt);ee,ww=eigs(C.T,k=3,sigma=1e-7,tol=1e-11);wc=ww[:,np.argmin(abs(ee))].real;wc/=wc@vr
        deriv=[]
        for h in [1e-6,5e-7,2.5e-7]:
            cp=s.characteristic(r,h,dt=dt);cm=s.characteristic(r,-h,dt=dt)
            d=complex(wc@((cp@vr-cm@vr)/(2*h)));assert abs(d.imag)<1e-7
            deriv.append(dict(lambda_step_per_ms=h,temporal_coefficient_ms=float(d.real)))
        temporal.append(dict(dt_ms=dt,right_null_relative=float(np.linalg.norm(C@vr)),left_null_relative=float(np.linalg.norm(C.T@wc)/np.linalg.norm(wc)),derivatives=deriv))
    a_values=np.array([x['parameter_coefficient'] for x in transversality]);b_values=np.array([x['quadratic_coefficient'] for x in coefficients])
    a_conv=float(abs(a_values[-1]-a_values[-2])/max(abs(a_values[-1]),1e-15));b_conv=float(abs(b_values[-1]-b_values[-2])/max(abs(b_values[-1]),1e-15))
    simple=bool(np.count_nonzero(abs(ev)<1e-5)==1)
    passed=bool(static['original_rate_residual_per_ms']<1e-11 and static['right_null_relative']<1e-7 and static['left_null_relative']<1e-7 and simple
        and abs(a_values[-1])>1e-7 and abs(b_values[-1])>1e-7 and a_conv<.01 and b_conv<.01
        and all(z['right_null_relative']<1e-7 and z['left_null_relative']<1e-7 and abs(z['derivatives'][-1]['temporal_coefficient_ms'])>1e-5
            and abs(z['derivatives'][-1]['temporal_coefficient_ms']/z['derivatives'][-2]['temporal_coefficient_ms']-1)<.01 for z in temporal))
    result=dict(status='GENERIC_EQUILIBRIUM_SADDLE_NODE_CERTIFIED' if passed else 'CERTIFICATE_INCOMPLETE',
        D_A=D,Z_A=1-D,native_CoreA_field_time_ms=tm,D_global=s.D,regional_rates_hz=s.regional_rates(r),global_rate_hz=s.global_rate(r),
        static=static,quadratic=coefficients,transversality=transversality,coefficient_relative_refinement=dict(parameter=a_conv,quadratic=b_conv),
        temporal=temporal,mode_energy_A_B_surround=[float(energy[s.geo['group_region']==j].sum()) for j in range(3)],
        scope='Equilibrium fold of the actual full spatial network with CoreA-only Z variation. Stability of other modes and relevance to observed irregular local transition not established by this certificate.',
        onset_bifurcation_type='NOT_ESTABLISHED',model_promoted=False)
    np.savez_compressed(DEST/'critical_point.npz',q=q,r=r,Z=s.Z,D_A=D,v_logit=v,left_logit=w,v_rate=vr,mode_energy=energy)
    write(DEST/'result.json',result);write(DEST/'jobs.json',dict(status='COMPLETE',pid=os.getpid(),scientific_result=result['status']));log('CORE A FOLD CERTIFICATE',result)


if __name__=='__main__':main()
