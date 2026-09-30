#!/usr/bin/env python3
"""Bounded pseudo-arclength continuation of the relevant static exit branch.

Numerical equilibrium turns are candidates only: dynamic stability and native
transition correspondence must be established separately, never inferred here.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres
from campaign import ROOT,read,write,sha
from target_stationary_root import TargetStationary,OUT as ROOT_POINT

OUT=ROOT/'target_exit_equilibrium_branch'


class Arc:
    def __init__(self):
        self.e=TargetStationary();self.shape=self.e.K/9.;self.weights=np.r_[self.e.sizes/self.e.sizes.sum()/.5**2,.01]

    def evaluate(self,y,jacobian=False):
        self.e.K=self.shape*y[-1]
        return self.e.evaluate(y[:-1],jacobian)

    def derivative_K(self,meta):
        e=self.e;pp=meta['physical'];g=meta['g'];_,grad=e.model.phi(pp,g);h=1+g
        den=1+.5*e.E/h*grad[:,0]
        value=self.shape*(grad[:,0]*(-30-pp[:,0])/h
                  -2*(grad[:,1]*pp[:,1]+grad[:,2]*pp[:,2])/h+grad[:,3])/den
        return e.S@value

    def border(self,J,pre,fk,t):
        normal=self.weights*t
        A=LinearOperator((self.e.P+1,self.e.P+1),matvec=lambda v:np.r_[J@v[:-1]+fk*v[-1],normal@v],dtype=float)
        last=normal[-1] if abs(normal[-1])>.02 else .1
        M=LinearOperator(A.shape,matvec=lambda v:np.r_[pre@v[:-1],v[-1]/last],dtype=float)
        return A,M

    def tangent(self,y,previous=None):
        _,J,pre,meta=self.evaluate(y,True);fk=self.derivative_K(meta)
        if previous is None:
            dr,info=gmres(J,-fk,M=pre,rtol=1e-9,atol=1e-12,restart=100,maxiter=3)
            t=np.r_[dr,1.]
        else:
            A,M=self.border(J,pre,fk,previous);rhs=np.zeros(len(y));rhs[-1]=1.
            t,info=gmres(A,rhs,M=M,rtol=1e-9,atol=1e-12,restart=100,maxiter=3)
        residual=np.r_[J@t[:-1]+fk*t[-1]]
        assert np.linalg.norm(residual)<1e-7,(info,np.linalg.norm(residual))
        t/=np.sqrt(np.sum(self.weights*t*t))
        if previous is not None and np.sum(self.weights*t*previous)<0:t=-t
        return t,dict(gmres_info=int(info),tangent_residual=float(np.linalg.norm(residual)),K_tangent=float(t[-1]))

    def correct(self,predict,t):
        y=predict.copy();history=[];normal=self.weights*t
        if np.min(y[:-1])<-1e-8:return None,dict(reason='Predictor left physical rate domain',history=history)
        y[:-1]=np.clip(y[:-1],0,self.e.cap*(1-1e-12))
        for iteration in range(18):
            if y[-1]<=0:return None,dict(reason='Nonpositive K',history=history)
            f,J,pre,meta=self.evaluate(y,True);arc=float(normal@(y-predict));residual=np.r_[f,arc]
            error=float(abs(f).max()*1000);history.append(dict(iteration=iteration,maximum_residual_Hz=error,arc_error=arc,K=float(y[-1])))
            if error<1e-6 and abs(arc)<1e-9:return y,dict(reason='Converged',history=history)
            fk=self.derivative_K(meta);A,M=self.border(J,pre,fk,t);errors=[]
            step,info=gmres(A,-residual,M=M,rtol=min(.02,max(1e-8,error*.001)),atol=1e-12,restart=100,maxiter=3,
                  callback=lambda q:errors.append(float(q)),callback_type='pr_norm')
            history[-1].update(gmres_info=int(info),gmres_iterations=len(errors))
            alpha=1.;accepted=False;norm=np.linalg.norm(residual)
            for attempt in range(10):
                trial=y+alpha*step
                trial[:-1]=np.clip(trial[:-1],0,self.e.cap*(1-1e-12))
                if trial[-1]>0:
                    ff,_=self.evaluate(trial);aa=float(normal@(trial-predict));rr=np.r_[ff,aa]
                    if np.linalg.norm(rr)<norm*(1-1e-4*alpha):y=trial;accepted=True;break
                alpha*=.5
            if not accepted:return None,dict(reason='No bounded correction decrease',history=history)
        return None,dict(reason='Corrector limit',history=history)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    result=read(ROOT_POINT/'result.json');assert result['status']=='NUMERICAL_ROOT'
    write(OUT/'contract.json',dict(status='REGISTERED_AFTER_ACTUAL_K9_ROOT',created_epoch=time.time(),
       question='How does the high equilibrium connected to the corresponding actual-field K9 state change as held K approaches the native K9-to12 exit interval?',
       design='One connected static-surrogate branch at Zmean.21 with actual16.7s spatialshape. K scales eachcellheldK, G remains selfconsistent fromR andM_i fromindividualrate. Sourcegraph3479groups/40000targets andfrozenv3response unchanged. Externalmeanfixed to paired5-10smean.',
       algorithm='Weighted pseudo-arclength, initialds.0125(Kstep~.125), maximum60acceptedpoints, atmost5halvingsperpoint, minarcstep.00025; max18Newtonsteps/point. DomainK8.5-12, stopafter6acceptedpost-turnpoints or90minwall. Allfailed attempts retained.',
       validation='AnalyticKderivative compared withsame-equation finite difference; tangent andequilibriumresiduals saved at every point. Accepted maxresidual<1e-6Hz. Numerical turning is NOT dynamicstability or nativebifurcation certification.',
       gate='Do not draw stable/unstable labels or certifyfold until actual-domain independentresponse andfull dynamic stability are established. Not anewMLP fit orunrelatedcommonfieldrootsearch.',
       root_producer_sha256=sha(__import__('target_stationary_root').__file__),producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    started=time.time();a=Arc()
    with np.load(ROOT_POINT/'root_candidate.npz') as z:y=np.r_[z['source_rate_per_ms'],9.]
    f,meta=a.evaluate(y);fk=a.derivative_K(meta);h=1e-4
    up=y.copy();up[-1]+=h;down=y.copy();down[-1]-=h
    fp,_=a.evaluate(up);fm,_=a.evaluate(down);fd=(fp-fm)/(2*h)
    error=float(np.linalg.norm(fk-fd)/np.linalg.norm(fd));assert error<3e-4,error
    write(OUT/'parameter_derivative_qa.json',dict(status='PASS',relative_error=error,step_K=h))
    tangent,tqa=a.tangent(y);rows=[];attempts=[];ds=.0125;post_turn=0;turn_seen=False;reason='Accepted point limit'
    for index in range(60):
        f,meta=a.evaluate(y);assert abs(f).max()*1000<1e-6
        cell=meta['cell_rate'];groups=a.e.S@cell;geo=a.e.geo
        regional=[float(np.average(groups[m],weights=a.e.sizes[m])*1000) for m in [a.e.groupE]+[a.e.groupE&(geo['group_region']==q) for q in range(3)]+[~a.e.groupE]]
        row=dict(index=index,K=float(y[-1]),rates_Hz_allE_A_B_other_I=regional,G=meta['G'],
            maximum_residual_Hz=float(abs(f).max()*1000),K_tangent=float(tangent[-1]),tangent_qa=tqa,
            elapsed_s=time.time()-started,stability='NOT_ESTABLISHED')
        rows.append(row)
        np.savez_compressed(OUT/f'point_{index:03d}.npz',source_rate_per_ms=y[:-1],K_mean=y[-1],cell_rate_per_ms=cell,
           physical=meta['physical'],g=meta['g'],Z=a.e.Z,K=a.e.K,G=meta['G'],tangent=tangent,residual_per_ms=f)
        write(OUT/'branch.json',dict(status='RUNNING',rows=rows,turn_candidate_seen=turn_seen,formal_bifurcation_allowed=False))
        write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),accepted_points=len(rows),latest=row,updated_epoch=time.time()))
        print('TARGET EQUILIBRIUM POINT',row,flush=True)
        if turn_seen:
            post_turn+=1
            if post_turn>=6:reason='Six post-turn points';break
        if index and not 8.5<=y[-1]<=12:reason='Bounded K range reached';break
        if time.time()-started>5400:reason='Bounded wall time';break
        accepted=None
        for halving in range(6):
            predict=y+ds*tangent
            accepted,detail=a.correct(predict,tangent)
            attempts.append(dict(from_index=index,ds=ds,halving=halving,**detail))
            write(OUT/'corrector_attempts.json',dict(attempts=attempts))
            if accepted is not None:break
            ds*=.5
            if ds<.00025:break
        if accepted is None:reason='Bounded corrector did not converge';break
        new_tangent,tqa=a.tangent(accepted,tangent)
        if tangent[-1]>0 and new_tangent[-1]<0:turn_seen=True
        y=accepted;tangent=new_tangent
        if len(detail['history'])<=5:ds=min(ds*1.2,.015)
        elif len(detail['history'])>=10:ds=max(ds*.8,.00025)
    final=dict(status='BOUNDED_BRANCH_COMPLETE',reason=reason,rows=rows,turn_candidate_seen=turn_seen,
           elapsed_s=time.time()-started,stability_established=False,formal_bifurcation_allowed=False)
    write(OUT/'branch.json',final);write(OUT/'progress.json',final);print('TARGET EQUILIBRIUM BRANCH COMPLETE',reason,flush=True)


if __name__=='__main__':main()
