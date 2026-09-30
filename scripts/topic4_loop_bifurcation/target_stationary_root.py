#!/usr/bin/env python3
"""Bounded equilibrium solve at the repaired, actual-exit-field K9 point.

Individual target adaptation is solved locally, then target rates are averaged
back to source groups. This is a static-surrogate root, not a stability result.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
from types import SimpleNamespace
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,gmres
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from stationary_native_diagnostic import Stationary,Response,load_models

AUDIT=ROOT/'target_stationary_response_audit'
OUT=ROOT/'target_stationary_root'


class TargetStationary:
    def __init__(self):
        torch.set_num_threads(3)
        self.geo=dict(np.load(OPS/'geometry.npz'));self.p=read(OPS/'prepared.json')['params']
        self.group=self.geo['cell_group'];self.P=len(self.geo['group_size']);self.N=len(self.group)
        self.E=np.arange(self.N)<32000;self.groupE=self.geo['population']==0;self.sizes=self.geo['group_size']
        self.S=sparse.coo_matrix((1/self.sizes[self.group],(self.group,np.arange(self.N))),shape=(self.P,self.N)).tocsr()
        self.W=[sparse.load_npz(AUDIT/f'{name}_dc.npz').tocsr() for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
        with np.load(AUDIT/'response.npz') as z:
            self.Z=z['Z'];self.K=z['K'];self.nu=z['mean_external_rate_per_ms'];self.initial=z['mean_source_rate_per_ms']
            self.initial_cell_rate=np.where(self.E,z['M']/1000.,z['predicted_target_rate_Hz']/1000.)
        self.tm=np.where(self.E,self.p['tau_m_E'],self.p['tau_m_I']);self.jext=np.where(self.E,self.p['J_ext_E'],self.p['J_ext_I'])
        self.area=np.array([.1/(self.p[n]*(1-np.exp(-.1/self.p[n]))) for n in ['tau_r_AMPA','tau_r_GABA']])
        self.causal_factor=.1/(15*(-np.expm1(-.1/15)))
        self.global_weights=np.where(self.groupE,self.sizes/32000.,0.)
        self.cap=np.where(self.groupE,.5,1.)
        self.model=Stationary.__new__(Stationary);self.model.s=SimpleNamespace(P=self.N,E=self.E,theta=self.geo['threshold_mv'][self.group])
        self.model.nets,self.model.bases,_=load_models();self.model.response=Response().double()
        self.model.response.load_state_dict(torch.load(ROOT/'conductance_static_v3/locked_model.pt',map_location='cpu',weights_only=False)['model']);self.model.response.eval()

    def evaluate(self,r,jacobian=False):
        assert np.min(r)>=0
        R=float(self.global_weights@r*1000*self.causal_factor);G=30*np.clip((R-200)/300,0,1)
        ZE=self.E*self.Z;g=ZE*G+self.K;h=1+g
        IE=self.tm*self.area[0]*(self.W[0]@r+self.jext*self.nu);II=self.tm*self.area[1]*(self.W[1]@r)
        mu0=(IE-self.Z*II+ZE*G*(-17.662847938268442)-30*self.K)/h
        ve=self.tm*self.area[0]**2*(self.W[2]@r+self.jext**2*self.nu)/h**2
        vi=self.tm*(self.Z*self.area[1])**2*(self.W[3]@r)/h**2
        c=.5*self.E/h;cell=self.initial_cell_rate.copy()
        for iteration in range(12):
            physical=np.c_[mu0-c*cell,ve,vi];rate,grad=self.model.phi(physical,g)
            den=1+c*grad[:,0];assert den.min()>.5
            f=rate-cell;err=float(abs(f).max())
            if err<1e-12:break
            cell=np.clip(cell+f/den,0,np.where(self.E,.5,1.))
        else:raise RuntimeError(f'Individual adaptation did not converge: {err}')
        value=self.S@cell-r
        meta=dict(cell_rate=cell,physical=physical,g=g,G=float(G),R_causal=float(R),adaptation_residual_per_ms=err,
                  local_iterations=iteration+1,min_local_adaptation_den=float(den.min()))
        if not jacobian:return value,meta
        coeff=np.array([grad[:,0]*self.tm*self.area[0]/h,
             -grad[:,0]*self.Z*self.tm*self.area[1]/h,
             grad[:,1]*self.tm*self.area[0]**2/h**2,
             grad[:,2]*self.tm*(self.Z*self.area[1])**2/h**2])/den
        u=(grad[:,0]*ZE*(-17.662847938268442-physical[:,0])/h
             -2*ZE/h*(grad[:,1]*ve+grad[:,2]*vi)+grad[:,3]*ZE)/den
        dG=100*self.causal_factor*self.global_weights if 200<R<500 else np.zeros(self.P)
        def jvp(v):
            local=sum(a*(w@v) for a,w in zip(coeff,self.W))+u*(dG@v)
            return self.S@local-v
        diagonal=-np.ones(self.P)
        # Target's source-group column gives the diagonal of S D W exactly.
        for a,w in zip(coeff,self.W):
            q=np.asarray(w[np.arange(self.N),self.group]).ravel()
            diagonal+=self.S@(a*q)
        diagonal+=(self.S@u)*dG
        safe=np.where(abs(diagonal)>.05,diagonal,-1.)
        return value,LinearOperator((self.P,self.P),matvec=jvp,dtype=float),LinearOperator((self.P,self.P),matvec=lambda v:v/safe,dtype=float),meta


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(AUDIT/'result.json')['status']=='COMPLETE_ONE_POINT_RESPONSE_AUDIT'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_ACTUAL_K9_ROOT',created_epoch=time.time(),
        question='Does the target-resolved static surrogate have a selfconsistent high equilibrium near the corresponding actual-field K9 state?',
        design='One workingpoint Zmean.21/Kmean9, originalactual16.7s fields, source3479groups with40000individualtargetZ/K/weights. Solve local steadyM_i=1000*r_i selfconsistently; externalinput is the completed5-10s exactpaired mean. G includes exact discrete15mscausalR DC factor.',
        algorithm='At most30Newtonsteps, matrixfreeGMRES200inneriterations withdiagonalpreconditioner, boundedpositive line search12trials. JVP checked againstsame-equation centraldifferences before solving. Residual max<1e-6Hz needed for a numerical root.',
        interpretation='Frozenv3staticEresponse andparentI. Staticworkingpointaudit showedsmallgroupmeanerrors butdynamic/DCvalidationnotcertified. Root,evenifconverged,isnotstability,nativebistability,orformalbifurcation.',
        stop='No automatic K sweep, newmodel fit, rootretry, or stability naming. Compare actualspatialfield/globalfeedback/domain and retainfailures.',
        model_sha256=sha(ROOT/'conductance_static_v3/locked_model.pt'),producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    started=time.time();e=TargetStationary();r=np.maximum(e.initial,1e-12)
    f,J,pre,meta=e.evaluate(r,True);rng=np.random.default_rng(928791);qa=[]
    for seed_scale in [1.,.1]:
        v=(r+1e-9)*rng.standard_normal(e.P)*.01*seed_scale;h=1e-4
        # Keep central perturbations in the nonnegative physical-rate domain.
        v=np.where(r<1e-10,0,v)
        plus,_=e.evaluate(r+h*v);minus,_=e.evaluate(r-h*v);measured=(plus-minus)/(2*h)
        predicted=J@v;relative=float(np.linalg.norm(predicted-measured)/max(np.linalg.norm(measured),1e-15))
        assert relative<3e-4,relative;qa.append(dict(scale=seed_scale,JVP_relative_error=relative))
    write(OUT/'implementation_qa.json',dict(status='PASS',rows=qa,causal_factor=float(e.causal_factor),
         local_adaptation_residual=meta['adaptation_residual_per_ms']))
    history=[];converged=False;reason='Iteration limit'
    for iteration in range(30):
        f,J,pre,meta=e.evaluate(r,True);error=float(abs(f).max()*1000)
        row=dict(iteration=iteration,max_residual_Hz=error,mean_E_Hz=float(e.global_weights@r*1000),G=meta['G'])
        history.append(row);write(OUT/'progress.json',dict(status='SOLVING',pid=os.getpid(),history=history,elapsed_s=time.time()-started))
        print('TARGET STATIC NEWTON',row,flush=True)
        if error<1e-6:converged=True;reason='Residual converged';break
        errors=[]
        step,info=gmres(J,-f,M=pre,rtol=min(.05,max(1e-7,error*.001)),atol=1e-12,restart=100,maxiter=2,
                       callback=lambda q:errors.append(float(q)),callback_type='pr_norm')
        row.update(gmres_info=int(info),gmres_iterations=len(errors),gmres_last_residual=errors[-1] if errors else None)
        if not np.isfinite(step).all():reason='Nonfinite Newton step';break
        norm=np.linalg.norm(f);alpha=1.;accepted=False
        for trial in range(12):
            candidate=np.clip(r+alpha*step,0,e.cap*(1-1e-12));test,_=e.evaluate(candidate)
            if np.linalg.norm(test)<norm*(1-1e-4*alpha):
                r=candidate;row['accepted_alpha']=alpha;accepted=True;break
            alpha*=.5
        if not accepted:reason='Bounded line search did not decrease residual';break
    residual,meta=e.evaluate(r);cell=meta['cell_rate'];geo=e.geo;regions=geo['group_region']
    groups=e.S@cell;regional=[]
    for label,mask in [('allE',e.groupE),('coreA',e.groupE&(regions==0)),('coreB',e.groupE&(regions==1)),('surround',e.groupE&(regions==2)),('I',~e.groupE)]:
        regional.append(dict(region=label,rate_Hz=float(np.average(groups[mask],weights=e.sizes[mask])*1000)))
    np.savez_compressed(OUT/'root_candidate.npz',source_rate_per_ms=r,cell_rate_per_ms=cell,residual_per_ms=residual,
                        physical=meta['physical'],g=meta['g'],Z=e.Z,K=e.K,G=meta['G'])
    result=dict(status='NUMERICAL_ROOT' if converged else 'BOUNDED_ROOT_NOT_CONVERGED',reason=reason,
       maximum_residual_Hz=float(abs(residual).max()*1000),regional=regional,G=meta['G'],causal_R_Hz=meta['R_causal'],
       history=history,elapsed_s=time.time()-started,root_is_static_surrogate_only=True,
       stability_established=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result);print(result,flush=True)


if __name__=='__main__':main()
