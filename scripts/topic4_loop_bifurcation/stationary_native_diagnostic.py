#!/usr/bin/env python3
"""Bounded spatial steady-state correspondence diagnosis; NO stability claims.

Use the already statically validated v3 response. Compare only three existing
native termination cuts before investing in further dynamic response fitting.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from campaign import ROOT,NATIVE,read,write,sha
from conductance_static_v3 import Response
from audit_topic4_loop_conductance_response import load_models,normalized_input,SCALE
from nonlinear_rate_response import normalized_jacobian
from common import model
import run_topic4_loop_zk_conditional as native

OUT=ROOT/'stationary_native_diagnostic'
EG=-17.662847938268442
EK=-30.


class Stationary:
    def __init__(self):
        torch.set_num_threads(3);self.s=model(20);s=self.s
        assert s.prep['graph_identity']==read(NATIVE/'protocol.json')['identity']
        self.nets,self.bases,_=load_models()
        self.response=Response().double();self.response.load_state_dict(torch.load(ROOT/'conductance_static_v3/locked_model.pt',map_location='cpu',weights_only=False)['model']);self.response.eval()
        self.a,self.b,self.qa,self.qb=s.matrices();self.eye=sparse.eye(s.P,format='csr')
        self.weights=np.zeros(s.P);self.weights[s.E]=s.mean_weights
        geo=np.load(NATIVE/'geometry.npz');self.counts=geo['cell_e_counts']
        assert np.array_equal(np.bincount(s.geo['group_cell'][s.E],weights=s.sizes[s.E],minlength=400),self.counts)
        self.regional=np.zeros((3,s.P));pos=geo['positions_e'];centers=geo['centers_mm']
        m=[np.linalg.norm(pos-c,axis=1)<1.75 for c in centers];m.append(~(m[0]|m[1]))
        assert [int(x.sum()) for x in m]==geo['region_counts'][:3].tolist()
        for j,mask in enumerate(m):self.regional[j]=np.bincount(s.members[mask],minlength=s.P)/mask.sum()

    def set_fields(self,zbar,kbar):
        zz,kk=native.fields(zbar,kbar);self.s.set_Z_cells(zz,source='Common native t20 conditional family')
        self.K=np.zeros(self.s.P);self.K[self.s.E]=self.s.project(kk)[self.s.E]

    def moments(self,r):
        s=self.s;R=float(r@self.weights*1000.);G=30*np.clip((R-200.)/300.,0,1)
        dG=100*self.weights if 200.<R<500. else np.zeros(s.P)
        ZE=s.Z*s.E;g=ZE*G+self.K;den=1+g
        IE=s.tm*s.area[0]*(self.a@r)+s.private_mu
        II=s.tm*s.area[1]*(self.b@r)
        mu=(IE-s.Z*II-.5*s.E*r+ZE*G*EG+self.K*EK)/den
        ve=(s.tm*s.area[0]**2*(self.qa@r)+s.private_ve)/den**2
        vi=(s.Z**2*s.tm*s.area[1]**2*(self.qb@r))/den**2
        assert ve.min()>=0 and vi.min()>=0
        return np.c_[mu,ve,vi],g,G,dG

    def phi(self,physical,g):
        s=self.s;rate=np.zeros(s.P);grad=np.zeros((s.P,4))
        for pop,mask in [('E',s.E),('I',~s.E)]:
            pp=physical[mask];theta=s.theta[mask];gg=g[mask]
            f=np.zeros((len(pp),39));f[:,:3]=normalized_input(pp,theta)/SCALE
            ft=torch.tensor(f,requires_grad=True);baseline,bg=self.bases[pop].evaluate(pp,theta,True)
            parent=self.nets[pop].logits(ft,torch.tensor(baseline));pg=torch.autograd.grad(parent.sum(),ft)[0].detach().numpy()
            ell=parent.detach().numpy();du=normalized_jacobian(pp,theta)
            local=bg+pg[:,:3]*du;cg=np.zeros(len(pp))
            if pop=='E':
                u=torch.tensor(np.c_[f[:,:3],np.log1p(gg)/3.],requires_grad=True)
                residual=self.response.layers(u).squeeze(-1);deriv=torch.autograd.grad(residual.sum(),u)[0].detach().numpy()
                ell+=np.log1p(gg)+residual.detach().numpy();local+=deriv[:,:3]*du;cg=(1+deriv[:,3]/3.)/(1+gg)
            ref=2. if pop=='E' else 1.;maximum=1000/ref
            prob=torch.sigmoid(torch.tensor(ell+np.log(ref/.1))).numpy();hz=maximum*prob
            rate[mask]=hz/1000.;grad[mask,:3]=(hz*(1-prob)/1000.)[:,None]*local
            grad[mask,3]=hz*(1-prob)/1000.*cg
        return rate,grad

    def evaluate(self,r,jacobian=False):
        s=self.s;physical,g,G,dG=self.moments(r);value,grad=self.phi(physical,g)
        if not jacobian:return value-r
        h=1+g;ZE=s.Z*s.E
        A=sparse.diags(grad[:,0]/h*s.tm*s.area[0])@self.a-sparse.diags(grad[:,0]/h*s.Z*s.tm*s.area[1])@self.b
        A+=sparse.diags(grad[:,1]/h**2*s.tm*s.area[0]**2)@self.qa
        A+=sparse.diags(grad[:,2]/h**2*s.Z**2*s.tm*s.area[1]**2)@self.qb
        A-=sparse.diags(1+.5*s.E*grad[:,0]/h)
        u=grad[:,0]*ZE*(EG-physical[:,0])/h-2*ZE/h*(grad[:,1]*physical[:,1]+grad[:,2]*physical[:,2])+grad[:,3]*ZE
        return value-r,A.tocsc(),u,dG,physical,g,G

    def solve(self,r):
        r=r.copy();trace=[]
        for iteration in range(80):
            f,A,u,v,physical,g,G=self.evaluate(r,True);error=float(abs(f).max());trace.append(error)
            if error<1e-9:return r,True,trace
            lu=splu(A);a=lu.solve(-f);b=lu.solve(u);den=1+v@b
            if abs(den)<1e-12:return r,False,trace
            step=a-b*(v@a)/den;alpha=1.
            for back in range(30):
                trial=r+alpha*step
                if trial.min()>=-1e-10 and np.all(trial<1/self.s.ref):
                    trial=np.maximum(trial,0.)
                    if abs(self.evaluate(trial)).max()<error:r=trial;break
                alpha*=.5
            else:return r,False,trace
        return r,False,trace


def main():
    assert not (OUT/'contract.json').exists();OUT.mkdir(exist_ok=True)
    write(OUT/'contract.json',dict(status='BOUNDED_DIAGNOSTIC_NO_STABILITY',created_epoch=time.time(),source_sha256=sha(__file__),
        question='Before more generic dynamic fitting, can the statically validated response even reproduce spatial steady activity near the actual native termination cuts?',
        scope='Only Z.21,K6/9/12, two existing native histories each; at most six Newton solves,80iterations each. No continuation,no stability or bifurcation type.',
        response='Frozen staticv3 E response; same conditioned parent I. Native graphg20 and same Z/K field family projected to original935groups. Native stationary G=30clip((globalEHz-200)/300),etaM*M=.5r_per_ms,E-onlyG/K. Mean externaldrive nu from originaloperator; colored-private-current moment closure approximate.',
        distinction='A deterministic mean-input stationary candidate is compared descriptively to noisy finite-window native responses. A residual root only establishes the approximate equation has a solution; it does not certify native correspondence or stability.',
        reflection='Dynamicv4 passes means but fails independent gains includingDC. Stop genericMLP expansion. Diagnose actual spatial/domain correspondence before choosing any further local calibration; failures remain and no tolerances are relaxed.',
        readouts='WholeE,matched1.75mm coreobserver and400nativecells; high/quiet history, completedshortevents,censoring retained. Report allEinputdomain extrapolation. No networktarget fitting.'))
    e=Stationary();s=e.s;e.set_fields(.21,6.)
    # Same-equation derivatives only: not independent physiological validation.
    rng=np.random.default_rng(927501);qa=[]
    for scale in [.01,.45]:
        r=np.where(s.E,scale,.6*scale);v=r*rng.standard_normal(s.P)*.01
        f,A,u,w,*_=e.evaluate(r,True);pred=A@v+u*(w@v);h=1e-4
        measured=(e.evaluate(r+h*v)-e.evaluate(r-h*v))/(2*h)
        relative=float(np.linalg.norm(pred-measured)/max(np.linalg.norm(measured),1e-15));assert relative<2e-4,relative
        qa.append(dict(E_initial_rate_Hz=scale*1000,JVP_relative_error=relative))
    write(OUT/'implementation_check.json',dict(status='PASS',rows=qa,graph_identity=s.prep['graph_identity'],field_counts_exact=True))
    table={r['name']:r for r in read(NATIVE/'extended_analysis_summary.json')['rows']};rows=[];start=time.time()
    for k in [6.,9.,12.]:
        e.set_fields(.21,k)
        for history in ['high','recovery']:
            name=f'exit_z0.21_k{k:g}_{history}';native_row=table[name];field=np.array(native_row['tail_native_readouts']['mean_field_Hz'])
            counts=[]
            for path in sorted((NATIVE/'runs'/name/'chunks').glob('*.npz')):
                with np.load(path) as z:counts.append(z['spikes_1ms'][:,1])
            meanI=float(np.concatenate(counts)[-10000:].sum()/8000/10.)
            initial=np.full(s.P,max(meanI/1000.,1e-8));initial[s.E]=np.maximum(field[s.geo['group_cell'][s.E]]/1000.,1e-8)
            r,converged,trace=e.solve(initial);physical,g,G,_=e.moments(r)
            predicted=s.cell_field(r);regional=e.regional@r*1000.;q=physical[s.E];scale=s.theta[s.E]-11.
            x=(q[:,0]-11)/scale;se=np.sqrt(q[:,1])/scale;si=np.sqrt(q[:,2])/scale
            outside=(x < -10)|(x > 30)|(se>12)|(si>12)|(g[s.E]>32)
            np.savez_compressed(OUT/f'{name}.npz',rate_per_ms=r,physical=physical,g=g,Z=s.Z,K=e.K,predicted_field_Hz=predicted,native_field_Hz=field)
            row=dict(name=name,K=k,history=history,converged=converged,maximum_residual_Hz=trace[-1]*1000,iterations=len(trace),native_allE_Hz=native_row['tail_mean_Hz'][0],candidate_allE_Hz=float(r@e.weights*1000.),native_core_Hz=native_row['tail_mean_Hz'][1:3],candidate_core_Hz=regional[:2].tolist(),native_I_Hz=meanI,candidate_I_Hz=float(np.average(r[~s.E],weights=s.sizes[~s.E])*1000.),candidate_G_raw=G,weighted_field_MAE_Hz=float(np.average(abs(predicted-field),weights=e.counts)),weighted_field_RMS_Hz=float(np.sqrt(np.average((predicted-field)**2,weights=e.counts))),E_cell_fraction_outside_static_training_domain=float(outside@s.mean_weights),native_finite_window_state=native_row['finite_window_state'],native_brief_events=native_row['tail_brief_events'],trace=trace)
            rows.append(row);write(OUT/'progress.json',dict(status='DIAGNOSING',pid=os.getpid(),completed=len(rows),total=6,latest={k:v for k,v in row.items() if k!='trace'},elapsed_s=time.time()-start))
    write(OUT/'result.json',dict(status='DIAGNOSTIC_COMPLETE_NO_STABILITY',rows=rows,elapsed_s=time.time()-start,native_correspondence_certified=False,formal_bifurcation_allowed=False))
    print([{k:v for k,v in r.items() if k!='trace'} for r in rows],flush=True)


if __name__=='__main__':main()
