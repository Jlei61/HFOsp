"""CPU/GPU and stationary identities for the experimental time integrator."""
from common import OUT, np, read, write, log
from onset_exponential_midpoint import ExponentialMidpointEngine
from fine_rate_frozen_Z_fields import capture, restore
from conditioned_refractory_rate import load_models, covariance_matrices
from nonlinear_rate_response import physical_from_features, normalized_input, SCALE
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from pathlib import Path
import argparse, os, gc

DEST=OUT/'core_a_bifurcation_type_20260924/numerical_checks/exponential_midpoint'
REF=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'


def cpu_step(e, base, arr):
    s=e.s; dt=e.dt; syn=base['syn']; old=base['local']; hist=base['history']; tick=int(base['clock'][0])
    pars=base['parameters']; con=e.transport.consts.get(); raw=np.array([
        s.tm*s.area[0]**2*arr[2]+pars[7], s.tm*s.area[1]**2*arr[3]])
    def synaptic(h):
        result=syn.copy()
        for c,(tr,td) in enumerate(zip(s.rise,s.decay)):
            a=np.exp(-h/tr); d=np.exp(-h/td); b=tr/(tr-td)*(a-d)
            force=s.tm*s.area[c]*arr[c]+(pars[6] if c==0 else 0.)
            result[2*c]=a*syn[2*c]+(1-a)*force
            result[2*c+1]=b*syn[2*c]+d*syn[2*c+1]+(1-d-b)*force
        return result
    mid=synaptic(dt/2); em=np.exp(-dt/(2*con[6]))
    mid[4]=em*syn[4]+(1-em)*.5*s.E*hist[tick%len(hist)]
    mean=mid[1]-syn[5]*mid[3]-mid[4]; localmid=old.copy(); result=old.copy(); u=np.zeros((3,s.P))
    for pop,mask in [('E',s.E),('I',~s.E)]:
        A,B,C,_,_=covariance_matrices(pop,dt/2)
        fullA,fullB,_,_,_=covariance_matrices(pop,dt)
        for c in range(2):
            localmid[3*c:3*c+3,mask]=A[c]@old[3*c:3*c+3,mask]+B[c,:,None]*raw[c,mask]
            result[3*c:3*c+3,mask]=fullA[c]@old[3*c:3*c+3,mask]+fullB[c,:,None]*raw[c,mask]
        scale=s.theta[mask]-11
        u[:,mask]=[np.arcsinh((mean[mask]-11)/scale)/3,
                   np.log1p(C[0]*localmid[2,mask]/scale**2)/2,
                   np.log1p(C[1]*localmid[5,mask]*syn[5,mask]**2/scale**2)/2]
    f=np.zeros((s.P,39)); f[:,:3]=u.T
    for c in range(3):
        for j,tau in enumerate([1.,4.,16.,64.]):
            k=6+c*12+j*3; h1,h2,h3=old[k:k+3]
            for h,target in [(dt/2,localmid),(dt,result)]:
                a=h/tau; dec=np.exp(-a)
                target[k]=dec*h1+(1-dec)*u[c]
                target[k+1]=dec*(h2+a*h1)+(1-dec*(1+a))*u[c]
                target[k+2]=dec*(h3+a*h2+.5*a*a*h1)+(1-dec*(1+a+.5*a*a))*u[c]
            f[:,3+c*12+j*3:6+c*12+j*3]=(localmid[k:k+3]-u[c]).T
    import torch
    nets,bases,_=load_models(); correction=read(OUT/'transient_response_correction/locked.json')['parameters']
    r=np.empty(s.P); hazards=np.empty(s.P)
    for pop,mask in [('E',s.E),('I',~s.E)]:
        logits=bases[pop].evaluate(physical_from_features(f[mask],s.theta[mask]),s.theta[mask])
        with torch.no_grad(): ell=nets[pop].logits(torch.tensor(f[mask]),torch.tensor(logits)).numpy()
        H=np.mean(f[mask,3:]**2,axis=1); c=correction[pop]; ell+=c['b']*H/(H+c['h'])
        rho=np.exp(ell)/.1; hazards[mask]=rho; nref=round(float(s.ref[mask][0])/dt)
        occupied=dt*hist[(tick+1-np.arange(1,nref+1))%len(hist)][:,mask].sum(0)
        release=hist[(tick+1-nref)%len(hist),mask]; p=-np.expm1(-rho*dt)
        r[mask]=(1-occupied)*p/dt+release*(1-p/(rho*dt))
    end=synaptic(dt); em=np.exp(-dt/con[6]); end[4]=em*syn[4]+(1-em)*.5*s.E*r
    return dict(syn=end,local=result,rate=r,mid_syn=mid,mid_local=localmid,mid_features=u,hazard=hazards)


def stationary(e):
    z=np.load(OUT/'physical_delay_conditional_drift_interface/baseline_equilibrium.npz'); r=z['r']; Z=z['Z']
    s=PhysicalDelayConditionalDrift(); s.set_Z(Z); a,b,qa,qb=s.matrices()
    F=s.tm*s.area[0]*(a@r)+s.private_mu; G=s.tm*s.area[1]*(b@r)
    ve=s.tm*s.area[0]**2*(qa@r)+s.private_ve; vi=s.tm*s.area[1]**2*(qb@r)
    syn=np.array([F,F,G,G,.5*s.E*r,Z]); inputs=np.column_stack([F-Z*G-syn[4],ve,Z*Z*vi])
    u=normalized_input(inputs,s.theta)/SCALE; local=np.zeros((42,s.P))
    for pop,mask in [('E',s.E),('I',~s.E)]:
        A,B,_,_,_=covariance_matrices(pop,e.dt)
        for c,v in enumerate([ve,vi]):
            local[3*c:3*c+3,mask]=np.linalg.solve(np.eye(3)-A[c],B[c])[:,None]*v[mask]
    for c in range(3): local[6+c*12:6+(c+1)*12]=u[:,c]
    e.reset(); e.syn[:]=e.cp.asarray(syn); e.local.state[:]=e.cp.asarray(local)
    e.local.history[:]=e.cp.asarray(r); e.local.rate[:]=e.cp.asarray(r); e.emitted[:]=e.cp.asarray(r)
    e.cp.cuda.get_current_stream().synchronize(); start=capture(e)
    for _ in range(round(2/e.dt)): e.step()
    e.cp.cuda.get_current_stream().synchronize(); end=capture(e)
    errors=dict(rate=float(abs(e.local.rate.get()-r).max()),syn=float(abs(end['syn']-syn).max()),
                local=float(abs(end['local']-local).max()))
    assert errors['rate']<1e-9 and max(errors.values())<1e-8, errors
    return errors


def main(device):
    assert read(DEST/'scalar_renewal_check.json')['status']=='MANUFACTURED_SCALAR_CHECK_COMPLETE'
    assert not (DEST/'spatial_jobs.json').exists(); jobs=dict(status='RUNNING',pid=os.getpid()); write(DEST/'spatial_jobs.json',jobs)
    rows=[]
    try:
        for dt,folder in [(.05,'matched_natural_short_entry_step'),(.025,'matched_natural_short_entry_step_fine')]:
            e=ExponentialMidpointEngine(dt=dt,device=device); e.graph()
            source=REF/folder/'held_short_field/checkpoint1000.npz'; base=dict(np.load(source)); restore(e,base)
            e.arrivals(); e.cp.cuda.get_current_stream().synchronize(); arr=e.transport.arr.get()
            expected=cpu_step(e,base,arr); e.step(); e.cp.cuda.get_current_stream().synchronize(); got=capture(e)
            errors=dict(syn=float(abs(got['syn']-expected['syn']).max()),
                local=float(abs(got['local']-expected['local']).max()),rate_hz=float(abs(got['rate']-expected['rate']).max()*1000),
                mid_syn=float(abs(e.mid_syn.get()-expected['mid_syn']).max()),
                mid_local=float(abs(e.mid_local.get()-expected['mid_local']).max()),
                mid_features=float(abs(e.mid_features.get()-expected['mid_features']).max()))
            assert errors['rate_hz']<1e-5 and max(v for k,v in errors.items() if k!='rate_hz')<1e-8, errors
            assert np.isfinite(got['rate']).all() and got['rate'].min()>=0
            assert np.array_equal(got['syn'][5],base['syn'][5])
            equilibrium=stationary(e)
            row=dict(dt_ms=dt,source=str(source),CPU_GPU_errors=errors,equilibrium_errors=equilibrium,
                     hazard_per_ms_quantiles=np.quantile(expected['hazard'],[0,.5,.9,.99,1]).tolist())
            rows.append(row); write(DEST/'spatial_identity_progress.json',rows); log('MIDPOINT IDENTITY',row)
            cp=e.cp; del e; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
        write(DEST/'spatial_identity.json',dict(status='PASS',rows=rows,
              scope='CPU/GPU numerical implementation and original stationary physical-law identity only. Full trajectory convergence and bifurcation remain untested.',model_promoted=False))
        jobs['status']='COMPLETE'; write(DEST/'spatial_jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc)); write(DEST/'spatial_jobs.json',jobs); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device',type=int,default=0); main(p.parse_args().device)
