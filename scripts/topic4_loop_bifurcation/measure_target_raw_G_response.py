#!/usr/bin/env python3
"""Direct physical shunt response, including voltage and noise rescaling."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
import cupy as cp
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from audit_target_root_response import density_condition
from measure_target_direct_response import run
import lif_mc

OUT=ROOT/'target_raw_G_response'
PARENT=ROOT/'target_direct_response'
EG=-17.662847938268442


def kernel():
    code=lif_mc.CODE
    changes={
      'void assay(':'void physical_shunt(',
      'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else gf=sqrt(1+o);':
      'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else if(channel==2)gf=sqrt(1+o);',
      'if(ref[side]==0){v[side]=p[18]*v[side]+(1-p[18])*cur;':
      'double decay=p[18];if(channel==3 && o!=0.){double h=1.+p[22];cur=(h*cur-17.662847938268442*o)/(h+o);decay=exp(-.1*(h+o)/p[23]);}\n   if(ref[side]==0){v[side]=decay*v[side]+(1-decay)*cur;'}
    for before,after in changes.items():
        assert code.count(before)==1;code=code.replace(before,after)
    return cp.RawKernel(code,'physical_shunt',options=('--fmad=false',))


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    cp.cuda.Device(0).use();params=read(OPS/'prepared.json')['params'];source=read(PARENT/'result.json')['rows']
    specifications=read(ROOT/'target_root_response_audit/result.json')['rows']
    base=[q for q in specifications if q['kernel']=='density_discrete' and q['channel']=='conductance' and q['amplitude_factor']==1]
    rows=[];pars=[]
    for q in base:
        p0,_=density_condition(q['physical'][:3],q['physical'][3],q['threshold_mV'],'E',params)
        p0[20],p0[22],p0[23]=3,q['physical'][3],params['tau_m_E']
        for freq in [0.,1.,5.,20.,80.]:
            for factor in [1.,.5]:
                p=p0.copy();p[4]=.002*(1+p[22])*factor;p[5]=2*np.pi*freq/1000*.1;pars.append(p)
                rows.append(dict(cell=q['cell'],region=q['region'],physical=q['physical'],frequency_Hz=freq,
                    amplitude_factor=factor,amplitude=float(p[4]),channel='physical_applied_G'))
    pars=np.array(pars)
    write(OUT/'contract.json',dict(status='FROZEN_BEFORE_PHYSICAL_G_RESPONSE',created_epoch=time.time(),
        question='What local response must enter the G feedback stability calculation when G changes the full membrane conductance rather than only its decay time?',
        equation='At delta appliedg: vinf(t)=[(1+g0)*(mu+IA-IG)+delta_g*EG]/(1+g0+delta_g); membrane decay exp(-dt*(1+g0+delta_g)/tm). Original unscaled currents stay unchanged. This includes instantaneous rescaling of existing current noise, unlike modulating its innovation intensity.',
        design=dict(targets=len(base),conditions=len(rows),replicas=8192,record_ms=4000,burn_ms=1000,seed=928841,
            device=0,amplitude='.002*(1+g0) and half, frequencies0/1/5/20/80Hz'),
        check='DC must agree with the independently measured chain rule for mu,varianceE,varianceI,and fixed-effective-input g; include paired covariance of those four prior channels. This chain rule is not assumed valid at nonzero frequency. Half-amplitude test uses the same10%DC/15%at80Hz and2SEM rule.',
        limits='AppliedG=Z_i*Graw, so networkglobal response still needs multiplying by heldZ_i. No physical Z restoration is simulated here. Full spatial stability not yet computed.',
        producer_sha256=sha(__file__),parent_sha256=sha(PARENT/'result.json'),formal_bifurcation_allowed=False,rows=rows))
    k=kernel();check=pars[:8].copy();check[:,4]=0
    a=lif_mc.run(check,64,100,30,928849,device=0,batch=8);b=run(k,check,64,100,30,928849)
    assert np.array_equal(a,b)
    write(OUT/'implementation_qa.json',dict(status='PASS',zero_modulation_bitwise=True))
    np.savez_compressed(OUT/'inputs.npz',pars=pars)
    started=time.time();allout=[]
    for lo in range(0,len(rows),8):
        write(OUT/'progress.json',dict(status='MEASURING',pid=os.getpid(),completed=lo,total=len(rows),elapsed_s=time.time()-started))
        out=run(k,pars[lo:lo+8],8192,4000,1000,928841)
        np.savez_compressed(OUT/f'mc_{lo:04d}.npz',observed=out);allout.append(out)
    out=np.concatenate(allout);gains=[]
    for i,q in enumerate(rows):
        z=(out[i,:,0]+1j*out[i,:,1])/(4*q['amplitude']);mu=z.mean()
        q.update(gain=[float(mu.real),float(mu.imag)],complex_SEM=float(np.sqrt(np.mean(abs(z-mu)**2)/len(z))))
        gains.append(z)
    dc={q['cell']:q for q in rows if q['frequency_Hz']==0 and q['amplitude_factor']==1}
    prior_obs=np.concatenate([np.load(p)['observed'] for p in sorted(PARENT.glob('mc_*.npz'))])
    comparisons=[];sensitivity=[]
    for i,q in enumerate(rows):
        if q['frequency_Hz']==0 and q['amplitude_factor']==1:
            physical=q['physical'];h=1+physical[3]
            coefficients=[(EG-physical[0])/h,-2*physical[1]/h,-2*physical[2]/h,1.]
            vector=np.zeros(8192,dtype=complex)
            for c,name in zip(coefficients,['mean_mV','variance_E','variance_I','conductance']):
                j=next(j for j,p in enumerate(source) if p['cell']==q['cell'] and p['channel']==name and p['frequency_Hz']==0 and p['amplitude_factor']==1)
                vector+=c*(prior_obs[j,:,0]+1j*prior_obs[j,:,1])/(4*source[j]['amplitude'])
            mean=vector.mean();sem=float(np.sqrt(np.mean(abs(vector-mean)**2)/len(vector)))
            combined=float(np.hypot(sem,q['complex_SEM']));error=float(abs(complex(*q['gain'])-mean));tol=max(.1*abs(mean),2*combined,1e-7)
            comparisons.append(dict(cell=q['cell'],chain_rule_gain=[float(mean.real),float(mean.imag)],
                measured_gain=q['gain'],combined_SEM=combined,error=error,tolerance=tol,passed=bool(error<=tol)))
        if q['amplitude_factor']==.5:
            delta=gains[i]-gains[i-1];sem=float(np.sqrt(np.mean(abs(delta-delta.mean())**2)/len(delta)))
            tol=max((.1 if q['frequency_Hz']<=20 else .15)*abs(complex(*dc[q['cell']]['gain'])),2*sem,1e-7)
            sensitivity.append(dict(cell=q['cell'],frequency_Hz=q['frequency_Hz'],difference=float(abs(delta.mean())),paired_SEM=sem,tolerance=tol,passed=bool(abs(delta.mean())<=tol)))
    result=dict(status='COMPLETE_PHYSICAL_G_RESPONSE',rows=rows,independent_DC_chain_rule=comparisons,
        amplitude_sensitivity=sensitivity,elapsed_s=time.time()-started,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result)
    summary=dict(status=result['status'],DC_chain_pass=sum(q['passed'] for q in comparisons),DC_chain_total=len(comparisons),
        amplitude_pass=sum(q['passed'] for q in sensitivity),amplitude_total=len(sensitivity),elapsed_s=result['elapsed_s'])
    write(OUT/'progress.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
