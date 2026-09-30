#!/usr/bin/env python3
"""Independent local test of a bounded conductance-aware hazard approximation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import hashlib
import time
from pathlib import Path
import numpy as np
from numba import njit, prange, set_num_threads

REPO=Path(__file__).resolve().parents[1]
MECHANISM=REPO/'scripts/topic4_zm_runaway_mechanism'
sys.path.insert(0,str(MECHANISM))
from common import OUT as OLD_OUT
from conditioned_refractory_rate import load_models
from nonlinear_rate_response import normalized_input, SCALE
from lif_mc import condition, PARAMS
import torch

OUT=Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/conductance_response')


def write(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


@njit(parallel=True,cache=True)
def monte_carlo(pars, conductances, replicates, steps, burn, seed):
    output=np.empty((len(pars),replicates),np.int32)
    for index in prange(len(pars)*replicates):
        case=index//replicates;rep=index%replicates;p=pars[case]
        # Common stochastic histories across conductance and input conditions.
        np.random.seed(seed+rep)
        qa=0.;ia=0.;qg=0.;ig=0.;v=p[21];ref=0;count=0
        decay=np.exp(-.1*(1+conductances[case])/20.)
        for step in range(-burn,steps):
            n=np.random.standard_normal(4)
            ia=p[7]*qa+p[8]*ia+p[12]*n[0]+p[13]*n[1]
            qa=p[6]*qa+p[11]*n[0]
            ig=p[9]*qg+p[10]*ig+p[15]*n[2]+p[16]*n[3]
            qg=p[17]*qg+p[14]*n[2]
            current=p[0]+ia-ig
            ref=max(0,ref-1)
            if ref==0:
                v=current+(v-current)*decay
                if v>=p[1]:
                    v=p[21];ref=int(p[19])
                    if step>=0:count+=1
            else:
                v=p[21]
        output[case,rep]=count
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'result.json').exists(), 'Finished local assay is immutable'
    set_num_threads(8);torch.set_num_threads(1)
    assert PARAMS['tau_m_E']==20. and PARAMS['V_reset']==11.
    rows=[];pars=[]
    for g in [0.,.5,2.,8.]:
        for x in [-.5,.5,1.2,3.]:
            for se,si in [(.5,1.),(2.,3.),(5.,2.)]:
                mu=11+7*x;ve=(7*se)**2;vi=(7*si)**2
                rows.append(dict(g=g,x=x,sigmaE=se,sigmaI=si,mu_effective_mV=mu,
                                 effective_ve=ve,effective_vi=vi,tau_m_effective_ms=20/(1+g)))
                pars.append(condition(mu,18.,ve,vi,'E'))
    pars=np.array(pars);g=np.array([row['g'] for row in rows])
    physical=pars[:,[0,2,3]]
    nets,bases,locked=load_models()
    f=np.zeros((len(pars),39));f[:,:3]=normalized_input(physical,18.)/SCALE
    with torch.no_grad():
        ell=nets['E'].logits(torch.tensor(f),torch.tensor(bases['E'].evaluate(physical,18.))).numpy()
    # Native discrete refractory ordering: ISI = ref + dt*exp(-logit).
    ordinary=1000/(2.+.1*np.exp(-ell))
    proposed=1000/(2.+.1*np.exp(-ell)/(1+g))
    assert np.array_equal(ordinary[g==0],proposed[g==0])
    predictions=dict(rows=[dict(**r,old_effective_current_rate_Hz=float(a),
                                  conductance_hazard_rate_Hz=float(b)) for r,a,b in zip(rows,ordinary,proposed)],
                     model_weights=locked['files'],producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write('predictions_locked.json',predictions)
    write('contract.json',dict(status='LOCKED_BEFORE_FRESH_TARGETS',created_epoch=time.time(),
        question='Can the existing local response be extended to conductance by effective-current conversion and intrinsic hazard time compression?',
        unit='One synthetic colored-LIF input condition;512 independent noise replicas. Not SNN seeds or patient observations.',
        physical_mean='mu_eff=(I_E-ZI_I-etaM*M+G*EG+K*EK)/(1+G+K)',
        physical_noise='Both effective current channels divide by1+G+K, so covariance intensities divide by its square. Input channel filters keep native physical time constants.',
        hypothesis='lambda_g=(1+G+K)*lambda_parent(mu_eff,vE_eff,vI_eff); unchanged absolute refractory time. Exact deterministic integration-time scaling motivates but does not prove noisy validity.',
        equation_test='Native exponential membrane step with tau_m/(1+g), same exact colored-current covariance update and reset/refractory ordering. Constant g isolates conductance response; not full dynamic-G/K or native-network correspondence.',
        design=dict(g=[0,.5,2,8],normalized_mean=[-.5,.5,1.2,3],normalized_noise_pairs=[[.5,1],[2,3],[5,2]],
                    replicates=512,dt_ms=.1,burn_ms=500,record_ms=2000,noise_seed=920924),
        gate='Each condition absolute mean-rate error <=max(2Hz,10%MC mean,3MC SEM). Report g0 baseline failures separately. Fixed approximation, no fit or tolerance adjustment after scoring.',
        next='If any g>0 condition fails, do not install this approximation for formal branches. Inspect error by g/input; prepare a conductance-calibrated response. Even a local pass requires transient and native spatial validation.',
        parent=str(OLD_OUT/'conditioned_refractory_rate'),new_training=False))
    start=time.time();counts=monte_carlo(pars,g,512,20000,5000,920924)
    rates=counts/2.
    mean=rates.mean(1);sem=rates.std(1,ddof=1)/np.sqrt(rates.shape[1])
    tolerance=np.maximum.reduce([np.full(len(mean),2.),.1*mean,3*sem])
    np.savez_compressed(OUT/'local_assay.npz',counts=counts,pars=pars,g=g,mean_Hz=mean,sem_Hz=sem,
                        old_prediction_Hz=ordinary,proposed_prediction_Hz=proposed)
    final=[]
    for i,row in enumerate(predictions['rows']):
        final.append(dict(**row,measured_Hz=float(mean[i]),SEM_Hz=float(sem[i]),tolerance_Hz=float(tolerance[i]),
                          error_Hz=float(proposed[i]-mean[i]),pass_condition=bool(abs(proposed[i]-mean[i])<=tolerance[i])))
    positive=[r for r in final if r['g']>0];zero=[r for r in final if r['g']==0]
    result=dict(status='LOCAL_STATIC_PASS' if all(r['pass_condition'] for r in final) else 'LOCAL_STATIC_FAIL',
                g_positive_pass=sum(r['pass_condition'] for r in positive),g_positive_total=len(positive),
                g_zero_pass=sum(r['pass_condition'] for r in zero),g_zero_total=len(zero),rows=final,
                seconds=time.time()-start,full_spatial_correspondence=False,formal_bifurcation_allowed=False)
    write('result.json',result)
    lines=['# G/K电导率响应的首轮局部检验','','此处检验的是一个明确的响应近似，不是把原生闭环换成拟合曲线。保持等效输入均值和噪声幅度相同，改变总电导以单独检验膜时间常数效应。', '',
           f"新增电导条件通过{result['g_positive_pass']}/{len(positive)}，零电导原模型通过{result['g_zero_pass']}/{len(zero)}。本轮没有拟合新系数；完整空间对应与分岔仍未认证。",'',
           '|g/gL|x|σE|σI|MC率Hz|预测Hz|误差Hz|通过|','|---|---|---|---|---|---|---|---|']
    for r in final:
        lines.append(f"|{r['g']}|{r['x']}|{r['sigmaE']}|{r['sigmaI']}|{r['measured_Hz']:.2f}|{r['conductance_hazard_rate_Hz']:.2f}|{r['error_Hz']:.2f}|{r['pass_condition']}|")
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
