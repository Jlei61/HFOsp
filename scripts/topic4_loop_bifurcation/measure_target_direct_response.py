#!/usr/bin/env python3
"""Direct frequency response of the validated density cell update, no rate fit."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import time
import numpy as np
import cupy as cp
from campaign import ROOT, read, write, sha
from audit_target_root_response import density_condition
from conditional_density_inputs import OPS
import lif_mc

OUT = ROOT / 'target_direct_response'
PARENT = ROOT / 'target_root_response_audit'


def kernel():
    code = lif_mc.CODE
    replacements = {
        'void assay(': 'void direct_response(',
        'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else gf=sqrt(1+o);':
        'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else if(channel==2)gf=sqrt(1+o);',
        'if(ref[side]==0){v[side]=p[18]*v[side]+(1-p[18])*cur;':
        'double decay=channel==3?exp(-.1*(1+p[22]+o)/p[23]):p[18];\n   if(ref[side]==0){v[side]=decay*v[side]+(1-decay)*cur;'
    }
    for before, after in replacements.items():
        assert code.count(before) == 1
        code = code.replace(before, after)
    return cp.RawKernel(code, 'direct_response', options=('--fmad=false',))


def run(k, pars, replicas, duration, burn, seed):
    n = len(pars) * replicas
    out = cp.zeros((n, 4))
    k(((n+127)//128,), (128,), (cp.asarray(pars), out, np.int32(replicas), np.int32(len(pars)),
        np.int32(round(duration/.1)), np.int32(round(burn/.1)), np.uint64(seed), np.int32(1)))
    return out.get().reshape(len(pars), replicas, 4)


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'contract.json').exists()
    cp.cuda.Device(0).use()
    params = read(OPS/'prepared.json')['params']
    source = read(PARENT/'result.json')
    base = [q for q in source['rows'] if q['kernel']=='density_discrete' and q['amplitude_factor']==1.]
    rows, pars = [], []
    for q in base:
        channel = ['mean_mV', 'variance_E', 'variance_I', 'conductance'].index(q['channel'])
        p0, _ = density_condition(q['physical'][:3], q['physical'][3], q['threshold_mV'], q['population'], params)
        p0[20], p0[22], p0[23] = channel, q['physical'][3], params['tau_m_'+q['population']]
        for frequency in [0., 1., 5., 20., 80.]:
            for factor in [1., .5]:
                p = p0.copy()
                p[4] = q['amplitude']*factor/(p[2 if channel==1 else 3] if channel in [1, 2] else 1.)
                p[5] = 2*np.pi*frequency/1000*.1
                pars.append(p)
                rows.append(dict(cell=q['cell'], population=q['population'], region=q['region'],
                    channel=q['channel'], frequency_Hz=frequency, amplitude=q['amplitude']*factor,
                    amplitude_factor=factor, physical=q['physical'],
                    old_static_predicted_gain_Hz=q['predicted_gain_Hz']))
    pars = np.array(pars)
    write(OUT/'contract.json', dict(status='FROZEN_BEFORE_DIRECT_RESPONSE', created_epoch=time.time(),
        question='What is the actual local susceptibility of the corresponding target-density cell dynamics, without inheriting the failed neural static/history derivative?',
        design=dict(conditions=len(rows), physical_targets=len(set(q['cell'] for q in rows)),
                    frequencies_Hz=[0,1,5,20,80], replicas=8192, duration_ms=4000, burn_ms=1000,
                    seed=928821, device=0, amplitudes='Same actual-input perturbations as the DC audit, plus half-amplitude at every frequency'),
        model='Original .1ms membrane/refractory update with fixed rootM/Z/K and exact density synaptic AR coefficients. Mean and variance channels perturb effective moments; conductance channel holds effective moments fixed and changes the membrane decay each step. NetworkG/M feedback must be reintroduced through their equations, not silently included here.',
        method='Paired +/- sinusoidal inputs with common independent-replica noise; complex demodulation. No learned predictor or network fitting. Zero-frequency responses independently repeat the fresh DC audit. A frequency family supplies local response data, not the full spatial characteristic determinant.',
        gate='Keep all nonestimable channels. DC SNR>=10; full/half-amplitude complex difference <=max(10%abs(DC),2pairedSEM,1e-7) through20Hz and15%at80Hz. Independent DC replication <=max(10%priorDC,2combinedSEM,1e-7). No dynamic pass inferred from an accurate mean.',
        native_engine_modified=False, producer_sha256=sha(__file__), parent_sha256=sha(PARENT/'result.json'),
        formal_bifurcation_allowed=False, rows=rows))
    k = kernel()
    # The unchanged mean/variance channels must reproduce the established kernel bitwise.
    check = pars[np.array([i for i,p in enumerate(pars) if p[20]<3])[:12]].copy()
    expected = lif_mc.run(check, 64, 100, 30, 928829, device=0, batch=12)
    observed = run(k, check, 64, 100, 30, 928829)
    assert np.array_equal(expected, observed)
    # With zero modulation, the new conductance channel is the same constant-g update.
    cg = pars[np.array([i for i,p in enumerate(pars) if p[20]==3])[:8]].copy();cg[:,4]=0.
    expected = lif_mc.run(cg, 64, 100, 30, 928829, device=0, batch=8)
    observed = run(k, cg, 64, 100, 30, 928829)
    assert np.array_equal(expected, observed)
    write(OUT/'implementation_qa.json', dict(status='PASS', original_channels_bitwise=True, constant_g_bitwise=True,
         density_filter_covariance_checked_in_parent=True))
    np.savez_compressed(OUT/'inputs.npz', pars=pars)
    started = time.time();allout=[]
    for lo in range(0,len(rows),8):
        write(OUT/'progress.json', dict(status='MEASURING', pid=os.getpid(), completed=lo,total=len(rows),elapsed_s=time.time()-started))
        out=run(k,pars[lo:lo+8],8192,4000,1000,928821)
        np.savez_compressed(OUT/f'mc_{lo:04d}.npz',observed=out);allout.append(out)
    out=np.concatenate(allout);gains=[]
    for i,row in enumerate(rows):
        z=(out[i,:,0]+1j*out[i,:,1])/(4*row['amplitude']);mean=z.mean()
        sem=float(np.sqrt(np.mean(abs(z-mean)**2)/len(z)))
        row.update(gain=[float(mean.real),float(mean.imag)],complex_SEM=sem,
            mean_rate_Hz=float(out[i,:,2:].mean()/4))
        gains.append(z)
    dc={(q['cell'],q['channel']):q for q in rows if q['frequency_Hz']==0 and q['amplitude_factor']==1}
    prior={(q['cell'],q['channel']):q for q in base};sens=[];replication=[]
    for i,row in enumerate(rows):
        reference=dc[row['cell'],row['channel']];d=abs(complex(*reference['gain']))
        row['DC_estimable']=bool(d/max(reference['complex_SEM'],1e-15)>=10)
        if row['frequency_Hz']==0 and row['amplitude_factor']==1:
            old=prior[row['cell'],row['channel']]
            combined=float(np.hypot(row['complex_SEM'],old['SEM']))
            tolerance=max(.1*abs(old['measured_gain_Hz']),2*combined,1e-7)
            error=abs(complex(*row['gain'])-old['measured_gain_Hz'])
            replication.append(dict(cell=row['cell'],channel=row['channel'],error=float(error),
                tolerance=tolerance,combined_SEM=combined,passed=bool(error<=tolerance)))
        if row['amplitude_factor']==.5:
            delta=gains[i]-gains[i-1];sem=float(np.sqrt(np.mean(abs(delta-delta.mean())**2)/len(delta)))
            tolerance=max((.1 if row['frequency_Hz']<=20 else .15)*d,2*sem,1e-7)
            sens.append(dict(cell=row['cell'],channel=row['channel'],frequency_Hz=row['frequency_Hz'],
                difference=float(abs(delta.mean())),paired_SEM=sem,tolerance=tolerance,
                passed=bool(abs(delta.mean())<=tolerance)))
    result=dict(status='COMPLETE_DIRECT_RESPONSE',rows=rows,independent_DC_replication=replication,
        amplitude_sensitivity=sens,elapsed_s=time.time()-started,formal_bifurcation_allowed=False,
        interpretation='Measured local susceptibility only; no spatial eigenvalues or stable/unstable labels.')
    write(OUT/'result.json',result)
    summary=dict(status=result['status'],elapsed_s=result['elapsed_s'],DC_pass=sum(q['passed'] for q in replication),
        DC_total=len(replication),amplitude_pass=sum(q['passed'] for q in sens),amplitude_total=len(sens))
    write(OUT/'progress.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
