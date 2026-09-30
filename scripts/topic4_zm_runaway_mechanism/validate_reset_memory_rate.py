"""Closed local predictions for the single-state reset-memory candidate.

These are reused diagnostic stimuli, never new independent validation.
No observed firing or reset trace is supplied at prediction time.
"""
from reset_memory_rate import *
from reset_memory_numerics import evaluate
from validate_nonlinear_rate_response import bin_readout
import os

VDIR=DEST/'validation'
ROOTS={'conditioned':OUT/'conditioned_refractory_rate','parent':OUT/'refractory_rate_response','legacy':OUT/'nonlinear_rate_response'}


def predict():
    torch.set_num_threads(2);nets,bases,locked=load_models();VDIR.mkdir(exist_ok=True)
    assert not (VDIR/'predictions_locked.json').exists();inputs=[]
    for kind in ['factorial_waveform','in_domain_waveform']:
        z=np.load(OUT/kind/'prepared.npz');T=float(z['T_ms'])
        for k in range(12):
            pop='E' if z['pars'][k,19]==20 else 'I';assert z['pars'][k,19] in [10,20]
            inputs.append(dict(label=f'{kind}_{k:02d}',kind=kind,index=k,pop=pop,theta=float(z['pars'][k,1]),wave=z['wave'][k],T=T,dt=.1,burn=round(5*T/.1),steps=round(20*T/.1)))
    for kind,src in ROOTS.items():
        z=np.load(src/'prepared.npz')
        for r in read(src/'profiles.json')['rows']:
            if r['split']=='validation':inputs.append(dict(label=f'{kind}{r["id"]:03d}',kind=kind,index=r['id'],pop=r['pop'],family=r['family'],startup=r['startup'],theta=18.,wave=z['wave'][r['id']],T=r['period_ms'],dt=.1,burn=r['burn_steps'],steps=r['record_steps']))
    jobs=dict(status='RUNNING',pid=os.getpid(),expected=len(inputs),completed=[]);assert not (VDIR/'jobs.json').exists();write(VDIR/'jobs.json',jobs);rows=[]
    for r in inputs:
        p=VDIR/f'{r["label"]}.npz';assert not p.exists()
        a,m=evaluate(nets[r['pop']],bases[r['pop']],r['wave'],r['T'],.1,r['burn'],r['steps'],r['theta']);prediction,exposure=bin_readout(a,r['T'],.1)
        fine,fm=evaluate(nets[r['pop']],bases[r['pop']],r['wave'],r['T'],.05,r['burn']*2,r['steps']*2,r['theta']);finepred,_=bin_readout(fine.reshape(-1,2).mean(1),r['T'],.1)
        np.savez_compressed(p,predicted_hz=prediction,fine_step_prediction_hz=finepred,exposure_ms=exposure,period_ms=r['T'],dt_ms=.1,theta_mv=r['theta'],minimum_available_mass=m,fine_minimum_available_mass=fm)
        rows.append({k:v for k,v in r.items() if k!='wave'});jobs['completed'].append(r['label']);write(VDIR/'jobs.json',jobs)
        if len(rows)%8==0:log('RESET MEMORY AUTONOMOUS VALIDATION',len(rows),len(inputs))
    write(VDIR/'predictions_locked.json',dict(status='PREDICTIONS_LOCKED_FOR_REUSED_DIAGNOSTICS',created_local=datetime.now().astimezone().isoformat(),weight_hashes=locked['files'],waveforms=rows,prediction_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in VDIR.glob('*.npz')},own_refractory_history=True,own_reset_trace=True,observed_firing_used=False,independent_fresh_validation=False))
    jobs['status']='COMPLETE';write(VDIR/'jobs.json',jobs)


def score():
    locked=read(VDIR/'predictions_locked.json');rows=[]
    for r in locked['waveforms']:
        p=VDIR/f'{r["label"]}.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==locked['prediction_hashes'][p.name]
        z=np.load(p);pred=z['predicted_hz'];w=z['exposure_ms']
        if r['kind'] in ROOTS:
            target=np.load(ROOTS[r['kind']]/f'local_data/profile{r["index"]:03d}.npz');ref=target['rate_hz'];assert np.array_equal(w,target['exposure_ms'])
        else:
            target=np.load(OUT/r['kind']/'response.npz');ref=target['measured_hz'][r['index']];assert np.allclose(w,target['occupancy_ms'],atol=1e-8,rtol=0)
        scale=max(np.linalg.norm(ref),np.sqrt(128));mean=np.average(ref,weights=w);err=float(np.linalg.norm(pred-ref)/scale);bias=float(abs(np.average(pred,weights=w)-mean)/max(mean,1));step=float(np.linalg.norm(pred-z['fine_step_prediction_hz'])/scale)
        rows.append(dict(**r,waveform_L2=err,mean_error=bias,time_step_error=step,passed=bool(err<=.15 and bias<=.1),numerical_pass=bool(step<=.02),reference_mean_hz=float(mean),predicted_mean_hz=float(np.average(pred,weights=w))))
    groups={k:dict(count=sum(r['kind']==k for r in rows),passed=sum(r['kind']==k and r['passed'] for r in rows),numerical_failed=sum(r['kind']==k and not r['numerical_pass'] for r in rows)) for k in list(ROOTS)+['factorial_waveform','in_domain_waveform']}
    ok=all(groups[k]['passed']>=58 for k in ROOTS) and all(groups[k]['passed']==12 for k in ['factorial_waveform','in_domain_waveform']) and all(r['numerical_pass'] for r in rows)
    write(VDIR/'result.json',dict(status='REUSED_WAVEFORMS_PASS_FRESH_AND_GAIN_VALIDATION_PENDING' if ok else 'LOCAL_CANDIDATE_FAIL',groups=groups,rows=rows,model_promoted=False,scope='Same fixed training data with one own-output reset-count trace. Reused waveform checks only; no gain, native spatial or bifurcation acceptance.'))
    log('RESET MEMORY RESULT',groups,ok)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['predict','score']);a=p.parse_args();{'predict':predict,'score':score}[a.command]()
