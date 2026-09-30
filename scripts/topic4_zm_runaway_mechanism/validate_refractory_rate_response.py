"""Autonomous local predictions: never substitute measured refractory history."""
from refractory_rate_response import *
from nonlinear_rate_response import physical_from_features
from validate_nonlinear_rate_response import bin_readout
from datetime import datetime
import argparse,hashlib

VDIR=DEST/'validation'

def load_models():
    locked=read(DEST/'fit/locked_weights.json');assert locked['status']=='FINAL_WEIGHTS_LOCKED_BEFORE_VALIDATION'
    nets={};bases={};torch.set_num_threads(2)
    for pop in 'EI':
        path=DEST/f'fit/{pop}_final.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==locked['files'][path.name]
        net=RefractoryReadout(pop).double();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);net.eval()
        nets[pop]=net;bases[pop]=BaseLogit(pop)
    return nets,bases,locked

def evaluate(net,base,wave,T,dt,burn,steps,theta=18.):
    f=features(wave,T,dt,burn,steps,net.pop,theta,include_burn=True);ell=[]
    with torch.no_grad():
        for start in range(0,len(f),4096):
            part=f[start:start+4096];b=base.evaluate(physical_from_features(part,theta),theta)
            ell.append(net.logits(torch.tensor(part),torch.tensor(b)).numpy())
    logits=np.concatenate(ell);rates,minimum=implicit_flux(logits,dt,net.ref)
    assert np.isfinite(rates).all() and rates.min()>=-1e-7
    return rates[burn:],minimum

def predict():
    nets,bases,locked=load_models();VDIR.mkdir(exist_ok=True);assert not (VDIR/'predictions_locked.json').exists()
    inputs=[]
    for src,kind in [(DEST,'fresh'),(OUT/'nonlinear_rate_response','reused')]:
        profiles=read(src/'profiles.json')['rows'];data=np.load(src/'prepared.npz')
        for row in profiles:
            if row['split']=='validation':
                inputs.append(dict(label=f'{kind}{row["id"]:03d}',kind=kind,index=row['id'],pop=row['pop'],family=row['family'],startup=row['startup'],
                    theta=18.,wave=data['wave'][row['id']],T=row['period_ms'],dt=.1,burn=row['burn_steps'],steps=row['record_steps']))
    for kind in ['factorial_waveform','in_domain_waveform']:
        data=np.load(OUT/kind/'prepared.npz');T=float(data['T_ms'])
        for index in range(12):
            pop='E' if data['pars'][index,19]==20 else 'I';assert data['pars'][index,19] in [10,20]
            inputs.append(dict(label=f'{kind}_{index:02d}',kind=kind,index=index,pop=pop,theta=float(data['pars'][index,1]),wave=data['wave'][index],
                T=T,dt=.1,burn=round(5*T/.1),steps=round(20*T/.1)))
    rows=[]
    for row in inputs:
        net,base=nets[row['pop']],bases[row['pop']];path=VDIR/f'{row["label"]}.npz';assert not path.exists()
        r,m=evaluate(net,base,row['wave'],row['T'],.1,row['burn'],row['steps'],row['theta']);pred,exposure=bin_readout(r,row['T'],.1)
        fine,mfine=evaluate(net,base,row['wave'],row['T'],.05,row['burn']*2,row['steps']*2,row['theta'])
        # Flux is an interval count/dt. Sum the two fine counts into the same
        # coarse observation interval before applying its phase-bin assignment.
        fine_aligned=fine.reshape(-1,2).mean(1);fine_pred,_=bin_readout(fine_aligned,row['T'],.1)
        np.savez_compressed(path,predicted_hz=pred,fine_step_prediction_hz=fine_pred,exposure_ms=exposure,
            period_ms=row['T'],dt_ms=.1,theta_mv=row['theta'],minimum_available_mass=m,fine_minimum_available_mass=mfine)
        rows.append({k:v for k,v in row.items() if k!='wave'})
        if len(rows)%8==0:log('REFRACTORY PREDICTION',len(rows),len(inputs))
    predictions=[]
    for case in read(OUT/'conditional_density_linear_response_contract.json')['cases']:
        q=case['workpoint'];p=np.array([[q['mu'],q['ve'],q['vi']]])
        f=np.zeros((1,39));f[:,:3]=normalized_input(p,q['theta'])/SCALE
        b,g=bases[q['pop']].evaluate(p,q['theta'],True);ig=normalized_jacobian(p,q['theta'])
        bank,cov,K=transfer_factors(q['pop'],[case['frequency_hz']])
        gain=nets[q['pop']].linear_response(torch.tensor(f),torch.tensor(b),torch.tensor(g),torch.tensor(ig),torch.tensor([case['channel']]),
            torch.tensor(bank),torch.tensor(cov),torch.tensor(K)).detach().numpy()[0]
        predictions.append(dict(id=case['id'],pop=q['pop'],gain=[float(gain.real),float(gain.imag)]))
    write(VDIR/'analytic_predictions.json',dict(rows=predictions,scope='Infinitesimal discrete gains; matched finite-amplitude assay remains separate.'))
    write(VDIR/'predictions_locked.json',dict(status='PREDICTIONS_LOCKED_BEFORE_FRESH_TARGET_SCORING',created_local=datetime.now().astimezone().isoformat(),
        weight_hashes=locked['files'],waveforms=rows,analytic_conditions=282,prediction_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in VDIR.glob('*.npz')},
        independent_fresh_targets_opened=False,measured_future_firing_used=False,own_refractory_history=True))

def score():
    locked=read(VDIR/'predictions_locked.json');rows=[]
    for info in locked['waveforms']:
        path=VDIR/f'{info["label"]}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==locked['prediction_hashes'][path.name]
        pred=np.load(path);rate=pred['predicted_hz'];exposure=pred['exposure_ms']
        if info['kind'] in ['fresh','reused']:
            root=DEST if info['kind']=='fresh' else OUT/'nonlinear_rate_response'
            target=np.load(root/f'local_data/profile{info["index"]:03d}.npz');measured=target['rate_hz'];assert np.array_equal(exposure,target['exposure_ms'])
        else:
            target=np.load(OUT/info['kind']/'response.npz');measured=target['measured_hz'][info['index']];assert np.allclose(exposure,target['occupancy_ms'],rtol=0,atol=1e-8)
        scale=max(np.linalg.norm(measured),np.sqrt(128.));mean=np.average(measured,weights=exposure)
        err=float(np.linalg.norm(rate-measured)/scale);bias=float(abs(np.average(rate,weights=exposure)-mean)/max(mean,1.))
        step=float(np.linalg.norm(rate-pred['fine_step_prediction_hz'])/scale)
        rows.append(dict(**info,waveform_L2=err,mean_error=bias,reference_mean_hz=float(mean),predicted_mean_hz=float(np.average(rate,weights=exposure)),
            near_zero_mean=bool(mean<1),passed=bool(err<=.15 and bias<=.1),time_step_error=step,numerical_pass=bool(step<=.02),minimum_available_mass=float(pred['minimum_available_mass'])))
    linear=[];cases=read(OUT/'conditional_density_linear_response_contract.json')['cases']
    for pred in read(VDIR/'analytic_predictions.json')['rows']:
        case=cases[pred['id']];ref=case['reference'];dc=abs(ref['dc_measured']);err=abs(complex(*pred['gain'])-complex(*ref['measured']))/dc if dc else None
        linear.append(dict(**pred,channel=case['channel'],frequency_hz=case['frequency_hz'],counted=ref['counted'],error=err,tolerance=ref['tol'],passed=bool(err<=ref['tol']) if ref['counted'] else None))
    groups={kind:[r for r in rows if r['kind']==kind] for kind in ['fresh','reused','factorial_waveform','in_domain_waveform']}
    counts={kind:dict(count=len(items),passed=sum(r['passed'] for r in items),numerical_failed=sum(not r['numerical_pass'] for r in items)) for kind,items in groups.items()}
    ok=counts['fresh']['passed']>=58 and counts['reused']['passed']>=58 and counts['factorial_waveform']['passed']==12 and counts['in_domain_waveform']['passed']==12 and all(r['numerical_pass'] for r in rows)
    ac=[r for r in linear if r['counted'] and r['frequency_hz']];dc=[r for r in linear if r['counted'] and not r['frequency_hz']]
    result=dict(status='LOCAL_WAVEFORM_PASS_FINITE_LINEAR_AND_SPATIAL_PENDING' if ok else 'LOCAL_CANDIDATE_FAIL',groups=counts,
        numerical_failures=sum(not r['numerical_pass'] for r in rows),analytic_AC_counted=len(ac),analytic_AC_failed=sum(not r['passed'] for r in ac),
        analytic_DC_counted=len(dc),analytic_DC_failed=sum(not r['passed'] for r in dc),rows=rows,analytic_rows=linear,model_promoted=False,
        scope='Continuous refractory population-rate approximation, local stage only. Analytic gains are not matched finite-amplitude measurements. No spatial propagation, onset or bifurcation acceptance follows from this score.')
    write(VDIR/'result.json',result);log('REFRACTORY VALIDATION RESULT',{k:v for k,v in result.items() if k not in ['rows','analytic_rows']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['predict','score']);a=p.parse_args();{'predict':predict,'score':score}[a.command]()
