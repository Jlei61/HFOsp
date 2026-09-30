"""Lock candidate predictions before opening fresh waveform targets.

Analytic linear gains are reported separately from the original finite-amplitude
assay. A waveform failure already rejects the candidate; it cannot be rescued
by a matching stationary curve. No training or parameter selection occurs here.
"""
from nonlinear_rate_response import *
from pathlib import Path
import argparse

VDIR=DEST/'validation'


def load_models():
    locked=read(DEST/'fit/locked_weights.json')
    assert locked['status']=='FINAL_WEIGHTS_LOCKED_BEFORE_VALIDATION'
    nets={};bases={};torch.set_num_threads(2)
    for pop in 'EI':
        p=DEST/f'fit/{pop}_final.pt'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==locked['files'][p.name]
        payload=torch.load(p,map_location='cpu',weights_only=False)
        net=RateReadout(pop).double();net.load_state_dict(payload['model']);net.eval()
        nets[pop]=net;bases[pop]=Baseline(pop)
    return nets,bases,locked


def evaluate_steps(net,base,features,theta):
    rates=[]
    with torch.no_grad():
        for start in range(0,len(features),4096):
            f=features[start:start+4096];b=base.evaluate(physical_from_features(f,theta),theta)
            rates.append(net(torch.tensor(f),torch.tensor(b)).numpy())
    rates=np.concatenate(rates)
    assert np.isfinite(rates).all() and rates.min()>=0 and rates.max()<=net.maximum
    return rates


def bin_readout(rates,period,dt,bins=128,quadrature=None):
    which=np.minimum(((((np.arange(len(rates))+1)*dt/period)%1)*bins).astype(int),bins-1)
    counts=np.bincount(which,minlength=bins)
    assert counts.min()>0
    if quadrature:
        values=[]
        for b in range(bins):
            times=np.flatnonzero(which==b)
            select=times[np.minimum(((np.arange(quadrature)+.5)*len(times)/quadrature).astype(int),len(times)-1)]
            values.append(rates[select].mean())
        return np.array(values),counts*dt
    return np.bincount(which,weights=rates,minlength=bins)/counts,counts*dt


def predict():
    nets,bases,locked=load_models();VDIR.mkdir(exist_ok=True)
    assert not (VDIR/'predictions_locked.json').exists()
    inputs=[]
    profiles=read(DEST/'profiles.json')['rows'];fresh=np.load(DEST/'prepared.npz')
    for row in profiles:
        if row['split']=='validation':
            inputs.append(dict(label=f'fresh{row["id"]:03d}',kind='fresh',index=row['id'],pop=row['pop'],
                family=row['family'],startup=row['startup'],theta=18.,wave=fresh['wave'][row['id']],T=row['period_ms'],
                dt=.1,burn=row['burn_steps'],steps=row['record_steps']))
    for dataset in ['factorial_waveform','in_domain_waveform']:
        data=np.load(OUT/dataset/'prepared.npz');T=float(data['T_ms'])
        for index in range(12):
            pop='E' if data['pars'][index,19]==20 else 'I'
            # Refractory count identifies E(2ms)/I(1ms) at this fixed0.1ms.
            assert data['pars'][index,19] in [10,20]
            inputs.append(dict(label=f'{dataset}_{index:02d}',kind=dataset,index=index,pop=pop,
                theta=float(data['pars'][index,1]),wave=data['wave'][index],T=T,dt=.1,
                burn=round(5*T/.1),steps=round(20*T/.1)))
    rows=[]
    for row in inputs:
        label=row['label'];path=VDIR/f'{label}.npz';assert not path.exists()
        net,base=nets[row['pop']],bases[row['pop']]
        f=history_features(row['wave'],row['T'],row['dt'],row['burn'],row['steps'],row['theta'])
        rates=evaluate_steps(net,base,f,row['theta']);pred,exposure=bin_readout(rates,row['T'],row['dt'])
        q16,_=bin_readout(rates,row['T'],row['dt'],quadrature=16)
        fine=history_features(row['wave'],row['T'],row['dt']/2,row['burn']*2,row['steps']*2,row['theta'])[1::2]
        fine_rates=evaluate_steps(net,base,fine,row['theta']);fine_pred,fine_exposure=bin_readout(fine_rates,row['T'],row['dt'])
        assert np.array_equal(exposure,fine_exposure)
        np.savez_compressed(path,predicted_hz=pred,fine_step_prediction_hz=fine_pred,quadrature16_prediction_hz=q16,
            exposure_ms=exposure,period_ms=row['T'],dt_ms=row['dt'],theta_mv=row['theta'])
        meta={k:v for k,v in row.items() if k!='wave'};rows.append(meta)
        log('RATE VALIDATION PREDICTION',len(rows),88,label)
    # All32oldworkpoints,282analyticresponses. These are infinitesimal gains,
    # not the finite-amplitude protocol, whose time-domain test is separate.
    contract=read(OUT/'conditional_density_linear_response_contract.json')
    predictions=[]
    for case in contract['cases']:
        q=case['workpoint'];physical=np.array([[q['mu'],q['ve'],q['vi']]])
        f=np.zeros((1,39));f[:,:3]=normalized_input(physical,q['theta'])/SCALE
        b,bg=bases[q['pop']].evaluate(physical,q['theta'],True);ig=normalized_jacobian(physical,q['theta'])
        gain=nets[q['pop']].linear_response(torch.tensor(f),torch.tensor(b),torch.tensor(bg),torch.tensor(ig),torch.tensor([case['frequency_hz']],dtype=torch.float64),torch.tensor([case['channel']])).detach().numpy()[0]
        predictions.append(dict(id=case['id'],pop=q['pop'],gain=[float(gain.real),float(gain.imag)]))
    write(VDIR/'analytic_predictions.json',dict(rows=predictions,scope='Infinitesimal gains; no target used for prediction; do not call this the matched finite-amplitude assay.'))
    write(VDIR/'predictions_locked.json',dict(status='PREDICTIONS_LOCKED_BEFORE_FRESH_TARGET_SCORING',created_local=datetime.now().astimezone().isoformat(),
        weight_hashes=locked['files'],waveforms=rows,analytic_conditions=282,
        prediction_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in VDIR.glob('*.npz')},
        independent_fresh_targets_opened=False))


def score():
    locked=read(VDIR/'predictions_locked.json');rows=[]
    for info in locked['waveforms']:
        path=VDIR/f'{info["label"]}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==locked['prediction_hashes'][path.name]
        pred=np.load(path);exposure=pred['exposure_ms'];rate=pred['predicted_hz']
        if info['kind']=='fresh':
            target=np.load(DEST/f'local_data/profile{info["index"]:03d}.npz')
            measured=target['rate_hz'];sem=target['sem_hz']
            assert np.array_equal(exposure,target['exposure_ms'])
        else:
            target=np.load(OUT/info['kind']/'response.npz');measured=target['measured_hz'][info['index']];sem=target['sem_hz'][info['index']]
            assert np.allclose(exposure,target['occupancy_ms'],rtol=0,atol=1e-8)
        scale=max(np.linalg.norm(measured),np.sqrt(128.));mean=np.average(measured,weights=exposure)
        error=float(np.linalg.norm(rate-measured)/scale);bias=float(abs(np.average(rate,weights=exposure)-mean)/max(mean,1.))
        quad=float(np.linalg.norm(rate-pred['quadrature16_prediction_hz'])/scale)
        step=float(np.linalg.norm(rate-pred['fine_step_prediction_hz'])/scale)
        row=dict(**info,waveform_L2=error,mean_error=bias,reference_mean_hz=float(mean),predicted_mean_hz=float(np.average(rate,weights=exposure)),
            near_zero_mean=bool(mean<1.),passed=bool(error<=.15 and bias<=.1),quadrature_error=quad,time_step_error=step,numerical_pass=bool(quad<=.02 and step<=.02))
        rows.append(row)
    cases=read(OUT/'conditional_density_linear_response_contract.json')['cases'];linear=[]
    for pred in read(VDIR/'analytic_predictions.json')['rows']:
        case=cases[pred['id']];ref=case['reference'];dc=abs(ref['dc_measured'])
        error=abs(complex(*pred['gain'])-complex(*ref['measured']))/dc if dc else None
        linear.append(dict(**pred,channel=case['channel'],frequency_hz=case['frequency_hz'],counted=ref['counted'],error=error,
            tolerance=ref['tol'],passed=bool(error<=ref['tol']) if ref['counted'] else None))
    fresh=[r for r in rows if r['kind']=='fresh'];old=[r for r in rows if r['kind']!='fresh']
    ac=[r for r in linear if r['frequency_hz'] and r['counted']];dc=[r for r in linear if not r['frequency_hz'] and r['counted']]
    waveform_pass=sum(r['passed'] for r in fresh)>=58 and all(r['passed'] for r in old)
    result=dict(status='LOCAL_CANDIDATE_WAVEFORM_PASS_LINEAR_FINITE_PROTOCOL_PENDING' if waveform_pass else 'LOCAL_CANDIDATE_FAIL',
        fresh_count=len(fresh),fresh_passed=sum(r['passed'] for r in fresh),fresh_required_passed=58,
        old_count=len(old),old_passed=sum(r['passed'] for r in old),old_required_passed=24,
        numerical_failures=sum(not r['numerical_pass'] for r in rows),
        analytic_AC_counted=len(ac),analytic_AC_failed=sum(not r['passed'] for r in ac),analytic_DC_counted=len(dc),analytic_DC_failed=sum(not r['passed'] for r in dc),
        analytic_scope='Analytic infinitesimal gains compared with original finite-amplitude estimates; not a replacement for the matched finite-amplitude protocol.',
        rows=rows,analytic_rows=linear,model_promoted=False,
        scope='Local LIF-calibrated candidate only. Held-out waveform failure rejects this fixed candidate. Even a local pass requires original finite-amplitude response and autonomous spatial/ZM validation before new bifurcation analysis.')
    write(VDIR/'result.json',result)
    write(DEST/'validation_separation.json',dict(status='SCORED_AFTER_WEIGHTS_AND_PREDICTIONS_LOCKED',train_ids=list(range(224)),validation_ids=list(range(224,288)),
        locked_predictions=str(VDIR/'predictions_locked.json'),training_modified_after_scoring=False))
    log('RATE VALIDATION RESULT',{k:v for k,v in result.items() if k not in ['rows','analytic_rows']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['predict','score']);a=p.parse_args()
    {'predict':predict,'score':score}[a.command]()
