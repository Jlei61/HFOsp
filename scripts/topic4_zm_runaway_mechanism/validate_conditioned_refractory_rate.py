"""New validation and paired old diagnostics for input-conditioned rate model."""
from conditioned_refractory_rate import DEST,PARENT,OUT,np,read,write,log,torch,load_models,BaseLogit,normalized_input,normalized_jacobian,SCALE,transfer_factors
from validate_refractory_rate_response import evaluate
from validate_nonlinear_rate_response import bin_readout
from datetime import datetime
import argparse,hashlib

VDIR=DEST/'validation'
ROOTS={'fresh':DEST,'parent':PARENT,'reused':OUT/'nonlinear_rate_response'}

def predict():
    nets,bases,locked=load_models();VDIR.mkdir(exist_ok=True);assert not (VDIR/'predictions_locked.json').exists();inputs=[]
    for kind,src in ROOTS.items():
        profiles=read(src/'profiles.json')['rows'];z=np.load(src/'prepared.npz')
        for row in profiles:
            if row['split']=='validation':
                inputs.append(dict(label=f'{kind}{row["id"]:03d}',kind=kind,index=row['id'],pop=row['pop'],family=row['family'],startup=row['startup'],
                    theta=18.,wave=z['wave'][row['id']],T=row['period_ms'],dt=.1,burn=row['burn_steps'],steps=row['record_steps']))
    for kind in ['factorial_waveform','in_domain_waveform']:
        z=np.load(OUT/kind/'prepared.npz');T=float(z['T_ms'])
        for index in range(12):
            pop='E' if z['pars'][index,19]==20 else 'I';assert z['pars'][index,19] in [10,20]
            inputs.append(dict(label=f'{kind}_{index:02d}',kind=kind,index=index,pop=pop,theta=float(z['pars'][index,1]),wave=z['wave'][index],T=T,dt=.1,burn=round(5*T/.1),steps=round(20*T/.1)))
    rows=[]
    for row in inputs:
        net,base=nets[row['pop']],bases[row['pop']];path=VDIR/f'{row["label"]}.npz';assert not path.exists()
        r,m=evaluate(net,base,row['wave'],row['T'],.1,row['burn'],row['steps'],row['theta']);pred,exposure=bin_readout(r,row['T'],.1)
        fine,mfine=evaluate(net,base,row['wave'],row['T'],.05,row['burn']*2,row['steps']*2,row['theta']);fine_pred,_=bin_readout(fine.reshape(-1,2).mean(1),row['T'],.1)
        np.savez_compressed(path,predicted_hz=pred,fine_step_prediction_hz=fine_pred,exposure_ms=exposure,period_ms=row['T'],dt_ms=.1,
            theta_mv=row['theta'],minimum_available_mass=m,fine_minimum_available_mass=mfine)
        rows.append({k:v for k,v in row.items() if k!='wave'})
        if len(rows)%8==0:log('CONDITIONED RATE PREDICTION',len(rows),216)
    analytic=[]
    for case in read(OUT/'conditional_density_linear_response_contract.json')['cases']:
        q=case['workpoint'];p=np.array([[q['mu'],q['ve'],q['vi']]]);f=np.zeros((1,39));f[:,:3]=normalized_input(p,q['theta'])/SCALE
        b,bg=bases[q['pop']].evaluate(p,q['theta'],True);ig=normalized_jacobian(p,q['theta']);bank,cov,K=transfer_factors(q['pop'],[case['frequency_hz']],dt=case['dt_ms'])
        g=nets[q['pop']].linear_response(torch.tensor(f),torch.tensor(b),torch.tensor(bg),torch.tensor(ig),torch.tensor([case['channel']]),torch.tensor(bank),torch.tensor(cov),torch.tensor(K)).detach().numpy()[0]
        analytic.append(dict(id=case['id'],pop=q['pop'],dt_ms=case['dt_ms'],gain=[float(g.real),float(g.imag)]))
    write(VDIR/'analytic_predictions.json',dict(rows=analytic,scope='Correct originalcase step; infinitesimal gains remain distinct from matched finite-amplitude measurements.'))
    write(VDIR/'predictions_locked.json',dict(status='PREDICTIONS_LOCKED_BEFORE_FRESH_TARGET_SCORING',created_local=datetime.now().astimezone().isoformat(),weight_hashes=locked['files'],waveforms=rows,
        prediction_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in VDIR.glob('*.npz')},independent_fresh_targets_opened=False,measured_future_firing_used=False))

def score():
    locked=read(VDIR/'predictions_locked.json');rows=[]
    for info in locked['waveforms']:
        path=VDIR/f'{info["label"]}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==locked['prediction_hashes'][path.name]
        pred=np.load(path);rate=pred['predicted_hz'];w=pred['exposure_ms']
        if info['kind'] in ROOTS:
            target=np.load(ROOTS[info['kind']]/f'local_data/profile{info["index"]:03d}.npz');ref=target['rate_hz'];assert np.array_equal(w,target['exposure_ms'])
        else:
            target=np.load(OUT/info['kind']/'response.npz');ref=target['measured_hz'][info['index']];assert np.allclose(w,target['occupancy_ms'],rtol=0,atol=1e-8)
        norm=max(np.linalg.norm(ref),np.sqrt(128.));mean=np.average(ref,weights=w)
        error=float(np.linalg.norm(rate-ref)/norm);bias=float(abs(np.average(rate,weights=w)-mean)/max(mean,1.));step=float(np.linalg.norm(rate-pred['fine_step_prediction_hz'])/norm)
        rows.append(dict(**info,waveform_L2=error,mean_error=bias,time_step_error=step,passed=bool(error<=.15 and bias<=.1),numerical_pass=bool(step<=.02),
            reference_mean_hz=float(mean),predicted_mean_hz=float(np.average(rate,weights=w))))
    groups={kind:dict(count=sum(r['kind']==kind for r in rows),passed=sum(r['kind']==kind and r['passed'] for r in rows),numerical_failed=sum(r['kind']==kind and not r['numerical_pass'] for r in rows)) for kind in list(ROOTS)+['factorial_waveform','in_domain_waveform']}
    ok=all(groups[k]['passed']>=58 for k in ROOTS) and all(groups[k]['passed']==12 for k in ['factorial_waveform','in_domain_waveform']) and all(r['numerical_pass'] for r in rows)
    cases=read(OUT/'conditional_density_linear_response_contract.json')['cases'];linear=[]
    for pred in read(VDIR/'analytic_predictions.json')['rows']:
        case=cases[pred['id']];ref=case['reference'];den=abs(ref['dc_measured']);err=abs(complex(*pred['gain'])-complex(*ref['measured']))/den if den else None
        linear.append(dict(**pred,counted=ref['counted'],frequency_hz=case['frequency_hz'],error=err,tolerance=ref['tol'],passed=bool(err<=ref['tol']) if ref['counted'] else None))
    ac=[r for r in linear if r['counted'] and r['frequency_hz']];dc=[r for r in linear if r['counted'] and not r['frequency_hz']]
    result=dict(status='LOCAL_WAVEFORMS_PASS_OTHER_GATES_PENDING' if ok else 'LOCAL_CANDIDATE_FAIL',groups=groups,
        analytic_AC_counted=len(ac),analytic_AC_failed=sum(not r['passed'] for r in ac),analytic_DC_counted=len(dc),analytic_DC_failed=sum(not r['passed'] for r in dc),
        rows=rows,analytic_rows=linear,model_promoted=False,scope='Same physical rate function class with invertible training input conditioning. No autonomous spatial or bifurcation claim.')
    write(VDIR/'result.json',result);log('CONDITIONED RATE VALIDATION',{k:v for k,v in result.items() if k not in ['rows','analytic_rows']})

def matched(command):
    import refractory_rate_linear_protocol as worker
    worker.DEST=DEST;worker.LDIR=DEST/'matched_linear_protocol';worker.load_models=load_models
    {'register':worker.register,'run':worker.run,'score':worker.score}[command]()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['predict','score','matched-register','matched-run','matched-score']);a=p.parse_args()
    if a.command.startswith('matched-'):matched(a.command.split('-',1)[1])
    else:{'predict':predict,'score':score}[a.command]()
