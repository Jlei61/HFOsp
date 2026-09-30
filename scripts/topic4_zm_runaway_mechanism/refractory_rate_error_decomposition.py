"""Read-only decomposition of the rejected candidate's remaining error.

No candidate weights, physics, validation gates or spatial parameters change.
Teacher availability is explicitly diagnostic and never a model prediction.
"""
from refractory_rate_response import *
from validate_refractory_rate_response import load_models
from validate_nonlinear_rate_response import bin_readout
from nonlinear_rate_response import physical_from_features
from datetime import datetime
from scipy.special import expit
import argparse,hashlib

DDIR=DEST/'error_decomposition'

def register():
    DDIR.mkdir(exist_ok=True);assert not (DDIR/'contract.json').exists()
    write(DDIR/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Separate refractory-feedback propagation error, conditional-flux error, and input-feature conditioning before changing physical state variables.',
        budget='Read-only analysis of2training arrays and64alreadyobserved fresh validation records. No simulation,new fitting,threshold change or spatial launch.',
        feature_check='Training-only centered covariance spectrum and feature scales; invertible linear conditioning does not add physical information. Condition number is a numerical diagnosis, not a model-accuracy proof.',
        refractory_counterfactual='Recompute the fixed network conditional p at all actual .1ms steps. Substitute OBSERVED available population fraction for its own refractory history only for diagnosis, then compare the phase waveform with the unchanged autonomous prediction.',
        waveform_metrics='Original normL2 andmeanerror retained; report paired error changes per condition, never use supplied observed history as an autonomous pass.',
        precision='For existing matched responses, report distance beyond original tolerance plus3SEM as a descriptive noise comparison; do not replace registered acceptance.',
        invariants=read(DEST/'fit/locked_weights.json')['files']))

def run():
    c=read(DDIR/'contract.json');nets,bases,locked=load_models();assert locked['files']==c['invariants']
    conditioning=[]
    for pop in 'EI':
        z=np.load(DEST/f'training_arrays/{pop}.npz');f=z['flux_features'].astype(float);center=f.mean(0);fc=f-center
        covariance=fc.T@fc/len(f);eigen,U=np.linalg.eigh(covariance);std=np.sqrt(np.diag(covariance))
        corr=covariance/np.outer(std,std);ce=np.linalg.eigvalsh(corr)
        row=dict(pop=pop,samples=len(f),smallest_covariance_eigenvalue=float(eigen[0]),largest_covariance_eigenvalue=float(eigen[-1]),
            covariance_condition=float(eigen[-1]/max(eigen[0],1e-30)),correlation_condition=float(ce[-1]/max(ce[0],1e-30)),
            feature_std_min=float(std.min()),feature_std_max=float(std.max()),std_by_feature=std.tolist(),
            covariance_eigenvalues=eigen.tolist(),correlation_eigenvalues=ce.tolist())
        conditioning.append(row);np.savez_compressed(DDIR/f'{pop}_feature_covariance.npz',center=center,covariance=covariance,eigenvalues=eigen,eigenvectors=U)
    profiles=read(DEST/'profiles.json')['rows'];inputs=np.load(DEST/'prepared.npz');validation=read(DEST/'validation/result.json')['rows'];lookup={r['index']:r for r in validation if r['kind']=='fresh'}
    rows=[]
    for row in profiles:
        if row['split']!='validation':continue
        k=row['id'];pop=row['pop'];burn=row['burn_steps'];steps=row['record_steps'];T=row['period_ms'];net=nets[pop];base=bases[pop]
        f=features(inputs['wave'][k],T,.1,burn,steps,pop,include_burn=True);ell=[]
        with torch.no_grad():
            for start in range(0,len(f),4096):
                fs=f[start:start+4096];b=base.evaluate(physical_from_features(fs))
                ell.append(net.logits(torch.tensor(fs),torch.tensor(b)).numpy())
        ell=np.concatenate(ell);rate,minimum=implicit_flux(ell,.1,net.ref);own,exposure=bin_readout(rate[burn:],T,.1)
        pred=np.load(DEST/f'validation/fresh{k:03d}.npz');replay_error=float(np.max(abs(own-pred['predicted_hz'])))
        assert replay_error<1e-9,(k,replay_error)
        data=np.load(DEST/f'local_data/profile{k:03d}.npz');p=expit(ell);available=data['available_counts']/row['replicates']
        supplied=available*p*1000/.1;teacher,_=bin_readout(supplied[burn:],T,.1)
        target=data['rate_hz'];norm=max(np.linalg.norm(target),np.sqrt(128.));error=float(np.linalg.norm(teacher-target)/norm)
        true_mean=np.average(target,weights=exposure);bias=float(abs(np.average(teacher,weights=exposure)-true_mean)/max(true_mean,1.))
        obs=lookup[k]
        rows.append(dict(id=k,pop=pop,family=row['family'],startup=row['startup'],prediction_replay_max_error_hz=replay_error,own_L2=obs['waveform_L2'],own_mean_error=obs['mean_error'],own_passed=obs['passed'],
            observed_availability_L2=error,observed_availability_mean_error=bias,observed_availability_within_gates=bool(error<=.15 and bias<=.1),
            L2_change_supplied_minus_own=error-obs['waveform_L2']))
        np.savez_compressed(DDIR/f'profile{k:03d}.npz',own_hz=own,observed_availability_diagnostic_hz=teacher,reference_hz=target,exposure_ms=exposure)
        if len(rows)%8==0:log('REFRACTORY ERROR DECOMPOSITION',len(rows),64)
    formal=read(DEST/'matched_linear_protocol/result.json')['rows'];reference=read(OUT/'conditional_density_linear_response_contract.json')['cases'];noise=[]
    for x in formal:
        if not x['counted']:continue
        ref=reference[x['id']]['reference'];normalized_sem=ref['sem']/abs(ref['dc_measured']) if ref['sem'] is not None else None
        noise.append(dict(id=x['id'],frequency_hz=x['frequency_hz'],passed=x['passed'],error=x['error'],tolerance=ref['tol'],normalized_SEM=normalized_sem,
            beyond_three_SEM_plus_tolerance=bool(x['error']>ref['tol']+3*normalized_sem) if normalized_sem is not None else None))
    result=dict(status='DIAGNOSTIC_COMPLETE_NO_MODEL_CHANGE',conditioning=conditioning,
        own_passed=sum(x['own_passed'] for x in rows),observed_availability_within_gates=sum(x['observed_availability_within_gates'] for x in rows),
        failed_to_within_gates=sum(not x['own_passed'] and x['observed_availability_within_gates'] for x in rows),
        median_L2_change=float(np.median([x['L2_change_supplied_minus_own'] for x in rows])),rows=rows,
        matched_failures=sum(not x['passed'] for x in noise),matched_failures_beyond_tolerance_plus_three_SEM=sum(not x['passed'] and x['beyond_three_SEM_plus_tolerance'] is True for x in noise),
        matched_failures_missing_SEM=sum(not x['passed'] and x['normalized_SEM'] is None for x in noise),noise_rows=noise,
        scope=c['question'],observed_history_supplied_in_diagnostic=True,model_promoted=False,weights_modified=False)
    write(DDIR/'result.json',result);log('ERROR DECOMPOSITION RESULT',{k:v for k,v in result.items() if k not in ['rows','noise_rows','conditioning']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run']);a=p.parse_args();{'register':register,'run':run}[a.command]()
