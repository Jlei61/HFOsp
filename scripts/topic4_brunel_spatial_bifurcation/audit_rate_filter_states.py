"""Check positivity of both physical rate filters, not only their mixture."""
from rate_field import *


def filter_state_minima(s,r,T,factor=4):
    N=len(r);M=factor*N;cf=np.fft.rfft(r,axis=0)/N
    lam=2j*np.pi*np.arange(N//2+1)[:,None]/T;H=s.filter_response(lam);result=[]
    for tau in [s.tf,s.ts]:
        pad=np.zeros((M//2+1,s.P),complex)
        pad[:len(cf)]=cf/(H*(1+lam*tau))*M;pad[len(cf)-1]*=.5
        y=np.fft.irfft(pad,n=M,axis=0)*1000;k=np.unravel_index(np.argmin(y),y.shape)
        result.append(dict(minimum_Hz=float(y[k]),phase_index=int(k[0]),group=int(k[1])))
    return dict(fast=result[0],slow=result[1],positive=all(q['minimum_Hz']>=-1e-9 for q in result),
                N=N,check_N=M)


def main():
    p=RATE_OUT/'periodic_completion';s=RateField();paths={}
    # Analytic positive sinusoidal target: recover both known first-order
    # responses from their mixture, including their distinct phase lags.
    n=128;T=200.;lam=2j*np.pi/T;theta=2*np.pi*np.arange(n)[:,None]/n
    xf=.001+.0004*np.real(np.exp(1j*theta)/(1+lam*s.tf))
    xs=.001+.0004*np.real(np.exp(1j*theta)/(1+lam*s.ts))
    control=filter_state_minima(s,s.alpha*xf+(1-s.alpha)*xs,T)
    fine=2*np.pi*np.arange(4*n)[:,None]/(4*n)
    expected=[float((1+.4*np.real(np.exp(1j*fine)/(1+lam*tau))).min()) for tau in [s.tf,s.ts]]
    error=max(abs(control[k]['minimum_Hz']-value) for k,value in zip(['fast','slow'],expected))
    assert error<1e-12 and control['positive'],(error,control)
    for f in p.glob('*_validation.json'):
        q=read(f)
        if q.get('status')=='VALIDATED_CYCLE_FOLD':paths[q['mesh_checks'][-1]['orbit']]=f.stem
    # Include the located but incompletely validated parents as well. Their
    # incomplete critical-mode status must not hide a physical-state defect.
    from plot_rate_periodic_completion import critical
    for q in critical():
        if q['label'].startswith('LPC') or q['label'].startswith('TR_'):
            validation=p/(q['label']+'_validation.json')
            paths[q['orbit']]=validation.stem if validation.exists() else q['label']
    for row in read(p/'composite_case_resolution.json')['rows']:paths[row['orbit']]='composite_'+row['case']
    for label in ['PD_double_low','PD_double_upper','PD_A_return']:
        qs=sorted([read(f) for f in p.glob(label+'_N*.json')],key=lambda q:q['N'])
        validation=read(p/(label+'_validation.json'))
        paths[validation.get('accepted_parent_orbit',qs[-1]['orbit'])]=label
    secondary=list(p.glob('PD_upper_child_next_amplitude_N*.json'))
    if secondary:
        latest=max(map(read,secondary),key=lambda q:q['N'])
        paths[latest['orbit']]='PD2_child_secondary_candidate'
    rows=[]
    for path,label in paths.items():
        z=np.load(path);row=dict(label=label,orbit=path,**filter_state_minima(s,z['r'],float(z['T'])))
        rows.append(row)
        if not row['positive']:print('NEGATIVE FILTER',row,flush=True)
    write(p/'rate_filter_state_positivity_audit.json',dict(
        status='PASS' if all(q['positive'] for q in rows) else 'FILTER_STATE_RESOLUTION_REVIEW_REQUIRED',rows=rows,
        diagnostic_threshold_Hz=-1e-9,analytic_positive_target_control_error_Hz=error,
        scope='CPU Fourier reconstruction of both constituent rate filters, which must be nonnegative for a physical exact periodic orbit driven by a nonnegative transfer function. Four-times oversampled diagnostic; numerical undershoot does not prove a wrong equation or a false critical point. Existing critical-mode evidence remains, but full physical-state waveform acceptance requires a finer-mesh follow-up.'))
    print('FILTER POSITIVITY',len(rows),sum(not q['positive'] for q in rows),flush=True)


if __name__=='__main__':main()
