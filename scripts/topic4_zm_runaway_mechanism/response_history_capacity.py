"""Fixed continuous-filter capacity screen on calibration frequency holdouts.

Only the original local-LIF calibration data are used. No native onset
trajectory, old waveform validation or random validation point is fitted.
"""
from common import OUT, BASE, np, read, write
from datetime import datetime

DEST = OUT / 'response_history_capacity'


def basis(frequency_hz, times, orders):
    lam = 2j*np.pi*np.asarray(frequency_hz)[:, None]/1000
    return np.column_stack([(1+lam[:, 0]*tau)**(-order)-1 for tau in times for order in orders])


def main():
    DEST.mkdir(exist_ok=True); path=DEST/'contract.json'; assert not path.exists()
    contract=dict(created_local=datetime.now().astimezone().isoformat(),
          question='Can one fixed36state causal filter bank represent local LIF frequency responses before a nonlinear finite-dimensional rate closure is trained?',
          equations='For each of3input channels and tau in[1,4,16,64]ms, tau*h1dot=u-h1; tau*h2dot=h1-h2; tau*h3dot=h2-h3. Constant inputs imply every h=u. Linear output correction is a weighted sum of h-u. Poles strictly negative; continuous finite-dimensional system.',
          taus_ms=[1.,4.,16.,64.],orders=[1,2,3],states_per_population=36,
          source=str(BASE/'dynamic_assay/rows.json'),fit_frequencies_hz=[2.,5.,20.,80.],heldout_frequencies_hz=[10.,40.],
          qualification='Original source DC absolute gain/SEM>=10. Frequencies and DC belong to the same fixed workpoint, not independent subjects.',
          fitting='DC fixed to measured calibration DC. Normalize residual by absolute DC. Weighted ridge with SEM floor0.02in normalized units and ridge coefficient1; single fixed choice, no tuning on held-out frequencies.',
          readout='Error/absoluteDC<=0.10at10Hz and<=0.15at40Hz; report noise precision separately. This is frequency interpolation at known calibration workpoints, not parameter interpolation, a new blind dataset, nonlinear waveform or network validation.',
          stop='One basis and one regularization, all eligible calibration workpoints. No nonlinear spatial model or bifurcation launched by this screen.')
    write(path,contract)
    source=read(BASE/'dynamic_assay/rows.json')['rows'];groups={}
    for row in source:
        key=tuple(row[k] for k in ['pop','x','sigma_E','sigma_I','channel'])
        groups.setdefault(key,{})[row['frequency_hz']]=row
    fit=np.array(contract['fit_frequencies_hz']);test=np.array(contract['heldout_frequencies_hz'])
    B=basis(fit,contract['taus_ms'],contract['orders']);Bt=basis(test,contract['taus_ms'],contract['orders'])
    rows=[];parameters=[];weights=[]
    for key,data in groups.items():
        dc=complex(*data[0.]['response']).real;dc_sem=data[0.]['sem']
        if abs(dc)/max(dc_sem,1e-15)<10:continue
        target=np.array([complex(*data[f]['response']) for f in fit])/abs(dc)-np.sign(dc)
        sigma=np.maximum([data[f]['sem']/abs(dc) for f in fit],.02)
        design=np.vstack([B.real/sigma[:,None],B.imag/sigma[:,None]])
        y=np.r_[target.real/sigma,target.imag/sigma]
        coeff=np.linalg.solve(design.T@design+np.eye(B.shape[1]),design.T@y)
        prediction=dc+abs(dc)*(Bt@coeff)
        parameters.append(key);weights.append(coeff)
        for k,f in enumerate(test):
            truth=complex(*data[f]['response']);error=abs(prediction[k]-truth)/abs(dc);tol=.1 if f<=25 else .15
            rows.append(dict(pop=key[0],x=key[1],sigma_E=key[2],sigma_I=key[3],channel=key[4],frequency_hz=float(f),
                             measured=[truth.real,truth.imag],prediction=[prediction[k].real,prediction[k].imag],
                             error=float(error),tolerance=tol,passed=bool(error<=tol),
                             normalized_reference_SEM=float(data[f]['sem']/abs(dc)),
                             outside_descriptive_three_SEM=bool(error>3*data[f]['sem']/abs(dc))))
    summary=[]
    for pop in 'EI':
        for channel in range(3):
            subset=[r for r in rows if r['pop']==pop and r['channel']==channel]
            summary.append(dict(pop=pop,channel=channel,heldout_count=len(subset),failed=sum(not r['passed'] for r in subset),
                                median_error=float(np.median([r['error'] for r in subset])) if subset else None,
                                failed_outside_three_SEM=sum(not r['passed'] and r['outside_descriptive_three_SEM'] for r in subset)))
    np.savez_compressed(DEST/'coefficients.npz',parameters=np.asarray(parameters,dtype=str),weights=np.array(weights),taus_ms=contract['taus_ms'],orders=contract['orders'])
    result=dict(status='CALIBRATION_FREQUENCY_HOLDOUT_CAPACITY_COMPLETE',workpoints_channels=len(parameters),heldout_count=len(rows),
                failed=sum(not r['passed'] for r in rows),summary=summary,rows=rows,
                model_promoted=False,scope=contract['readout'],
                remaining='Smooth parameter dependence, static gain consistency, nonlinear reset/recovery memory, truly held-out local inputs, autonomous spatial propagation/ZM and bifurcation remain unvalidated.')
    write(DEST/'result.json',result)
    print({k:v for k,v in result.items() if k!='rows'})


if __name__=='__main__':main()
