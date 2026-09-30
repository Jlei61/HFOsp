"""Training-only temporal-basis capacity test; never modifies the frozen closure."""
from common import *


def fit(f,H,sem,taus,mask):
    w=2j*np.pi*f[:,None]/1000;z=w*np.asarray(taus)[None,:]
    A=z/(1+z);target=H-1;weight=1/np.maximum(sem,.005)
    B=A[mask]*weight[mask,None];y=target[mask]*weight[mask]
    coef=np.linalg.lstsq(np.vstack([B.real,B.imag]),np.r_[y.real,y.imag],rcond=None)[0]
    return 1+A@coef,coef


def main():
    c=read(OUT/'response_filter_capacity_contract.json')
    source=BASE/'dynamic_assay/shapes_and_fits.json';data=read(source)['rows']
    params=read(ROOT/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/prepared.json')['params']
    tausyn=[0.,params['tau_r_AMPA']+params['tau_d_AMPA'],params['tau_r_GABA']+params['tau_d_GABA']]
    rows=[]
    for point in data:
        for channel,q in point['channels'].items():
            if q['snr']<10:continue
            ch=int(channel);f=np.array(q['frequencies']);sel=f>0
            H=np.array([complex(*v) for v in q['H']]);sem=np.array(q['norm_sem'])
            if not np.isfinite(H).all() or not np.isfinite(sem).all():continue
            f=f[sel];H=H[sel];sem=sem[sel]
            assert len(f)==6
            if ch:sem*=abs(1+2j*np.pi*f/1000*tausyn[ch]/2)
            variants=[]
            for poles in c['bases_ms']:
                full,coef=fit(f,H,sem,poles,np.ones(len(f),bool))
                errors=[]
                for j in range(len(f)):
                    mask=np.arange(len(f))!=j
                    pred,_=fit(f,H,sem,poles,mask);errors.append(float(abs(pred[j]-H[j])))
                variants.append(dict(poles_ms=poles,coefficients=coef.tolist(),
                    training_RMS=float(np.sqrt(np.mean(abs(full-H)**2))),
                    LOO_errors=errors,LOO_interior_RMS=float(np.sqrt(np.mean(np.array(errors[1:-1])**2))),
                    LOO_endpoint_RMS=float(np.sqrt(np.mean(np.array(errors)[[0,-1]]**2)))))
            rows.append(dict(pop=point['pop'],x=point['x'],sigma_E=point['sigma_E'],sigma_I=point['sigma_I'],
                channel=ch,DC_snr=q['snr'],frequencies_hz=f.tolist(),variants=variants))
    summary={}
    for ch,label in enumerate(['mean','variance_E','variance_I']):
        subset=[r for r in rows if r['channel']==ch];summaries=[]
        for j,poles in enumerate(c['bases_ms']):
            values=[r['variants'][j] for r in subset]
            summaries.append(dict(poles_ms=poles,n_workpoints=len(subset),
                median_training_RMS=float(np.median([v['training_RMS'] for v in values])),
                median_LOO_interior_RMS=float(np.median([v['LOO_interior_RMS'] for v in values])),
                p90_LOO_interior_RMS=float(np.percentile([v['LOO_interior_RMS'] for v in values],90)),
                median_LOO_endpoint_RMS=float(np.median([v['LOO_endpoint_RMS'] for v in values])),
                p90_max_absolute_coefficient=float(np.percentile([max(abs(np.array(v['coefficients']))) for v in values],90))))
        summary[label]=summaries
    out=dict(status='TRAINING_CAPACITY_DIAGNOSTIC_COMPLETE',source=str(source),
        n_workpoint_channels=len(rows),summary=summary,rows=rows,
        scope='Per-workpoint frequency interpolation only. Signed filters are not a fitted spatial response table or a nonlinear population model, and no SNN/validation/bifurcation acceptance is changed.')
    write(OUT/'response_filter_capacity.json',out)
    log('FILTER CAPACITY',summary)


if __name__=='__main__':main()
