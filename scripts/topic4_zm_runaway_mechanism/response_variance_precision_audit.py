"""Independently reaggregate the precision repeat and evaluate fixed predictions."""
from response_variance_precision import DEST
from response_voltage_units_audit import response
from response_bank import tables
from common import *


def physical_key(q,channel):
    return tuple(q[k] for k in ['pop','theta','mu','ve','vi'])+(channel,)


def main():
    c=read(OUT/'response_variance_precision_contract.json');q=read(DEST/'result.json')
    assert q['status']=='COMPLETE' and read(DEST/'checkpoint.json')['status']=='COMPLETE'
    raw=np.load(DEST/'observations.npy',mmap_mode='r');s=model();bank=tables()
    source=read(OUT/'response_error_decomposition.json')['rows'];oldref=read(BASE/'dynamic_assay/validation_result.json')['rows']
    measured={};DC={}
    for i,r in enumerate(q['rows']):
        point=r['workpoint'];ch=r['channel'];amp=c['variance_relative_amplitude']*point['ve' if ch==1 else 'vi']
        vals=(raw[i,:,0]+1j*raw[i,:,1])*1000/(c['duration_ms']*amp)
        mean=vals.mean();sem=float(np.sqrt(np.mean(abs(vals-mean)**2)/c['replicates']))
        assert abs(mean-complex(*r['measured']))<1e-10*max(1,abs(mean))
        assert abs(sem-r['complex_SEM'])<1e-12
        key=physical_key(point,ch)
        if r['frequency_hz']==0:DC[key]=(mean.real,sem)
        else:measured[(key,r['frequency_hz'])]=(mean,sem)
    tau=np.array(read(OUT/'response_bank_candidate_contract.json')['filter_times_ms']);rows=[]
    for old,ref in zip(source,oldref):
        for k in ['kind','pop','channel','frequency_hz','counted']:assert old[k]==ref[k]
        if not old['counted'] or old['channel']=='mean':continue
        point=old['workpoint'];ch=1 if old['channel']=='variance_E' else 2;f=old['frequency_hz'];key=physical_key(point,ch)
        value,sem=measured[(key,f)];dc,dcsem=DC[key];den=max(abs(dc),1e-12)
        mu=np.array([point['mu']]);ve=np.array([point['ve']]);vi=np.array([point['vi']]);th=np.array([point['theta']])
        g=s.spline[point['pop']].evaluate(mu,ve,vi,th);b=bank[point['pop']].evaluate(mu,ve,vi,th)[:,:,0]
        w=2j*np.pi*f/1000;basis=w*tau/(1+w*tau)
        bankpred=(g['d_ve' if ch==1 else 'd_vi'][0]*1000+g['d_mu'][0]*1000*(b[ch]@basis)/(point['theta']-11))/(1+w*s.tau[ch-1]/2)
        predictions=dict(frozen_v3=response(s,point,f,ch,False),unit_corrected=response(s,point,f,ch,True),bank_candidate=bankpred)
        errors={k:float(abs(p-value)/den) for k,p in predictions.items()}
        bound=old['tol']*(1+3*dcsem/den)+3*sem/den
        oldvalue=complex(*ref['measured']);combined=np.sqrt(sem**2+ref['sem']**2)
        rows.append(dict(kind=old['kind'],pop=old['pop'],channel=old['channel'],workpoint=point,
            frequency_hz=f,new_measured=[value.real,value.imag],new_DC=dc,new_DC_SNR=abs(dc)/max(dcsem,1e-12),
            new_AC_SEM_over_DC=sem/den,old_AC_SEM_over_old_DC=ref['sem']/max(abs(ref['dc_measured']),1e-12),
            old_new_difference_in_combined_SEM=float(abs(value-oldvalue)/max(combined,1e-12)),
            tolerance=old['tol'],errors=errors,failed={k:v>old['tol'] for k,v in errors.items()},
            exceeds_three_SEM_margin={k:v>bound for k,v in errors.items()}))
    assert len(rows)==88
    summary={}
    for ch in ['variance_E','variance_I']:
        ss=[r for r in rows if r['channel']==ch]
        summary[ch]=dict(n=len(ss),new_DC_SNR_below10=sum(r['new_DC_SNR']<10 for r in ss),
            median_AC_SEM_over_DC=float(np.median([r['new_AC_SEM_over_DC'] for r in ss])),
            median_old_new_difference_in_combined_SEM=float(np.median([r['old_new_difference_in_combined_SEM'] for r in ss])),
            failures={k:sum(r['failed'][k] for r in ss) for k in predictions},
            exceeds_three_SEM={k:sum(r['exceeds_three_SEM_margin'][k] for r in ss) for k in predictions})
    write(DEST/'independent_audit.json',dict(status='PRECISION_REPEAT_AUDITED',rows=rows,summary=summary,
        scope='All88originallyeligible variance responses retained. Old measurements/decisions preserved. No coefficients fitted to this repeat; strong-waveform and network validation remain separate.'))
    log('VARIANCE PRECISION AUDIT',summary)


if __name__=='__main__':main()
