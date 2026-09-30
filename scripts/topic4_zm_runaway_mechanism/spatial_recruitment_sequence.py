"""Locate the first persistent activity along the matched near-critical runs.

Use the existing spatial thresholds on sliding one-second windows. This is a
finite-time recruitment readout, not a bifurcation or causal-mode classifier.
"""
from native_path import *


def first_windows(rate,threshold,duty):
    n=len(rate)//10;bins=rate[:10*n].reshape(n,10,-1).mean(1)
    sums=np.vstack([np.zeros((1,bins.shape[1]),dtype=int),np.cumsum(bins>=threshold,axis=0)])
    occupied=sums[100:]-sums[:-100]>=int(round(100*duty))
    first=np.where(occupied.any(0),(np.argmax(occupied,axis=0)+100)*10,np.nan)
    return first,occupied


def main():
    s=model();runs=OUT/'runs'
    specs={
        'low_D_matched_12s':['rate_tail_D14497_D0.1449700_dt0.05'],
        'high_D_matched_12s':['rate_critical_finer_restart_D0.1449750_dt0.05','rate_tail_D144975_D0.1449750_dt0.05'],
    }
    rows=[];arrays={}
    for label,folders in specs.items():
        fields=[];regional=[]
        for name in folders:
            z=np.load(runs/name/'trajectory.npz');fields.append(z['field_E_hz'])
            groups=z['group_rate_hz'];reg=[]
            for i in range(3):
                use=s.E&(s.geo['group_region']==i)
                reg.append(groups[:,use]@(s.sizes[use]/s.sizes[use].sum()))
            regional.append(np.stack(reg,axis=1));counts=z['cell_counts']
        field=np.concatenate(fields)[:12000];region=np.concatenate(regional)[:12000]
        assert len(field)==12000
        for threshold,duty in [(20,.8),(20,.9),(50,.8),(50,.9)]:
            first,occupied=first_windows(field,threshold,duty)
            regfirst,_=first_windows(region,threshold,duty)
            population=occupied@(counts/counts.sum())
            recruits={}
            for fraction in [.01,.05,.1,.25]:
                ids=np.flatnonzero(population>=fraction)
                recruits[str(fraction)]=int((ids[0]+100)*10) if len(ids) else None
            cells=np.flatnonzero(np.isfinite(first));order=cells[np.argsort(first[cells])]
            q=dict(label=label,threshold_hz=threshold,duty=duty,window_ms=1000,
                   first_regional_mean_persistence_A_B_surround_ms=[float(t) if np.isfinite(t) else None for t in regfirst],
                   first_population_fraction_times_ms=recruits,
                   first_cells=[dict(cell=int(k),x_mm=float(k%20+.5),y_mm=float(k//20+.5),time_ms=float(first[k])) for k in order[:12]],
                   maximum_E_population_fraction=float(population.max()))
            rows.append(q);log('SPATIAL RECRUITMENT',q)
            if threshold==20 and duty==.8:arrays[label+'_first_persistent_ms']=first
    np.savez_compressed(OUT/'spatial_recruitment_sequence.npz',**arrays)
    write(OUT/'spatial_recruitment_sequence.json',dict(status='COMPLETE',
        history_match=read(OUT/'matched_critical_history_audit.json'),rows=rows,
        time_definition='End of first qualifying sliding 1-s window, relative to the common initial history',
        regional_caution='Regional mean persistence and individual cell persistence are distinct spatial observables',
        claim='Where persistence first appears in this matched trajectory pair, not which region causes a bifurcation'))


if __name__=='__main__':main()
