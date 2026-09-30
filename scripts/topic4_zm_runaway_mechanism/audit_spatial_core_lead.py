"""Check physical core onset order against the existing whole-field direction."""
from common import OUT,BASE,model,np,read,write,log
from native_readouts import readouts
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from datetime import datetime

DEST=OUT/'native_current_memory'
NATIVE=OUT.parents[0]/'fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/chunks'


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'core_lead.json').exists()
    write(DEST/'core_lead_contract.json',dict(status='REGISTERED_BEFORE_SCORING',created_local=datetime.now().astimezone().isoformat(),
        question='Is the one-direction rate result also present in physicalcore timing, or only in the previously defined globalcentroid readout?',
        readout='Exact754A/786B Ecell-count mean rates,5ms smoothing. Within each alreadyeligiblecompleteevent, first50percentownpeak crossing; requireeachpeak>=20Hz. lag=tB-tA,with+/-2mstie band. Report all events, including ineligiblecorepairs.',
        comparison='Native andboth0.5mm runs, common1-9.42swindow, same existingglobalcompleteeventdefinition. Newdiagnostic does not replace/waive olddirectionmetric.'))
    s20=model(20);count=np.bincount(s20.geo['group_cell'][s20.E],weights=s20.sizes[s20.E],minlength=400)
    n=np.load(BASE/'native_reference/seed9108401_readouts.npz');chunks=[np.load(p) for p in sorted(NATIVE.glob('*.npz'))]
    t_native=np.concatenate([z['time_ms'] for z in chunks]);cores=np.concatenate([z['regions_1ms'][:,:3] for z in chunks]).astype(float)/np.array([754,786,30460])*1000
    assert np.array_equal(t_native,n['t']);assert np.max(abs(cores@np.array([754,786,30460])/32000-n['allE']))<1e-4
    sources=[('native',20,None)]+[(label,40,OUT/'conditioned_refractory_spatial_resolution'/label) for label in ['recorded_drive_expected','recorded_drive_binomial_seed1']]
    rows=[];summary=[]
    for label,grid,source in sources:
        if source is None:t=t_native;field=n['rate_cells'];core=cores[:,:2]
        else:
            z=np.load(source/'trajectory.npz');t=z['time_ms'];field=z['field_E_hz'];s=model(grid);r=z['group_rate_hz'].astype(float);core=[]
            for k in [0,1]:
                mask=s.E&(s.geo['group_region']==k);core.append(r[:,mask]@s.sizes[mask]/s.sizes[mask].sum())
            core=np.column_stack(core)
        whole=field.astype(float)@(count/count.sum());sm=uniform_filter1d(whole,10,mode='nearest');core=uniform_filter1d(core,5,axis=0,mode='nearest')
        events,_,_,_=readouts(t,field,count,label);selected=[]
        for e in events:
            if not 1000<=e['start_ms']<9420:continue
            a=int(np.searchsorted(t,e['start_ms']));b=a+int(e['duration_ms'])
            if not(a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all()):continue
            segment=core[a:b];peak=segment.max(0);eligible=bool((peak>=20).all())
            times=[float(t[a+np.argmax(segment[:,j]>=.5*peak[j])]) for j in [0,1]] if eligible else [None,None]
            lag=times[1]-times[0] if eligible else None
            item=dict(label=label,start_ms=e['start_ms'],duration_ms=e['duration_ms'],centroid_direction_mm=e['direction_axis_mm'],peak_core_hz=peak.tolist(),eligible_core_pair=eligible,core_halfpeak_time_ms=times,lag_B_minus_A_ms=lag)
            selected.append(item);rows.append(item)
        lags=[r['lag_B_minus_A_ms'] for r in selected if r['eligible_core_pair']]
        summary.append(dict(label=label,events=len(selected),eligible_core_pairs=len(lags),A_first=sum(x>2 for x in lags),B_first=sum(x<-2 for x in lags),ties=sum(abs(x)<=2 for x in lags),median_lag_ms=float(np.median(lags)) if lags else None))
    write(DEST/'core_lead.json',dict(status='CORE_ORDER_DIAGNOSTIC_COMPLETE',rows=rows,summary=summary,scope='Descriptiveonehistory. Coreorder andwholefielddirection answerdifferentquestions; neitherestablishes causalpropagation or bifurcation.'))
    log('CORE ORDER',summary)


if __name__=='__main__':main()
