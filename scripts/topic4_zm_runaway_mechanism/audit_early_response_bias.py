"""Independent original-readout audit of one response-bias counterfactual."""
from common import OUT,BASE,model,np,read,write,log
from native_readouts import readouts,window_stats
from refractory_spatial_resolution import mapping,projections
from scipy.ndimage import uniform_filter1d

DEST=OUT/'early_response_bias_diagnostic/causal_counterfactual'
EARLY=OUT/'native_early_surround_inputs'


def main():
    assert read(DEST/'network_jobs.json')['status']=='COMPLETE'
    s=model(40);coarse=model(20);parent,_=mapping(coarse,s);projection=projections(s,coarse,parent)
    C,count=projection[20];weights=count/count.sum();groups=np.load(EARLY/'membership.npz')['selected_groups']
    parts=[]
    for path in sorted((EARLY/'inputs').glob('*.npz')):
        with np.load(path) as z:parts.append(z['spikes'].reshape(500,10,s.P).sum(1,dtype=np.uint32))
    native_counts=np.concatenate(parts);native=native_counts/s.sizes*1000;del parts
    original=np.load(BASE/'native_reference/seed9108401_readouts.npz')
    native_field=(C@native.T).T
    assert np.max(abs(native_field-original['rate_cells'][:3000]))<1e-4
    native_t=original['t'][:3000]
    assert np.array_equal(native_t,np.arange(3000)+.5)
    nt=EARLY/'replay/runs/eta0.0005_s9108401/checkpoints/t3000ms.npz'
    with np.load(nt) as z:
        cells=s.geo['cell_group'];total=np.bincount(cells,minlength=s.P)
        nz=np.bincount(cells,weights=z['slow__z'],minlength=s.P)/total
        nm=.0005*np.bincount(cells,weights=z['slow__m'],minlength=s.P)/total
    rows=[];trace={};qa=[]
    for label,source in [('native',None),('parent',OUT/'physical_delay_count_rate/recorded_drive_binomial_seed1/trajectory.npz'),('bias_counterfactual',DEST/'trajectory.npz')]:
        if source is None:
            r=native;field=native_field;zf=nz;mf=nm;t=native_t
        else:
            z=np.load(source);r=z['group_rate_hz'][:3000].astype(float);field=z['field_E_hz'][:3000].astype(float)
            t=z['time_ms'][:3000];assert np.array_equal(t,np.arange(1,3001.))
            reconstructed=(C@r.T).T;spatial_error=float(np.max(abs(reconstructed-field)));assert spatial_error<1e-4
            global_error=float(np.max(abs(field@weights-z['global_E_hz'][:3000])));assert global_error<1e-4
            index=np.flatnonzero(z['state_time_ms']==3000);assert len(index)==1
            zf=z['Z'][index[0]].astype(float);mf=z['M_current'][index[0]].astype(float)
            assert np.isfinite(r).all() and r.min()>=0 and zf.min()>=0 and zf.max()<=1
            assert np.array_equal(zf[~s.E],np.ones((~s.E).sum())) and mf.min()>=0 and mf[s.E].max()>0
            D_error=abs(1-zf[s.E]@s.mean_weights-z['D'][index[0]]);assert D_error<1e-6
            counts=r*s.sizes/1000;integer_error=float(np.max(abs(counts-np.rint(counts))))
            assert integer_error<1e-4
            for mask,span in [(s.E,2),(~s.E,1)]:
                x=counts[:-1,mask]+counts[1:,mask] if span==2 else counts[:,mask]
                assert np.max(x-s.sizes[mask])<1e-4
            if label=='bias_counterfactual':
                assert int(z['final_tick'][0])==60000
                assert np.array_equal(z['final_own_history'],z['final_emitted_history'])
                h=z['final_own_history']*s.sizes*.05;assert abs(h-np.rint(h)).max()<1e-9
            qa.append(dict(label=label,spatial_error_hz=spatial_error,global_error_hz=global_error,D_error=float(D_error),count_integer_error=integer_error,physical_bounds=True))
        events,summary,whole,sm=readouts(t,field,count,label);strict=[]
        for e in events:
            a=int(np.searchsorted(t,e['start_ms']));b=a+int(e['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():strict.append(e)
        keep=(t>=500)&(t<3000)
        regional=[]
        for name,mask in [('All E',s.E)]+[(name,s.E&(s.geo['group_region']==k)) for k,name in enumerate(['Core A','Core B','Surround'])]+[('I',~s.E)]:
            w=s.sizes[mask]/s.sizes[mask].sum();rate=r[:,mask]@w
            regional.append(dict(region=name,mean_rate_hz=float(rate[keep].mean()),Z3000=float(zf[mask]@w),M3000_mv=float(mf[mask]@w)))
        item=dict(label=label,window_ms=[500,3000],original_active_episodes=window_stats(events,500,3000),
            strict_complete_events=window_stats(strict,500,3000),quiet_fraction=float((sm[keep]<5).mean()),
            high_entry_ms=summary['high_onset_ms'],regional=regional,D3000=float(1-zf[s.E]@s.mean_weights),
            primary_tail_persistent_fraction=float(weights[(field[-1000:]>50).mean(0)>=.9].sum()))
        rows.append(item)
        trace[label+'_global_hz']=whole
        trace[label+'_time_ms']=t
        trace[label+'_mean_field_hz']=field[keep].mean(0)
        trace[label+'_selected50ms_counts']=r[:,groups].reshape(60,50,len(groups)).sum(1)[10:]*s.sizes[groups]/1000
    # Selection-window bins use raw interval endpoints [500,3000) as in the
    # earlier local assay; the primary native readout uses its original t clock.
    fixed=np.load(EARLY/'fixed_readout.npz')
    assert np.max(abs(trace['native_selected50ms_counts']-fixed['native_counts']))<1e-8
    np.savez_compressed(DEST/'audit_readouts.npz',time_ms=t,cell_counts=count,**trace)
    write(DEST/'result.json',dict(status='READOUT_AUDIT_PASS',rows=rows,numerical_checks=qa,
        original_readout_definitions_preserved=True,statistical_unit='One native trajectory and one paired seed1 rate counterfactual. No independent-event population inference.',
        local_transfer='FAIL remains. Network run is separately registered causal sensitivity, not a passed model-validation branch.',
        clock='Native original1ms bin centers0.5,1.5,...; rate originalbin endpoints1,2,... retained inprimaryreadouts. Same rawtime intervals; no eventalignment or silentclocktranslation.',
        source_parameter_target='Only earlyconditionalLIF responses; no native onset/rate outputs used for fitting.',
        model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    log('EARLY BIAS COUNTERFACTUAL',[(r['label'],r['strict_complete_events'],r['quiet_fraction'],r['D3000']) for r in rows])


if __name__=='__main__':main()
