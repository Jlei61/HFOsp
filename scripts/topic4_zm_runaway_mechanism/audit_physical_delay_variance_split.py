"""Independent diagnosis of delay-index units in private variance splitting."""
from common import OUT,model,np,read,write,log
from physical_delay_variance_split import physical_split,covariance_kernel
from shared_variance_network_sensitivity import split
from scipy.integrate import quad
from scipy import sparse
from datetime import datetime

DEST=OUT/'physical_delay_variance_split'


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'result.json').exists()
    s=model(40);correct,qa=physical_split(s)
    old={dt:split(s,dt)[0] for dt in [.1,.05,.025]}
    checks=[];rows=[];saved={}
    source=OUT/'fine_rate_frozen_Z_fields'
    probes={}
    for tm in [9000,9420,9870]:
        z=np.load(source/f'native_Z{tm}_held/trajectory.npz')
        probes[tm]=z['group_rate_hz'][-1000:].astype(float).mean(0)/1000
    for k,kind in enumerate(['ampa','gaba']):
        tr=s.rise[k];td=s.decay[k];row,col,a=s.raw[k];total_pair=np.asarray(s.raw[k+2][2].sum(1)).ravel()
        # Independent continuous-time impulse autocorrelation integral.
        def impulse(t):return (np.exp(-t/td)-np.exp(-t/tr))/(td-tr)
        norm=quad(lambda t:impulse(t)**2,0,np.inf,epsabs=1e-12,epsrel=1e-11)[0]
        quadrature=[]
        for lag in [0.,.1,1.,3.,float(s.delays.max()-s.delays.min())]:
            integral=quad(lambda t:impulse(t)*impulse(t+lag),0,np.inf,epsabs=1e-12,epsrel=1e-11)[0]/norm
            analytic=float(covariance_kernel(lag,tr,td));error=abs(integral-analytic)
            assert error<1e-10;quadrature.append(dict(lag_ms=lag,error=error))
        # Actual nonzero delay pairs, without the preassembled kernel matrix.
        selected=np.unique(np.linspace(0,a.shape[0]-1,31).astype(int));manual=[]
        private=correct[kind].tocoo();basekeys=row*s.P+col;where=np.searchsorted(basekeys,private.row*s.P+private.col%s.P)
        remaining=np.bincount(where,weights=private.data,minlength=len(row))
        for j in selected:
            start,end=a.indptr[j:j+2];d=a.indices[start:end];weight=a.data[start:end]
            shared=0.
            for u in range(len(d)):
                for v in range(len(d)):
                    lag=abs(s.delays[d[u]]-s.delays[d[v]])
                    kv=quad(lambda t:impulse(t)*impulse(t+lag),0,np.inf,epsabs=1e-12,epsrel=1e-10)[0]/norm
                    shared+=weight[u]*weight[v]*kv/s.sizes[col[j]]
            error=abs(remaining[j]+shared-total_pair[j])/max(total_pair[j],1e-30)
            assert error<1e-10;manual.append(float(error))
        full=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr();fixed=correct[kind]
        legacy_1_error=float(np.max(abs(fixed.data-old[.1][kind].data)))/float(np.max(full.data))
        assert legacy_1_error<1e-12
        diff=fixed.data-old[.05][kind].data
        assert diff.min()>-1e-12
        checks.append(dict(synapse=kind,quadrature=quadrature,actual_pair_quadrature_relative_max=max(manual),
            agrees_with_legacy_at_native_step_relative_error=legacy_1_error,
            legacy_dt05_corrected_max_operator_difference=float(np.max(abs(diff))),
            legacy_dt05_dt025_max_difference=float(np.max(abs(old[.05][kind].data-old[.025][kind].data)))))
        for tm,rate in probes.items():
            history=np.tile(rate,len(s.delays));total=full@history;new=fixed@history
            for dt in [.05,.025]:
                previous=old[dt][kind]@history;bias=(new-previous)/np.maximum(total,1e-30)
                saved[f'{kind}_t{tm}_dt{dt}_variance_deficit_relative_full']=bias
                for region,name in enumerate(['Core A','Core B','Surround E']):
                    mask=s.E&(s.geo['group_region']==region);w=s.sizes[mask]/s.sizes[mask].sum()
                    rows.append(dict(synapse=kind,probe_field_ms=tm,legacy_dt_ms=dt,region=name,
                        mean_total_recurrent_variance_deficit=float(w@bias[mask]),
                        mean_private_variance_increase_fraction=float(w@((new-previous)/np.maximum(previous,1e-30))[mask]),
                        largest_total_recurrent_variance_deficit=float(bias[mask].max())))
    np.savez_compressed(DEST/'probe_fields.npz',**saved)
    for kind in ['ampa','gaba']:sparse.save_npz(DEST/f'physical_private_{kind}.npz',correct[kind])
    write(DEST/'result.json',dict(status='PHYSICAL_DELAY_UNIT_ERROR_CONFIRMED',created_local=datetime.now().astimezone().isoformat(),
        physical_delay_spacing_ms=.1,legacy_mistake='Covariance used abs(delay_index_difference)*integrator_dt instead of actualphysicaldelaydifference. Transport still uses0.1msdelaybins.',
        consequence='Atdt.05/.025 old sharedvariance is too large and remainingprivatevariance too small. Reducing dt changes the modeled variance split as well as numericalaccuracy.',
        corrected='Evaluate normalized filter autocovariance at actual s.delays differences. No integration timestep argument in physical_split.',
        checks=checks,closure_checks=qa,rows=rows,
        scope='Exact unit andstationaryPoisson identity audit. Rates are imposedmeans fromcompletedoldmodel; not trajectories orproofthatthisaccountsfornativeDerror. Renewal and cross-population covariance approximation remains.',
        running_baseline_changed=False,new_network_runs=0,model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    log('PHYSICAL DELAY SPLIT AUDIT COMPLETE',checks)


if __name__=='__main__':main()
