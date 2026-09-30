"""Independent review of the frozen rate closure; does not alter its equations.

Reuses native runs only when their parameter/input contracts match. Local
colored-LIF assays are not full-network validation. CPU Fourier reconstruction
checks the full nine-state RHS between, as well as at, collocation nodes.
"""
from rate_field import *
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import argparse

AUDIT = RATE_OUT / 'model_audit_20260918'
PER = RATE_OUT / 'periodic_completion'


def local_assays(s):
    result = []
    for pop, source in [('E', 'local_response_check_v2'), ('I', 'local_response_check_I')]:
        rows = read(OUT / source / 'result.json')['rows']
        dc = {(r['core'], r['group']): r for r in rows if not r['frequency_hz']}
        for row in rows:
            g = row['group']; f = row['frequency_hz']; base = dc[row['core'], g]
            obs = complex(*row['measured_chi_hz_per_mv'])
            pred = complex(*base['predicted_chi_hz_per_mv']) * s.filter_response(2j*np.pi*f/1000)[g]
            result.append(dict(population=pop, channel='mean', core=row['core'], group=g,
                frequency_hz=f, measured=obs, current_closure=pred,
                relative_complex_error=abs(pred-obs)/abs(obs)))
    rows = read(OUT / 'local_variance_response_check/result.json')['rows']
    dc = {(r['population'], r['core'], r['group'], r['input_variance']): r for r in rows if not r['frequency_hz']}
    for row in rows:
        pop, core, g, kind = [row[k] for k in ['population', 'core', 'group', 'input_variance']]
        f = row['frequency_hz']; lam = 2j*np.pi*f/1000; base = dc[pop, core, g, kind]
        obs = complex(*row['measured_chi_hz_per_mv2'])
        # The assay modulates diffusion drive upstream of the synaptic filter.
        pred = complex(*base['predicted_chi_hz_per_mv2']) * s.filter_response(lam)[g] / (1+lam*s.tau['EI'.index(kind)]/2)
        result.append(dict(population=pop, channel='variance_'+kind, core=core, group=g,
            frequency_hz=f, measured=obs, current_closure=pred,
            relative_complex_error=abs(pred-obs)/abs(obs)))
    summaries = []
    for pop in 'EI':
        for channel in ['mean', 'variance_E', 'variance_I']:
            for frequencies in [[6.], [2.,4.,6.,10.,20.,40.]]:
                rr = [q for q in result if q['population']==pop and q['channel']==channel and q['frequency_hz'] in frequencies]
                e = np.array([q['relative_complex_error'] for q in rr])
                summaries.append(dict(population=pop, channel=channel, frequencies_hz=frequencies,
                    n=len(e), median_relative_complex_error=np.median(e), maximum_relative_complex_error=e.max()))
    write(AUDIT/'local_assay_recheck.json', dict(rows=result, summaries=summaries,
        unit='Previously simulated local colored-Gaussian LIF assay, at its recorded working point; not native-network events',
        independent_validation=False, limitation='Mean-response data informed the earlier susceptibility fit. Variance was not fit by the current two-filter closure.'))
    for q in summaries: print('ASSAY', q, flush=True)


def matched_native(s):
    from expanded_readouts import OLD, observer, smooth2, describe, compare
    contract = read(OLD/'observer_firing.json'); names = contract['contact_names']; rows = []
    for J in [.6, .942, 1.3, 1.6, 2.]:
        candidates = list((RATE_OUT/'runs/main').glob(f'J{J:.7f}/trajectory.npz')) + list((RATE_OUT/'runs/pilot').glob(f'J{J:.7f}/trajectory.npz'))
        native = (AUDIT if J==.942 else OUT/'expanded')/f'native/J{J:.4f}_s199771'
        if not candidates or not (native/'exact_readout.npz').exists(): continue
        rate = candidates[0]; rd=np.load(rate); nd=np.load(native/'trajectory.npz'); exact=np.load(native/'exact_readout.npz')
        rc=read(rate.parent/'contract.json'); nc=read(native/'contract.json')
        assert rc['J_EE_core']==nc['J_EE_core']==J and rc['Z']==nc['Z']=='fixed 1'
        assert nc['common_OU']=='zero fluctuation' and 'reset' in nc['initial_state']
        assert abs(nc['private_input_per_ms']-s.nu)<1e-12
        assert rd['contact_names'].tolist()==exact['contact_names'].tolist()==names
        end = min(len(rd['time_ms']), len(nd['time_ms'])); burn = 2000
        if end <= burn+1000: continue
        pair = {}
        for label, z, contact in [('rate',rd,rd['contact_rate_hz']), ('native',nd,exact['contact_rate_hz'])]:
            nr=end//2*2; env=smooth2(contact[:nr].reshape(-1,2,15).sum(1)/1000)
            ob=observer.observe(env.T,2.,contract); mu=np.asarray(ob['centroid_ms'],float).reshape(-1,15)
            ids=np.array([i for i in ob['primary_event_indices'] if ob['events'][i]['window_ms'][0]>=burn and ob['events'][i]['window_ms'][1]<=end],int)
            smooth=gaussian_filter1d(z['regional_rates_hz'][:end,:3],5,axis=0); dyn=[]
            for k in range(2):
                y=smooth[burn:,k]; peaks=find_peaks(y,height=20,prominence=10,distance=60)[0]
                iei=np.diff(peaks); dyn.append(dict(peaks=len(peaks), median_peak_interval_ms=np.median(iei) if len(iei) else None,
                    peak_interval_CV=np.std(iei)/np.mean(iei) if len(iei)>1 else None,
                    fraction_below_5Hz=np.mean(y<5)))
            pair[label]=dict(mean_E_rates_A_B_surround_Hz=z['regional_rates_hz'][burn:end,:3].mean(0),
                dynamics=dyn, metrics=describe(mu[ids],names), qualified_event_centroids_ms=mu[ids])
        rows.append(dict(J_EE_core=J, window_ms=[burn,end], native_source=str(native), rate_source=str(rate),
            native_contract=nc, rate_contract=rc, **pair,
            metric_differences=compare(pair['rate']['metrics'],pair['native']['metrics'],names)))
        print('MATCH', J, {k:dict(mean=v['mean_E_rates_A_B_surround_Hz'],dynamics=v['dynamics'],events=v['metrics']['N']) for k,v in pair.items()},rows[-1]['metric_differences'],flush=True)
    write(AUDIT/'matched_native_recheck.json',dict(rows=rows,contact_names=names,
        statistical_unit='One native seed and one deterministic trajectory per J, same physical input moments, frozen Z, dynamic M, zero common OU fluctuation',
        limitation='Descriptive check only: no new acceptance threshold, no independent seed interval, not proof of full dynamic equivalence. Undefined SCL timing is not counted as successful recovery.'))


def conditional_orders():
    from expanded_readouts import describe
    result=read(AUDIT/'matched_native_recheck.json');names=result['contact_names'];out=[]
    for row in result['rows']:
        if row['J_EE_core'] not in [.942,1.3]:continue
        for kind in ['native','rate']:
            f=Path(row['native_source'])/'trajectory.npz' if kind=='native' else Path(row['rate_source'])
            z=np.load(f);v=gaussian_filter1d(z['regional_rates_hz'],5,axis=0)
            a,b=[find_peaks(v[:,k],height=20,prominence=10,distance=60)[0]+1 for k in range(2)]
            d=a[:,None]-b[None,:];pairs=[]
            for i in range(len(a)):
                if not len(b):break
                j=np.argmin(abs(d[i]))
                if abs(d[i,j])<=150 and np.argmin(abs(d[:,j]))==i and min(a[i],b[j])>=2000:pairs.append((a[i],b[j]))
            pairs=np.array(pairs).reshape(-1,2);mu=np.array(row[kind]['qualified_event_centroids_ms'],float).reshape(-1,15);labels=[]
            for e in mu:
                if not len(pairs):labels.append('?');continue
                dist=abs(pairs.mean(1)-np.nanmean(e));j=np.argmin(dist)
                labels.append(('A' if pairs[j,0]<pairs[j,1] else 'B') if dist[j]<150 else '?')
            rec=dict(J_EE_core=row['J_EE_core'],kind=kind,pairs=pairs,event_labels=labels,
                conditional={k:describe(mu[np.array(labels)==k],names) for k in 'AB'})
            out.append(rec);print('DIRECTION',row['J_EE_core'],kind,'core pairs A/B',int((pairs[:,0]<pairs[:,1]).sum()),int((pairs[:,0]>pairs[:,1]).sum()),flush=True)
    differences=[]
    for J in [.942,1.3]:
        matched=[x for x in out if x['J_EE_core']==J]
        if len(matched)!=2:continue
        for lead in 'AB':
            aa,bb=[x['conditional'][lead] for x in matched]
            for shaft in ['ICL','SCL']:
                ix=np.array([i for i,n in enumerate(names) if n.startswith(shaft)])
                for metric in ['mean_rank','within_shaft_order_probability','participation']:
                    diff=abs(np.array(aa[metric],float)-np.array(bb[metric],float))
                    diff=diff[np.ix_(ix,ix)] if diff.ndim==2 else diff[ix];valid=np.isfinite(diff)
                    differences.append(dict(J_EE_core=J,lead_core=lead,shaft=shaft,metric=metric,
                        jointly_estimable_entries=int(valid.sum()),mean_absolute_difference=float(diff[valid].mean()) if valid.any() else None))
    write(AUDIT/'core_order_conditioned_recheck.json',dict(rows=out,differences=differences,
        definition='Mutually nearest A/B peaks within 150ms, common 5ms rate smoothing; contact events assigned to closest core-pair midpoint within 150ms.',
        limitation='Core lead-order diagnostic, not patient TA/TB labels. One seed; no distribution matching requirement imposed.'))


def inventory():
    from plot_rate_periodic_completion import families, critical
    fs=families(); crit=critical(); rows=[]
    mon=[]
    for path in (PER/'floquet').glob('*.json'):
        q=read(path);mon.append(dict(path=str(path),orbit=q['orbit'],J=q['J_EE_core'],T=q['T_ms'],
            phase_defect=q.get('phase_tangent_relative_defect'),requested_multipliers=q['requested_multipliers']))
    spectral=[dict(path=str(f), **read(f)) for f in (PER/'spectral_floquet').glob('*.json')]
    for key, rr in fs.items():
        hit=[]
        for i,q in enumerate(rr):
            if any(abs(q['J_EE_core']-m['J'])<1e-8 and abs(q['T_ms']-m['T'])<.01 for m in mon):hit.append(i)
        rows.append(dict(family=key,points=len(rr),points_with_matching_monodromy=len(hit),matching_indices=hit,
            endpoint_sources=[rr[0]['path'],rr[-1]['path']]))
    missing=[]
    for q in crit:
        meshes={read(f)['N'] for f in PER.glob(q['label']+'_N*.json')}
        if len(meshes)<2:missing.append(dict(label=q['label'],meshes=sorted(meshes)))
    write(AUDIT/'coverage_recheck.json',dict(families=rows,monodromy_files=mon,
        spectral_files=len(spectral),single_mesh_critical_points=missing,
        counting_rule='Monodromy match by J within 1e-8 and period within .01ms; a match is evidence of computation, not an automatic stability certificate.',
        complete=False))
    print('COVERAGE', rows,'SINGLE_MESH',missing,flush=True)


def spatial_check(s):
    """Direction-conditioned field check, on the same 1-mm physical grid.

    The two amplitude cutoffs are descriptive sensitivity checks, not fitted
    acceptance criteria. Align only to the leading core peak, with no warping.
    """
    from scipy.stats import spearmanr
    rows=[q for q in read(AUDIT/'core_order_conditioned_recheck.json')['rows'] if q['J_EE_core']==.942]
    if len(rows)!=2:return
    xy=s.geo['original_positions'][:32000];ij=np.floor(xy*2).astype(int)
    counts=np.bincount(ij[:,1]*40+ij[:,0],minlength=1600).reshape(40,40)
    coarse=counts.reshape(20,2,20,2).sum((1,3));curves={};arrays={};offset=np.arange(-80,161)
    mask=s.E&(s.geo['group_region']<2)
    core_cells=np.unique(s.geo['group_cell'][mask]);outside=np.ones(400,bool);outside[core_cells]=False
    for q in rows:
        kind=q['kind'];path=(AUDIT/'native/J0.9420_s199771/trajectory.npz' if kind=='native' else RATE_OUT/'runs/main/J0.9420000/trajectory.npz')
        z=np.load(path);field=z['field_E_hz'].astype(float)
        if kind=='native':
            field=(field.reshape(-1,40,40)*counts).reshape(-1,20,2,20,2).sum((2,4))/coarse
            field=field.reshape(-1,400)
        field=gaussian_filter1d(field,5,axis=0)
        pairs=np.asarray(q['pairs'],int)
        for lead in 'AB':
            use=(pairs[:,0]<pairs[:,1]) if lead=='A' else (pairs[:,1]<pairs[:,0])
            anchors=pairs[use].min(1)-1;anchors=anchors[(anchors>=80)&(anchors+160<len(field))]
            mean=np.array([field[a+offset] for a in anchors]).mean(0)
            baseline=mean[offset<-40].mean(0);signal=np.maximum(mean-baseline,0.)
            curves[kind,lead]=signal;arrays[kind+'_'+lead]=mean
    diagnostics=[]
    for lead in 'AB':
        a=curves['rate',lead];b=curves['native',lead]
        for threshold in [10.,20.]:
            active=(a.max(0)>=threshold)&(b.max(0)>=threshold)&outside
            ta=(offset[:,None]*a).sum(0)/np.maximum(a.sum(0),1e-12)
            tb=(offset[:,None]*b).sum(0)/np.maximum(b.sum(0),1e-12)
            diagnostics.append(dict(lead_core=lead,peak_threshold_hz=threshold,common_active_surround_cells=int(active.sum()),
                centroid_order_spearman=float(spearmanr(ta[active],tb[active]).statistic),
                median_absolute_centroid_difference_ms=float(np.median(abs(ta[active]-tb[active])))))
    np.savez_compressed(AUDIT/'direction_mean_spatial_fields.npz',offset_ms=offset,**arrays)
    write(AUDIT/'direction_mean_spatial_fields.json',dict(rows=diagnostics,
        definition='Native 0.5mm E-rate field averaged to 1mm by actual E-cell counts. Means conditioned on leading-core identity, aligned to leading peak, 5ms smoothing, no temporal warping.',
        unit='Single native seed; direction means from all detected paired core events, not independent network replicates.',
        limitation='Exploratory spatial check; 10/20Hz common-participation thresholds are sensitivity diagnostics, not predeclared acceptance thresholds. Cells intersecting either core excluded from spatial-order correlation.'))
    print('SPATIAL',diagnostics,flush=True)


def full_rhs_check(s, path, factor=4):
    z=np.load(path);r=z['r'];N=len(r);M=N*factor;J=float(z['J']);T=float(z['T']);K=N//2+1
    rf=np.fft.rfft(r,axis=0)/N; cf=np.empty((9,K,s.P),complex);af=np.empty((4,K,s.P),complex)
    for k,v in enumerate(rf):
        lam=2j*np.pi*k/T
        af[:,k]=[a@v for a in s.matrices(J,lam)]
        a,b,aa,bb=af[:,k];target=v/s.filter_response(lam)
        qa=s.tm*s.area[0]*a/(1+lam*s.rise[0]);ia=qa/(1+lam*s.decay[0])
        qg=s.tm*s.area[1]*b/(1+lam*s.rise[1]);ig=qg/(1+lam*s.decay[1])
        cf[:,k]=[target/(1+lam*s.tf),target/(1+lam*s.ts),qa,ia,qg,ig,
            s.tm*s.area[0]**2*aa/(1+lam*s.tau[0]/2),s.tm*s.area[1]**2*bb/(1+lam*s.tau[1]/2),.5*s.E*v/(1+lam*1000)]
    def at_mesh(c):
        padded=np.zeros((M//2+1,s.P),complex);padded[:K]=c*M;padded[K-1]*=.5
        return np.fft.irfft(padded,n=M,axis=0)
    yy=np.array([at_mesh(c) for c in cf]);aa=np.array([at_mesh(c) for c in af])
    lam=2j*np.pi*np.arange(K)[:,None]/T
    dydt=np.array([at_mesh(c*lam) for c in cf]);res=np.empty_like(yy)
    phi=np.empty((M,s.P));quad=SpatialBrunel(quadrature=96);qerr=0.
    for i in range(M):
        res[:,i]=s.rhs(yy[:,i],aa[:,i])-dydt[:,i]
        mu=yy[3,i]-yy[5,i]-yy[8,i]+s.private_mu;ve=yy[6,i]+s.private_ve;vi=yy[7,i]
        phi[i]=s.phi(mu,ve,vi)
        if i%max(1,M//128)==0:qerr=max(qerr,float(abs(quad.phi(mu,ve,vi)-phi[i]).max()*1000))
    rr=s.alpha*yy[0]+(1-s.alpha)*yy[1];freq=2j*np.pi*np.arange(M//2+1)[:,None]/T
    H=s.alpha/(1+freq*s.tf)+(1-s.alpha)/(1+freq*s.ts)
    integrated=rr-np.fft.irfft(np.fft.rfft(phi,axis=0)*H,n=M,axis=0)
    original_H=s.alpha/(1+lam*s.tf)+(1-s.alpha)/(1+lam*s.ts)
    original_discrete=rr[::factor]-np.fft.irfft(np.fft.rfft(phi[::factor],axis=0)*original_H,n=N,axis=0)
    regional_errors=[]
    for k in range(3):
        mask=s.E&(s.geo['group_region']==k)
        regional_errors.append(float(abs(np.average(integrated[:,mask],axis=1,weights=s.geo['group_size'][mask])).max()*1000))
    row=dict(path=str(path),J=J,T_ms=T,N=N,check_N=M,stored_collocation_error_Hz=float(z['residual']),
        original_mesh_CPU_collocation_error_Hz=float(abs(original_discrete).max()*1000),
        oversampled_residual_at_original_nodes_Hz=float(abs(integrated[::factor]).max()*1000),
        between_nodes_integrated_error_Hz=float(abs(integrated).max()*1000),
        between_nodes_regional_error_A_B_surround_Hz=regional_errors,
        full_RHS_rate_error_Hz_per_ms=float(abs(res[:2]).max()*1000),
        full_RHS_linear_state_max_abs=float(abs(res[2:]).max()),
        transfer_48_vs_96_quadrature_max_Hz=qerr,
        minimum_interpolated_group_rate_Hz=float(rr.min()*1000),
        refractory_rate_bound_excess_Hz=float(np.maximum(rr-1/s.ref,0).max()*1000))
    print('CPU_RHS',row,flush=True);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--orbits',nargs='*');p.add_argument('--base',action='store_true');args=p.parse_args()
    s=RateField();AUDIT.mkdir(exist_ok=True)
    assert np.all(s.Z==1)
    if args.base:local_assays(s);matched_native(s);conditional_orders();spatial_check(s);inventory()
    if args.orbits:
        rows=[]
        for name in args.orbits:
            path=PER/'orbits'/name;rows.append(full_rhs_check(s,path))
            write(AUDIT/'independent_full_rhs_checks.json',dict(rows=rows,method='Independent CPU harmonic state/delay reconstruction and full RHS check on 4x temporal mesh; no clipping, no re-solving.'))


if __name__=='__main__':main()
