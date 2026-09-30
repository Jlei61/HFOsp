"""Check scientific identities used by the expanded figures, without refitting."""
from common import *
from model import SpatialBrunel
from response import characteristic,white
import mpmath as mp

def complex_response_check(s,r,J,lam,v):
    # Independent special-function evaluation at growing complex roots, in
    # addition to the earlier imaginary-axis checks of the wide-bound solver.
    d=s.phi(*s.moments(r,J),details=True);mp.mp.dps=60;rows=[]
    ids={int(np.argmin(d[1])),int(np.argmax(d[2])),int(np.argmax(np.where(s.E,abs(v),0))),int(np.argmax(np.where(~s.E,abs(v),0)))}
    for k in sorted(ids):
        lo,hi,sig=[float(a[k]) for a in d[1:4]];tm=float(s.tm[k]);rate=float(d[0][k]);z=mp.mpc(lam*tm)
        def u(y):
            return mp.hyperu(z/2,mp.mpf('.5'),y*y) if y<=0 else (mp.sqrt(mp.pi)/mp.gamma((1+z)/2)*mp.hyp1f1(z/2,mp.mpf('.5'),y*y)
                +2*mp.sqrt(mp.pi)*y/mp.gamma(z/2)*mp.hyp1f1((1+z)/2,mp.mpf('1.5'),y*y))
        den=u(hi)-u(lo)
        ref=np.array([complex(rate/sig*(mp.diff(u,hi)-mp.diff(u,lo))/den/(1+z)),
            complex(rate/sig**2*(mp.diff(u,hi,2)-mp.diff(u,lo,2))/den/(2+z))])
        got=white(lam,np.array([lo]),np.array([hi]),np.array([sig]),np.array([tm]),np.array([rate]))[:,0]
        err=float(np.linalg.norm(got-ref)/max(np.linalg.norm(ref),1e-300));assert err<1e-7,(J,k,lo,hi,err)
        rows.append(dict(group=k,bounds=[lo,hi],relative_error=err))
    return rows

def main():
    base=OUT/'expanded';s=SpatialBrunel(response='calibrated_full');summary=read(base/'readouts/result.json')
    geo=np.load(OUT/'operators/g40/geometry.npz');e=geo['population']==0;size=geo['group_size'];reg=geo['group_region']
    counts=np.bincount(geo['group_cell'][e],weights=size[e],minlength=1600)
    regions=np.array([size[e&(reg==k)].sum() for k in range(3)])
    assert regions.tolist()==[754,786,30460] and sum(counts)==32000
    runs=[]
    for q in summary['rows']:
        rec=read(q['source']);z=np.load(rec['trajectory']);x=np.load(rec['exact_readout'])
        assert np.array_equal(z['time_ms'],np.arange(1,10001))
        assert z['field_E_hz'].shape==(10000,1600) and x['lfp_raw'].shape==(20000,15)
        assert x['contact_names'].tolist()==summary['contact_names']
        total=z['field_E_hz'].astype(float)@counts/1000
        reference=z['regional_rates_hz'][:,:3]@regions/1000
        mismatch=float(abs(total-reference).max());assert mismatch<.002
        for kind in ['firing','current_hfo']:
            ob=q[kind];order=np.array(ob['within_shaft_order_probability'],float)
            part=np.array(ob['participation'],float);rank=np.array(ob['mean_rank'],float)
            good=np.isfinite(order)
            if good.any():assert np.max(abs((order+order.T-1)[good]))<1e-12
            if ob['N']==0:assert np.isnan(part).all() and np.isnan(rank).all()
            else:assert np.isnan(rank[part==0]).all()
        for d in q['dynamics']:
            assert d['peak_count']==len(d['burst_windows_ms'])
            assert (d['IEI_CV'] is None)==(d['peak_count']<3)
        exact=read(Path(rec['exact_readout']).with_suffix('.json'))
        if 'same_trajectory_regional_rate_max_error' in exact:assert exact['same_trajectory_regional_rate_max_error']<1e-10
        runs.append(dict(tag=q['tag'],J_EE_core=q['J_EE_core'],maximum_field_count_mismatch=mismatch,
            eligible_events={k:q[k]['N'] for k in ['firing','current_hfo']}))
    folds=read(base/'folds/result.json')['rows'];assert len(folds)==11
    for q in folds:
        assert q['equilibrium_residual']<1e-9 and q['eigenvector_residual']<1e-8
        assert abs(q['transversality'])>1e-7 and abs(q['quadratic_coefficient'])>1e-7
    modes=[]
    for J in [1.3,1.6,2.]:
        z=np.load(base/f'modes/upper_J{J:g}/modes.npz');r=z['rates'];errors=[]
        assert abs(s.residual(r,J)).max()<1e-9
        for lam,v in zip(z['roots'],z['vectors']):
            err=float(np.linalg.norm(characteristic(s,r,J,lam)@v)/np.linalg.norm(v));errors.append(err)
            assert lam.real>0 and err<1e-8
        i=int(np.argmax(z['roots'].real));independent=complex_response_check(s,r,J,z['roots'][i],z['vectors'][i])
        modes.append(dict(J_EE_core=J,positive_complex_pairs=len(errors),largest_eigenvector_residual=max(errors),independent_complex_response_checks=independent))
    trace=read(base/'persistent_mode/result.json');assert trace['status']=='COMPLETE'
    overlap=min(q['previous_vector_overlap'] for q in trace['rows'][1:]);assert overlap>.99
    assert all(q['lambda_per_ms'][0]>0 for q in trace['rows'])
    out=dict(status='PASS',native_trajectories=runs,additional_stationary_folds=11,original_fold=1,high_branch_eigenpairs=modes,
        traced_points=len(trace['rows']),minimum_neighboring_mode_overlap=overlap,
        scope='Numerical and readout consistency only; not nonlinear rate/SNN equivalence or high-rate response calibration',
        continuation_disposition={'low_arclength':'Stopped after positivity-related step shrinkage; plotted prefix steps 0-781; continued in low_arclength_v2',
            'high_arclength':'Superseded numerical attempt; not plotted','low_arclength_v2':'Bounded 1200-step continuation',
            'high_arclength_v2':'Bounded 1200-step continuation','persistent_mode':'Complete 29-point mode continuation'},human_visual_acceptance=False)
    write(base/'delivery_checks.json',out);print(json.dumps(clean(out),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
