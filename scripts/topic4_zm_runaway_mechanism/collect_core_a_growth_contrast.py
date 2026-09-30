"""Compare actual-state perturbation growth on the same physical time window.

The two numerical steps start from the same retained physical state, with
fine physical-lag interpolation. They are numerical controls, not replicates.
"""
from common import OUT,np,read,write,model
from onset_state_continuation import regional_weights
from scipy.ndimage import uniform_filter1d

BASE=OUT/'core_a_bifurcation_type_20260924'


def main():
    s=model(40);W=regional_weights(s);rows=[]
    for name,parent,labels in [
        ('returning_D0328122',BASE/'actual_returning_growth',['coarse','fine']),
        ('late_return_D0343158',BASE/'actual_sustained_growth',['coarse','fine']),
        ('strong_depletion',BASE,['depleted_coarse','depleted_fine'])]:
        conditions=read(parent/'conditions.json')
        assert conditions[labels[0]]['initial']==conditions[labels[1]]['initial']
        original=np.load(conditions[labels[0]]['initial'])['syn'][5]
        A=s.E&(s.geo['group_region']==0)
        d=float(1-np.average(original[A],weights=s.sizes[A]))
        for label in labels:
            folder=parent/label;c=conditions[label]
            assert read(folder/'jobs.json')['status']=='COMPLETE'
            assert read(folder/'local_state_audit.json')['status']=='AUDIT_PASS'
            assert read(folder/'implementation_check.json')['status']=='PASS'
            g=read(folder/'tangent_result.json');assert g['status']=='COMPLETE'
            # Exactly the same first5s after the declared common source,
            # rather than each run's different last5s window.
            z=np.load(folder/'block00.npz');assert np.array_equal(z['Z'],original)
            rates=z['group_rate_hz'].astype(float)@W.T;assert len(rates)==5000
            assert np.max(abs(rates-z['regional_rate_hz']))<1e-9
            sm=uniform_filter1d(rates[:,1],10,mode='nearest')
            edge=np.diff(np.r_[False,sm<5,False].astype(int))
            quiet=[[int(a),int(b)] for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if b-a>=20]
            blocks=np.asarray(g['one_second_blocks_per_s'])
            row=dict(condition=name,label=label,D_A=d,Z_A=1-d,dt_ms=c['dt_ms'],
                common_source=str(c['initial']),source_prior_same_field_ms=c['prior_same_field_elapsed_ms'],
                growth_window_after_source_ms=[2000,5000],
                common_window_growth_per_s=float(blocks[2:5].mean()),
                common_one_second_growth_per_s=blocks[2:5].tolist(),
                all_post_alignment_growth_per_s=g['finite_time_growth_per_s'],
                all_post_alignment_duration_ms=g['duration_ms']-2000,
                activity_window_after_source_ms=[0,5000],
                quiet_intervals_after_source_ms=quiet,
                minimum_Core_A_10ms_hz=float(sm.min()),
                mean_rates_global_A_B_surround=rates.mean(0).tolist(),
                Z_identity=True,all_M_dynamic=True)
            rows.append(row)
    write(BASE/'actual_growth_same_window_contrast.json',dict(
        status='MATCHED_WINDOW_READOUT_PASS',rows=rows,
        observation_unit='One actual complete-state history at each local Z field; dt.05/.025 are paired numerical controls, not independent runs or seeds.',
        definition='Full original delayed-state variational growth,100ms normalization; discard2s directional alignment, compare exactly the common2--5s interval. Quiet applies original10ms mean<5Hz for>=20ms to the same first5s after each source.',
        interpretation='Signs describe finite-time expansion on actual trajectories. An unstable short-cycle root is a separate invariant object. No asymptotic chaos, crisis, stable cycle or complete bifurcation certificate follows from these signs alone.',
        model_promoted=False))


if __name__=='__main__':main()
