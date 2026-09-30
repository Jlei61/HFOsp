"""Independent readout of the paired early/late M interventions."""
from common import OUT,np,read,write,model
from onset_state_continuation import regional_weights
from scipy.ndimage import uniform_filter1d

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'


def intervals(sm,time,quiet=True):
    on=sm<5 if quiet else sm>=5
    edge=np.diff(np.r_[False,on,False].astype(int));a=np.flatnonzero(edge==1);b=np.flatnonzero(edge==-1)
    result=[]
    for first,last in zip(a,b):
        if quiet and last-first<20:continue
        result.append(dict(start_ms=int(time[0]-1+first),end_ms=int(time[0]-1+last),
            duration_ms=int(last-first),left_censored=bool(first==0),right_censored=bool(last==len(sm))))
    return result


def main():
    s=model(40);W=regional_weights(s);rows=[]
    for name in ['first_long_M_intervention','first_long_M_intervention_from2800']:
        p=BASE/name;assert read(p/'jobs.json')['status']=='COMPLETE'
        contract=read(p/'contract.json');start=contract['source_elapsed_ms'];duration=contract['duration_ms']
        source=np.load(contract['source']);expected=None;arms=[]
        for label in ['dynamic_M','held_M']:
            data=np.load(p/label/'trajectory.npz');rr=data['group_rate_hz'].astype(float)@W.T
            assert np.max(abs(rr-data['regional_rate_hz']))<1e-10
            tm=data['elapsed_time_ms'];assert np.array_equal(tm,np.arange(start+1,start+duration+1.))
            assert np.array_equal(data['Z'],source['syn'][5])
            assert np.all(data['M_current'][:,~s.E]==0)
            if label=='held_M':assert np.array_equal(data['M_current'],np.broadcast_to(source['syn'][4],data['M_current'].shape))
            old=read(p/label/'result.json');sm=uniform_filter1d(rr[:,1],10,mode='nearest')
            quiet=intervals(sm,tm);assert quiet==old['quiet_intervals']
            # Join across sub20ms dips using the registered quiet separators.
            complete=[]
            for left,right in zip(quiet[:-1],quiet[1:]):
                first=left['end_ms'];last=right['start_ms']
                if last>first:complete.append(dict(start_ms=first,end_ms=last,duration_ms=last-first))
            first_on=next((int(tm[j]-1) for j in range(len(sm)) if sm[j]>=5),None)
            post_entry_quiet=next((q for q in quiet if first_on is not None and q['start_ms']>first_on),None)
            if label=='dynamic_M':
                source_folder=BASE/'actual_D0300/from_interictal_history'
                block=np.concatenate([np.load(source_folder/f'block{j:02d}.npz')['group_rate_hz'] for j in range((start+duration+4999)//5000)])
                assert np.array_equal(data['group_rate_hz'],block[start:start+duration])
            row=dict(label=label,first_active_ms=first_on,first_following_20ms_quiet=post_entry_quiet,
                complete_between_quiet_activities=complete,quiet_intervals=quiet,
                quiet_fraction=float((sm<5).mean()),mean_coreA_rate_hz=float(rr[:,1].mean()),
                M_A_initial=float(source['syn'][4]@W[1]),M_A_final=float(data['M_current'][-1]@W[1]),
                M_held=label=='held_M',normal_arm_original_rates_bitwise=True if label=='dynamic_M' else None)
            arms.append(row)
        row=dict(start_ms=start,duration_ms=duration,source=str(p),arms=arms)
        rows.append(row);write(p/'independent_audit.json',dict(status='PASS',**row,
            scope='A single matched conditional deterministic trajectory, not independent replication. All-group M clamping does not isolate Core-A M, and no finite censoring is a permanent-state or bifurcation certificate.'))
    write(BASE/'M_intervention_comparison.json',dict(status='PAIRED_READOUT_AUDIT_PASS',rows=rows,
        bifurcation_type='NOT_ESTABLISHED',model_promoted=False))


if __name__=='__main__':main()
