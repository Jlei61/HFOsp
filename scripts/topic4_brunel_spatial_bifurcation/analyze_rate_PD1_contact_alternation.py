"""Frozen three-contact-metric comparison of the two halves of one PD1 child.

Repeated exact cycles provide equal phase coverage, not independent samples.
Matching the first and second event positions within each half prevents
pooled opposite directions from hiding changes. Bin origins are a numerical
observer sensitivity check; thresholds and event eligibility stay frozen.
"""
from plot_rate_branch_completion import *
from expanded_readouts import observer,smooth2,describe,compare,OLD
from scipy.interpolate import CubicSpline
from scipy.stats import rankdata
from collections import Counter


def main():
    source=DATA/'PD1_physical_child.json';display=read(source)
    snapshot=read(display['source']);path=Path(display['orbit'])
    row=next(v for v in snapshot['rows'] if Path(v['orbit']).resolve()==path.resolve())
    assert row['physical_check']['filter_state_check']['positive']
    assert row['physical_check']['maximum_group_defect_Hz']<1e-6
    contract_source=OLD/'observer_firing.json';contract=read(contract_source)
    names=contract['contact_names'];s=RateField();z=np.load(path);T=float(z['T'])
    geometry=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    assert geometry['contact_names'].tolist()==names
    # The display uses this same phase for both halves. Linear contact
    # projection commutes with periodic spline interpolation of the rates.
    r=np.roll(z['r'],-display['phase_shift_samples'],axis=0)
    contact=r@s.geo['contact_rate_weights']*1000;N=len(r)
    spline=CubicSpline(np.arange(N+1)*T/N,np.r_[contact,contact[:1]],bc_type='periodic')
    stop=int(16*T)//2*2;records=[]
    for offset in [0.,.5,1.,1.5]:
        t=(np.arange(stop*4)+.5)/4+offset
        rates=spline(t%T).reshape(stop,4,15).mean(1)
        env=smooth2(rates.reshape(-1,2,15).sum(1)/1000)
        ob=observer.observe(env.T,2.,contract)
        centroids=np.asarray(ob['centroid_ms'],float).reshape(-1,15)+offset
        anchors=np.array([np.mean(e['window_ms'])+offset for e in ob['events']])
        ids=np.flatnonzero((anchors>=4*T)&(anchors<12*T))
        primary=np.array([i for i in ids if ob['events'][i]['primary_eligible']],int)
        quarter=np.floor((anchors%T)/(T/4)).astype(int)
        groups={}
        selections={'first_half':primary[quarter[primary]<2],
                    'second_half':primary[quarter[primary]>=2]}
        for k in range(4):selections[f'event_position_{k+1}']=primary[quarter[primary]==k]
        for name,indices in selections.items():
            groups[name]=dict(event_indices=indices,metrics=describe(centroids[indices],names))
        contrasts={}
        for name,left,right in [('whole_halves','first_half','second_half'),
                                ('matched_first_event','event_position_1','event_position_3'),
                                ('matched_second_event','event_position_2','event_position_4')]:
            metrics=compare(groups[left]['metrics'],groups[right]['metrics'],names)
            contrasts[name]=dict(left=left,right=right,shaft_balanced_absolute_differences=metrics)
        # Check actual matched events as well as pooled metrics, so opposite
        # rank changes could not disappear through averaging across repeats.
        pairs=[];missing_pairs=[];cycle=np.floor(anchors/T).astype(int)
        same_shaft=np.array([[left[:3]==right[:3] for right in names] for left in names])
        for repetition in range(4,12):
            for event_position in range(2):
                left=primary[(cycle[primary]==repetition)&(quarter[primary]==event_position)]
                right=primary[(cycle[primary]==repetition)&(quarter[primary]==event_position+2)]
                if len(left)!=1 or len(right)!=1:
                    missing_pairs.append(dict(repetition=repetition,event_position=event_position,left=left,right=right));continue
                x,y=centroids[left[0]],centroids[right[0]]
                vx,vy=np.isfinite(x),np.isfinite(y);joint=vx&vy
                ranks=[]
                for value,mask in [(x,vx),(y,vy)]:
                    rank=np.full(15,np.nan);rank[mask]=(rankdata(value[mask])-1)/(mask.sum()-1);ranks.append(rank)
                supported=np.triu(same_shaft&joint[:,None]&joint[None,:],1)
                ox=np.sign(x[None,:]-x[:,None]);oy=np.sign(y[None,:]-y[:,None])
                pairs.append(dict(repetition=repetition,event_position=event_position,
                    left_event=int(left[0]),right_event=int(right[0]),participating_set_equal=bool(np.array_equal(vx,vy)),
                    maximum_joint_normalized_rank_difference=float(np.max(abs(ranks[0][joint]-ranks[1][joint]))) if joint.any() else None,
                    supported_within_shaft_pairs=int(supported.sum()),changed_within_shaft_pairs=int(np.sum(ox[supported]!=oy[supported])),
                    participating_contacts=[names[i] for i in np.flatnonzero(joint)]))
        record=dict(bin_origin_ms=offset,qualified_events=len(primary),detected_events=len(ids),
            excluded=Counter(reason for i in ids for reason in ob['events'][i]['primary_exclusion_reasons']),
            quarter_counts=[int(np.sum(quarter[primary]==k)) for k in range(4)],
            groups=groups,contrasts=contrasts,contact_centroids_ms=centroids,
            matched_event_checks=pairs,missing_or_ambiguous_event_pairs=missing_pairs,
            observation=ob,interior_event_indices=ids,qualified_event_indices=primary)
        records.append(record)
        print('PD1 CONTACTS',offset,record['quarter_counts'],contrasts,flush=True)
    ranges={}
    for name in records[0]['contrasts']:
        rows=[v['contrasts'][name]['shaft_balanced_absolute_differences'] for v in records]
        keys=['mean_rank_difference','within_shaft_order_difference','participation_difference']
        ranges[name]={}
        for key in keys:
            values=[v[key] for v in rows if v is not None and v[key] is not None]
            ranges[name][key]=dict(estimable_bin_origins=len(values),minimum=min(values) if values else None,
                maximum=max(values) if values else None)
    write(DATA/'PD1_consecutive_half_contact_metrics.json',dict(status='FROZEN_OBSERVER_COMPARISON_COMPLETE',
        orbit=str(path),J_EE_core=float(z['J']),T_ms=T,physical_check=row['physical_check'],
        phase_shift_samples=display['phase_shift_samples'],observer=str(contract_source),contact_names=names,
        records=records,bin_origin_sensitivity_ranges=ranges,
        statistical_unit='One exact deterministic doubled orbit. Eight interior repetitions ensure equal phase coverage; neither detected events nor four bin origins are independent realizations.',
        contrast='Consecutive halves at identical J and on the same child orbit; additionally compare the first and second event positions separately. This is not a root-versus-child parameter comparison.',
        missingness='A shaft without jointly estimable contact order yields an undefined shaft-balanced metric, not zero error.',
        scope='The original mean normalized rank, within-shaft jointly participating contact order and contact participation metrics, with the frozen firing observer. These describe this periodic orbit; no patient distribution fit, SNN equivalence, child stability or propagation-template bifurcation is inferred.'))


if __name__=='__main__':main()
