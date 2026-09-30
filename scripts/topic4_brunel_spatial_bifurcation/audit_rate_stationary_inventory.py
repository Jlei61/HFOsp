"""Check whether registered local crossings can explain full root-count changes.

This is a necessary accounting test, not a completeness proof: opposite
crossings can cancel. Unaccounted roots imply missing crossings or a numerical
inconsistency; they are not assigned a bifurcation type without eigenvectors.
"""
from plot_rate_periodic_completion import *
from datetime import datetime, timezone


def main():
    z=np.load(RATE_OUT/'equilibrium_branch.npz')
    root=read(PERIODIC_OUT/'stationary_root_counts/summary.json')
    sites=sorted([r for r in root['rows'] if r['branch_index'] is not None
                  and r['status']=='PAIRED_OFFSET_AGREEMENT'],key=lambda r:r['branch_index'])
    folds=[]
    # The stationary arc is the actual continuation order, not sorted J.
    for k,q in enumerate(read(OUT/'critical_revision/fold_audit.json')['rows']):
        distance=np.linalg.norm(z['regional']-np.array(q['rates_hz']),axis=1)
        index=int(np.argmin(distance))
        folds.append(dict(label=f'LP{k+1}',approximate_branch_index=index,
            J_EE_core=q['J_EE_core']))
    hopfs=[];crossings={h['label'].split('_')[0]:h for h in additional_hopfs()}
    # Root-count accounting requires validated linear crossings, not the
    # subsequent nonlinear Hopf criticality/child-cycle classification.
    for path in (PERIODIC_OUT/'stationary_root_counts').glob('H*_full_state_validation.json'):
        check=read(path)
        if check.get('status')=='VALIDATED_IMAGINARY_PAIR_CROSSING':
            label=check['label']
            if label not in crossings:
                h=read(path.with_name(label+'_highorder_half.json'))
                h['validation_status']='LINEAR_CROSSING_VALIDATED_NONLINEAR_PENDING'
                crossings[label]=h
    for h in crossings.values():
        lo,hi=sorted(h['source_branch_indices'])
        coord=0 if 'Core-A' in h['transversality']['coordinate'] else 1
        direction=np.sign(z['regional'][hi,coord]-z['regional'][lo,coord])
        derivative=h['transversality']['real_exponent_derivative_per_ms_per_Hz']
        hopfs.append(dict(label=h['label'].split('_')[0],branch_bracket=[lo,hi],
            signed_root_change=int(2*np.sign(direction*derivative)),
            validation_status=h.get('validation_status','VALIDATED_LOCAL_HOPF')))
    rows=[]
    for a,b in zip(sites[:-1],sites[1:]):
        lo,hi=a['branch_index'],b['branch_index']
        hh=[h for h in hopfs if lo<=h['branch_bracket'][0] and h['branch_bracket'][1]<=hi]
        ff=[f for f in folds if lo<f['approximate_branch_index']<hi]
        observed=b['unstable_root_count']-a['unstable_root_count']
        remaining=observed-sum(h['signed_root_change'] for h in hh)
        # Every known generic stationary fold changes the count by at most 1.
        # This deliberately permits either sign instead of assuming it.
        minimum=max(0,abs(remaining)-len(ff))
        rows.append(dict(branch_indices=[lo,hi],J_endpoints=[a['J_EE_core'],b['J_EE_core']],
            observed_unstable_root_counts=[a['unstable_root_count'],b['unstable_root_count']],
            observed_change=observed,known_hopfs=hh,known_folds=ff,
            minimum_unaccounted_root_change=minimum,
            status='REGISTERED_CROSSINGS_INSUFFICIENT' if minimum else 'ACCOUNTING_NOT_CONTRADICTED'))
    out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),intervals=rows,
        status='INVENTORY_INCOMPLETE' if any(r['minimum_unaccounted_root_change'] for r in rows) else 'NECESSARY_ACCOUNTING_PASSES_ONLY',
        total_minimum_unaccounted_root_change=sum(r['minimum_unaccounted_root_change'] for r in rows),
        stationary_fold_locations=folds,hopf_crossings=hopfs,
        meaning='Lower bound conditional on the resolved contour counts and generic known crossings. A deficit is missing crossings or a numerical inconsistency, never an automatically named Hopf. Zero deficit does not exclude cancelling crossings. Fold branch positions use nearest full regional equilibrium and lie well inside the listed count intervals.')
    write(PERIODIC_OUT/'stationary_inventory_balance.json',out)
    print(out['status'],'minimum root deficit',out['total_minimum_unaccounted_root_change'],flush=True)
    for r in rows:
        if r['minimum_unaccounted_root_change']:print('UNACCOUNTED',r,flush=True)


if __name__=='__main__':main()
