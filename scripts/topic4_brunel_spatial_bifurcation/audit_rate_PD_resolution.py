"""Check positivity of every accepted PD parent and its classified child.

This CPU audit can withdraw insufficient evidence, never promote a root.
Original root/mode files remain intact for temporal refinement.
"""
from rate_periodic import *
import shutil


def main():
    rows=[]
    history=PERIODIC_OUT/'stability_coverage/attempt_history';history.mkdir(exist_ok=True)
    for label in ['PD_double_low','PD_double_upper','PD_A_return']:
        path=PERIODIC_OUT/(label+'_validation.json');q=read(path)
        roots=sorted([read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')],key=lambda v:v['N'])
        root=roots[-1];z=np.load(root['orbit']);r=z['r'];minimum=float(resample(r,4*len(r),axis=0).min()*1000)
        row=dict(label=q['label'],orbit=root['orbit'],N=root['N'],check_N=4*root['N'],minimum_rate_Hz=minimum,
                 prior_status=q['status'],source=str(path))
        if minimum < -1e-9:
            if q['status']=='VALIDATED_PD':
                archived=history/(label+f'_before_positive_profile_review_{time.time_ns()}.json')
                shutil.copy2(path,archived)
                q.update(status='VALIDATION_RESOLUTION_REVIEW_PENDING',prior_validation_source=str(archived),
                    criticality='RESOLUTION_REVIEW_PENDING',continuous_positivity_check=row,
                    scope='Existing antiperiodic and monodromy evidence retained; parent time mesh must be refined to resolve negative between-node rates before renewed acceptance.')
                write(path,q)
            row['status']='RESOLUTION_REVIEW_PENDING'
        else:row['status']='POSITIVE_PROFILE';row['scope']='Positivity only; existing root/mode acceptance is separate.'
        rows.append(row)
    child=PERIODIC_OUT/'PD_child_classification.json';q=read(child)
    if q.get('status')=='SUBCRITICAL_PD' and q.get('minimum_group_rate_hz',0)<-1e-9:
        archived=history/f'PD1_child_before_positive_profile_review_{time.time_ns()}.json';shutil.copy2(child,archived)
        q.update(status='CRITICALITY_RESOLUTION_REVIEW_PENDING',prior_classification_source=str(archived),
                 child_at_checked_orbit='UNRESOLVED_PENDING_POSITIVE_PROFILE_REFINEMENT',
                 numerical_note='The old child endpoint has negative Fourier undershoot. Its stability and this criticality label are withheld until temporal refinement and independent checks finish.')
        write(child,q)
    write(PERIODIC_OUT/'PD_resolution_review.json',dict(rows=rows,child_status=q['status'],
          scope='A negative numerical interpolation is a resolution failure, not a new physical state. No equations changed and no clipping. Withdrawal does not prove absence of the candidate PD.'))
    print('PD RESOLUTION REVIEW',rows,flush=True)


if __name__=='__main__':main()
