"""Select and refine a rate recurrence as a seed, never as a certificate."""
from common import model, np, read, write, log
from onset_state_continuation import DEST
from scipy.optimize import minimize_scalar
import argparse


def seed(label,tail_ms=5000):
    folder=DEST/label;audit=read(folder/'independent_audit.json')
    assert audit['status']=='AUDIT_PASS'
    minima=audit['recurrence'].get('best_local_minima',[])
    jobs=read(folder/'jobs.json');block=jobs['completed_blocks'][-1]
    x=np.load(folder/f'block{block:02d}.npz')['group_rate_hz'].astype(float)[-tail_ms:]
    s=model(40)
    if tail_ms!=5000:
        from audit_onset_state_continuation import recurrence
        rec=recurrence(x,s);minima=rec.get('best_local_minima',[])
    # This numerical screen only avoids launching a cycle solver from an
    # obviously nonrecurrent trace; it is not an acceptance threshold.
    good=[r for r in minima if r['relative_MSE']<1e-3]
    if not good:
        write(folder/'period_seed_screen.json',dict(status='NO_CLOSE_RATE_RECURRENCE',
            candidates=minima,scope='No periodicity or nonperiodicity conclusion.'))
        log('NO CLOSE RATE SEED',label,minima[:3]);return
    first=min(good,key=lambda r:r['lag_ms'])['lag_ms']
    w=s.sizes/s.sizes.sum();den=((x-x.mean(0))**2@w).mean()
    width=int(np.ceil(first+2));n=len(x)
    def objective(T):
        k=int(T);a=T-k
        y=(1-a)*x[k:k+n-width]+a*x[k+1:k+1+n-width]
        return float(((y-x[:n-width])**2@w).mean()/den)
    fit=minimize_scalar(objective,bounds=(first-1.,first+1.),method='bounded',options={'xatol':1e-9})
    result=dict(source=f'{label}/block{block:02d}.npz',period_ms=float(fit.x),
        tail_ms=tail_ms,
        relative_rate_MSE=float(fit.fun),integer_candidate_ms=first,
        definition='Linear interpolation of1ms averaged full E/I group rates; period seed only, not full delayed-state closure or Floquet certificate.')
    write(folder/'rate_period_seed.json',result)
    write(folder/'period_seed_screen.json',dict(status='RATE_SEED_AVAILABLE_NOT_CERTIFIED',result=result))
    log('ONSET RATE PERIOD SEED',label,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--tail-ms',type=int,default=5000)
    a=p.parse_args();assert 1000<=a.tail_ms<=5000;seed(a.label,a.tail_ms)
