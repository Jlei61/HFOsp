"""Second pass: 4x replicates (2048, same CRN streams 0..2047 include the original 0..511) for grid
points with first-pass rate < 30 Hz, where MC relative error is largest. Replaces those entries."""
from build_transfer_table import *
import argparse
def main(a):
    folder=DEST/'transfer_table'
    for pop in a.pops:
        path=folder/f'table_{pop}.npz'
        while not path.exists():time.sleep(20)
        z=dict(np.load(path,allow_pickle=True))
        if int(z.get('replicates_low',0))>=a.replicates:log('already refined',pop);continue
        X,SE,SI,pars=conditions(pop);rate=z['rate_hz'].ravel();sel=np.flatnonzero(rate<a.threshold);log(pop,'refining',len(sel),'of',len(rate))
        counts=np.zeros((len(sel),a.replicates),np.uint16);started=time.time();ck=folder/f'refine_checkpoint_{pop}.npz';done=np.zeros(len(sel),bool)
        if ck.exists():
            q=np.load(ck);counts[:]=q['counts'];done[:]=q['done']
        for start in range(0,len(sel),a.batch):
            idx=np.arange(start,min(len(sel),start+a.batch))
            if done[idx].all():continue
            obs=run(pars[sel[idx]],a.replicates,z['duration_ms'],z['burn_ms'],int(z['seed']),crn=True,device=a.device)
            counts[idx]=obs[:,:,2].astype(np.uint16);done[idx]=True;np.savez(ck,counts=counts,done=done);log(pop,f'{done.sum()}/{len(sel)}',f'{time.time()-started:.0f}s')
        # consistency: first 512 streams must reproduce the original counts exactly (CRN)
        orig=z['counts'][sel];assert np.array_equal(orig,counts[:,:orig.shape[1]]),'CRN stream mismatch'
        T=float(z['duration_ms']);rate=z['rate_hz'].ravel().copy();sem=z['sem_hz'].ravel().copy()
        rate[sel]=counts.mean(1)/T*1000;sem[sel]=counts.std(1)/np.sqrt(a.replicates)/T*1000
        shape=z['rate_hz'].shape
        np.savez_compressed(path,**{k:v for k,v in z.items() if k not in('rate_hz','sem_hz')},rate_hz=rate.reshape(shape),sem_hz=sem.reshape(shape),
                            replicates_low=a.replicates,low_rate_threshold_hz=a.threshold,refined_indices=sel,refined_counts=counts)
        ck.unlink();log('REFINED',pop)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pops',nargs='+',default=['E','I']);p.add_argument('--replicates',type=int,default=2048);p.add_argument('--threshold',type=float,default=30.)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--device',type=int,default=0);main(p.parse_args())
