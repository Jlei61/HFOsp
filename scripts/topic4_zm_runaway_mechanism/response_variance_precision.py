"""New-seed, higher-precision repeat of all originally eligible variance assays."""
from common import *
from lif_mc import condition,run
import argparse

DEST=OUT/'response_variance_precision'


def main(device):
    c=read(OUT/'response_variance_precision_contract.json');DEST.mkdir(exist_ok=True)
    source=read(OUT/'response_error_decomposition.json')['rows'];conditions={}
    for r in source:
        if not r['counted'] or r['channel']=='mean':continue
        q=r['workpoint'];channel=1 if r['channel']=='variance_E' else 2
        key=tuple(q[k] for k in ['pop','theta','mu','ve','vi'])+(channel,)
        if key not in conditions:conditions[key]=dict(workpoint=q,channel=channel,frequencies={0.})
        conditions[key]['frequencies'].add(r['frequency_hz'])
    meta=[];pars=[]
    for j,item in enumerate(conditions.values()):
        q=item['workpoint'];channel=item['channel']
        for freq in sorted(item['frequencies']):
            meta.append(dict(workpoint_id=j,workpoint=q,channel=channel,frequency_hz=freq))
            pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],
                      amplitude=c['variance_relative_amplitude'],freq_hz=freq,channel=channel))
    pars=np.array(pars);P=len(pars);R=c['replicates'];duration=c['duration_ms']
    checkpoint=DEST/'checkpoint.json';data=DEST/'observations.npy'
    if checkpoint.exists():
        progress=read(checkpoint);assert progress['meta']==meta
        done=np.array(progress['done'],bool);obs=np.load(data,mmap_mode='r+')
        assert obs.shape==(P,R,4)
    else:
        assert not data.exists(),'Unregistered partial observations need review before reusing'
        done=np.zeros(P,bool);obs=np.lib.format.open_memmap(data,mode='w+',dtype=np.float64,shape=(P,R,4))
        write(checkpoint,dict(status='RUNNING',meta=meta,done=done))
    log('VARIANCE PRECISION START',P,'conditions',R,'replicates','completed',sum(done))
    for start in range(0,P,c['batch_conditions']):
        ids=np.arange(start,min(P,start+c['batch_conditions']));todo=ids[~done[ids]]
        if not len(todo):continue
        values=run(pars[todo],R,duration,c['burn_ms'],c['seed'],device=device)
        obs[todo]=values;obs.flush();done[todo]=True
        write(checkpoint,dict(status='RUNNING',meta=meta,done=done));log('VARIANCE PRECISION',int(done.sum()),'/',P)
    rows=[]
    for i,item in enumerate(meta):
        p=pars[i];amplitude=p[4]*p[2 if item['channel']==1 else 3]
        samples=(obs[i,:,0]+1j*obs[i,:,1])/(duration*amplitude)*1000
        mean=samples.mean();sem=float(np.sqrt(np.mean(abs(samples-mean)**2)/R))
        rows.append(dict(**item,measured=[mean.real,mean.imag],complex_SEM=sem,
                         rate_hz=float((obs[i,:,2]+obs[i,:,3]).mean()/2/duration*1000)))
    write(DEST/'result.json',dict(status='COMPLETE',rows=rows,contract=c,
          scope='New-seed paired +/- local colored-LIF response, no network or parameter change. Original eligibility fixed before this run.'))
    write(checkpoint,dict(status='COMPLETE',meta=meta,done=done));log('VARIANCE PRECISION COMPLETE',P)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    main(p.parse_args().device)
