#!/usr/bin/env python3
"""Verify every retained physical observation against immutable natural source."""
import time
import numpy as np
from campaign import ROOT,read,write,sha
import native_campaign as n


def main():
    root=ROOT/'exit_state_reconstruction';run=root/'runs/source10_to16p70'
    source=n.native.SOURCE/'runs'/n.native.NAME
    assert read(run/'result.json')['status']=='COMPLETE'
    checks=[]
    for stream in n.STREAMS:
        if not (source/stream).exists():continue
        for path in sorted((run/stream).glob('*.npz')):
            start,end=map(int,path.stem.split('_'))
            matches=[q for q in (source/stream).glob('*.npz') if int(q.stem.split('_')[0])<=start and int(q.stem.split('_')[1])>=end]
            assert len(matches)==1,(stream,path)
            old=matches[0];a,b=map(int,old.stem.split('_'))
            with np.load(path) as x,np.load(old) as y:
                assert set(x.files)==set(y.files),(stream,x.files,y.files)
                for key in x.files:
                    if key in ['start_step','end_step']:
                        assert int(x[key])==(start if key=='start_step' else end);continue
                    if key in ['keys','region_names','variables']:expected=y[key]
                    else:
                        stride=(b-a)//len(y[key]);assert stride*len(y[key])==b-a
                        assert (start-a)%stride==0 and (end-a)%stride==0
                        expected=y[key][(start-a)//stride:(end-a)//stride]
                    assert np.array_equal(x[key],expected,equal_nan=x[key].dtype.kind in 'fc'),(stream,path.name,key)
                    checks.append(dict(stream=stream,chunk=path.name,key=key,elements=int(x[key].size),bitwise=True))
    state=n.native.read_pickle(run/'checkpoint.pkl')['engine'];z=state['slow']['z'][:32000];k=state['termination_mechanism']['sahp_g']
    gate=dict(status='PASS',source=str(source),replay=str(run),interval_s=[10.,16.7],arrays=len(checks),checks=checks,
              checkpoint_sha256=sha(run/'checkpoint.pkl'),meanZ=float(z.mean()),meanK=float(k.mean()),G_raw=float(30*state['global_feedback_response']['global_state']),
              verified_epoch=time.time(),scope='All retained physical observations over6.7s match bitwise. No independent source checkpoint exists at16.7s; whole-state continuation to20s is a separate stronger check. Diagnostic replay is not a new autonomous seed.')
    write(root/'observation_gate.json',gate)
    print({k:v for k,v in gate.items() if k!='checks'},flush=True)


if __name__=='__main__':main()
