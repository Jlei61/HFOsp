"""Bounded one-second observation audit at the already fitted parameters."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_two_state_filter import worker,OUT

def main():
    best={}
    for p in (RUN/'joint_two_state_v1_20/fits').glob('b1_*.json'):
        r=json.loads(p.read_text());j=r['job'];assert r['success'] and r['state_converged'];key=(j['scope'],j['coupled'])
        if key not in best or r['loglik']>best[key][1]['loglik']:best[key]=(p,r)
    assert len(best)==8;jobs=[]
    for (scope,c),(p,r) in best.items():
        for q in [40,80]:
            for rf in [False,True]:jobs.append(dict(id=f'b1_{scope}_c{int(c)}_q{q}_rf{int(rf)}',source=str(p),seconds=1,scope=scope,coupled=c,order=q,rate_first=rf))
    write_json(OUT/'fine_contract.json',dict(n_jobs=32,question='Does reducing observation bins from 5 to 1 second change the approximate forward advantage?',scope='Existing trained parameters only; 40/80 nodes, both projection orders. Does not resolve Gaussian approximation bias.'))
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();assert r['status']=='COMPLETE';print(json.dumps(dict(done=i+1,total=32,id=r['job']['id'],status=r['status'])),flush=True)
    write_json(OUT/'fine_status.json',dict(status='COMPLETE',n_jobs=32))

if __name__=='__main__':main()
