"""Check zero-horizon equivalence and future-data exclusion in memory forecasts."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.horizon_memory_controls import predict

def main():
    table=pd.read_csv(RUN/'two_scale_horizon_v1_16/predictions.csv.gz');table=table[table.horizon_minutes==0];best={};checks=[]
    for directory,kind in [('two_timescale_free_tau_v1_13','ou2'),('brownian_fast_marks_v1_12','brownian')]:
        for p in (RUN/directory/'fits').glob('fold*.json'):
            r=json.loads(p.read_text());j=r['job']
            if r['status']!='COMPLETE' or j['history'] or (kind=='ou2' and j['method']!='adf') or (kind=='brownian' and j['drift']):continue
            k=(int(j['scope'][4:]),f'{kind}_carry{int(j["carry"])}')
            if k not in best or r['loglik']>best[k][0]['loglik']:best[k]=(r,p)
    for (fold,model),(r,p) in best.items():
        g=table[(table.fold==fold)&(table.model==model)];actual=np.load(p.with_suffix('.npz'))['predict_tb'][g['index'].to_numpy()];error=np.max(abs(actual-g.p_tb.to_numpy()));assert error<2e-6;checks.append(dict(fold=fold,model=model,n=len(g),max_probability_difference=error))
    d=np.load(RUN/'observations.npz');y=d['y'].copy();target=np.array([2000]);previous=np.array([1000]);elapsed=np.array([d['t'][2000]-d['t'][1000]]);same=np.array([True]);theta=np.array([np.log(1.),np.log(10.),-.8]);p0=predict(y,d['dt'],d['reset'],target,previous,elapsed,same,theta)[0];y[1001:]=1-y[1001:];p1=predict(y,d['dt'],d['reset'],target,previous,elapsed,same,theta)[0];assert np.array_equal(p0,p1)
    write_json(RUN/'two_scale_horizon_v1_16/numerical_audit.json',dict(status='PASS',zero_horizon_checks=checks,future_label_perturbation_changes_memory_prediction=False,scope='Zero-horizon two-state forecasts match original forward predictions; later labels do not enter stored earlier memory counts'))
    print(max(c['max_probability_difference'] for c in checks))
if __name__=='__main__':main()
