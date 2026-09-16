"""Keep record-time confounding visible after changing the state estimator."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scripts.patient_state_v1.common import RUN,write_json
from scripts.explore_e1146_seizure_interictal_association import blocked_rank_test
def main():
    root=RUN/'joint_preseizure_v1_22';df=pd.read_csv(root/'state_readouts.csv');df=df[df.coupled].reset_index(drop=True);inv=json.loads((RUN/'seizures.json').read_text());origin=float(np.load(RUN/'observations.npz')['origin_epoch']);times=np.array([(inv[int(s)-1]['onset']-origin)/3600 for s in df.sz]);label=df.label.to_numpy()=='TB';rows=[]
    for anchor,shift in [('first_seizure_original',(inv[0]['onset']-origin)/3600),('record_origin_sensitivity',0.)]:
        for width in [6.,12.,24.]:
            for measure in ['pre15_observed_tb_rate','pre15_uniform_tb_probability','pre15_event_weighted_tb_share']:
                test=blocked_rank_test(df[measure].to_numpy(),label,times-shift,width);estimable=test['n_mixed_time_blocks']>0;rows.append(dict(anchor=anchor,width_hours=width,measure=measure,status='LOW_RESOLUTION_EXPLORATORY' if estimable else 'NOT_ESTIMABLE',n_assignments=test['n_permutations'],n_mixed_strata=test['n_mixed_time_blocks'],rank_sum_residual=test['rank_sum_residual'],p=test['p'] if estimable else None))
    primary=[r for r in rows if r['anchor']=='first_seizure_original' and r['width_hours']==12 and r['measure']=='pre15_observed_tb_rate'][0];assert np.isclose(primary['p'],1/3)
    write_json(root/'record_time_control.json',dict(status='COMPLETE',results=rows,interpretation='State inference uses the same event-rate information as the original association, so a small unstratified p is not independent mechanism confirmation. Coarse time blocking has few or no exchangeable label assignments; temporal confounding is unresolved.',strata_definition='Primary inherited from original producer: fixed elapsed-time bins from first seizure, centered rank sum. Recording-origin phase is an explicitly separate sensitivity; neither anchor is selected for a favorable p.',original_12h_rate_result_reproduced=True))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
