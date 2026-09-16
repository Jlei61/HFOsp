"""Read-only crosswalk of the SNN FIT label bank to the current patient labels."""
import sys,json,pickle
from pathlib import Path
ROOT=Path('/home/honglab/leijiaxin/HFOsp');WT=ROOT/'.worktrees/topic4-substrate-autapse-fix';sys.path.insert(0,str(WT));sys.path.insert(1,str(ROOT))
import numpy as np,pandas as pd
RUN=ROOT/'results/topic5_patient_state_inference/e1146_drift_hypothesis_v1/overnight_20260909'
def main():
    evaluator_path=WT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl'
    with evaluator_path.open('rb') as f:ev=pickle.load(f)
    # Only FIT indices and labels are consumed. No CAL/PROBE outcome or waveform access.
    ix=np.asarray(ev.index['FIT'],int);m=np.asarray(ev.fit_labels,int)
    target_path=ROOT/'results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/shaft_aware_patient_training_target.npz';target=np.load(target_path)
    rawid=target['patient_train_event_indices'][ix];blocks=target['patient_train_block_ids'][ix];assert np.array_equal(blocks,np.asarray(ev.blocks)[ix])
    z=np.load(ROOT/'results/topic5_preseizure_template_share/epilepsiae_1146/event_index.npz');field=json.loads((ROOT/'results/interictal_propagation_masked/template_gradient_fields_all_events_timing_plus_space/per_subject/epilepsiae_1146.json').read_text())
    print(field['template_discovery'].keys())
    labels=np.asarray(field['template_discovery']['event_labels'],int)
    print('label sample',np.unique(z['template_label'],return_counts=True))
    a=pd.DataFrame(dict(raw_id=rawid,block=blocks,snn_mode=m));b=pd.DataFrame(dict(raw_id=z['source_event_index'],block=z['source_block_id'],source_label=z['template_label'],event_abs_time=z['event_abs_time']))
    joined=a.merge(b,on=['raw_id','block'],how='left',validate='one_to_one');assert joined.source_label.notna().all()
    # Verified current preparation consumes labels from the Timing+Space field, not an old bank.
    assert np.array_equal(np.asarray(field['template_discovery']['sampled_event_indices']),z['source_event_index'])
    if labels is not None:
        mapping=dict(zip(z['source_event_index'].tolist(),labels.tolist()));joined['current_field_label']=joined.raw_id.map(mapping)
    joined.to_csv(RUN/'snn_fit_label_crosswalk.csv',index=False)
    tables={}
    for column in ['source_label']+(['current_field_label'] if labels is not None else []):tables[column]=pd.crosstab(joined.snn_mode,joined[column]).to_dict()
    current=pd.read_csv(RUN/'events.csv');matched=joined.merge(current[['start_epoch','label_tb']],left_on='event_abs_time',right_on='start_epoch',how='inner',validate='one_to_one');assert np.array_equal(matched.current_field_label,matched.label_tb)
    ictal=np.zeros(len(joined),bool)
    for seizure in json.loads((RUN/'seizures.json').read_text()):ictal|=(joined.event_abs_time.to_numpy()<seizure['offset'])&(joined.event_abs_time.to_numpy()+.25>seizure['onset'])
    assert (~ictal).sum()==len(matched),'Every unmatched FIT event must be accounted for by the clinical exclusion'
    assert set(joined.loc[~ictal,'raw_id'])==set(matched.raw_id)
    joined['overlaps_clinical_ictal_window']=ictal;joined.to_csv(RUN/'snn_fit_label_crosswalk.csv',index=False)
    out=dict(status='COMPLETE',n_fit=len(ix),n_joined=len(joined),n_current_interictal_matched=len(matched),n_fit_overlapping_clinical_seizures=int(ictal.sum()),source_tables=tables,current_interictal_table=pd.crosstab(matched.snn_mode,matched.label_tb).to_dict(),evaluator=str(evaluator_path),target=str(target_path),not_opened='No CAL/PROBE event readouts or validation waveforms consumed',note='Current-field 0=TA, 1=TB, verified against current prepared interictal labels. Legacy source_label is retained only to expose bank differences; SNN baseline remains unchanged.')
    (RUN/'snn_fit_label_crosswalk.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
