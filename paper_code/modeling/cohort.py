"""New per-patient fits with the copied shaft-balanced cohort objective.

manifest: [{subject, observations, configuration, contact_contract}]. Observation NPZ keys:
fit_centroid_ms, cal_centroid_ms, cal_block_ids, contact_names, contact_xy_mm, coords_mm.
Per-patient configurations fix geometry/priors BEFORE optimization. This runner
uses the notebook optimizer, not the historical recurrent-EE expansion search.
"""
import argparse,json
from pathlib import Path
import numpy as np
from modeling.computational import build_substrate,simulate,score_simulation,save_run
from modeling.experiments import search_core_counts,simulate_selected
from modeling.panel_analysis import model_patient_comparison,fit_patient_templates
from modeling.cohort_objective import CohortObjective,descriptive,summaries
from modeling.src.topic4_observation_repaired import calibrate_reference
from workflows.common import write_json


def run(manifest,output):
    records=json.loads(Path(manifest).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if len({r['subject'] for r in records})!=len(records):raise ValueError('Duplicate patients')
    table=[]
    for row in records:
        out=output/row['subject'];out.mkdir();config=json.loads(Path(row['configuration']).read_text())
        with np.load(row['observations'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        config['contact_names']=data['contact_names'].tolist()
        obj=CohortObjective(data['contact_names'].tolist(),seed=20261015,n_fourier=config['n_fourier'])
        obj.fit_maps(data['fit_centroid_ms']);obj.fit_targets(data['fit_centroid_ms']);obj.calibrate(data['cal_centroid_ms'],data['cal_block_ids'])
        obj.reference_summary=summaries(data['fit_centroid_ms'],obj.names)
        contract=json.loads(Path(row['contact_contract']).read_text())
        patient_model=fit_patient_templates(data['fit_centroid_ms'],obj.names,data['coords_mm'])
        sub=build_substrate(config);reference=simulate(sub,config['centers_mm'],data['contact_xy_mm'],seed=config['training_seed'])
        observer=calibrate_reference(reference['envelope'],2.,obj.names,cal_stop_ms=config['parameters']['T'])
        observer['burnin_ms']=config['burnin_ms'];write_json(out/'observer.json',observer)
        searches=search_core_counts(sub,data['contact_xy_mm'],observer,obj,out/'core_search',
            core_counts=(2,),budget=config['optimization_budget'],restarts=config['restarts'])
        write_json(out/'scores.json',[r for s in searches for r in s['rows']])
        result=dict(subject=row['subject'],status='NOT_ESTIMABLE',J=None)
        if any(s['best'] is not None for s in searches):
            _,best,selected=simulate_selected(config,searches,data['contact_xy_mm'],observer,obj)
            result.update(status='TRAINING_SELECTED',J=selected['J'],**descriptive(best['primary_centroids_ms'],obj.reference_summary,obj.names))
            save_run(out/'selected_training_simulation.npz',best)
            matched=model_patient_comparison(best['primary_centroids_ms'],data['fit_centroid_ms'],obj.names,contract,patient_model)
            result['training_selected_semantic_rho']=np.diag(matched['matrix']).tolist()
            write_json(out/'contact_crossfit.json',{k:v for k,v in matched.items() if k!='within_shaft_null'})
            np.savez_compressed(out/'contact_null.npz',null=matched['within_shaft_null'])
            # Confirmation noise is never used to reselect a candidate.
            vec=selected['vector'];check_sub=build_substrate(config,angle_offset=vec[-1]) if vec[-1] else sub
            check=simulate(check_sub,np.asarray(vec[:4]).reshape(2,2),data['contact_xy_mm'],outward=vec[-2],seed=928401)
            result['new_noise_confirmation']=score_simulation(check,observer,obj,config['burnin_ms'])
            save_run(out/'new_noise_confirmation.npz',check)
        table.append(result);write_json(output/'patient_results.json',table)
    finite=[r['training_selected_semantic_rho'] for r in table if r.get('training_selected_semantic_rho') is not None]
    write_json(output/'fig4I_patient_units.json',dict(unit='patient',n_patients=len(finite),semantic_rho=finite,
        interpretation='Training-selected models, with contact cross-fitting; not held-out-patient model identification.'))
    return table

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--output',required=True);a=p.parse_args();run(a.manifest,a.output)
