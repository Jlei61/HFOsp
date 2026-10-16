"""Recompute Fig1/2 per patient from raw recording configurations, then aggregate.

Usage: python -m workflows.cohort manifest.json --output ../paper_outputs/cohort
manifest is a JSON list of {subject, config, input_root}; patient IDs unique.
"""
import argparse,json
from pathlib import Path
import numpy as np
from data_processing.raw_to_events import process_record,join_blocks
from analysis.from_events import cluster_events,patient_statistics,cohort_statistics,align_coordinates,direction_and_crossfit,template_fields,spatial_templates,shared_field_similarity
from workflows.common import write_json


def run(manifest,output):
    rows=json.loads(Path(manifest).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if len({r['subject'] for r in rows})!=len(rows):raise ValueError('Duplicate patients')
    stats={};gains={};datasets={}
    for row in rows:
        config=json.loads(Path(row['config']).read_text());root=Path(row['input_root']);subject=row['subject'];out=output/subject;out.mkdir()
        datasets[subject]=config.get('dataset','provided_cohort')
        events=join_blocks([process_record(r,config,root)[2] for r in config['records']])
        clustered=cluster_events(events);stats[subject]=patient_statistics(events,clustered)
        np.savez_compressed(out/'events.npz',**events);write_json(out/'statistics.json',stats[subject])
        xyz=align_coordinates(events,json.loads((root/config['coordinates']).read_text()))
        try:
            directions,gain=direction_and_crossfit(events,xyz);gains[subject]=gain
            field_cluster=spatial_templates(events,xyz,directions)
            field=template_fields(events,field_cluster,xyz,subject_id=subject,subject=subject)
            write_json(out/'shared_field_similarity.json',shared_field_similarity(field))
            write_json(out/'spatial.json',gain);write_json(out/'field_record.json',field)
        except ValueError as exc:
            write_json(out/'spatial_exclusion.json',dict(status='NOT_ESTIMABLE',reason=str(exc)))
    write_json(output/'fig1_cohort.json',cohort_statistics(stats,datasets))
    write_json(output/'fig2_cohort.json',gains)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--output',required=True);a=p.parse_args();print(run(a.manifest,a.output))
