"""Verify newly delivered figures and numerical consistency, not scientific success."""
import csv
import json
from pathlib import Path
from PIL import Image
from scripts import analyze_topic4_propagation_recovery_night as s


def main():
    n=s.night.OUT
    roots=['analysis_final_B','main_review_final_B','main_review_final_B_primary','distribution_review_final_B',
        'parameter_response_final_B','temporal_support_final_B','core_timing_final_B','continuous_native_final_B',
        'core_ou_effect_final_B','core_ou_propagation_response','core_window_alias_final_B','early_contact_geometry',
        'core_to_contact_timing','core_prior_extent_audit','parameter_response_summary']
    media=[];gif_frames=0
    for name in roots:
        for p in sorted((n/name).rglob('*')):
            if p.suffix not in ['.png','.gif']:continue
            with Image.open(p) as im:
                if p.suffix=='.gif':
                    count=im.n_frames
                    for i in range(count):im.seek(i);im.load()
                    gif_frames+=count
                else:im.load();count=1
                dimensions=im.size
            media.append(dict(path=str(p),sha256=s.rt.sha(p),dimensions=dimensions,frames=count))
    with (n/'analysis_final_B/run_observations.csv').open() as f:full=list(csv.DictReader(f))
    rapid=s.rt.read(n/'final_B_rapid_observations.json');maximum=0.
    for r in rapid['rows']:
      for mode,values in r['observations'].items():
        actual=next(q for q in full if q['base_id']==r['candidate'] and int(q['seed'])==r['seed'] and q['mode']==mode and q['layer']=='primary')
        for key,value in values.items():
            if isinstance(value,(float,int)) and actual.get(key) not in ('',None):
                error=abs(float(actual[key])-value);assert error<1e-10;maximum=max(error,maximum)
        if mode=='ALL':assert abs(float(actual['L_search'])-r['score']['L_search'])<1e-10
    for phase in ['wave1','long','final_A','final_B']:
        audit=s.rt.read(n/f'local_burst_audit_{phase}/audit.json')
        assert all(r['event_and_mask_identity'] for r in audit['checks'])
    reports=[]
    main=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo')
    for folder in ['overnight_key_results_20260911','model_collaborator_report_v8_2026-09-11']:
        current=s.rt.read(main/folder/'current.json');assert s.rt.sha(current['report'])==current['sha256']
        assert current['renderer_warnings']==''
        reports.append(dict(current_json=str(main/folder/'current.json'),report=current['report'],pages=current['pages'],sha256=current['sha256']))
    result=dict(status='PASS_DELIVERY_CHECKS',png_count=sum(p['path'].endswith('.png') for p in media),
        gif_count=sum(p['path'].endswith('.gif') for p in media),decoded_gif_frames=gif_frames,media=media,
        reports=reports,numerical_maximum_rapid_full_difference=maximum,
        observer_replay_event_and_mask_identity=True,scientific_acceptance=False,user_visual_acceptance=False,
        interpretation='Image decoding and consistent statistics do not establish patient propagation recovery. PDF page rendering was already checked by each report producer; raw arrays were verified by the physical execution chain.',
        producer=__file__,producer_sha256=s.rt.sha(__file__))
    s.rt.write(n/'delivery_checks.json',result)
    print({k:result[k] for k in ['status','png_count','gif_count','decoded_gif_frames','numerical_maximum_rapid_full_difference','reports']},flush=True)


if __name__=='__main__':main()
