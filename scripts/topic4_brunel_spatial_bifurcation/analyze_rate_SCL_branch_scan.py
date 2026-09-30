"""Frozen SCL observer on every physically checked point of the main families.

Keep continuation order and explicit gaps. Individual recruitment and
qualified group-event participation remain different observables.
"""
from plot_rate_sameJ_burst_pair import *
from audit_rate_survey_filter_states import fingerprint
import argparse,os,time

FOLDER=DATA/'SCL_branch_scan'


def assemble():
    fs=families();mapping={};rejected=[]
    def add(source,orbit,check,evidence):
        source,orbit=Path(source),Path(orbit)
        if not (check.get('maximum_group_defect_Hz',1)<.001 and
                check.get('minimum_rate_Hz',-1)>=-1e-9 and
                check.get('filter_state_check',{}).get('positive',False)):
            rejected.append(dict(source=str(source),evidence=str(evidence)));return
        record=dict(original_orbit=str(source),orbit=str(orbit),profile_evidence=str(evidence),
            maximum_group_defect_Hz=check['maximum_group_defect_Hz'])
        mapping[str(source.resolve())]=record;mapping[str(orbit.resolve())]=record
    previous=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_small_burst_connection_20260919')
    # The spatial readout question concerns all already checked branches,
    # including global extensions beyond the compact main-figure display.
    # Admit only individually checked profiles, never every member of a
    # segment merely because its sampled checks have passed.
    extension_sources=[]
    for family in ['A','B']:
        for stage in [3,4,5]:
            label=f'arc{family}connectionStage{stage}'+('_20260920' if family=='B' and stage>=4 else '')
            file=PERIODIC_OUT/(label+'_accuracy.json')
            if not file.exists():continue
            segment=read(file)
            if segment.get('status')!='SAMPLED_PASS':continue
            extension_sources.append(str(file))
            for row in segment.get('checks',[]):
                add(row.get('source',row['orbit']),row['orbit'],row,file)
    for file in [previous/'displayed_H1_path_continuous_check.json',DATA/'H1_display_extension.json',
                 DATA/'H1_to_PD3_display_extension.json',DATA/'H2_display_return.json']:
        for row in read(file)['rows']:
            if row['status']=='PASS':add(row['source'],row['orbit'],row['check'],file)
    for file in sorted((DATA/'sites').glob('[0-9][0-9][0-9].json')):
        row=read(file)
        if row.get('resolution',{}).get('status')!='RESOLUTION_CHECKED':continue
        if row.get('relative_waveform_refinement_change',1)>.02:continue
        if row.get('relative_period_change',1)>.001:continue
        add(row['original_orbit'],row['analyzed_orbit'],row['resolution'],file)
    for file in sorted((DATA/'Aleading_profile_gaps').glob('corrected_site_*.json')):
        row=read(file)
        if row.get('status')=='SAME_J_PHYSICAL_PROFILE_CHECKED':
            assert row['relative_waveform_change']<.02 and row['relative_period_change']<.001
            add(row['original_orbit'],row['orbit'],row['resolution'],file)
    file=PERIODIC_OUT/'arcBleadingConnection_20260920_accuracy.json'
    for row in read(file)['checks']:add(row['orbit'],row['orbit'],row,file)
    file=DATA/'Bleading_extension/nearest_target_refinement.json'
    for row in read(file)['rows']:
        if row.get('same_branch_refinement_pass'):
            add(row['original_pair']['second_orbit'],row['corrected_target'],row['resolution'],file)
    s=RateField();file=PERIODIC_OUT/'composite_case_resolution.json';cases=read(file)
    for row in cases['rows']:
        check=dict(row['resolution'])
        if 'filter_state_check' not in check:
            z=np.load(row['orbit']);check['filter_state_check']=filter_state_minima(s,z['r'],float(z['T']))
        add(row['original_orbit'],row['orbit'],check,file)
    points=[];missing={};breaks={}
    for family in ['A','B','double','Bleading','single']:
        rr=fs[family];breaks[family]=continuation_breaks(rr);missing[family]=[]
        for index,row in enumerate(rr):
            key=str(Path(row['path']).resolve())
            if key not in mapping:
                missing[family].append(dict(index=index,orbit=row['path'],J_EE_core=row['J_EE_core']));continue
            record=mapping[key];meta=read(Path(record['orbit']).with_suffix('.json'))
            assert abs(meta['J_EE_core']-row['J_EE_core'])<1e-10
            points.append(dict(**record,family=family,index=index,J_EE_core=meta['J_EE_core'],
                T_ms=meta['T_ms'],profile_fingerprint=fingerprint(record['orbit'])))
    return dict(model='Frozen 400-cell / 935-population spatial rate DDE',points=points,
        unverified_family_samples=missing,continuation_breaks=breaks,rejected_profile_checks=rejected,
        global_extension_sources=extension_sources,
        observer_source=str(OLD/'observer_firing.json'),
        scope='Every eligible checked point is retained. Missing physical checks are explicit gaps, not absence of SCL. Stability is not inferred from readout.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assemble-only',action='store_true');args=parser.parse_args()
    FOLDER.mkdir(exist_ok=True);manifest=assemble();write(FOLDER/'manifest.json',manifest)
    print('MANIFEST',Counter(q['family'] for q in manifest['points']),flush=True)
    if args.assemble_only:return
    contract=read(manifest['observer_source']);s=RateField();names=contract['contact_names']
    scl=np.array([n.startswith('SCL') for n in names]);rows=[]
    for point in manifest['points']:
        output=FOLDER/'points'/f'{point["family"]}_{point["index"]:04d}.json'
        cached=read(output) if output.exists() else None
        if cached and cached.get('profile_fingerprint')==point['profile_fingerprint']:
            result=cached
        else:
            write(FOLDER/'worker.json',dict(status='FROZEN_CONTACT_OBSERVER',pid=os.getpid(),
                completed=len(rows),expected=len(manifest['points']),family=point['family'],index=point['index']))
            z=np.load(point['orbit']);assert fingerprint(point['orbit'])==point['profile_fingerprint']
            ob=observe_cycle(z['r'],float(z['T']),s,contract);records=[]
            for q in ob['records']:
                record={k:q[k] for k in ['bin_origin_ms','detected_events','qualified_events',
                    'SCL_qualified_events','sustained_contact_names','sustained_SCL_contact_names',
                    'contact_peak_to_threshold','metrics']}
                record['SCL_fraction_in_qualified_events']=q['SCL_qualified_events']/q['qualified_events'] if q['qualified_events'] else None
                records.append(record)
            result=dict(**point,records=records,interior_cycles=ob['interior_cycles']);write(output,result)
        records=result['records'];ratios=np.array([q['contact_peak_to_threshold'] for q in records])[:,scl]
        summary={k:point[k] for k in ['family','index','orbit','J_EE_core','T_ms','profile_evidence']}
        summary.update(source=str(output),SCL_contacts=[q['sustained_SCL_contact_names'] for q in records],
            SCL_count=[len(q['sustained_SCL_contact_names']) for q in records],
            SCL_peak_threshold_ratio_min=ratios.min(0),SCL_peak_threshold_ratio_max=ratios.max(0),
            qualified_groups=[q['qualified_events'] for q in records],
            SCL_fraction=[q['SCL_fraction_in_qualified_events'] for q in records])
        rows.append(summary)
        print('SCL',point['family'],point['index'],point['J_EE_core'],summary['SCL_count'],summary['qualified_groups'],flush=True)
    boundaries=[]
    for left,right in zip(rows[:-1],rows[1:]):
        family=left['family']
        if right['family']!=family or right['index']!=left['index']+1:continue
        if right['index'] in manifest['continuation_breaks'][family]:continue
        robust=lambda q:all(v==q['SCL_contacts'][0] for v in q['SCL_contacts'])
        if robust(left) and robust(right) and left['SCL_contacts'][0]!=right['SCL_contacts'][0]:
            boundaries.append(dict(family=family,left=left,right=right,
                type='Individual contact recruitment change across adjacent checked samples; not a dynamical bifurcation'))
    write(FOLDER/'summary.json',dict(status='CHECKED_POINT_OBSERVER_SCAN_COMPLETE',rows=rows,
        manifest=str(FOLDER/'manifest.json'),contact_names=names,SCL_names=np.array(names)[scl],
        recruitment_change_brackets=boundaries,
        statistical_unit='One exact periodic solution per continuation point. Eight repeated cycles and four bin origins are not independent realizations.',
        scope='Physical periodic solutions, including unstable or unclassified branches. Individual SCL recruitment and qualified-group participation are reported separately. Gaps remain wherever full physical profile checks are missing; this is not an exhaustive stable-attractor parameter window.'))
    write(FOLDER/'worker.json',dict(status='COMPLETE',pid=os.getpid(),completed=len(rows),expected=len(rows)))


if __name__=='__main__':main()
