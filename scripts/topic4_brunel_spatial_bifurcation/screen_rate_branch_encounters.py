"""Screen already traced Hopf/burst families for a common-orbit encounter.

Regional metadata only select candidate pairs. The reported distance uses
all 935 population waveforms and one common phase, including their means.
An encounter still needs correction at the same parameter and independent
orbit-resolution checks; a negative screen is not a global exclusion.
"""
from plot_rate_periodic_completion import *
from compare_rate_torus_periodic_targets import distances
from functools import lru_cache


def main():
    p=argparse.ArgumentParser();p.add_argument('--pd-children',action='store_true')
    p.add_argument('--provisional-segment',help='Screen a frozen snapshot of newly converged points without accepting it for plotting')
    p.add_argument('--provisional-family',choices=['A','B','single','Bleading','double'],default='A')
    p.add_argument('--include-higher-hopfs',action='store_true',help='Also screen the two first Hopf families against computed children of later Hopfs')
    p.add_argument('--accepted-segment',help='Restrict source-family screening to this checked continuation segment')
    p.add_argument('--phase-samples',type=int,default=256)
    p.add_argument('--output-name',help='Separate JSON filename for a resolution check, preserving the original screen')
    options=p.parse_args()
    if options.output_name:
        assert Path(options.output_name).name==options.output_name and options.output_name.endswith('.json')
    assert options.phase_samples>=64
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();fs=families();rows=[]
    provisional=None
    if options.provisional_segment:
        source=PERIODIC_OUT/(options.provisional_segment+'_continuation.json');segment=read(source)
        added=[read(Path(q['path']).with_suffix('.json')) for q in segment['rows']]
        assert added and all(q['status']=='CONVERGED' for q in added)
        seen={q['path'] for q in fs[options.provisional_family]}
        fs[options.provisional_family]+=[q for q in added if q['path'] not in seen]
        provisional=dict(source=str(source),family=options.provisional_family,
                         orbit_snapshot=[q['path'] for q in added],figure_acceptance=False)
    pairs=[('A','B')]+[(name,burst) for name in ['A','B'] for burst in ['single','Bleading','double']]
    if options.include_higher_hopfs:
        pairs += [(name,child) for name in ['A','B'] for child in fs if child.startswith('H') and child[1:].isdigit()]
    accepted=None
    if options.accepted_segment:
        assert not options.provisional_segment and not options.pd_children
        source=PERIODIC_OUT/(options.accepted_segment+'_accuracy.json');segment=read(source)
        assert segment['status']=='SAMPLED_PASS'
        family=options.provisional_family
        previous=fs[family]
        fs[family]=[read(Path(path).with_suffix('.json')) for path in segment['included_orbits']]
        # A B-only extension must also be compared with A. Filtering the
        # pre-existing [('A','B'), ...] list by its first member omitted it.
        targets=['A','B','single','Bleading','double']
        if options.include_higher_hopfs:
            targets += [name for name in fs if name.startswith('H') and name[1:].isdigit()]
        pairs=[(family,other) for other in targets if other!=family]
        # Check whether a long folded extension revisits a remote part of
        # its own family. Omit the immediate predecessor neighborhood.
        current_paths={str(Path(q['path']).resolve()) for q in fs[family]}
        previous=[q for q in previous if str(Path(q['path']).resolve()) not in current_paths]
        fs[family+'_previous']=previous[:-20]
        pairs.append((family,family+'_previous'))
        accepted=dict(source=str(source),family=family,orbit_snapshot=segment['included_orbits'])
    ratio=1.10
    if options.pd_children:
        pairs=[(name,burst) for name in ['PDreturnchild','PDupperchild'] for burst in ['single','Bleading','double']]
        ratio=1.25
    @lru_cache(maxsize=80)
    def profile(path):
        z=np.load(path);return resample(z['r']*1000,options.phase_samples,axis=0)
    def amplitude(x):return float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1))))
    for first,second in pairs:
        candidates=[]
        for i,a in enumerate(fs[first]):
            for j,b in enumerate(fs[second]):
                dj=abs(a['J_EE_core']-b['J_EE_core']);dt=abs(np.log(a['T_ms']/b['T_ms']))
                if dj>.01 or dt>np.log(ratio):continue
                am=np.array(a['mean_rates_hz']);bm=np.array(b['mean_rates_hz'])
                aa=np.array(a['max_rates_hz'])-np.array(a['min_rates_hz'])
                ba=np.array(b['max_rates_hz'])-np.array(b['min_rates_hz'])
                scale=max(np.linalg.norm(aa),np.linalg.norm(ba),.001)
                score=(np.linalg.norm(am-bm)+np.linalg.norm(aa-ba))/scale+10*dt+10*dj
                candidates.append((float(score),i,j))
        # Retain diverse parameter locations as well as the best metadata
        # matches. This shortlist is deliberately recorded, not called a
        # complete search through the intervening continuous branches.
        ordered=sorted(candidates);selected=ordered[:40]
        for binindex in range(10):
            local=[v for v in ordered if int(np.floor(fs[first][v[1]]['J_EE_core']*100))%10==binindex]
            selected.extend(local[:5])
        unique={(i,j):(score,i,j) for score,i,j in selected};checks=[]
        for score,i,j in unique.values():
            a,b=fs[first][i],fs[second][j];x,y=profile(a['path']),profile(b['path'])
            d,phase=distances(x[:,None,:],y,weights);scale=max(amplitude(x),amplitude(y))
            checks.append(dict(first_index=i,second_index=j,first_orbit=a['path'],second_orbit=b['path'],
                first_J=a['J_EE_core'],second_J=b['J_EE_core'],first_T_ms=a['T_ms'],second_T_ms=b['T_ms'],
                metadata_shortlist_score=score,full_group_RMS_difference_Hz=float(d[0]),
                relative_waveform_difference=float(d[0]/scale),common_phase_cycles=float(phase[0])))
        checks.sort(key=lambda q:q['relative_waveform_difference'])
        row=dict(families=[first,second],metadata_window_pairs=len(candidates),fully_compared_pairs=len(checks),
                 nearest=checks[:10],candidate_same_parameter_corrections=[v for v in checks if v['relative_waveform_difference']<.05])
        rows.append(row);print('BRANCH ENCOUNTER SCREEN',first,second,len(candidates),len(checks),checks[:1],flush=True)
    control=profile(fs['A'][20]['path']);d,_=distances(np.roll(control,37,axis=0)[:,None,:],control,weights)
    assert d[0]<1e-7,d
    filename='PD_child_branch_encounter_screen.json' if options.pd_children else 'branch_encounter_screen.json'
    if provisional:filename=options.provisional_segment+'_encounter_screen.json'
    if accepted:filename=options.accepted_segment+'_accepted_encounter_screen.json'
    if options.include_higher_hopfs:filename=filename.replace('.json','_with_higher_hopfs.json')
    if options.output_name:filename=options.output_name
    write(PERIODIC_OUT/filename,dict(status='SCREEN_COMPLETE',rows=rows,
        provisional_segment=provisional,
        accepted_source_segment=accepted,
        identity_phase_control_Hz=float(d[0]),window=dict(J_absolute_difference=.01,period_ratio=ratio),
        phase_samples=options.phase_samples,observable='Neuron-weighted full-population waveform RMS after one common phase; normalized by the larger waveform temporal RMS. Means retained.',
        scope='Metadata shortlist over traced families only. A small distance is a candidate for same-J BVP correction, not proof of identity. A large minimum does not exclude unsampled or untraced connections. No stability classification.'))


if __name__=='__main__':main()
