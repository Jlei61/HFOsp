"""Associate sampled stability changes with computed roots in waveform space.

An association is a coverage diagnostic. It neither rules out additional
crossings inside a bracket nor validates a previously unresolved root.
"""
from plot_rate_periodic_completion import *
from compare_rate_torus_periodic_targets import distances
from functools import lru_cache
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--coverage',type=Path,default=PERIODIC_OUT/'stability_coverage/summary.json')
    parser.add_argument('--output',type=Path,default=PERIODIC_OUT/'stability_bracket_root_associations.json')
    parser.add_argument('--profile-map',type=Path)
    args=parser.parse_args()
    s=RateField();w=s.geo['group_size']/s.geo['group_size'].sum()
    fs=families();roots=critical();coverage=read(args.coverage)
    mapping={}
    if args.profile_map:
        for q in read(args.profile_map)['points']:
            mapping[str(Path(q['original_orbit']).resolve())]=q['orbit']
    for q in coverage.get('sites',[]):
        mapping[str(Path(q['original_orbit']).resolve())]=q['analyzed_orbit']
    N=512;frequency=np.fft.fftfreq(N)*N;rows=[]
    @lru_cache(maxsize=100)
    def profile(path):
        actual=mapping.get(str(Path(path).resolve()),path)
        return resample(np.load(actual)['r']*1000,N,axis=0)
    def inner(x,y):return float(np.mean(np.sum(x*y*w,axis=1)))
    brackets=coverage.get('opposite_or_dimension_change_brackets',coverage.get('stability_change_brackets',[]))
    for bracket in brackets:
        left,right=bracket['ends'];branch=fs[bracket['family']][left['index']:right['index']+1]
        assert Path(branch[0]['path']).resolve()==Path(left.get('original_orbit',left.get('orbit'))).resolve()
        assert Path(branch[-1]['path']).resolve()==Path(right.get('original_orbit',right.get('orbit'))).resolve()
        js=np.array([q['J_EE_core'] for q in branch]);ts=np.array([q['T_ms'] for q in branch])
        padj=max(abs(np.diff(js)));padt=max(abs(np.diff(ts)));matches=[]
        for root in roots:
            if not (js.min()-padj<=root['J_EE_core']<=js.max()+padj and
                    ts.min()-padt<=root['T_ms']<=ts.max()+padt):continue
            target=profile(root['orbit']);aligned=[]
            for orbit in branch:
                x=profile(orbit['path']);_,phase=distances(x[:,None,:],target,w)
                aligned.append(np.fft.ifft(np.fft.fft(x,axis=0)*
                    np.exp(2j*np.pi*frequency*phase[0])[:,None],axis=0).real-target)
            segments=[]
            for i,(x,y) in enumerate(zip(aligned[:-1],aligned[1:])):
                delta=y-x;size=inner(delta,delta)
                if size<1e-24:continue
                fraction=-inner(x,delta)/size
                residual=x+np.clip(fraction,0,1)*delta
                segments.append(dict(left_index=left['index']+i,fraction=fraction,
                    RMS_distance_Hz=np.sqrt(inner(residual,residual)),
                    distance_over_local_waveform_step=np.sqrt(inner(residual,residual)/size)))
            best=min(segments,key=lambda q:q['RMS_distance_Hz'])
            validation=PERIODIC_OUT/(root['label']+'_validation.json')
            evidence=read(validation) if validation.exists() else {}
            matches.append(dict(internal_label=root['label'],display_label=CRITICAL_LABELS.get(root['label'],root['label']),
                J_EE_core=root['J_EE_core'],T_ms=root['T_ms'],orbit=root['orbit'],
                validation_status=evidence.get('status','NOT_AVAILABLE'),nearest_segment=best,
                within_sampled_polyline_neighborhood=bool(0<=best['fraction']<=1 and
                    best['distance_over_local_waveform_step']<.2)))
        row=dict(family=bracket['family'],ends=bracket['ends'],continued_points=len(branch),
            J_range=[js.min(),js.max()],T_range_ms=[ts.min(),ts.max()],nearby_roots=matches)
        rows.append(row)
        print('STABILITY BRACKET',row['family'],[(q['display_label'],q['within_sampled_polyline_neighborhood'],
            q['nearest_segment'],q['validation_status']) for q in matches],flush=True)
    test=profile(roots[0]['orbit']);d,phase=distances(np.roll(test,37,axis=0)[:,None,:],test,w)
    aligned=np.fft.ifft(np.fft.fft(np.roll(test,37,axis=0),axis=0)*
        np.exp(2j*np.pi*frequency*phase[0])[:,None],axis=0).real
    error=np.sqrt(inner(aligned-test,aligned-test));assert error<1e-7,error
    write(args.output,dict(status='ASSOCIATION_SCREEN_COMPLETE',coverage_source=str(args.coverage),
        corrected_profile_map_source=str(args.profile_map) if args.profile_map else None,
        rows=rows,phase_alignment_control_RMS_Hz=error,phase_samples=N,
        definition='All 935 rate waveforms, one common temporal phase, neuron-count-weighted RMS. Root compared with adjacent branch-waveform line segments; 0<=fraction<=1 and residual <0.2 local step is a descriptive association screen.',
        scope='Locations of sampled stability changes relative to existing roots. Does not count every crossing, establish unique root association, replace root validation, or certify a whole interval.'))


if __name__=='__main__':main()
