"""Locate known critical profiles inside sampled stability-change brackets.

Use continuation order and full phase-aligned spatial waveforms. Sorting by
J alone loses a fold when both sampled endpoints lie on its same side.
"""
from plot_rate_periodic_completion import *
from compare_rate_torus_periodic_targets import distances


def family_for(name):
    if name.startswith(('LPC_A','LPC_resonance','TR_A')):return 'A'
    if name.startswith(('LPC_burst','LPC_single_upper')):return 'single'
    if name.startswith('LPC_Bleading'):return 'Bleading'
    if name.startswith('LPC_B'):return 'B'
    if name.startswith(('LPC_double','PD_double')):return 'double'


def main():
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();fs=families();out=[]
    located={}
    brackets=read(PERIODIC_OUT/'stability_coverage/summary.json')['stability_change_brackets']
    needed={q['family'] for q in brackets}
    for q in critical():
        name=family_for(q['label'])
        if name not in needed:continue
        rr=fs[name];meta=read(Path(q['orbit']).with_suffix('.json'))
        feature=np.array([q['J_EE_core'],q['T_ms'],*meta['mean_rates_hz']])
        features=np.array([[r['J_EE_core'],r['T_ms'],*r['mean_rates_hz']] for r in rr])
        scale=np.array([max(1e-4,feature[0]*.005),feature[1]*.02,*np.maximum(feature[2:],.2)*.05])
        ids=np.argsort(np.sum(((features-feature)/scale)**2,axis=1))[:24]
        z=np.load(q['orbit']);target=resample(z['r']*1000,256,axis=0);nearest=[]
        for i in ids:
            zz=np.load(rr[i]['path']);wave=resample(zz['r']*1000,256,axis=0)
            d,shift=distances(wave[:,None,:],target,weights)
            nearest.append(dict(index=int(i),orbit=rr[i]['path'],waveform_distance_Hz=float(d[0]),phase_shift_cycles=float(shift[0])))
        nearest.sort(key=lambda x:x['waveform_distance_Hz'])
        located[q['label']]=dict(label=CRITICAL_LABELS[q['label']],internal_label=q['label'],family=name,
            J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],source=q['orbit'],nearest_profiles=nearest[:3],
            profile_RMS_Hz=float(np.sqrt(np.mean(target**2@weights))))
    for bracket in brackets:
        name=bracket['family'];left,right=[q['index'] for q in bracket['ends']];rr=fs[name][left:right+1]
        matches=[]
        for q in located.values():
            if q['family']!=name:continue
            compatible=[]
            for rank,profile in enumerate(q['nearest_profiles']):
                index=profile['index']
                if not left<=index<=right:continue
                sample=fs[name][index]
                # A refined restart may revisit a critical orbit already
                # crossed by the earlier, coarser segment. The unique global
                # nearest sample must not erase that earlier visit. Require
                # full-waveform, period and parameter agreement for an
                # alternate visit; do not match by J alone.
                alternative_ok=(profile['waveform_distance_Hz']<.01*q['profile_RMS_Hz'] and
                    abs(sample['T_ms']/q['T_ms']-1)<.005 and
                    abs(sample['J_EE_core']-q['J_EE_core'])<.001*max(1,abs(q['J_EE_core'])))
                if rank==0 or alternative_ok:compatible.append(profile)
            if compatible:matches.append(dict(q,bracket_profile=min(compatible,key=lambda r:r['waveform_distance_Hz'])))
        out.append(dict(**bracket,known_critical_profiles=matches,
            assessment='KNOWN_CRITICAL_PROFILES_IN_BRACKET' if matches else 'NO_KNOWN_CRITICAL_PROFILE_MATCH',
            scope='Continuation-order localization with full waveforms, not a proof of unique crossings or absence of additional points.'))
        print('STABILITY BRACKET',name,left,right,[q['label'] for q in matches],flush=True)
    write(PERIODIC_OUT/'stability_coverage/known_critical_brackets.json',dict(rows=out,critical_profile_locations=list(located.values()),
        scope='Do not count an opposite-stability bracket as an additional bifurcation before checking the known critical points and the interior continuation path.'))


if __name__=='__main__':main()
