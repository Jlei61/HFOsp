"""Compare whole periodic waveforms at adjacent fold temporal resolutions."""
from plot_rate_periodic_completion import *
from scipy.signal import resample


def main():
    s=RateField();weight=s.geo['group_size']/s.geo['group_size'].sum();rows=[]
    for root in critical():
        label=root['label']
        if not label.startswith('LPC'):continue
        versions=sorted([read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')],key=lambda q:q['N'])
        if len(versions)<2:continue
        lo,hi=versions[-2:];low=np.load(lo['orbit']);high=np.load(hi['orbit'])
        x=resample(low['r'],len(high['r']),axis=0);y=high['r'];n=len(y)
        correlation=np.fft.irfft(np.sum(np.fft.rfft(y,axis=0)*np.conj(np.fft.rfft(x,axis=0))*weight,axis=1),n=n)
        shift=int(correlation.argmax());x=np.roll(x,shift,axis=0)
        scale=np.sqrt(np.mean(np.sum(y*y*weight,axis=1)))
        error=np.sqrt(np.mean(np.sum((x-y)**2*weight,axis=1)))
        period_change=abs(hi['T_ms']-lo['T_ms'])/hi['T_ms']
        row=dict(label=CRITICAL_LABELS[label],internal_label=label,
            source_orbits=[lo['orbit'],hi['orbit']],N=[lo['N'],hi['N']],
            J_absolute_change=abs(hi['J_EE_core']-lo['J_EE_core']),
            period_relative_change=period_change,phase_shift_grid=shift,
            neuron_weighted_waveform_RMS_difference_Hz=error*1000,
            neuron_weighted_waveform_relative_change=error/max(scale,1e-20),
            maximum_group_time_difference_Hz=float(abs(x-y).max()*1000))
        row['status']='MESH_WAVEFORM_AGREEMENT' if row['neuron_weighted_waveform_relative_change']<1e-3 and period_change<1e-4 else 'MESH_WAVEFORM_REVIEW_REQUIRED'
        rows.append(row)
    write(PERIODIC_OUT/'cycle_fold_mesh_waveform_audit.json',dict(rows=rows,
        diagnostic_thresholds=dict(weighted_waveform_relative_change=1e-3,period_relative_change=1e-4),
        scope='Nearest two temporal resolutions of each located root, with a single common time shift. A small root-parameter change alone need not imply the same whole-space orbit. This added waveform diagnostic does not replace continuous-equation, physical-filter, critical-mode or stability checks.'))
    print('FOLD MESH WAVEFORMS',len(rows),'review required',sum(q['status']!='MESH_WAVEFORM_AGREEMENT' for q in rows),flush=True)
    for q in rows:
        if q['status']!='MESH_WAVEFORM_AGREEMENT':print(q,flush=True)


if __name__=='__main__':main()
