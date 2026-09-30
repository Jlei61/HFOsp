"""Check whether dense cycle folds are merely repeated-period descriptions.

A nonzero first Fourier harmonic of the full population-rate field excludes
an exact T/d subperiod for every integer d > 1. Test that observable at both
stored meshes; also retain direct divisor mismatches and nearby waveform
distances. This does not supply bifurcation or stability evidence by itself.
"""
from plot_rate_periodic_completion import *
from compare_rate_torus_periodic_targets import distances


def describe(path, weights):
    with np.load(path) as z:
        r=z['r']*1000
    N=len(r);cf=np.fft.fft(r,axis=0)/N
    energy=np.sum(abs(cf)**2*weights,axis=1)
    total=float(energy[1:].sum());assert total>0
    frequency=np.fft.fftfreq(N)*N
    mismatch=lambda fraction: float(np.sqrt(np.sum(
        energy*abs(1-np.exp(2j*np.pi*frequency*fraction))**2)/total))
    # A whole-period shift must reproduce the complete spatial field.
    assert mismatch(1.)<1e-10
    return dict(orbit=str(path),N=N,temporal_RMS_Hz=np.sqrt(total),
        first_harmonic_RMS_fraction=np.sqrt((energy[1]+energy[-1])/total),
        nyquist_energy_fraction=float(energy[N//2]/total),
        period_divisor_mismatches=[dict(divisor=d,relative_difference=mismatch(1/d))
                                  for d in [2,3,4,5,6]])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',nargs='+')
    parser.add_argument('--output',type=Path,default=PERIODIC_OUT/'cycle_primitive_period_audit.json')
    args=parser.parse_args()
    s=RateField();w=s.geo['group_size']/s.geo['group_size'].sum();rows=[]
    for root in critical():
        name=root['label']
        if not name.startswith('LPC'):continue
        if args.labels and name not in args.labels:continue
        versions=sorted([read(f) for f in PERIODIC_OUT.glob(name+'_N*.json')],
                        key=lambda q:q['N'])[-2:]
        checks=[describe(q['orbit'],w) for q in versions]
        h=np.array([q['first_harmonic_RMS_fraction'] for q in checks])
        mesh_drift=float(abs(h[-1]-h[-2])) if len(h)>1 else None
        resolved=bool(len(h)==2 and min(h)>max(.001,10*mesh_drift)
                      and max(q['nyquist_energy_fraction'] for q in checks)<1e-8)
        rows.append(dict(label=CRITICAL_LABELS[name],internal_label=name,
            J_EE_core=root['J_EE_core'],T_ms=root['T_ms'],checks=checks,
            first_harmonic_mesh_drift=mesh_drift,
            status='PRIMITIVE_PERIOD_SUPPORTED' if resolved else 'PRIMITIVITY_UNRESOLVED'))
    if args.labels:assert {q['internal_label'] for q in rows}==set(args.labels)
    # Dense H2 folds can share J and T to plotting precision. Compare the
    # complete spatial field after a single common phase, never separate
    # shifts for the two cores. Different J is not an exact identity test.
    dense=[q for q in rows if q['internal_label'] in [f'LPC_B{i}' for i in range(3,9)]]
    pairs=[];profiles={q['label']:resample(np.load(q['checks'][-1]['orbit'])['r']*1000,512,axis=0)
                       for q in dense}
    for i,a in enumerate(dense):
        for b in dense[i+1:]:
            x=profiles[a['label']];y=profiles[b['label']]
            d,phase=distances(x[:,None,:],y,w)
            scale=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*w,axis=1)))
            pairs.append(dict(labels=[a['label'],b['label']],
                J_difference=abs(a['J_EE_core']-b['J_EE_core']),
                period_difference_ms=abs(a['T_ms']-b['T_ms']),
                common_phase_shift_cycles=float(phase[0]),
                full_group_RMS_difference_Hz=float(d[0]),
                relative_waveform_difference=float(d[0]/scale)))
    out=dict(status='AUDIT_COMPLETE',rows=rows,dense_H2_pair_comparisons=pairs,
        observable='Neuron-weighted full 935-group rate field, including E and I groups; temporal RMS around each group mean.',
        criteria=dict(minimum_first_harmonic_RMS_fraction=.001,
            mesh_drift_safety_factor=10,maximum_nyquist_energy_fraction=1e-8),
        scope='Resolved first-harmonic content excludes simple repeated-period representations at these stored roots. It does not validate a fold, its adjacent stability, or an entire connecting branch. Pair distances compare different parameters and cannot establish or rule out a global branch connection.')
    write(args.output,out)
    print('PRIMITIVE PERIODS',len(rows),sum(q['status']=='PRIMITIVE_PERIOD_SUPPORTED' for q in rows),flush=True)
    print('FIRST HARMONIC MINIMUM',min(q['checks'][-1]['first_harmonic_RMS_fraction'] for q in rows),flush=True)
    if pairs:print('DENSE FOLD NEAREST',min(pairs,key=lambda q:q['relative_waveform_difference']),flush=True)


if __name__=='__main__':main()
