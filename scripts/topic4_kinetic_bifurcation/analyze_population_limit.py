"""Late-window check before attributing early grouping differences to closure."""
from compare_density_spatial import *


def main():
    base=OUT/'particle_controls/selected_g40';rows=[];prefix=[];maps={}
    density=OUT/'recurrence_searches/D225_5to15s'
    for label,folder in [('grouped',base/'D0.225000_Nscale16_seed1901_12000ms'),
                         ('individual',base/'D0.225000_Nscale16_seed1901_12000ms_microscopic'),
                         ('density',density)]:
        assert json.load(open(folder/'status.json'))['status']=='COMPLETE',folder
        # Use explicit paths rather than assuming a new run continued an old
        # checkpoint. Both controls rerun the same seed from reset.
        if label!='density':
            short=base/('D0.225000_Nscale16_seed1901_4000ms'+('_microscopic' if label=='individual' else ''))
            with np.load(short/'trajectory.npz') as a,np.load(folder/'trajectory.npz') as b:
                check={};errors={}
                for k in a.files:
                    y=b[k] if k=='count_e' else b[k][:len(a[k])]
                    check[k]=bool(np.array_equal(a[k],y));errors[k]=float(np.max(abs(a[k]-y)))
            # Spatial GPU bincount is a floating atomic readout only; its
            # summation order may change at a few pixels without feeding back
            # into the network. Dynamic rate and M records remain bitwise equal.
            passed=all(check[k] for k in check if k!='field_1ms') and errors['field_1ms']<1e-10
            prefix.append(dict(model=label,pass_replay=passed,pass_all_arrays_bitwise=all(check.values()),arrays=check,
                maximum_absolute_errors=errors,field_readout_roundoff_tolerance_hz=1e-10))
            assert passed,(check,errors)
        windows=[summarize(folder,a,b) for a,b in ([(1000,4000),(4000,8000),(8000,12000)] if label!='density' else [(8000,12000)])]
        late=windows[-1]
        if late['finite_events']:
            spatial=extract(folder,.225,8000,12000);maps[label]=spatial
            lags=spatial['core_B_minus_A_crossing_ms']
        else:lags=[]
        rows.append(dict(model=label,source=str(folder),windows=windows,late_core_B_minus_A_crossing_ms=lags,
            note='Density trajectory uses the validated mixed noise-product search; whole-cycle FP64 return evidence is reported separately' if label=='density' else 'Exact finite particles with the stated within-group heterogeneity'))
    comparison={name:compare(maps['density'],maps[name]) for name in ('grouped','individual') if name in maps and 'density' in maps}
    result=dict(status='COMPLETE_BOUNDED_LATE_WINDOW_DIAGNOSTIC',D=.225,rows=rows,prefix_replay=prefix,
        late_spatial_comparisons=comparison,statistical_unit='One input realization per condition; events are nested within a trajectory',
        acceptance='Descriptive test of transient versus grouping effects; not automatic complete propagation or bifurcation acceptance')
    (OUT/'population_limit_long_comparison.json').write_text(json.dumps(safe(result),indent=2)+'\n')
    print(json.dumps(safe(result),indent=2))


if __name__=='__main__':main()
