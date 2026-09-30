"""Analyze the bounded long-window heterogeneity controls with prefix replay QA."""
from compare_density_spatial import *


def main():
    folder=OUT/'population_pair_replication'
    contract=json.load(open(folder/'long_factorial_contract.json'))
    base=OUT/'particle_controls/selected_g40';rows=[];checks=[]
    for flags,suffix in [(0,''),(2,'_individual_flags2'),(5,'_individual_flags5'),(7,'_microscopic')]:
        source=base/f'D0.225000_Nscale16_seed1901_12000ms{suffix}'
        assert json.load(open(source/'status.json'))['status']=='COMPLETE'
        if flags in (2,5):
            short=base/f'D0.225000_Nscale16_seed1901_4000ms{suffix}'
            with np.load(short/'trajectory.npz') as a,np.load(source/'trajectory.npz') as b:
                identical={};error={}
                for key in a.files:
                    v=b[key] if key=='count_e' else b[key][:len(a[key])]
                    identical[key]=bool(np.array_equal(a[key],v))
                    error[key]=float(np.max(abs(a[key]-v)))
            passed=all(v for k,v in identical.items() if k!='field_1ms') and error['field_1ms']<1e-10
            checks.append(dict(flags=flags,pass_replay=passed,arrays_bitwise=identical,maximum_absolute_errors=error))
            assert passed,checks[-1]
        windows=[summarize(source,*w) for w in contract['windows_ms']]
        rows.append(dict(individual_flags=flags,source=str(source.resolve()),windows=windows))
    result=dict(status='LONG_FACTORIAL_COMPLETE',D=.225,seed=1901,population_multiplier=16,
        rows=rows,prefix_checks=checks,
        interpretation='Complementary heterogeneity controls in one matched realization; compare observed long-episode and quiet-time changes before assigning a closure mechanism.',
        statistical_unit=contract['statistical_unit'],model_acceptance='NOT_ESTABLISHED',critical_type='NOT_INFERRED')
    (folder/'long_factorial_result.json').write_text(json.dumps(safe(result),indent=2)+'\n')
    for row in rows:
        late=row['windows'][-1]
        print(row['individual_flags'],late['category'],'mean',late['mean_E_hz'],'quiet',late['quiet_fraction'],
              'events',late['finite_events'],'median duration',late['event_duration_median_ms'])


if __name__=='__main__':
    main()
