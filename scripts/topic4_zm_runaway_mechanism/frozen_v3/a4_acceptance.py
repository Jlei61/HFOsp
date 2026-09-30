"""Stage A4 acceptance: evaluate the registered a4_contract.json criteria on the comparison table."""
from common_v3 import *
def main():
    c=read(DEST/'a4_contract.json');t=read(DEST/'a4_comparison/comparison_table.json')['table'];nat=read(DEST/'native_reference/summary.json')
    zD=np.load(DEST/'runs/A4_det_meandrive/trajectory.npz')['D'];zS=np.load(DEST/'runs/A4_stoch_seed9108401/trajectory.npz')['D']
    natD=read(DEST/'native_reference/checkpoint_projections.json')
    rows={}
    for arm in ['A4_det_meandrive','A4_stoch_seed9108401','A4_stoch_meandrive']:
        r=t.get('rate_'+arm)
        if r is None:continue
        w=r['windows']['1000-9420'];D=zD if arm=='A4_det_meandrive' else (zS if arm=='A4_stoch_seed9108401' else np.load(DEST/f'runs/{arm}/trajectory.npz')['D'])
        D987=float(D[min(len(D)-1,987)]);D80=float(D[min(len(D)-1,800)])
        checks=dict(
            self_limited_events=dict(passed=bool(w['n']>0 and 50<=w['median_duration_ms']<=200 and r['quiet_fraction']>=.15),n=w['n'],median_duration_ms=w['median_duration_ms'],quiet_fraction=r['quiet_fraction']),
            two_core_participation=dict(passed=bool(w['n']>0 and w['both_cores']/w['n']>=.5),fraction=w['both_cores']/max(w['n'],1)),
            surround_recruitment=dict(passed=bool(w['median_area'] is not None and .3<=w['median_area']<=1.),median_area=w['median_area'],median_surround_cells=w['median_surround_cells']),
            propagation=dict(passed=bool(w['forward']>0 and w['reverse']>0 and w['median_extent_mm'] is not None and 5<=w['median_extent_mm']<=20),forward=w['forward'],reverse=w['reverse'],median_extent_mm=w['median_extent_mm'],
                             note='deterministic arm cannot alternate ignition side without noise' if arm=='A4_det_meandrive' else ''),
            entry=dict(passed=bool(r['high_onset_ms'] is not None and 7000<=r['high_onset_ms']<=13000),high_onset_ms=r['high_onset_ms'],native_ms=nat['seed9108401']['high_onset_ms']),
            D_track=dict(passed=bool(abs(D987-natD['9870']['D'])<=.05),D_at_9870ms=D987,native=natD['9870']['D'],D_at_8000ms=D80,native_8000=natD['8000']['D']))
        rows[arm]=dict(checks=checks,n_passed=sum(v['passed'] for v in checks.values()),n_total=len(checks))
    verdict={'A4_det_meandrive':'skeleton','A4_stoch_seed9108401':'primary stochastic contrast','A4_stoch_meandrive':'noise-only contrast'}
    out=dict(status='COMPLETE',arms=rows,native_basis=c['tolerance_basis'],roles=verdict,
        overall='PASS' if all(rows[a]['n_passed']==rows[a]['n_total'] for a in rows if a=='A4_stoch_seed9108401') and rows.get('A4_det_meandrive',{}).get('n_passed',0)>=5 else 'PARTIAL')
    write(DEST/'a4_comparison/acceptance.json',out);print(json.dumps(clean(out),indent=1)[:5000])
if __name__=='__main__':main()
