"""Read the declared nested time meshes in common physical coordinates."""
from common import np, read, write, model, log
from check_spatial_midpoint_convergence import DEST


def main():
    assert read(DEST/'jobs.json')['status']=='COMPLETE'
    s=model(40); w=s.sizes/s.sizes.sum(); steps=[.05,.025,.0125,.00625]
    slices=[('AMPA',slice(0,2)),('GABA',slice(2,4)),('M',slice(4,5)),
            ('covariance',slice(5,11)),('input_memory',slice(11,47)),('history',slice(47,None))]
    gates=dict(last_midpoint_difference_ratio_interval=[3.2,4.8],
               last_cross_method_difference_ratio_interval=[1.6,2.4],
               times_ms=[20,60,120],
               scope='Short-window accuracy diagnostic only, not long-time attractor/critical-parameter convergence.')
    write(DEST/'analysis_gates.json',gates)
    def folder(method,h):return DEST/(method+'_dt'+str(h).replace('.','p'))
    results=[]
    for tm in gates['times_ms']:
        states={method:[np.load(folder(method,h)/f'common_state{tm}.npz')['state'] for h in steps]
                for method in ['old_endpoint','exponential_midpoint']}
        reference=states['exponential_midpoint'][-1]
        scales={name:float(np.sqrt(np.mean((reference[ix]**2)@w))) for name,ix in slices}
        def distance(a,b):
            d={name:float(np.sqrt(np.mean(((a[ix]-b[ix])**2)@w)))/max(scales[name],1e-12) for name,ix in slices}
            return dict(block_relative_rms=d,combined_relative_rms=float(np.sqrt(np.mean(np.array(list(d.values()))**2))))
        method_rows={}
        for method,x in states.items():
            differences=[distance(a,b) for a,b in zip(x[:-1],x[1:])]
            values=[q['combined_relative_rms'] for q in differences]
            ratios=[a/b for a,b in zip(values[:-1],values[1:])]
            method_rows[method]=dict(differences=differences,ratios=ratios)
        cross=[distance(a,b) for a,b in zip(states['old_endpoint'],states['exponential_midpoint'])]
        cv=[q['combined_relative_rms'] for q in cross]; cr=[a/b for a,b in zip(cv[:-1],cv[1:])]
        ratio=method_rows['exponential_midpoint']['ratios'][-1]
        passed=3.2<=ratio<=4.8 and 1.6<=cr[-1]<=2.4 and all(a>b for a,b in zip(cv[:-1],cv[1:]))
        row=dict(time_ms=tm,methods=method_rows,cross_method_differences=cross,cross_method_ratios=cr,
                 order_and_common_limit_gate=bool(passed))
        results.append(row);log('MIDPOINT COMMON LIMIT',tm,'newratios',method_rows['exponential_midpoint']['ratios'],
                                 'oldratios',method_rows['old_endpoint']['ratios'],'cross',cr,'gate',passed)
    status='SHORT_WINDOW_ORDER_AND_COMMON_LIMIT_PASS' if all(q['order_and_common_limit_gate'] for q in results) else 'SHORT_WINDOW_CONVERGENCE_NOT_PASSED'
    write(DEST/'analysis.json',dict(status=status,rows=results,gates=gates,
          interpretation='All comparisons share exactly the same bin-integrated initial history. Passing authorizes further numerical controls only; neither a phase-independent periodic orbit nor an onset bifurcation is established.',model_promoted=False))


if __name__=='__main__':main()
