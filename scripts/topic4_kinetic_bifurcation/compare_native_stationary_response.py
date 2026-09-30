"""Compare conditional-density local responses with independent native MC."""
from stationary_response import *


def run(args):
    source=OUT/'stationary_native_mc'/args.source;truth=read(source/'result.json')
    theta=np.array([r['threshold_mv'] for r in truth['conditions']]);u=np.array([r['current_mv'] for r in truth['conditions']])
    pop=np.array([r.get('population',0) for r in truth['conditions']],dtype=np.uint8)
    rows=[];folder=source/args.label;folder.mkdir(parents=True,exist_ok=False)
    for degree,dv in [(d,args.dv) for d in args.degrees]+([(max(args.degrees),args.dv/2)] if args.voltage_refinement else []):
        m=LocalStationaryDensity(theta,pop,u,degree,dv,args.device,basis_mode='high_precision')
        def progress(it,residual,r):
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),degree=degree,dv=dv,iteration=it,residual=float(residual.max())))
        r,_,diag,_=m.solve(10000,progress,tolerance=1e-10,acceleration=True)
        native=np.array([a['rate_hz'] for a in truth['conditions']]);se=np.array([a['standard_error_hz'] for a in truth['conditions']])
        derivatives=(r[2:len(r)-1:3]-r[:len(r)-1:3])/.04
        row=dict(degree=degree,dv=dv,rate_hz=r,native_rate_hz=native,native_standard_error_hz=se,
                 difference_hz=r-native,relative_difference=(r-native)/native,
                 central_derivative_hz_per_mv=derivatives,qa=diag)
        rows.append(row);write(folder/f'degree{degree}_dv{dv:g}.json',row)
        print('native response comparison',degree,dv,'maxrel',np.max(abs((r-native)/native)),diag,flush=True)
        del m;cp.get_default_memory_pool().free_all_blocks()
    write(folder/'result.json',dict(status='COMPLETE',rows=rows,scope='Native local law versus numerical conditional-density response; no direct network bifurcation type inferred'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='critical_response_v1');ap.add_argument('--label',default='density_comparison')
    ap.add_argument('--degrees',type=int,nargs='+',default=[6,8,12,16,20]);ap.add_argument('--voltage-refinement',action='store_true')
    ap.add_argument('--dv',type=float,default=.125)
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
