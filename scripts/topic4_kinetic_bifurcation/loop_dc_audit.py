"""Independent DC check of delay/M susceptibility elimination."""
from dynamic_loop_spectrum import *
from equilibrium_predictor import EquilibriumProblem


def run(args):
    loop=DynamicLoop(args.response)
    problem=EquilibriumProblem(Path(loop.config.get('table',OUT/'stationary_response/degree6_dv0.125')))
    with np.load(Path(args.response)/'susceptibility.npz') as z:
        derivative=z['integrated_derivative_hz_per_mv']
    n=loop.P;P=problem.P
    incidence=sparse.csr_matrix((np.ones(P),(np.arange(P),loop.group_cell)),shape=(P,n))
    average=sparse.csr_matrix((loop.w,(loop.group_cell,np.arange(P))),shape=(n,P))
    local=derivative/(1+.0005*problem.e*derivative)
    K=average@sparse.diags(local)@(problem.A-sparse.diags(loop.Z)@problem.G)@incidence
    actual=loop.matrix(0.)
    error=(K-actual).tocsr()
    matrix_error=float(np.max(abs(error.data))) if error.nnz else 0.
    rng=np.random.default_rng(9108);v=rng.normal(size=n)
    matvec_error=float(np.max(abs(K@v-actual@v)))
    report=dict(D=loop.config['D'],maximum_matrix_element_error=matrix_error,
        maximum_random_vector_error=matvec_error,
        pass_check=bool(matrix_error<1e-10 and matvec_error<1e-9),
        definition='Independent stationary A/G gains and dynamic M DC elimination agree with delayed frequency loop at zero frequency')
    write(Path(args.response)/'dc_operator_audit.json',report);print(report)
    assert report['pass_check']


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--response',type=Path,required=True);run(ap.parse_args())
