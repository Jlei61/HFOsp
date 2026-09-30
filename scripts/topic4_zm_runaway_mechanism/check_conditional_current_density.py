"""Independent Gaussian integration, synaptic, and refractory timing checks."""
from conditional_current_density import *
from common import OUT, write
from lif_mc import condition
from scipy.integrate import quad


def main():
    rng=np.random.default_rng(920)
    max_quad=0.; max_partition=0.;max_synaptic=0.
    p=condition(0.,18.,1.,1.,'E')
    for index in range(12):
        mean=rng.normal(size=4); L=rng.normal(size=(4,4)); C=L@L.T+.5*np.eye(4)
        mass=.1+.1*index;raw=np.empty((5,5));raw[0,0]=mass
        raw[0,1:]=raw[1:,0]=mass*mean;raw[1:,1:]=mass*(C+np.outer(mean,mean))
        direction=rng.normal(size=4)*.1
        lo,hi=sorted(rng.normal(size=2)*2)
        moments=normal_integrals(lo,hi)
        intercept=.7;slope=.03
        actual=weighted_raw(raw,mean,direction,moments,intercept,slope)
        numerical=np.zeros((5,5))
        def vec(z):return np.r_[1.,mean+direction*z]
        residual=np.zeros((5,5));residual[1:,1:]=C-np.outer(direction,direction)
        for i in range(5):
            for j in range(5):
                numerical[i,j]=quad(lambda x:mass*(vec(x)[i]*vec(x)[j]+residual[i,j])*(intercept+slope*x)*
                    np.exp(-x*x/2)/np.sqrt(2*np.pi),lo,hi,epsabs=1e-12,epsrel=1e-12)[0]
        max_quad=max(max_quad,float(abs(actual-numerical).max()))
        A=np.zeros((5,5));A[0,0]=1.;A[1,1]=p[6];A[2,1]=p[7];A[2,2]=p[8]
        A[3,3]=p[17];A[4,3]=p[9];A[4,4]=p[10]
        B=np.zeros((5,4));B[1,0]=p[11];B[2,:2]=p[12:14]
        B[3,2]=p[14];B[4,2:]=p[15:17]
        v=np.array([1.3,1.3,.7,.7]);expected=A@raw@A.T+mass*(B*v)@B.T
        max_synaptic=max(max_synaptic,float(abs(advance(raw,p,1.3,.7)-expected).max()))
        grid=voltage_grid(18.,11.,128);out=np.zeros((len(grid),5,5));spikes=np.zeros((5,5))
        cut,under,negative=membrane_remap(raw,17.8,40.*(index-5),p,grid,out,spikes)
        max_partition=max(max_partition,float(abs(out.sum(0)+spikes-raw).max()))
        assert out[:,0,0].min()>-1e-12 and spikes[0,0]>=0 and cut==0
    assert max_quad<1e-10 and max_partition<1e-9 and max_synaptic<1e-10
    # With a sufficiently strong deterministic input, every released cohort
    # fires immediately. All mass must spike exactly every ref_steps.
    wave=np.zeros((3,2));wave[0]=1e6
    p=condition(0.,18.,0.,0.,'E'); grid=voltage_grid(18.,11.,64)
    answer=simulate(p,wave,grid,.1,10.,0,200,10)
    expected=np.zeros(200);expected[::int(p[19])]=10000.
    assert np.array_equal(answer[3],expected)
    assert np.max(abs(answer[6][:4]))<1e-10
    result=dict(status='CONDITIONAL_CURRENT_DENSITY_IMPLEMENTATION_PASS',
        quadrature_max_error=max_quad,partition_raw_moment_max_error=max_partition,
        linear_synaptic_max_error=max_synaptic,deterministic_refractory_timing_exact=True,
        scope='Algebra, conservation and native discrete refractory ordering only; not a Gaussian closure accuracy or spatial-model acceptance.')
    write(OUT/'conditional_current_density_implementation_check.json',result);print(result)


if __name__=='__main__':main()
