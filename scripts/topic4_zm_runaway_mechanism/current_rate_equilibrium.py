"""Exact static layer of the current conditioned39 expected-rate equations.

Physical spatial operators, thresholds, delays, resource fields and M law are
unchanged. This class uses FULL diffusion, as the existing expected-rate arm;
it is not the finite-count conditional drift. No old-v3 response is reused.
No stability label follows from solving these static equations.
"""
from common import model, np
from model_v3 import SpatialRateV3
from conditioned_refractory_rate import load_models, DT0, SCALE, torch
from nonlinear_rate_response import normalized_input, normalized_jacobian
from scipy.special import expit


class CurrentRateEquilibrium(SpatialRateV3):
    def __init__(self, grid=40):
        physical=model(grid)
        self.__dict__.update(physical.__dict__)
        self.Z=physical.Z.copy()
        self.nets,self.bases,self.response_lock=load_models()
        self.response_kind='conditioned39 absolute-refractory rate; full diffusion'

    def local_operating(self,mu,ve,vi):
        physical=np.column_stack([mu,ve,vi])
        assert physical.shape==(self.P,3) and np.isfinite(physical).all()
        assert np.all(physical[:,1:]>=0)
        result=dict(rate=np.zeros(self.P),gradient=np.zeros((self.P,3)),
            log_hazard=np.zeros(self.P),feature_gradient=np.zeros((self.P,39)),
            input_normalization_gradient=np.zeros((self.P,3)),base_gradient=np.zeros((self.P,3)))
        for pop,mask in [('E',self.E),('I',~self.E)]:
            x=physical[mask];theta=self.theta[mask]
            f=np.zeros((len(x),39));f[:,:3]=normalized_input(x,theta)/SCALE
            ft=torch.tensor(f,dtype=torch.float64,requires_grad=True)
            correction=self.nets[pop].network(ft).squeeze(-1)
            grad=torch.autograd.grad(correction.sum(),ft)[0].detach().numpy()
            base,bg=self.bases[pop].evaluate(x,theta,True)
            ell=base+correction.detach().numpy();ref=self.ref[mask]
            # r=rho/(1+rho*t_ref),rho=exp(ell)/DT0; rates per millisecond.
            rate=expit(ell+np.log(ref/DT0))/ref
            du=normalized_jacobian(x,theta)
            derivative=rate[:,None]*(1-rate*ref)[:,None]*(bg+grad[:,:3]*du)
            result['rate'][mask]=rate;result['gradient'][mask]=derivative
            result['log_hazard'][mask]=ell;result['feature_gradient'][mask]=grad
            result['input_normalization_gradient'][mask]=du;result['base_gradient'][mask]=bg
        return result

    def phi(self,mu,ve,vi,order=1):
        if order not in (0,1):
            raise NotImplementedError('No second derivative or fold certificate is supplied by this static adapter.')
        r=self.local_operating(mu,ve,vi);g=r['gradient']
        return dict(rate=r['rate'],d_mu=g[:,0],d_ve=g[:,1],d_vi=g[:,2])

    def characteristic(self,*args,**kwargs):
        raise NotImplementedError('Current-response temporal characteristic has not yet been implemented and verified; do not use old-v3 stability.')
