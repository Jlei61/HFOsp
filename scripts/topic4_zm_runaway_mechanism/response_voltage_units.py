"""Explicit unit-corrected response facade; never mutates the frozen v3 code.

Calibration uses theta=18 mV and reset=11 mV. Its alpha/a_E/a_I are
dimensionless, while eta_E/eta_I multiply variance (mV^2) to give mV.
For voltage scale k=(theta-reset)/7, eta(theta)=eta(reference)/k.
The theta is a fixed group parameter; all moment derivatives get the same
factor. This does not repair the shape or nonlinear-history approximation.
"""
from common import *
from response_tables import CUDA_RESP

REFERENCE_GAP_MV=7.
RESET_MV=11.


class VoltageScaledResponseTable:
    def __init__(self, reference):
        self.reference=reference

    def __getattr__(self,name):
        return getattr(self.reference,name)

    def evaluate(self,mu,ve,vi,theta):
        theta=np.asarray(theta,float)
        if np.any(theta<=RESET_MV):
            raise ValueError('LIF threshold must exceed reset')
        weights,gradients=self.reference.evaluate(mu,ve,vi,theta)
        factor=REFERENCE_GAP_MV/(theta-RESET_MV)
        weights[3:]*=factor[None,:]
        gradients[3:]*=factor[None,None,:]
        return weights,gradients

    def device_block(self):
        return self.reference.device_block()


assert CUDA_RESP.count('void resp_weights(')==1
CUDA_RESP_VOLTAGE_UNITS=CUDA_RESP.replace('void resp_weights(', 'void resp_weights_reference(')+r'''
__device__ void resp_weights(const double* S,double mu,double ve,double vi,double theta,double* w,double* grad){
 resp_weights_reference(S,mu,ve,vi,theta,w,grad);
 double factor=7./(theta-11.);
 for(int p=3;p<5;p++){
  w[p]*=factor;
  for(int j=0;j<3;j++)grad[3*p+j]*=factor;
 }
}
'''
