"""FP32 PDF transport for orbit-initial-guess searches only.

Recurrent currents, delayed firing history, M and the weighted firing readout
remain FP64. The PDF/noise product and voltage transport are FP32, with the
known private-noise marginal restored every step. This implementation requires
a paired trajectory pilot and must never supply final critical/Floquet labels.
"""
from mixed_precision_pilot import MixedDensity
from autonomous_density import *
from population_field_gpu import CODE


class FastSearchDensity(MixedDensity):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.F=self.F.astype(cp.float32);self.Q=cp.empty_like(self.F)
        self.flux32=cp.empty((self.P,self.K),dtype=cp.float32)
        self.de32,self.dc32,self.dw32,self.nodes32,self.dr32,self.decay32=[
            x.astype(cp.float32) for x in (self.de,self.dc,self.dw,self.nodes,self.dr,self.decay)]
        code=CODE.replace('void voltage(','void voltage32(').replace('double','float')
        for old,new in [('fmin(','fminf('),('fmax(','fmaxf('),('fabs(','fabsf('),('copysign(','copysignf(')]:
            code=code.replace(old,new)
        mod=cp.RawModule(code=code,options=('--fmad=false',),name_expressions=['voltage32'])
        self.voltage32=mod.get_function('voltage32')
        code=CODE.replace('void observe(','void observe_mixed(')
        old='const double* Q,const double* flux,const double* mass'
        assert code.count(old)==1
        code=code.replace(old,'const float* Q,const float* flux,const double* mass')
        mod=cp.RawModule(code=code,options=('--fmad=false',),name_expressions=['observe_mixed'])
        self.observe_mixed=mod.get_function('observe_mixed')

    def mix_noise(self):
        moved=cp.matmul(self.transition32,self.F)
        total=cp.sum(moved,axis=2,dtype=cp.float64)
        moved*=(self.fixed_marginal[None,:]/total).astype(cp.float32)[:,:,None]
        return cp.ascontiguousarray(moved)

    def advance_step(self,extra_drive=None):
        i32=np.int32;step=self.step_index
        self.recurrent((self.P,),(128,),(*self.operators,self.history,self.dp,self.tm,self.dr,
            self.qa,self.ia,self.qg,self.ig,self.qe,self.ie,self.Z,self.M,self.drive,
            i32(self.P),i32(self.D),i32(step),*self.synpars))
        if extra_drive is not None:self.drive+=cp.asarray(extra_drive)
        moved=self.mix_noise();drive32=self.drive.astype(cp.float32)
        self.voltage32((self.P*self.K,),(128,),(moved,self.Q,self.flux32,
            self.de32,self.dc32,self.dw32,self.nodes32,self.dr32,self.decay32,drive32,
            self.drefs,i32(self.K),i32(self.nv),i32(self.width)))
        self.after_voltage()
        self.observe_mixed((self.P,),(128,),(self.Q,self.flux32,self.mass,self.pop,self.ig,
            self.Z,self.M,self.history,self.activity,self.maxneg,self.minflux,self.masserror,
            i32(self.K),i32(self.nv),i32(self.width),i32(self.P),i32(self.D),i32(step)))
        self.Z[:]=self.z_clamp;self.F,self.Q=self.Q,self.F
        self.positive_emitted+=cp.maximum(self.activity,0.);self.negative_emitted+=cp.maximum(-self.activity,0.)
        self.maximum_lower_mass=cp.maximum(self.maximum_lower_mass,self.F[:,:,0].astype(cp.float64)@self.mass)
        self.minimum_drive_bound=cp.minimum(self.minimum_drive_bound,self.drive+self.dr*self.noise_min)
        self.step_index+=1
        return self.activity

    def after_voltage(self):
        pass


class ConservativeFastSearchDensity(FastSearchDensity):
    """Also restore the exact modal marginal after FP32 voltage transport.

Each conditional voltage/reset transport conserves its total modal mass.
This projection removes roundoff in that invariant; it neither clips signed
coefficients nor modifies a voltage distribution's shape within a noise mode.
The correction size is recorded, and paired FP64 validation is still required.
"""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.maximum_transport_marginal_relative_error=cp.asarray(0.)

    def after_voltage(self):
        total=cp.sum(self.Q,axis=2,dtype=cp.float64)
        error=cp.max(abs(total/self.fixed_marginal[None,:]-1.))
        self.maximum_transport_marginal_relative_error=cp.maximum(self.maximum_transport_marginal_relative_error,error)
        factor=(self.fixed_marginal[None,:]/total).astype(cp.float32)
        self.Q*=factor[:,:,None];self.flux32*=factor

    def diagnostics(self):
        report=super().diagnostics()
        report['maximum_transport_modal_mass_relative_correction']=float(self.maximum_transport_marginal_relative_error.get())
        return report
