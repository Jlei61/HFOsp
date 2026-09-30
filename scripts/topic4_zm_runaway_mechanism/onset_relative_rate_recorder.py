"""Passive rate bins relative to a possibly fractional-ms orbit start.

The native engine's output bins use its absolute clock modulo10ms. Those
bins cannot be concatenated as relative time from arbitrary shooting nodes.
This recorder reads the same emitted/expected flux after each original step
and never feeds a value back into the physical network.
"""
from common import np


class RelativeRateRecorder:
    def __init__(self,e):
        self.e=e;cp=e.cp;self.start=int(e.local.clock.get()[0]);P=e.s.P
        self.accumulator=cp.zeros((2,P));self.output=cp.zeros((10,2,P))
        self.integral=cp.zeros((2,P))
        code=r'''
        extern "C" __global__ void relative_record(const double* emitted,const double* expected,
          double* accumulator,double* output,double* integral,const int* clock,
          int start,int P,int per_ms,double dt){
          int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
          int n=clock[0]-start;if(n<=0)return;
          double a=emitted[g]*dt,b=expected[g]*dt;
          accumulator[g]+=a;accumulator[P+g]+=b;integral[g]+=a;integral[P+g]+=b;
          if(n%per_ms==0){
            int row=(n/per_ms-1)%10;
            output[(row*2)*P+g]=accumulator[g]*1000.;
            output[(row*2+1)*P+g]=accumulator[P+g]*1000.;
            accumulator[g]=0.;accumulator[P+g]=0.;
          }
        }
        '''
        self.module=cp.RawModule(code=code,options=('--fmad=false',),name_expressions=['relative_record'])
        self.kernel=self.module.get_function('relative_record');self.record();cp.cuda.get_current_stream().synchronize()
        self.stream=cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.stream.begin_capture()
            for _ in range(round(10/e.dt)):e.step();self.record()
            self.graph=self.stream.end_capture()

    def record(self):
        e=self.e;P=e.s.P
        self.kernel(((P+127)//128,),(128,),
            (e.emitted,e.local.rate,self.accumulator,self.output,self.integral,e.local.clock,
             np.int32(self.start),np.int32(P),np.int32(round(1/e.dt)),e.dt))

    def chunk(self):
        self.graph.launch(self.stream);self.stream.synchronize();return self.output.get()

    def step(self):
        self.e.step();self.record();self.e.cp.cuda.get_current_stream().synchronize()

    def read_period(self,T):
        """Complete relative1ms bins plus a flux quadrature over exactT.

        The quadrature uses the original emitted flux per actual step;
        the final partial step has its corresponding fractional weight.
        Its time-step error remains subject to ordinary mesh refinement.
        """
        from onset_segment_flow import split_step
        e=self.e;n,a=split_step(T,e.dt);chunk_steps=round(10/e.dt)
        whole,tail=divmod(n,chunk_steps);rates=[];M=[]
        for _ in range(whole):rates.extend(self.chunk()[:,0]);M.append(e.syn[4].get())
        for _ in range(tail):self.step()
        complete_bins=tail//round(1/e.dt)
        if complete_bins:rates.extend(self.output.get()[:complete_bins,0])
        integral=self.integral.get()
        if a:
            self.step()
            integral+=a*e.dt*np.array([e.emitted.get(),e.local.rate.get()])
            # integral was captured BEFORE this step, so only its fractional
            # contribution is added; the recorder's internal total is unused.
        return np.array(rates),np.array(M),integral*1000./T
