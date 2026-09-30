"""Audit variance clipping and table support on the actual periodic orbit."""
from native_path import *
from streaming_periodic import StreamGalerkin,StreamPeriodic
import argparse,gc


def main(a):
    s=model();attach_rate_entry_path(s);rows=[]
    for file in a.orbits:
        z=np.load(file);N=len(z['r']);s.set_D(float(z['D']))
        if N%2:
            StreamGalerkin.harmonic_block=33;o=StreamGalerkin(s,N,max(16384,2*N),a.device)
        else:
            StreamPeriodic.harmonic_block=33;o=StreamPeriodic(s,N,a.device)
        o.cache_mean_operators=False;cp=o.cp;cp.fft.config.get_plan_cache().set_size(0)
        inp=o.inputs(cp.asarray(z['r']),float(z['T']),s.Z);ph=o.phi(inp)
        mu,ve,vi,mus,vEf,vIf,vEv,vIv=inp;w=ph[5:10]
        me=w[0]*mu+(1-w[0])*mus+w[3]*(ve-vEf)+w[4]*(vi-vIf)
        ee=w[1]*ve+(1-w[1])*vEv;ii=w[2]*vi+(1-w[2])*vIv
        n=inp.shape[1];weights=cp.asarray(s.sizes*s.E)[None,:];rate=ph[0]
        mass=cp.sum(weights*rate);active=rate>.01
        def stats(mask):
            return dict(E_cell_time_fraction=float(cp.sum(weights*mask)/(n*weights.sum())),
                        E_rate_mass_fraction=float(cp.sum(weights*rate*mask)/mass),
                        active_E_cell_time_fraction=float(cp.sum(weights*mask*active)/cp.sum(weights*active)))
        q=dict(source=file,D=float(z['D']),T_ms=float(z['T']),N=N,sampled_times=n,
               negative_effective_variance_E=stats(ee<0),negative_effective_variance_I=stats(ii<0),
               min_effective_variance_E=float(cp.min(ee)),min_effective_variance_I=float(cp.min(ii)))
        for source_name,m,vE,vI,tables in [('response',mu,ve,vi,s.resp.tables),('static',me,cp.maximum(ee,0),cp.maximum(ii,0),s.spline)]:
            outside=cp.zeros_like(m,dtype=bool)
            components={}
            for pop,mask in [('E',s.E),('I',~s.E)]:
                table=tables[pop];sc=cp.asarray(s.theta[mask]-11.)[None,:]
                u=cp.arcsinh((m[:,mask]-11)/sc);sigmaE=cp.sqrt(cp.maximum(vE[:,mask],0))/sc;sigmaI=cp.sqrt(cp.maximum(vI[:,mask],0))/sc
                outside[:,mask]=(u<table.u[0])|(u>table.u[-1])|(sigmaE>table.sEmax)|(sigmaI>table.sImax)
                if pop=='E':
                    for name,condition in [('mu_low',u<table.u[0]),('mu_high',u>table.u[-1]),('sigma_E_high',sigmaE>table.sEmax),('sigma_I_high',sigmaI>table.sImax)]:
                        fullmask=cp.zeros_like(outside);fullmask[:,mask]=condition;components[name]=stats(fullmask)
                    components['table_bounds']=dict(x=[float(np.sinh(table.u[0])),float(np.sinh(table.u[-1]))],sigma_E_max=float(table.sEmax),sigma_I_max=float(table.sImax))
            q[source_name+'_outside_tabulated_domain']=stats(outside)
            q[source_name+'_outside_components']=components
            if source_name=='response':
                massout=cp.sum(weights*rate*outside)
                q['response_outside_rate_weighted_weights']=[float(cp.sum(weights*rate*outside*w[i])/massout) for i in range(5)]
        q['claim']='Support and nonsmooth-clipping audit; boundary clipping is not by itself a proven onset mechanism'
        rows.append(q);write(OUT/'orbit_domain_audit.json',dict(status='RUNNING',rows=rows));log('ORBIT DOMAIN',q)
        del o,inp,ph,w,me,ee,ii,mu,ve,vi,mus,vEf,vIf,vEv,vIv,weights,rate,active,outside
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(OUT/'orbit_domain_audit.json',dict(status='COMPLETE',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbits',nargs='+');p.add_argument('--device',type=int,default=1);main(p.parse_args())
