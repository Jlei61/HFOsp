"""Independent recurrence and population-mass audit of trained reset rate."""
from reset_memory_rate import *
from reset_memory_numerics import direct,equilibria
from scipy.special import expit


def main():
    torch.set_num_threads(2);nets,bases,locked=load_models();rows=[]
    for pop in 'EI':
        net=nets[pop];src=OUT/'refractory_rate_response';allrows=read(src/'profiles.json')['rows']
        row=next(x for x in allrows if x['split']=='train' and x['pop']==pop and x['startup'])
        wave=np.load(src/'prepared.npz')['wave'][row['id']]
        for dt in [.1,.05]:
            f=features(wave,row['period_ms'],dt,0,round(50/dt),pop)
            b=bases[pop].evaluate(physical_from_features(f));r,qs,minimum=direct(net,f,b,dt)
            fired=np.zeros(len(r));q=0.;a=np.exp(-dt/net.tau);nref=round(net.ref/dt);trace=[];minimum_available=1.
            with torch.no_grad():
                for k in range(len(r)):
                    avail=1-fired[max(0,k-nref+1):k].sum();ff=np.r_[f[k],np.log1p(q)/3]
                    ell=float(net.logits(torch.tensor(ff[None]),torch.tensor([b[k]]))[0])
                    fired[k]=avail*expit(ell+np.log(dt/DT0));trace.append(q);q=a*(q+fired[k]);minimum_available=min(minimum_available,avail-fired[k])
            rate_error=float(np.max(abs(r-fired*1000/dt)));trace_error=float(np.max(abs(qs-trace)))
            convolution=reset_trace(r*dt/1000,dt,net.tau);conv_error=float(abs(convolution-qs).max())
            assert rate_error<1e-7 and trace_error<1e-9 and conv_error<1e-12
            assert minimum_available>=-1e-12 and abs(minimum-minimum_available)<1e-10
            rows.append(dict(pop=pop,dt_ms=dt,independent_rate_error_hz=rate_error,trace_error=trace_error,convolution_error=conv_error,minimum_available_mass=minimum_available))
        # Verify analytic partial derivatives at nonzero q, including coordinate transform.
        a=np.load(DEST/f'training_arrays/{pop}.npz');ff=torch.tensor(a['flux_features'][::4096].astype(float),requires_grad=True)
        derivative=torch.autograd.grad(net.correction(ff).sum(),ff)[0].detach().numpy()[:,39]
        x=ff.detach().numpy();h=1e-6;plus=x.copy();minus=x.copy();plus[:,39]+=h;minus[:,39]-=h
        with torch.no_grad():fd=(net.correction(torch.tensor(plus))-net.correction(torch.tensor(minus))).numpy()/(2*h)
        err=float(np.max(abs(fd-derivative)/np.maximum(1,abs(derivative))));assert err<1e-7
        rows.append(dict(pop=pop,learned_q_derivative_relative_error=err,derivative_min=float(derivative.min()),derivative_max=float(derivative.max())))
    write(DEST/'independent_implementation_audit.json',dict(status='PASS',weights=locked['files'],rows=rows,
        scope='Trained nonzero-feedback implementation, causal history and mass conservation only. No waveform, gain, spatial or onset validation follows.'))
    log('RESET TRAINED INDEPENDENT AUDIT PASS',rows)


if __name__=='__main__':main()
