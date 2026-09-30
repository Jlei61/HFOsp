"""Independent post-fit state/count/derivative audit; no new targets or fit."""
from voltage_memory_rate import *
from voltage_memory_numerics import direct
from scipy.special import expit


def main():
    torch.set_num_threads(2); nets,bases,locked=load_models(); rows=[]
    for pop in 'EI':
        net=nets[pop]; src=OUT/'refractory_rate_response'; profiles=read(src/'profiles.json')['rows']
        case=next(x for x in profiles if x['split']=='train' and x['pop']==pop and x['startup'])
        wave=np.load(src/'prepared.npz')['wave'][case['id']]
        for dt in [.1,.05]:
            for theta in [18.,16.5]:
                f=features(wave,case['period_ms'],dt,0,round(50/dt),pop,theta)
                physical=physical_from_features(f,theta); b=bases[pop].evaluate(physical,theta)
                predicted,vpre,minimum=direct(net,f,b,dt,theta)
                fired=np.zeros(len(predicted)); observed=np.zeros(len(predicted)); v=VR; a=np.exp(-dt/net.tau)
                nref=round(net.ref/dt); remain=1.
                with torch.no_grad():
                    for k in range(len(fired)):
                        used=fired[max(0,k-nref+1):k].sum()
                        # Expanded mean-voltage balance, not the fused update.
                        prior=a*v+(1-a)*physical[k,0]-(1-a)*(physical[k,0]-VR)*used
                        observed[k]=prior
                        ff=np.r_[f[k],np.arcsinh((prior-VR)/(theta-VR))/3]
                        ell=float(net.logits(torch.tensor(ff[None]),torch.tensor([b[k]]))[0])
                        fired[k]=(1-used)*expit(ell+np.log(dt/DT0))
                        v=prior-(theta-VR)*fired[k]; remain=min(remain,1-used-fired[k])
                re=float(abs(predicted-fired*1000/dt).max()); ve=float(abs(vpre-observed).max())
                from_trace=voltage_trace(physical[:,0].copy(),predicted*dt/1000,dt,net.tau,net.ref,theta)
                te=float(abs(from_trace-vpre).max())
                assert re<1e-7 and ve<1e-9 and te<1e-10
                assert remain>=-1e-12 and abs(minimum-remain)<1e-10
                rows.append(dict(pop=pop,theta_mV=theta,dt_ms=dt,rate_max_error_hz=re,voltage_max_error_mV=ve,
                    independent_trace_error_mV=te,minimum_available_mass=remain))
        d=np.load(DEST/f'training_arrays/{pop}.npz'); x=d['flux_features'][::4096].astype(float)
        ff=torch.tensor(x,requires_grad=True); derivative=torch.autograd.grad(net.correction(ff).sum(),ff)[0].detach().numpy()[:,39]
        h=1e-6; plus=x.copy(); minus=x.copy(); plus[:,39]+=h; minus[:,39]-=h
        with torch.no_grad(): fd=(net.correction(torch.tensor(plus))-net.correction(torch.tensor(minus))).numpy()/(2*h)
        error=float((abs(fd-derivative)/np.maximum(1,abs(derivative))).max()); assert error<1e-7
        rows.append(dict(pop=pop,voltage_feature_derivative_error=error,
            derivative_min=float(derivative.min()),derivative_max=float(derivative.max())))
    write(DEST/'independent_implementation_audit.json',dict(status='PASS',rows=rows,weights=locked['files'],
        scope='Trained nonzero memory feedback, refractory bounds, units and independent implementation. No local response or spatial acceptance follows.'))
    log('TRAINED VOLTAGE MEMORY AUDIT PASS',rows)


if __name__=='__main__': main()
