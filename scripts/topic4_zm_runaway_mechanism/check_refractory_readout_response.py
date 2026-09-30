"""Check the whole nonlinear readout's analytic gain against its own dynamics."""
from refractory_rate_response import *
from nonlinear_rate_response import physical_from_features
from validate_refractory_rate_response import evaluate

def check():
    torch.set_num_threads(2);torch.manual_seed(920066);rows=[]
    for pop in 'EI':
        net=RefractoryReadout(pop).double()
        with torch.no_grad():
            net.network[-1].weight.normal_(0,.02);net.network[-1].bias.fill_(.01)
        base=BaseLogit(pop);physical=np.array([[20.,120.,150.]])
        for theta in [14.2,18.]:
            f=np.zeros((1,39));f[:,:3]=normalized_input(physical,theta)/SCALE
            b,bg=base.evaluate(physical,theta,True);ig=normalized_jacobian(physical,theta)
            for frequency in [10.,80.]:
                T=1000/frequency;W=16384;phase=2*np.pi*np.arange(W)/W
                bank,cov,K=transfer_factors(pop,[frequency],dt=.1)
                for channel in range(3):
                    amp=.001 if channel==0 else .01
                    wave=np.tile(physical[0,:,None],(1,W));wave[channel]+=amp*np.sin(phase)
                    # At80Hz,12.5ms/0.1ms is integral; all acquisitions contain
                    # complete cycles after an identical zero-state burn.
                    burn=round(2000/.1);steps=round(1000/.1)
                    r,m=evaluate(net,base,wave,T,.1,burn,steps,theta)
                    t=(np.arange(steps)+1)*.1
                    measured=2*np.mean(r*(np.sin(2*np.pi*frequency*t/1000)+1j*np.cos(2*np.pi*frequency*t/1000)))/amp
                    predicted=net.linear_response(torch.tensor(f),torch.tensor(b),torch.tensor(bg),torch.tensor(ig),torch.tensor([channel]),torch.tensor(bank),torch.tensor(cov),torch.tensor(K)).detach().numpy()[0]
                    err=float(abs(measured-predicted)/max(abs(predicted),1e-8))
                    assert err<2e-4,(pop,theta,frequency,channel,measured,predicted,err)
                    rows.append(dict(pop=pop,theta=theta,frequency_hz=frequency,channel=channel,relative_error=err))
    write(DEST/'full_readout_response_check.json',dict(status='PASS',conditions=len(rows),maximum_relative_error=max(r['relative_error'] for r in rows),rows=rows,
        scope='Same mathematical candidate, random nonzero residual readout. This verifies derivatives and clocks, not agreement with LIF data.'))
    log('REFRACTORY FULL READOUT RESPONSE PASS',max(r['relative_error'] for r in rows))

if __name__=='__main__':check()
