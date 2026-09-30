#!/usr/bin/env python3
"""Implementation and development diagnostics, separate from fresh validation."""
import numpy as np
import torch
from campaign import ROOT,read,write,sha
import conductance_dynamic_v4 as v


def main():
    torch.set_num_threads(3)
    assert read(v.OUT/'fit_locked.json')['status']=='FROZEN_BEFORE_FRESH_TARGETS'
    model=v.ResponseV4();model.load_state_dict(torch.load(v.OUT/'locked_model.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    rows=read(v.OUT/'validation_rows.json');data=v.tensors(v.OUT/'validation_inputs.npz');_,gain=model.linear(data)
    ix=np.array([i for i,r in enumerate(rows) if r['frequency_Hz']==0. and r['primary']]);ch=np.array([rows[i]['channel'] for i in ix])
    pars=np.load(v.OUT/'validation_inputs.npz')['pars'][ix];physical=pars[:,[0,2,3]];g=np.array([rows[i]['g'] for i in ix])
    h=1e-5*np.maximum(1.,abs(physical[np.arange(len(ix)),ch]));plus=physical.copy();minus=physical.copy()
    plus[np.arange(len(ix)),ch]+=h;minus[np.arange(len(ix)),ch]-=h
    pp=np.concatenate([plus,minus]);gg=np.tile(g,2)
    pars=np.array([v.condition(mu,18.,ve,vi,'E') for mu,ve,vi in pp]);f,b=v.base_features(pars,gg)
    with torch.no_grad():r=model.static(torch.tensor(f),torch.tensor(b)).numpy()
    fd=(r[:len(ix)]-r[len(ix):])/(2*h);linear=gain.detach().numpy()[ix]
    assert np.allclose(fd,linear.real,rtol=2e-4,atol=2e-6)
    assert np.max(abs(linear.imag))<1e-10
    write(v.OUT/'stationary_derivative_implementation.json',dict(status='PASS',DC_components=len(ix),maximum_absolute_difference=float(np.max(abs(fd-linear.real))),same_stationary_function=True,scope='Finite differences of candidate versus its analytic derivative; no MonteCarlo targets and no scientific derivative pass.'))
    rows=read(v.OUT/'training_rows.json');d=v.tensors(v.OUT/'training_inputs.npz');t=np.load(v.OUT/'training_targets.npz');_,h=model.linear(d)
    ratio=abs(h.detach().numpy()-t['target'])/t['norm'];groups=[]
    for label,cond in [('g_zero',np.array([r['g']==0 for r in rows])),('g_positive',np.array([r['g']>0 for r in rows])),('DC',np.array([r['frequency_Hz']==0 for r in rows])),('nonzero_frequency',np.array([r['frequency_Hz']>0 for r in rows]))]:
        selected=cond&t['estimable'];a=ratio[selected]
        groups.append(dict(label=label,total=int(selected.sum()),within_tolerance=int((a<=1).sum()),ratio_quantiles=np.quantile(a,[.5,.9,.99]).tolist()))
    failures=[]
    for i in np.argsort(np.where(t['estimable'],ratio,-1))[-30:][::-1]:failures.append(dict(**rows[i],error_over_tolerance=float(ratio[i]),measured=[float(t['target'][i].real),float(t['target'][i].imag)],predicted=[float(h.detach().numpy()[i].real),float(h.detach().numpy()[i].imag)]))
    write(v.OUT/'training_diagnostic.json',dict(status='DEVELOPMENT_ONLY',groups=groups,worst=failures,question='Which domain and response terms still limit this frozen candidate?',weights_sha256=sha(v.OUT/'locked_model.pt'),not_independent_validation=True))
    print(groups,flush=True)


if __name__=='__main__':main()
