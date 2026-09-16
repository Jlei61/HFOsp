"""Known-truth instruments. These independent synthetic episodes are not patients.

The strong-signal tier tests interfaces first. Real-record support/missingness
calibration is a separate tier, not inferred from success on these instruments.
"""
from __future__ import annotations
import numpy as np


CASES = ('fixed_history', 'linear_nonlinear_observation', 'nonlinear_transition',
         'background_only', 'separate_states', 'no_event_feedback')


def generate(case, *, seed=39001, n=3072, strength=1., steps=96):
    if case not in CASES: raise ValueError(case)
    rng = np.random.default_rng(seed)
    dt = 8 / steps
    raw = rng.normal(size=(n, steps, 4)) * .22
    missing = rng.random((n, steps)) < .08
    context = rng.normal(size=(n, 2))
    state = np.zeros((n, 4))
    matrix = np.array([[-.3,-.65,0,0],[.65,-.3,0,0],[0,0,-.45,-.3],[0,0,.3,-.45]])
    if case == 'fixed_history': matrix = -np.diag([2., .5, .25, .125])
    def drift(s):
        out = s @ matrix.T
        if case == 'nonlinear_transition':
            # Nonlinear drift only; the event write remains exactly additive.
            out += .8*np.tanh(s[:, [1,0,3,2]]*1.7)
        return out
    def advance(s, elapsed):
        pieces = max(1, int(np.ceil(elapsed/(1/48))))
        h = elapsed/pieces
        for _ in range(pieces):
            a=drift(s); b=drift(s+h*a/2); c=drift(s+h*b/2); d=drift(s+h*c)
            s=s+h*(a+2*b+2*c+d)/6
        return s
    if case == 'no_event_feedback': state = rng.normal(size=(n,4))
    observed = raw.copy()
    for j in range(steps):
        state = advance(state,dt)
        if case == 'no_event_feedback':
            # Events reveal a common evolving cause and never enter its drift.
            state += rng.normal(size=(n,4))*.09
            observed[:,j] = .22*state + rng.normal(size=(n,4))*.1
        else:
            state += raw[:,j]
    observed[missing] = 0
    # A separate availability observation distinguishes missingness from a zero mark.
    inputs = np.concatenate((observed, (~missing)[...,None].astype(float)/steps),axis=-1)
    outputs={}
    for lead in (0.,2.,6.):
        future = advance(state.copy(),lead+.25)
        if case == 'background_only':
            count_latent = context[:,0]; recruitment_latent = context[:,1]
        else:
            count_latent = future[:,0] + .5*future[:,1]**2
            recruitment_latent = future[:,2] if case == 'separate_states' else future[:,0]-.5*future[:,1]
        log_mu = np.clip(1.3 + strength*.8*count_latent,-4,5)
        mu=np.exp(log_mu); dispersion=5.
        counts=rng.negative_binomial(dispersion,dispersion/(dispersion+mu)).astype(np.float32)
        logits=np.stack((strength*recruitment_latent,-strength*recruitment_latent+.2),axis=-1)
        probability=1/(1+np.exp(-np.clip(logits,-30,30)))
        recruits=(rng.random(probability.shape)<probability).astype(np.float32)
        # Reserved continuous fine-expression endpoint shares the recruitment cause.
        fine=recruitment_latent+rng.normal(size=n)*.4
        outputs[str(int(lead))]={'count':counts,'recruitment':recruits,'fine':fine.astype(np.float32),
                                'oracle_log_mean':log_mu.astype(np.float32),'oracle_probability':probability.astype(np.float32)}
    return {'inputs':inputs.astype(np.float32),'dt_hours':np.full((n,steps),dt,np.float32),
            'context':context.astype(np.float32), 'targets':outputs,
            'fit_end':int(n*2/3),'inner_end':int(n*5/6),
            'truth':dict(case=case,strength=strength,event_changes_true_target_state=case not in ('no_event_feedback','background_only'),
                         observed_event_information_predictive=case!='background_only',
                         shared_count_recruitment_cause=case!='separate_states',
                         nonlinear_transition=case=='nonlinear_transition',
                         tier='independent_episode_instrument',
                         target_width_hours=.5, target_generation='distribution of a 30-minute aggregate conditioned on midpoint state',
                         real_patient_power_calibrated=False, seed=seed)}
