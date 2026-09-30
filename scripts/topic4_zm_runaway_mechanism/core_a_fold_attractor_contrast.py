"""Does the certified equilibrium SN locate the observed loss of termination?

Both fields start from the same complete, actually observed quiet history.
This tests relevance of the fold; it does not infer a basin from a rate label.
"""
from common import OUT,np,read,write,log,model
from core_a_equilibrium_branch import Family
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
import onset_state_continuation as flow
import core_a_resource_branch as local
from pathlib import Path
import argparse

DEST=OUT/'core_a_bifurcation_type_20260924/fold_attractor_contrast'
flow.DEST=DEST;local.DEST=DEST


def register():
    cert=read(DEST.parent/'fold_certificate/result.json');assert cert['status']=='GENERIC_EQUILIBRIUM_SADDLE_NODE_CERTIFIED'
    source=OUT/'core_a_transition_continuation_20260924/mid1_from_upper_ext_long'
    assert read(source/'local_state_audit.json')['status']=='AUDIT_PASS'
    DEST.mkdir(exist_ok=True);assert not (DEST/'conditions.json').exists()
    s=PhysicalDelayConditionalDrift();family=Family(s);fields={};conditions={};coordinates={}
    for label,delta in [('below',-.01),('above',.01)]:
        D=cert['D_A']+delta;tm=family.set(D);fields[label]=s.Z.copy()
        coordinates[label]=dict(D_A=D,Z_A=1-D,D_global=s.D,native_time_ms=tm)
        conditions[label]=dict(label=label,field=label,initial=str(source/'final_state.npz'),duration_ms=20000,previous_elapsed_ms=0,dt_ms=.05,source_dt_ms=.05)
    assert np.array_equal(fields['below'][~family.A],fields['above'][~family.A])
    np.savez_compressed(DEST/'fields.npz',**fields);write(DEST/'conditions.json',conditions)
    write(DEST/'contract.json',dict(question='Does crossing the certified high-equilibrium recruitment fold change actual self-termination when starting from the same complete interictal-like quiet history?',
        reference_fold=cert['D_A'],coordinates=coordinates,
        equations='Unchanged full spatial conditional drift, only native within-Core-A Z pattern varies; all non-Core-A Z fixed native9s and all M dynamic. Same complete initial state for both fields, original constant input, no future count innovations.',
        source_history='Last full state of the already audited30s midpoint trajectory that returned to quiet. Neither equilibrium nor ad hoc high/low rate reset is used.',
        interpretation='A field above this SN that still returns to quiet rejects identifying this equilibrium fold with irreversible local activation for this history. Contrasting finite-window outcomes supports a candidate correspondence but does not certify a new attractor or permanence. Identical outcomes do not prove absence of a bifurcation elsewhere.',
        duration_ms=20000,additional_readout='Record actual AMPA current, raw GABA current and M every10ms to distinguish changing slow adaptation from fast inhibitory interruption of activity.',model_promoted=False))


def run(label,device):
    original_initialize=flow.initialize;c=read(DEST/'conditions.json')[label];folder=DEST/label
    def initialize(e,condition):
        Z=original_initialize(e,condition);original=e.chunk;inputs=[];calls=0
        def chunk():
            nonlocal calls,inputs
            out=original();inputs.append(np.vstack([e.syn.get()[[1,3,4]],e.local.physical.get()[0]]));calls+=1
            if calls%500==0:
                block=calls//500-1
                np.savez_compressed(folder/f'input_block{block:02d}.npz',state_time_ms=np.arange(block*5000+10,(block+1)*5000+1,10),
                    currents_ampa_rawgaba_M_mu=np.array(inputs),Z=Z,
                    channel_names=np.array(['AMPA','raw_GABA','M_after_step','mean_drive_used_by_response']))
                inputs=[]
            return out
        e.chunk=chunk;return Z
    flow.initialize=initialize
    try:flow.run(label,device)
    finally:flow.initialize=original_initialize


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit']);p.add_argument('--label',choices=['below','above']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    {'register':register,'check':lambda:flow.check(a.device),'run':lambda:run(a.label,a.device),'audit':lambda:local.audit(a.label)}[a.command]()
