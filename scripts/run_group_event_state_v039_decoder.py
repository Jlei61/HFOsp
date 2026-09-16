#!/usr/bin/env python3
"""Run the accepted tissue decoder with the new FIT-only contact vocabulary."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.topic5_group_event_state.v035.contracts import atomic_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();start=time.time();folder=a.root/'decoder_rebuilt'/a.subject;fit=a.subject+'__anatomy';arm='L3_LOCAL_PLUS_LEARNED_LR'
    if a.output.exists():raise FileExistsError(a.output)
    command=[sys.executable,str(ROOT/'scripts/train_topic5_lbss_unit_v0_2.py'),'--fit-id',fit,'--arm',arm,'--seed',str(a.seed),
        '--out-root',str(folder),'--device',a.device,'--epochs-freeze','300','--unit-root-name','formal_units','--contract-label','v039_fit_sensor_calibration_decoder']
    subprocess.run(command,check=True)
    unit=folder/'formal_units'/fit/arm/f'seed{a.seed}';metrics=json.loads((unit/'metrics.json').read_text())
    if not (unit/'DONE.json').exists() or not metrics['best_checkpoint_eligible']:raise ValueError('Decoder checkpoint is not frozen-mask eligible')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    atomic_json(a.output,dict(status='COMPLETE',subject=a.subject,seed=a.seed,unit_dir=str(unit),cache_dir=str(folder/'cache'/fit),
        metrics=metrics,checkpoint_sha256=sha(unit/'weights.pt'),metrics_sha256=sha(unit/'metrics.json'),elapsed_seconds=time.time()-start,
        selection_scope='calibration-prefix validation only',training_command=command,source_sha256=sha(__file__),
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False))
