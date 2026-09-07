#!/usr/bin/env python3
"""Resume the authorized formal round using a dependency-scoped source lock.

Completed canaries and their broad historical snapshots stay byte-for-byte intact.
Only a phase with no result artifacts or surviving workers may migrate its snapshot.
"""
from pathlib import Path
import ast
import fcntl
import json
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_topic4_multidimensional_pilot as pilot
from scripts import analyze_topic4_multidimensional_pilot as analysis
OUT=pilot.OUT
read,write,sha=pilot.read,pilot.write,pilot.sha


def dependency_paths(root, seeds):
    """Local static import closure plus literal dynamically loaded Python files."""
    root=Path(root);seen=set();pending=list(seeds)
    def include_module(name):
        bits=name.split('.')
        for end in range(1,len(bits)+1):
            rel=Path(*bits[:end])
            for p in (root/rel.with_suffix('.py'),root/rel/'__init__.py'):
                if p.is_file():pending.append(str(p.relative_to(root)))
        if len(bits)==1:
            p=root/'src/snn_engine'/f'{name}.py'
            if p.is_file():pending.append(str(p.relative_to(root)))
    while pending:
        rel=pending.pop()
        if rel in seen:continue
        p=root/rel
        if not p.is_file():raise RuntimeError(f'missing dependency: {rel}')
        seen.add(rel);tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for a in node.names:include_module(a.name)
            elif isinstance(node,ast.ImportFrom):
                name=node.module or ''
                if node.level:
                    parent=list(Path(rel).parent.parts)
                    name='.'.join(parent[:len(parent)-node.level+1]+([name] if name else []))
                include_module(name)
                for a in node.names:
                    if a.name!='*':include_module(name+'.'+a.name)
            elif isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py') and '\n' not in node.value:
                # Includes os.path.join('scripts', 'run_*_readout.py') loaders.
                for candidate in (root/node.value,root/'scripts'/node.value,root/'src/snn_engine'/node.value,p.parent/node.value):
                    if candidate.is_file():pending.append(str(candidate.relative_to(root)))
    return seen


def verified_canaries():
    records=[];modules=set()
    for phase,n in [('canary',7),('observer_parity',1)]:
        folder=OUT/'execution'/phase;sp=folder/'runtime_snapshot.json'
        snapshot=read(sp)
        for name in ('execution_config.json','candidate_manifest.json'):
            p=folder/name
            if snapshot['input_hashes'].get(str(p.resolve()))!=sha(p):raise RuntimeError('historical canary input changed')
        workers=sorted((folder/'workers').glob('*.json'))
        if len(workers)!=n:raise RuntimeError('incomplete historical canaries')
        for wp in workers:
            if not pilot.worker_complete(wp,sp):raise RuntimeError('canary artifact/provenance mismatch')
            d=read(wp)
            for stage in ('runtime_module_sha256','runtime_module_sha256_at_completion'):
                for rel,digest in d['provenance'][stage].items():
                    if sha(ROOT/rel)!=digest:raise RuntimeError(f'actual canary runtime changed: {rel}')
                    modules.add(rel)
            records.append({'path':str(wp),'sha256':sha(wp),'array_path':d['arrays']['path'],
                            'array_sha256':d['arrays']['sha256'],'snapshot_sha256':sha(sp)})
    if not read(OUT/'observer_parity.json')['pass'] or not read(OUT/'parameter_canary_audit.json')['pass']:
        raise RuntimeError('canary checks did not pass')
    # Verify the recorded baseline parity directly again without rewriting it.
    import numpy as np
    left=OUT/'execution/canary/workers/historical__baseline_seed_2511.npz'
    right=OUT/'execution/observer_parity/workers/historical__baseline_seed_2511.npz'
    if sha(left)!=sha(right):raise RuntimeError('baseline arrays differ')
    return records,modules


def prepare_resume():
    path=OUT/'resume_dependency_lock.json'
    if path.exists():
        contract=read(path);pilot.base.verify_sources(contract['source_hashes'])
        for p,h in contract['input_hashes'].items():
            if sha(p)!=h:raise RuntimeError(f'resume input changed: {p}')
        return contract
    records,modules=verified_canaries()
    modules.update({'scripts/resume_topic4_multidimensional_pilot.py',
                    'scripts/run_topic4_multidimensional_pilot.py',
                    'scripts/run_topic4_rev12_node_worker.py',
                    'scripts/analyze_topic4_multidimensional_pilot.py',
                    'scripts/render_topic4_multidimensional_raw.py',
                    'src/topic4_interictal_pilot_evaluation.py'})
    paths=dependency_paths(ROOT,modules)
    old=read(OUT/'source_lock.json')
    changed=[p for p,h in old['source_hashes'].items() if sha(ROOT/p)!=h]
    if set(changed)&paths:raise RuntimeError('changed files belong to actual dependency closure')
    lock={p:sha(ROOT/p) for p in sorted(paths)}
    # No simulation/model/evaluator/design changes are allowed in this resumption.
    for p in paths & set(old['source_hashes']):
        if lock[p]!=old['source_hashes'][p]:raise RuntimeError(f'pilot dependency changed: {p}')
    folder=OUT/'execution/paired_round1'
    if any((folder/'workers').glob('*')):
        raise RuntimeError('formal phase already has result artifacts; requires a different migration')
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if str(folder/'execution_config.json').encode() in args:
                raise RuntimeError('a formal-phase worker is still alive')
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    archive=OUT/'setup_attempts/002_unrelated_audit_lock_failure';archive.mkdir(parents=True,exist_ok=True)
    for src in (folder/'runtime_snapshot.json',OUT/'status.json'):
        if src.exists() and not (archive/src.name).exists():shutil.copy2(src,archive/src.name)
    interrupted_logs=list((folder/'run_logs').glob('*'))
    if any(p.stat().st_size for p in interrupted_logs):
        raise RuntimeError('nonempty interrupted worker logs require investigation')
    if interrupted_logs:
        shutil.move(str(folder/'run_logs'),str(archive/'interrupted_formal_run_logs'))
    inputs=[OUT/'design.json',OUT/'evaluation_manifest.json',OUT/'evaluator.pkl',
            OUT/'observer_parity.json',OUT/'parameter_canary_audit.json',
            ROOT/'config/topic4_joint_xy_kernel_v4.json',ROOT/'config/topic4_joint_xy_kernel_v3.json',
            folder/'execution_config.json',folder/'candidate_manifest.json']
    ev=read(OUT/'evaluation_manifest.json')
    inputs += [Path(ev['patient_training_path']),Path(ev['patient_xy_path'])]
    for f,key in [('design.json','design_sha256'),('evaluation_manifest.json','evaluation_manifest_sha256')]:
        if sha(OUT/f)!=old[key]:raise RuntimeError('original scientific input changed')
    contract={'status':'SCOPED_DEPENDENCY_RESUMPTION_VERIFIED','created_unix':time.time(),
        'source_hashes':lock,'input_hashes':{str(p.resolve()):sha(p) for p in inputs},
        'old_lock_sha256':sha(OUT/'source_lock.json'),'excluded_changed_audit_sources':changed,
        'verified_canaries':records,'historical_results_rewritten':False,
        'interrupted_formal_attempts_without_artifacts':len(interrupted_logs),
        'scientific_design_changed':False,'parameters_or_thresholds_changed':False}
    snapshot=read(folder/'runtime_snapshot.json');snapshot['source_hashes']=lock
    snapshot['resume_reason']='Exclude unrelated audit scripts; actual dependencies unchanged; interrupted formal attempts produced no artifacts; logs archived.'
    write(folder/'runtime_snapshot.json',snapshot)
    write(path,contract)
    return contract


def main():
    guard=open(OUT/'controller.lock','a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract=prepare_resume()
    if '--prepare-only' in sys.argv:
        print(json.dumps({'status':contract['status'],'locked_dependencies':len(contract['source_hashes']),
                         'excluded_changed_sources':contract['excluded_changed_audit_sources']}));return
    d=read(OUT/'design.json')
    pilot.run_phase('paired_round1',d['candidates'],d['network_and_dynamics_seeds'],d['duration_ms'],
                    contract['source_hashes'],d['maximum_workers'])
    pilot.status('ANALYZING_COMPLETED_MULTIDIMENSIONAL_ROUND')
    subprocess.run([pilot.base.PYTHON,str(ROOT/'scripts/analyze_topic4_multidimensional_pilot.py')],
                   cwd=ROOT,env=pilot.base.ENV,check=True)
    pilot.status('ROUND1_COMPLETE_PENDING_SCIENTIFIC_REVIEW',n_candidates=len(d['candidates']),
                 n_jobs=d['n_jobs'],automatic_next_round=False,final_substrate_frozen=False,fig5_released=False)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        pilot.status('ERROR',error=repr(exc));raise
