#!/usr/bin/env python
"""Source and numerical-check snapshot for the v0.3.11 root."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
REPO=Path(__file__).resolve().parents[1]
FILES=sorted([str(p.relative_to(REPO)) for p in (REPO/'src'/'topic5_group_event_state'/'v0311').glob('*.py')]+
             [f'scripts/{n}' for n in ('build_group_event_state_v0311_packets.py',
              'run_group_event_state_v0311_cell.py','schedule_group_event_state_v0311.py',
              'export_group_event_state_v0311_frozen.py','score_group_event_state_v0311_seizure.py',
              'calibrate_group_event_state_v0311.py','aggregate_group_event_state_v0311.py',
              'plot_group_event_state_v0311.py','run_group_event_state_v0311_synthetic.py',
              'postprocess_group_event_state_v0311.py','watchdog_group_event_state_v0311.py',
              'build_group_event_state_v0311_manifests.py','profile_group_event_state_v0311.py',
              'snapshot_group_event_state_v0311_source.py')]+
             [f'tests/{n}' for n in ('test_group_event_state_v0311_numerics.py',
              'test_group_event_state_v0311_contract.py',
              'test_group_event_state_v0311_scoring.py')]+
             ['scripts/rescore_group_event_state_v0311.py','scripts/handoff_group_event_state_v0311.py'])

if __name__=='__main__':
    run_tests='--tests' in sys.argv
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True).stdout.strip()
    dirty=subprocess.run(['git','status','--porcelain'],cwd=REPO,capture_output=True,text=True).stdout
    files={}
    for f in FILES:
        p=REPO/f
        if p.exists():files[f]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    checks=None
    if run_tests:
        r=subprocess.run([sys.executable,'-m','pytest','tests/test_group_event_state_v0311_numerics.py',
                          'tests/test_group_event_state_v0311_contract.py',
                          'tests/test_group_event_state_v0311_scoring.py','-q','--tb=no',
                          '-p','no:cacheprovider'],cwd=REPO,capture_output=True,text=True)
        checks=dict(returncode=r.returncode,summary=r.stdout.strip().splitlines()[-1] if r.stdout else '',
                    stdout_tail=r.stdout.strip()[-1500:])
    out=dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),git_commit=git,
             worktree_dirty=bool(dirty.strip()),files=files,numerical_and_contract_checks=checks,
             note=('the v0.3.11 sources are uncommitted in this worktree; the sha256 list is the '
                   'authoritative record of what produced these results'))
    (ROOT/'source_snapshot.json').write_text(json.dumps(out,indent=1))
    print(json.dumps(dict(git=git,n_files=len(files),checks=None if checks is None else checks['summary'])))
