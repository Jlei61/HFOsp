#!/usr/bin/env python3
"""Drain the owned v1 controller at a completed batch before loss revision.

Never terminates a live simulation. The controller is stopped first to prevent
new dispatch; any already-started descendants finish before service retirement.
"""
from pathlib import Path
import json
import os
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v2'
SERVICE = 'codex-t4-joint-kernel-xy-20260906.service'


def proc(pid):
    try:
        p = Path('/proc')/str(pid)
        fields = p.joinpath('stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'parent': int(fields[1]), 'start': fields[19],
                'command': p.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except (FileNotFoundError, ProcessLookupError): return None


def descendants(parent):
    rows = [proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    rows = [r for r in rows if r]; selected = {parent}
    while True:
        new = {r['pid'] for r in rows if r['parent'] in selected}
        if new.issubset(selected): break
        selected |= new
    return [r for r in rows if r['pid'] in selected and r['pid'] != parent and r['state'] != 'Z']


def record(**kwargs):
    p = OUT/'optimizer_revision_batch_guard.json'; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'service': SERVICE, 'updated_unix': time.time(), **kwargs}, indent=2)+'\n')
    os.replace(tmp, p)


def main():
    pid = int(subprocess.check_output(['systemctl', '--user', 'show', SERVICE, '-p', 'MainPID', '--value'], text=True))
    identity = proc(pid)
    if not identity or 'scripts/run_topic4_joint_xy_kernel_search.py' not in identity['command']:
        raise RuntimeError('owned adaptive controller is not live')
    marker = OUT/'rounds/000/analysis.json'
    record(status='WAITING_FIRST_BATCH_ANALYSIS', controller=identity,
           reason='Eight-network results expose optimistic two-network ranking. Finish all racers, then restrict local anchors to completed common-seed candidates; keep objective and acceptance unchanged.')
    while not marker.exists():
        now = proc(pid)
        if not now or now['start'] != identity['start']:
            record(status='CONTROLLER_EXITED_BEFORE_BATCH_BOUNDARY'); return
        time.sleep(2)
    now = proc(pid)
    if not now or now['start'] != identity['start']: raise RuntimeError('controller identity changed')
    os.kill(pid, signal.SIGSTOP)
    while True:
        now = proc(pid)
        if not now or now['start'] != identity['start']: raise RuntimeError('stopped controller identity changed')
        live = descendants(pid)
        record(status='DRAINING_EXISTING_CHILDREN_FOR_OPTIMIZER_REVIEW', controller=now, children=live)
        if not live: break
        time.sleep(5)
    # A deliberate stop suppresses Restart=on-failure. Continue only to deliver
    # systemd's pending termination after all descendants have finished.
    subprocess.run(['systemctl', '--user', 'stop', '--no-block', SERVICE], check=True)
    try: os.kill(pid, signal.SIGCONT)
    except ProcessLookupError: pass
    subprocess.run(['systemctl', '--user', 'disable', SERVICE], check=True)
    record(status='BATCH_DRAIN_COMPLETE_V2_RETIRED_FOR_OPTIMIZER_REVIEW', controller_pid=pid,
           completed_batch=str(marker), no_live_simulations_terminated=True,
           goal_remains_active=True)


if __name__ == '__main__': main()
