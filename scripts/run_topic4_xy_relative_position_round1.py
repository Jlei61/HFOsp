#!/usr/bin/env python3
"""Round 1: existing signed-direction fit plus a core-relative-position prior.

Reuses the frozen VTH-only workers and memory-limited phase scheduler. A
controller handoff drains existing work and never mutates running sources.
"""
from pathlib import Path
import argparse
import fcntl
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_topic4_xy_direction_research as previous
from src.topic4_xy_relative_position_round import (relative_position_ranking,
                                                   primary_and_unconstrained_nominees)
base=previous.base
OUT=base.OUT
CONTRACT=OUT/'round1_relative_position_contract.json'
ROUND_SOURCES=['src/topic4_xy_relative_position_round.py',
               'scripts/run_topic4_xy_relative_position_round1.py',
               'scripts/paper_figures/plot_topic4_xy_relative_position_round1.py']


def prepare_round():
    direction=previous.prepare()
    weight=direction['weak_prior_weight']
    if weight != 0.1:
        raise RuntimeError('the previously prespecified 0.1 weight changed')
    expected={'source_hashes':{str(ROOT/p):base.sha(ROOT/p) for p in ROUND_SOURCES},
              'direction_objective_sha256':base.sha(previous.OBJECTIVE_PATH),
              'original_search_design_sha256':base.sha(OUT/'search_design.json'),
              'v2_runtime_lock_sha256':base.sha(OUT/'runtime_lock_direction_v2.json')}
    if CONTRACT.exists():
        result=base.read(CONTRACT)
        if result['locked_inputs']!=expected:
            raise RuntimeError('round 1 input or source changed')
        return direction,result
    if (OUT/'global/aggregate.json').exists() or (OUT/'refinement_design.json').exists():
        raise RuntimeError('round 1 must be fixed before global selection or refinement')
    contract={'status':'ROUND1_RELATIVE_POSITION_SOFT_CONSTRAINT_FIXED', 'round':1,
        'created_unix':time.time(),'locked_inputs':expected,
        'primary_score':'J_round1 = J_direction + 0.1 * sin(core_line_angle - patient_training_axis_angle)^2',
        'primary_core_prior_weight':weight,
        'relative_axis_reference_deg':direction['patient_direction_summary']['axial_angle_deg'],
        'structural_EE_axis_deg':direction['structural_ee_axis_deg'],
        'constraint_kind':'soft relative-position penalty; not a hard angle cutoff',
        'free_parameters':['x1','y1','x2','y2'],
        'midpoint':'free within existing geometry domain',
        'separation':'free, subject to the existing >=4 mm and disjoint-core geometry rules',
        'absolute_core_locations':'not fixed to old bottom cores or manual endpoints',
        'stage':'VTH core location search; VTH depth parameters fixed, EE/EtoI/ZM learning off',
        'objective_change':'promote the already prespecified weak positional prior to primary for this round',
        'unconstrained_comparison':'retain no-prior ranking, best no-prior shortlist and paired confirmation nominee per domain',
        'primary_refinement':'two best J_round1 geometries per domain; all four XY coordinates remain free',
        'simulation_reuse':'identical source hashes, manifests, seeds, duration and contact readout; only selection objective changes',
        'maximum_workers':24,'memory_reserve_gib':previous.RESERVE_GIB,
        'patient_heldout_opened':False,'ictal_opened':False,
        'final_substrate_frozen':False,'old_s39_status':'HISTORICAL_CONTROL_ONLY'}
    base.write(CONTRACT,contract)
    return direction,contract


def retire_previous_dispatcher():
    path=OUT/'round1_relative_position_handoff.json'; handoff=base.read(path)
    if handoff['status']=='DRAIN_COMPLETE_OLD_CONTROLLER_RETIRED': return
    pid=handoff['old_controller_pid']; parent=Path(f'/proc/{pid}')
    if parent.exists():
        if b'run_topic4_xy_direction_research.py' not in (parent/'cmdline').read_bytes():
            raise RuntimeError('old controller PID no longer belongs to this search')
        state=next(l for l in (parent/'status').read_text().splitlines() if l.startswith('State:'))
        if not state.split()[1]=='T': raise RuntimeError('old dispatcher is not paused')
    while True:
        live=[p for p in handoff['child_pids'] if previous.proc_memory(p) is not None]
        if not live:break
        base.write(OUT/'status.json',{'status':'ROUND1_REUSING_ACTIVE_TRAJECTORIES_DURING_HANDOFF',
            'round':1,'phase':'global','primary_score':'J_round1','running':len(live),
            'complete':len(list((OUT/'global/workers').glob('*.json'))),'total':264,
            'active_pids':live,'updated_unix':time.time(),'final_substrate_frozen':False})
        time.sleep(5)
    current=int(subprocess.check_output(['systemctl','--user','show',handoff['old_service'],
                                        '-p','MainPID','--value'],text=True))
    if current not in (0,pid): raise RuntimeError('old service was replaced')
    if current:
        subprocess.run(['systemctl','--user','kill','--signal=SIGKILL',handoff['old_service']],check=True)
        subprocess.run(['systemctl','--user','stop',handoff['old_service']],check=True)
    handoff.update(status='DRAIN_COMPLETE_OLD_CONTROLLER_RETIRED',retired_unix=time.time())
    base.write(path,handoff)


def aggregate(phase,rows,direction):
    report=previous.aggregate(phase,rows,direction)
    base.write(OUT/phase/'aggregate_direction_v2.json',report)
    report=relative_position_ranking(report,weight=direction['weak_prior_weight'])
    report['round1_contract_sha256']=base.sha(CONTRACT)
    base.write(OUT/phase/'aggregate.json',report)
    return report


def shortlist(reports,rows):
    # Preserve distribution/component optima and the no-prior comparator,
    # while explicitly including the best two positional-prior candidates.
    chosen=[r['candidate_id'] for r in previous.shortlist(reports,rows)]
    scored={r['candidate_id']:r for report in reports for r in report['candidates'] if r['selection_eligible']}
    for domain in ('whole_sheet','interior'):
        selected=sorted([r for r in scored.values() if r['domain']==domain],
                        key=lambda r:(r['J_round1'],r['candidate_id']))[:2]
        chosen.extend(r['candidate_id'] for r in selected)
    byid={r['candidate_id']:r for r in rows}
    return [byid[c] for c in dict.fromkeys(chosen)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--maximum-workers',type=int,default=24);args=parser.parse_args()
    if not 1<=args.maximum_workers<=24:raise ValueError('worker maximum must be 1..24')
    direction,contract=prepare_round()
    worker_lock=base.read(OUT/'source_lock.json')['source_hashes']
    v2_lock=base.read(OUT/'runtime_lock_direction_v2.json')['hashes']
    base.verify_sources(worker_lock);previous.verify_amendment(v2_lock)
    if args.prepare_only:
        print({'status':contract['status'],'contract':str(CONTRACT)});return
    dispatcher_lock=open(OUT/'round1_controller.lock','a');fcntl.flock(dispatcher_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active_path=OUT/'active_search_contract.json'
    if not (OUT/'active_search_contract_direction_v2.json').exists():
        base.write(OUT/'active_search_contract_direction_v2.json',base.read(active_path))
    active=base.read(active_path)
    active.update(round=1,round_contract=str(CONTRACT),round_contract_sha256=base.sha(CONTRACT),
        primary_score='J_round1',primary_core_prior_weight=.1,
        controller_sha256=base.sha(Path(__file__)),maximum_workers=args.maximum_workers)
    base.write(active_path,active)
    retire_previous_dispatcher()
    controller_lock=open(OUT/'controller.lock','a');fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Attach the active round to scheduler status without changing the frozen
    # worker/scheduler files. This adapter is part of the round source lock.
    original_write=base.write
    def write_with_round(path,payload):
        if Path(path)==OUT/'status.json':
            payload={**payload,'round':1,'primary_score':'J_round1','primary_core_prior_weight':.1}
        original_write(path,payload)
    base.write=write_with_round
    def run(phase,rows):
        prepare_round()
        previous.run_phase(phase,rows,worker_lock,v2_lock,args.maximum_workers)
        prepare_round()
        return aggregate(phase,rows,direction)
    rows=base.prepare_design()['candidates'];report=run('global',rows)
    refined=base.refinement(report,rows)
    base.write(OUT/'refinement_design.json',{'candidates':refined,'round':1,
        'primary_score':'J_round1','parents_selected_from_global_only':True,
        'all_four_XY_dimensions_free':True})
    reports=[report];all_rows=rows+refined
    if refined:reports.append(run('refinement',refined))
    selected_rows=shortlist(reports,all_rows)
    base.write(OUT/'development_shortlist.json',{'candidates':selected_rows,'round':1,'primary_score':'J_round1'})
    selected=run('selection',selected_rows)
    names=primary_and_unconstrained_nominees(selected)
    if not names:
        base.write(OUT/'status.json',{'status':'ROUND1_COMPLETE_NO_QUALIFIED_GEOMETRY','round':1,'final_substrate_frozen':False});return
    names+=['control_old_edge','control_historical_matched']
    byid={r['candidate_id']:r for r in all_rows};nominees=[byid[n] for n in dict.fromkeys(names)]
    base.write(OUT/'confirmation_nominees.json',{'candidates':nominees,'round':1,
        'selection_complete_before_confirmation':True,'primary_score':'J_round1',
        'nominee_rule':'per-domain constrained and unconstrained optimum plus historical controls',
        'final_substrate_frozen':False})
    final=run('confirmation',nominees)
    base.write(OUT/'final_search_report.json',{'status':'ROUND1_COMPLETE_AWAITING_SCIENTIFIC_REVIEW',
        'round':1,'round_contract_sha256':base.sha(CONTRACT),'confirmation':final,
        'final_substrate_frozen':False,'global_optimum_established':False,
        'next_step':'Review constrained versus unconstrained measured behavior before any VTH freeze or EE/EtoI/ZM learning.'})
    subprocess.run([base.PYTHON,str(ROOT/'scripts/paper_figures/plot_topic4_xy_relative_position_round1.py')],cwd=ROOT,env=base.ENV,check=True)
    base.write(OUT/'status.json',{'status':'ROUND1_COMPLETE_AWAITING_SCIENTIFIC_REVIEW','round':1,'final_substrate_frozen':False})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        base.write(OUT/'status.json',{'status':'FAILED','round':1,'reason':str(exc),'updated_unix':time.time()})
        raise
