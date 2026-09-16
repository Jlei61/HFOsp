#!/usr/bin/env python3
"""Recompute the nine-run budget extension without opening new human outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.topic5_group_event_state.v039.instrument_audit import convergence_audit, sha256
from src.topic5_group_event_state.v039.synthetic import generate


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    result = convergence_audit(root)
    if result is None:
        raise ValueError('Registered audit absent')
    result['current_source_matches'] = {p: sha256(ROOT/p) == h for p, h in result['source_hashes'].items()}
    if not all(result['current_source_matches'].values()):
        raise ValueError('Regeneration requires the recorded training/generator sources')
    original_path = root/'final_reports/summary_main.json'
    summary = json.loads(original_path.read_text())
    result['original_summary_sha256'] = sha256(original_path)
    old = {}
    for row in summary['instruments']:
        if row['case'] != 'nonlinear_transition' or row['experiment'] != 'joint' or row['family'] not in ('F', 'L', 'N'):
            continue
        family = row['family']
        card = json.loads(Path(row['source']).read_text())
        new_path = root/f'instruments_convergence_audit/nonlinear_{family}_seed{card["seed"]}'
        new = json.loads(new_path.read_text())
        matches = card['training_curve'] == [x for x in new['training_curve'] if x['step'] <= card['steps_run']]
        old[family] = dict(source=row['source'], source_sha256=sha256(row['source']),
                           held_out=card['held_out'], selected_step=card['selected_step'],
                           stop_reason=card['stop_reason'], same_curve_through_old_budget=matches,
                           same_source=card['source_hashes'] == new['source_hashes'],
                           same_input=card['input_sha256'] == new['input_sha256'],
                           config_differences={k:[card['config'].get(k), new['config'].get(k)]
                                               for k in card['config'] if card['config'].get(k) != new['config'].get(k)})
        if not all(old[family][k] for k in ('same_curve_through_old_budget', 'same_source', 'same_input')):
            raise ValueError('Old/new runs do not share the original training trajectory')
    result['original_480_runs'] = old
    result['original_contrasts'] = {f'{b}_over_{w}': old[w]['held_out']['total']-old[b]['held_out']['total']
                                    for b,w in [('N','L'), ('N','F'), ('L','F')]}
    config = json.loads(Path(result['records'][0]['source']).read_text())['config']
    data = generate('nonlinear_transition', seed=result['data_seed'], n=config['episodes'], strength=config['strength'])
    partial = hashlib.sha256(data['inputs'].tobytes()+data['context'].tobytes()).hexdigest()
    if partial != result['input_sha256']:
        raise ValueError('Regenerated synthetic inputs differ')
    hashes = {}
    def bind(name, value):
        if isinstance(value, np.ndarray):
            header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
            hashes[name] = hashlib.sha256(header+np.ascontiguousarray(value).tobytes()).hexdigest()
        elif isinstance(value, dict):
            for k in sorted(value): bind(name+'/'+k, value[k])
        else:
            hashes[name] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    bind('synthetic', data)
    result['regenerated_full_data_hashes'] = hashes
    result['regeneration_boundary'] = 'Full dt/targets/split hashes added now from matching generator; not contemporaneous original target attestation.'
    (output/'instrument_budget_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (output/'instrument_budget_paired_seeds.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['optimization_seed','data_seed','F','L','N','N_over_L','N_over_F','L_over_F'])
        writer.writeheader()
        for seed in result['seeds']:
            scores = {r['family']:r['held_out']['total'] for r in result['records'] if str(r['seed']) == seed}
            writer.writerow(dict(optimization_seed=seed,data_seed=result['data_seed'],**scores,
                                 N_over_L=scores['L']-scores['N'],N_over_F=scores['F']-scores['N'],L_over_F=scores['F']-scores['L']))
    print(json.dumps({k:result[k] for k in ('status','all_patience_stopped','n_data_seeds','N_over_L','N_over_F','L_over_F')},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
