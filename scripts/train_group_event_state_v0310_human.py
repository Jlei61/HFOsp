#!/usr/bin/env python3
"""v0.3.10 human training cell: one subject x family x recipe x seed x horizon.

Clause C0. Every knob the frozen spec names is a real argument that reaches
every layer; the parameter inventory on the card is exported from the
instantiated model, not from the arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v0310.trainer import CellConfig, STAGE_BUDGETS, run_cell


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--family', choices=('F', 'L', 'N'), required=True)
    p.add_argument('--recipe-id', default='R0')
    p.add_argument('--state-width', type=int, default=16)
    p.add_argument('--transition-rank', type=int, default=None,
                   help='defaults to the registered min(width/2,16)')
    p.add_argument('--readout-hidden', type=int, default=32)
    p.add_argument('--physical-batch', type=int, default=128)
    p.add_argument('--effective-batch', type=int, default=128)
    p.add_argument('--history-hours', type=float, default=8.0)
    p.add_argument('--source-mode', choices=('event_only', 'background_conditional'), default='event_only')
    p.add_argument('--view', choices=('joint', 'count', 'recruitment'), default='joint')
    p.add_argument('--seed', type=int, default=20260905)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--event-budget', type=int, default=STAGE_BUDGETS['event'])
    p.add_argument('--foundation-budget', type=int, default=STAGE_BUDGETS['background'])
    p.add_argument('--constant-budget', type=int, default=STAGE_BUDGETS['refitted_constant'])
    p.add_argument('--activation-checkpoint-chunk', type=int, default=32)
    p.add_argument('--min-history-coverage', type=float, default=0.0)
    p.add_argument('--deadline-epoch', type=float, default=0.0)
    p.add_argument('--no-resume', action='store_true')
    args = p.parse_args()
    if args.effective_batch % args.physical_batch:
        raise SystemExit('Effective batch must be a whole number of physical batches')
    rank = args.transition_rank if args.transition_rank is not None else min(args.state_width // 2, 16)
    if args.family in ('L', 'N') and rank != min(args.state_width // 2, 16):
        raise SystemExit('transition_rank must follow the registered min(width/2,16) rule')
    cfg = CellConfig(data=str(args.data), output_dir=str(args.output_dir), family=args.family,
                     state_width=args.state_width, transition_rank=rank, readout_hidden=args.readout_hidden,
                     physical_batch=args.physical_batch, effective_batch=args.effective_batch,
                     history_hours=args.history_hours, source_mode=args.source_mode, view=args.view,
                     seed=args.seed, device=args.device, lr=args.lr,
                     stage_budget=dict(background=args.foundation_budget, event=args.event_budget,
                                       refitted_constant=args.constant_budget),
                     activation_checkpoint_chunk=args.activation_checkpoint_chunk,
                     min_history_coverage=args.min_history_coverage,
                     deadline_epoch=args.deadline_epoch, resume=not args.no_resume,
                     recipe_id=args.recipe_id)
    card = run_cell(cfg)
    print(json.dumps({k: card.get(k) for k in ('status', 'subject', 'family', 'recipe', 'seed',
                                               'history_hours', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
