"""Topic 4 MZ slow–fast dynamical transition — scientific runner.

*** THIS RUNS SIMULATIONS. *** Nothing runs on import; sim subcommands gated by --confirm-run.
Design contract (BINDING): docs/superpowers/specs/2026-07-20-topic4-mz-slow-fast-transition-design.md

Freeze the natural MZ slow state {z_i, m_i} at registered checkpoints and evolve ONLY the fast spiking system.
Reuse (not reinvent): src.topic4_mz_onset_dynamics (MZOnsetProbe, run_loop checkpoint/resume, score_runaway,
epsilon_c_from_ladder), run_m4_phaseplane.build_substrate, src.topic4_mz_slow_fast_transition (pure helpers).
NO engine edits (6 guarded files read-only).

Subcommands (all resumable via per-(cond,seed,state) JSON):
  pilot      1 cond x 1 seed x 1 checkpoint x few replays -> peak RSS + wall/step (resource probe)
  run        per-(cond,seed) job: natural trajectory + checkpoints + P_runaway / epsilon_c / tau_rec +
             counterfactuals + matched-D; --all fans out over the 12 units (memory-gated Pool)
  aggregate  (no sim) combine per-(cond,seed,state) JSON -> summary CSV/JSON + classification + STATUS
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse   # noqa: E402
import sys        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slow_fast_transition")
CFG_PATH = os.path.join(ROOT, "config", "topic4_mz_slow_fast_transition.yaml")
DT = 0.1


def load_cfg():
    import yaml
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


# stubs raise (CLAUDE.md §6: a stub must fail loudly, never return plausible values).
def cmd_pilot(args, cfg):
    raise NotImplementedError("pilot: implemented in Task 6")


def cmd_run(args, cfg):
    raise NotImplementedError("run: implemented in Tasks 5/7")


def cmd_aggregate(args, cfg):
    raise NotImplementedError("aggregate: implemented in Task 7")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Topic 4 MZ slow–fast dynamical transition runner (design 2026-07-20).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pilot", "run", "aggregate"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true")
        sp.add_argument("--seeds", default=None, help="comma-separated seed subset override")
        sp.add_argument("--conditions", default=None, help="comma-separated condition subset override")
        sp.add_argument("--only", default=None, help="run: single unit 'cond:seed'")
        sp.add_argument("--all", action="store_true", help="run: fan out over all (cond,seed) units")
        sp.add_argument("--workers", type=int, default=None, help="run --all: worker count (default memory-gated)")
        sp.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    if args.cmd in ("pilot", "run") and not args.confirm_run:
        print(f"REFUSING: '{args.cmd}' runs simulations. Pass --confirm-run.", file=sys.stderr)
        sys.exit(2)
    os.makedirs(OUT, exist_ok=True)
    {"pilot": cmd_pilot, "run": cmd_run, "aggregate": cmd_aggregate}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
