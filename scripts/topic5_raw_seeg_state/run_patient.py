#!/usr/bin/env python
"""Train one patient / one arm of the Raw-SEEG evolvable prediction-state model.

Writes the stage-C per-subject artifact set of the execution plan into
``results/epi_prssm/raw_seeg_state/r0_1/per_subject/<subject>/``.

Example
-------
    LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:$LD_LIBRARY_PATH \
    /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
      scripts/topic5_raw_seeg_state/run_patient.py \
      --subject epilepsiae_1073 --arm full --seed 0 --resume

Exit codes: 0 ok, 17 OOM downgrade chain exhausted (the queue runner re-queues
once), 1 anything else.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_raw_seeg_state import analysis, contract, train as T  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True)
    p.add_argument("--arm", default="full", choices=sorted(T.ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", default=None, help="per-subject output dir")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--job-id", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--lambda-cons", type=float, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--skip-analysis", action="store_true",
                   help="train only; do not run the state-swap / consistency passes")
    return p


def git_dirty_file_count() -> int:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      cwd=str(contract.REPO_ROOT), text=True)
        return len([ln for ln in out.splitlines() if ln.strip()])
    except Exception:
        return -1


def decoder_weight(model) -> np.ndarray:
    for attr in ("decode", "decoder", "readout"):
        obj = getattr(model, attr, None)
        w = getattr(obj, "weight", None)
        if w is not None:
            return w.detach().float().cpu().numpy()
    raise AttributeError("model exposes no linear decoder weight (decode/decoder/readout)")


#: Peak activation memory scales with (batch x contacts): measured on the 3090 at
#: 0.0385 GB per unit (batch 4 x 87 contacts -> 13.4 GB, batch 4 x 139 -> 21.4 GB).
#: Capping the product at 440 keeps every subject under ~17 GB on a 24 GB card,
#: which leaves room for the other work that shares this GPU. Contact counts in
#: this cohort run 24 to 183, so a single fixed batch size either wastes the card
#: on the small subjects or OOMs on the large ones.
MAX_BATCH_CONTACT_PRODUCT = 440
#: Measured on the 3090 at batch 8 x 31 contacts (product 248): the plain
#: Transformer peaks at 9.8 GB, the Conformer at 17.9 GB, the capacity-matched
#: wide Transformer at 14.3 GB. The Conformer keeps a residual stream through an
#: extra convolution module in every block. The original conformer cap 220 still
#: selected batch 2 at 87 contacts and produced an asynchronous CUDA OOM during
#: backward on 2026-08-24. Cap 160 forces batch 1 for both 82- and 87-contact
#: paths while retaining batch 5 for the 31-contact low end.
MAX_BATCH_CONTACT_PRODUCT_BY_ENCODER = {
    "transformer": 440,
    "transformer_wide": 300,
    "conformer": 160,
}


def batch_contact_cap(encoder_kind: str = "transformer", d_model: int = 128) -> int:
    if str(encoder_kind) == "conformer":
        return MAX_BATCH_CONTACT_PRODUCT_BY_ENCODER["conformer"]
    if int(d_model) > 128:
        return MAX_BATCH_CONTACT_PRODUCT_BY_ENCODER["transformer_wide"]
    return MAX_BATCH_CONTACT_PRODUCT_BY_ENCODER["transformer"]


def auto_batch_size(n_contacts: int, lo: int = 1, hi: int = 8,
                    encoder_kind: str = "transformer", d_model: int = 128) -> int:
    if n_contacts <= 0:
        return hi
    cap = batch_contact_cap(encoder_kind, d_model)
    return int(max(lo, min(hi, cap // int(n_contacts))))


def subject_contact_count(subject: str) -> int:
    import pandas as pd
    con = pd.read_parquet(contract.DATA_DIR / "contact_metadata.parquet")
    return int((con.subject == subject).sum())


def main(argv=None) -> int:
    import torch

    args = build_parser().parse_args(argv)
    batch_size = args.batch_size
    auto_note = None
    if batch_size is None:
        arm_spec = T.ARMS.get(args.arm, {})
        enc = str(arm_spec.get("encoder_kind", "transformer"))
        dm = int(arm_spec.get("d_model", 128))
        n_c = subject_contact_count(args.subject)
        batch_size = auto_batch_size(n_c, encoder_kind=enc, d_model=dm)
        auto_note = (f"auto batch {batch_size} from {n_c} contacts "
                     f"(encoder={enc} d_model={dm} cap {batch_contact_cap(enc, dm)})")
        print(f"[run_patient] {auto_note}", flush=True)
    cfg = T.resolve_arm(
        args.subject, args.arm, seed=args.seed,
        batch_size=batch_size, grad_accum=args.grad_accum, lr=args.lr,
        max_epochs=args.max_epochs, max_steps=args.max_steps, patience=args.patience,
        ckpt_every=args.ckpt_every, num_workers=args.num_workers, device=args.device,
        lambda_cons=args.lambda_cons, out_dir=args.out, log_dir=args.log_dir,
        job_id=args.job_id or f"{args.subject}__{args.arm}__s{args.seed}",
    )
    if args.no_amp:
        cfg = T.replace(cfg, amp=False)
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    contract.atomic_write_json(out_dir / "config.json", cfg.to_json())

    t0 = time.time()
    result = T.train_subject(cfg, resume=args.resume)

    if result["status"] == "ok" and not args.skip_analysis and result.get("latent_cache"):
        _write_analyses(cfg, out_dir, result.pop("model"), result["latent_cache"],
                        T.default_loss_bundle())

    manifest = {
        "revision": contract.REVISION,
        "contract_version": contract.CONTRACT_VERSION,
        "code_revision": contract.code_revision(),
        "package_hash": contract.package_hash(contract.r0_1_source_files()),
        "git_dirty_file_count": git_dirty_file_count(),
        "config": cfg.to_json(),
        "status": result["status"],
        "reason": result["reason"],
        "selected_epoch": result["selected_epoch"],
        "best_val_forecast_loss": result["best_val_forecast_loss"],
        "global_step": result["global_step"],
        "epochs_run": result["epochs_run"],
        "batch_size_final": result["batch_size"],
        "batch_size_rule": auto_note or "explicit --batch-size",
        "grad_accum_final": result["grad_accum"],
        "use_checkpoint_final": result["use_checkpoint"],
        "oom_events": result["oom_events"],
        "oom_halvings": result["oom_halvings"],
        "oom_rung": result["oom_rung"],
        "oom_ladder": result["oom_ladder"],
        "nonfinite_steps": result["nonfinite_steps"],
        "latent_collapse": result.get("latent_collapse"),
        "n_val_windows_scored": result.get("n_val_windows_scored"),
        "n_train_windows": result["n_train_windows"],
        "n_val_windows": result["n_val_windows"],
        "amp_mode": result["amp_mode"],
        "determinism": result["determinism"],
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "peak_memory_bytes": result["peak_memory_bytes"],
        "wall_time_sec": time.time() - t0,
        "dev_end_epoch": contract.dev_end_epoch(cfg.subject),
        "sealed_partition_touched": False,
        "scientific_boundary": (
            "R0.1 predicts a contact x log-frequency power field only; it says "
            "nothing about seizure risk, IED generation, or >100 Hz activity."
        ),
    }
    contract.atomic_write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in
                      ("status", "reason", "selected_epoch", "best_val_forecast_loss",
                       "global_step", "oom_halvings", "nonfinite_steps")}, indent=2))
    if result["status"] == "ok":
        return 0
    return T.EXIT_OOM_BUDGET if "oom_budget" in str(result["reason"]) else 1


def _write_analyses(cfg, out_dir: Path, model, cache, bundle) -> None:
    """Spec 8.2 analyses 3/4/5, all decoded from the single cached encoder pass."""
    import pandas as pd

    analysis.save_latent_cache(cache, out_dir / "latent_trajectory.zarr")

    cons = analysis.state_consistency(
        model, cache, loss_bundle=bundle,
        out_path=out_dir / "state_consistency.parquet")
    contract.atomic_write_json(out_dir / "state_consistency_summary.json", cons["summary"])

    swap = analysis.matched_state_swap(model, cache, horizons=cfg.horizons)
    frame = pd.DataFrame(swap.pop("rows"))
    tmp = out_dir / "state_swap_results.parquet.tmp"
    frame.to_parquet(tmp, index=False)
    tmp.replace(out_dir / "state_swap_results.parquet")
    contract.atomic_write_json(out_dir / "state_swap_summary.json", swap)

    readout = analysis.mode_readout(model.dynamics, decoder_weight(model),
                                    int(cache["n_contacts"]), int(cache["n_freq"]))
    loading = readout.pop("loading")
    np.save(out_dir / "decoder_loading.npy", loading.astype(np.float32))
    contract.atomic_write_json(out_dir / "dynamics_modes.json", readout)


if __name__ == "__main__":
    raise SystemExit(main())
