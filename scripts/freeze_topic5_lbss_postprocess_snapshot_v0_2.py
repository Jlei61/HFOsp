#!/usr/bin/env python3
"""Freeze a self-contained, read-only LBSS postprocess source snapshot."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ENTRY_FILES = (
    "scripts/analyse_topic5_lbss_interictal_v0_2.py",
    "scripts/build_topic5_lbss_fields_v0_2.py",
    "scripts/analyse_topic5_lbss_pathways_v0_2.py",
    "scripts/run_topic5_lbss_attenuation_v0_2.py",
    "scripts/prepare_topic5_lbss_early_ictal_unseal_v0_2.py",
    "scripts/score_topic5_lbss_early_ictal_v0_2.py",
    "scripts/summarize_topic5_lbss_claims_v0_2.py",
    "scripts/plot_topic5_lbss_figure6_v0_2.py",
    "scripts/run_topic5_lbss_postprocess_pipeline_v0_2.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_import(module: str, importer: Path) -> Path | None:
    dotted = module.replace(".", "/") + ".py"
    candidates = [ROOT / dotted]
    if importer.parts[0] == "scripts":
        candidates.append(ROOT / "scripts" / (module.replace(".", "/") + ".py"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.relative_to(ROOT)
    return None


def dependency_closure(entries: tuple[str, ...]) -> list[Path]:
    pending = [Path(value) for value in entries]
    found: set[Path] = {Path("src/__init__.py")}
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(relative)
        tree = ast.parse(path.read_text(), filename=str(path))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        for module in modules:
            dependency = local_import(module, relative)
            if dependency is not None and dependency not in found:
                pending.append(dependency)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--snapshot-name", default="postprocess_snapshot_v2")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="freeze the exact reviewed worktree and record its diff hash")
    args = parser.parse_args()
    out = args.out_root.resolve()
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists() or (out / "TARGET_ACCESS_AUDIT.json").exists():
        raise RuntimeError("postprocess source cannot be changed after target unseal")
    snapshot = out / args.snapshot_name
    if snapshot.exists():
        raise FileExistsError(snapshot)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if status and not args.allow_dirty:
        raise RuntimeError(f"postprocess snapshot requires a clean worktree:\n{status}")
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    files = dependency_closure(ENTRY_FILES)
    records = {}
    for relative in files:
        source = ROOT / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records[str(relative)] = sha256(destination)
    payload = {
        "contract": "topic5_lbss_self_contained_postprocess_snapshot_v0_2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_worktree_explicitly_allowed": bool(args.allow_dirty and status),
        "git_status_porcelain": status.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "files": records,
        "entry_files": list(ENTRY_FILES),
        "self_contained_local_import_closure": True,
        "target_values_read": False,
        "immutable": True,
    }
    manifest = snapshot / "SNAPSHOT_MANIFEST.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    for path in snapshot.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((value for value in snapshot.rglob("*") if value.is_dir()), reverse=True):
        path.chmod(0o555)
    snapshot.chmod(0o555)
    print(json.dumps({"snapshot": str(snapshot), "n_files": len(files), "manifest_sha256": sha256(manifest)}))


if __name__ == "__main__":
    main()
