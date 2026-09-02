#!/usr/bin/env python3
"""Freeze true, wrong-position, and degree-matched ECoG graph contracts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_ecog_physical_neighborhood_v0_1 import (  # noqa: E402
    coordinate_array,
    degree_class_permutation,
    degree_preserving_random_mask,
    graph_audit,
    true_grid_mask,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=("958", "1084"))
    parser.add_argument("--n-graphs", type=int, default=31)
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--feasibility-root", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/feasibility"))
    parser.add_argument("--output-root", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/graphs"))
    args = parser.parse_args()

    rows = list(csv.DictReader((args.feasibility_root / "GRID_CHANNELS.csv").open()))
    names = [row["contact"] for row in rows if row["subject"] == args.subject and row["is_bad_config"] == "False"]
    names.sort(key=lambda name: (name[1], int(name[2:])))
    true_mask = true_grid_mask(names, diagonal=args.diagonal)
    output = args.output_root / args.subject / ("eight_neighbour" if args.diagonal else "four_neighbour")
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    def save(graph_id: str, family: str, mask: np.ndarray, seed: int) -> None:
        audit = graph_audit(mask, true_mask)
        path = output / f"{graph_id}.npz"
        temporary = output / f"{graph_id}.tmp.npz"
        np.savez_compressed(
            temporary,
            schema_version=np.asarray("topic5_ecog_graph_v0.1"),
            subject=np.asarray(args.subject),
            graph_id=np.asarray(graph_id),
            family=np.asarray(family),
            seed=np.asarray(seed),
            channel_names=np.asarray(names),
            coordinates=coordinate_array(names).astype(np.float32),
            mask=np.asarray(mask, dtype=np.uint8),
            true_mask=true_mask.astype(np.uint8),
        )
        temporary.replace(path)
        manifest.append({
            "subject": args.subject,
            "graph_id": graph_id,
            "family": family,
            "seed": seed,
            "path": str(path),
            "sha256": sha256_file(path),
            "n_nodes": audit["n_nodes"],
            "n_directed_edges": audit["n_directed_edges"],
            "minimum_degree": audit["minimum_degree"],
            "maximum_degree": audit["maximum_degree"],
            "connected": audit["connected"],
            "true_edge_overlap_fraction": audit["true_edge_overlap_fraction"],
            "true_edge_jaccard": audit["true_edge_jaccard"],
        })
        (output / f"{graph_id}.audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    save("TRUE_GRID", "TRUE_GRID", true_mask, 0)
    seen_wrong: set[bytes] = set()
    seen_random: set[bytes] = set()
    attempt = 0
    while len(seen_wrong) < args.n_graphs:
        seed = stable_seed(f"wrong-grid|{args.subject}|{attempt}|{int(args.diagonal)}")
        graph = degree_class_permutation(true_mask, seed)
        key = graph.tobytes()
        audit = graph_audit(graph, true_mask)
        attempt += 1
        if key in seen_wrong or float(audit["true_edge_overlap_fraction"]) >= 0.55:
            continue
        seen_wrong.add(key)
        save(f"WRONG_GRID_{len(seen_wrong) - 1:02d}", "WRONG_GRID", graph, seed)

    attempt = 0
    while len(seen_random) < args.n_graphs:
        seed = stable_seed(f"degree-random|{args.subject}|{attempt}|{int(args.diagonal)}")
        graph = degree_preserving_random_mask(true_mask, seed)
        key = graph.tobytes()
        audit = graph_audit(graph, true_mask)
        attempt += 1
        if key in seen_random or float(audit["true_edge_overlap_fraction"]) >= 0.55:
            continue
        seen_random.add(key)
        save(f"DEGREE_RANDOM_{len(seen_random) - 1:02d}", "DEGREE_RANDOM", graph, seed)

    fields = list(manifest[0])
    with (output / "GRAPH_MANIFEST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    payload = {
        "schema": "topic5_ecog_graph_manifest_v0.1",
        "subject": args.subject,
        "diagonal": bool(args.diagonal),
        "n_wrong": len(seen_wrong),
        "n_degree_random": len(seen_random),
        "channel_names": names,
        "manifest_sha256": sha256_file(output / "GRAPH_MANIFEST.csv"),
    }
    (output / "GRAPH_MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
