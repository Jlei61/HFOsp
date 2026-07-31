#!/usr/bin/env python3
"""Phase-C final adjudication with explicit v2 analysis-amendment lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import scripts.adjudicate_topic4_zm_phasec as V1  # noqa: E402
import scripts.analyze_topic4_zm_phasec1_v2 as A2  # noqa: E402


def _relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(CODE_ROOT))
    except ValueError:
        return str(path)


def _analysis_fields():
    amendment = A2._read_amendment()
    return {
        "analysis_amendment_path": _relative(A2.AMENDMENT),
        "analysis_amendment_file_sha256": V1._sha(A2.AMENDMENT),
        "analysis_amendment_sha256": amendment["amendment_sha256"],
        "analysis_producer_file_sha256": A2._analysis_producers(),
        "adjudication_v2_wrapper_file_sha256": {
            str(Path(__file__).resolve().relative_to(CODE_ROOT)):
            V1._sha(Path(__file__).resolve())
        },
    }


def _require_binding(payload, fields, *, label):
    for key in (
        "analysis_amendment_path",
        "analysis_amendment_file_sha256",
        "analysis_amendment_sha256",
        "analysis_producer_file_sha256",
    ):
        if payload.get(key) != fields[key]:
            raise ValueError(f"{label} lacks v2 analysis binding: {key}")


def build_final_inputs(**kwargs):
    fields = _analysis_fields()
    c1_native = V1._read(Path(kwargs["c1_native_path"]))
    c1_gate = V1._read(Path(kwargs["c1_gate_path"]))
    _require_binding(c1_native, fields, label="C1 native summary")
    _require_binding(c1_gate, fields, label="C1 resolution gate")
    inputs = V1.build_final_inputs(**kwargs)
    return {
        name: {**payload, **fields}
        for name, payload in inputs.items()
    }


def adjudicate_files(**kwargs):
    fields = _analysis_fields()
    for name in ("c1_primary_path", "c1_shell_path", "coverage_path"):
        path = kwargs.get(name)
        if path is None or not Path(path).is_file():
            raise ValueError(f"missing v2 final input: {name}")
        _require_binding(
            V1._read(Path(path)),
            fields,
            label=name,
        )
    payload = V1.adjudicate_files(**kwargs)
    return {**payload, **fields}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-inputs", action="store_true")
    parser.add_argument("--c0-gate", type=Path)
    parser.add_argument("--c0-native", type=Path)
    parser.add_argument("--c1-gate", type=Path)
    parser.add_argument("--c1-native", type=Path)
    parser.add_argument("--modal-summary", type=Path)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--c0", type=Path)
    parser.add_argument("--c1-primary", type=Path)
    parser.add_argument("--c1-shell", type=Path)
    parser.add_argument("--modal", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--trigger", type=Path)
    parser.add_argument("--phasec-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.build_inputs:
        required = {
            "--c0-gate": args.c0_gate,
            "--c0-native": args.c0_native,
            "--c1-gate": args.c1_gate,
            "--c1-native": args.c1_native,
            "--modal-summary": args.modal_summary,
            "--inputs-dir": args.inputs_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("--build-inputs requires " + ", ".join(missing))
        inputs = build_final_inputs(
            phasec_manifest_path=args.phasec_manifest,
            c0_gate_path=args.c0_gate,
            c0_native_path=args.c0_native,
            c1_gate_path=args.c1_gate,
            c1_native_path=args.c1_native,
            modal_path=args.modal_summary,
        )
        paths = V1.write_final_inputs(args.inputs_dir, inputs)
        print(json.dumps({
            "status": "final_inputs_locked",
            "analysis_amendment_sha256": _analysis_fields()[
                "analysis_amendment_sha256"
            ],
            "paths": {name: str(path) for name, path in paths.items()},
        }, sort_keys=True))
        return
    if args.output is None:
        parser.error("adjudication mode requires --output")
    payload = adjudicate_files(
        c0_path=args.c0,
        c1_primary_path=args.c1_primary,
        c1_shell_path=args.c1_shell,
        modal_path=args.modal,
        coverage_path=args.coverage,
        trigger_path=args.trigger,
        phasec_manifest_path=args.phasec_manifest,
    )
    V1._atomic_write(args.output, payload)
    print(json.dumps({
        "verdict": payload["verdict"],
        "next_route": payload["next_route"],
        "output": str(args.output),
        "analysis_amendment_sha256": payload[
            "analysis_amendment_sha256"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
