#!/usr/bin/env python3
"""Verify or reproduce the author-accepted Figure 1 release.

The exact layout is rebuilt from frozen native PDF/PNG layers, with no raw-data
reanalyis. Panel-level scientific inputs and original producers are backed up
in the package for subsequent explicitly requested scientific revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(package):
    manifest = json.loads((package / "release_manifest.json").read_text())
    assert manifest["status"] == "AUTHOR_ACCEPTED_FINAL"
    for name, expected in manifest["files"].items():
        p = package / name
        assert p.is_file(), f"Missing release asset: {name}"
        assert digest(p) == expected["sha256"], f"Changed release asset: {name}"
    meta = json.loads((package / "metadata.json").read_text())
    with np.load(package / "data/ce/display_ranks_18.npz") as z:
        ranks,mask,labels=z["ranks"],z["participating"],z["labels"]
        assert ranks.shape == (18,18190)
        assert np.array_equal(np.isfinite(ranks),mask)
        assert np.bincount(labels).tolist() == [13160,5030]
        counts=mask.sum(axis=0)
        ordered=np.sort(np.where(mask,ranks,np.inf),axis=0)
        for i in range(18):
            assert np.all(ordered[i,counts>i] == i+1)
    with np.load(package / "data/ce/original_arrays_26.npz") as z:
        for key,expected in meta["original_array_hashes"].items():
            assert hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest()==expected,key
    records=json.loads((package / "data/df/cohort_plot_records.json").read_text())
    assert len(records)==40 and all(r["legacy_mi"]["masked"] for r in records)
    for dataset in ("yuquan","epilepsiae"):
        rows=[r for r in records if r["dataset"]==dataset]
        assert len(rows)==20
        expected=meta["summaries"]["D"]["statistics"][dataset]
        np.testing.assert_allclose(np.median([r["legacy_mi"]["mi_mean"] for r in rows]),expected["data_median"],rtol=0,atol=1e-14)
    contract=json.loads((package / "spectrum_contract.json").read_text())
    assert [(r["record"],r["event_index"]) for r in contract["display_events"]]==[
        ("FA134AX6",1559),("FA134AX6",1562),("FA134AXF",1494)]
    return dict(status="PASS",files_verified=len(manifest["files"]),
        original_26_channel_arrays_preserved=True,display_channels=18,events=18190,
        cluster_counts=[13160,5030],cohort_n=40,release_version=manifest["version"])


def rebuild(package, output):
    assert output.resolve()!=package.resolve(), "Use a separate output directory."
    if output.exists() and any(output.iterdir()):
        raise ValueError("The rebuild destination must be empty.")
    code=package / "source/code_snapshot/scripts/paper_figures"
    # Verification/rebuild must leave the accepted source snapshot untouched.
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location("frozen_fig1_row_layout",code / "revise_fig1_lower_row_gap.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.BASE=package / "source/layout_input"
    module.OUT=output
    module.main()
    assert digest(output / "figures/fig1-complete-layout.png")==digest(package / "figures/fig1-complete-layout.png")
    for name in "abcdef":
        for ext in ("png","pdf"):
            fn=f"fig1-panel{name}.{ext}"
            assert digest(output / "figures" / fn)==digest(package / "figures" / fn)
    (output / "rebuild_validation.json").write_text(json.dumps(dict(status="PASS",
        complete_png_byte_identical=True,standalone_panels_byte_identical=True,
        method="Frozen scientific PDF/PNG layers and rigid final row translation; no raw analysis rerun."),indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,default=ROOT / "results/paper-ready-figure/fig1")
    parser.add_argument("--output-dir",type=Path,help="Empty destination for exact layout reconstruction; omit for verification only.")
    args=parser.parse_args()
    report=verify(args.package)
    if args.output_dir:
        rebuild(args.package,args.output_dir)
        report["rebuild"]=str(args.output_dir)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
