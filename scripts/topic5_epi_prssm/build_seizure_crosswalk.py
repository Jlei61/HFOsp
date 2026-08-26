#!/usr/bin/env python3
"""Task A: a verified one-to-one crosswalk from H2B seizures to the canonical
inventory, and from there to the frozen z-ER subtype labels.

The two sides do not share an identifier scheme, and the mismatch is silent:

  Epilepsiae  H2B ``107300100102``      inventory ``107300100102``   -- same string
  Yuquan      H2B ``FA0013KQ_0``        inventory ``gaolan_sz_001``  -- no overlap at all

A naive string join therefore keeps every Epilepsiae seizure and drops every Yuquan
one without raising anything.  Position-based joins are worse: the two tables are
not in the same order and do not have the same length.

So the Yuquan side is mapped explicitly -- the H2B identifier is parsed as
``<record code>_<index within record>``, the record code is looked up in the
inventory's own ``record`` column, and the index selects among that record's
seizures ordered by onset.  Every match on both sides is then *verified* against
the onset timestamp of the matching kind, and the residual differences are
reported rather than assumed to be zero.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from _common import OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision  # noqa: E402
from _common import package_hash  # noqa: E402

ROOT = OUTPUT_ROOT.parents[2]
EPI_INVENTORY = ROOT / "results/epilepsiae_seizure_inventory.csv"
YUQ_INVENTORY = ROOT / "results/dataset_inventory/yuquan_seizure_inventory.csv"
SUBTYPE_ROOT = ROOT / ("results/data_driven_soz/layer_a_ictal_er_rank/"
                       "seizure_clusters/per_subject")
H2B = OUTPUT_ROOT / "seizure_link_preictal"
OUT = OUTPUT_ROOT / "seizure_crosswalk"

#: an onset match beyond this is reported as a discrepancy rather than accepted
TOLERANCE_SECONDS = 5.0
YUQUAN_ID = re.compile(r"^(?P<record>[A-Z]{2}[0-9A-Z]+)_(?P<index>\d+)$")


def h2b_seizures(layer: str, lead: str) -> pd.DataFrame:
    path = H2B / f"preictal_effects__{layer}__{lead}.csv"
    frame = pd.read_csv(path)
    keep = ["subject", "dataset", "seizure_id", "onset_epoch", "onset_kind",
            "n_events_in_lookback_2h", "lookback_stratum",
            "preictal_observation_premise_met", "anchor_gap_to_cutoff_seconds",
            "nuisance_coverage"]
    return frame[[c for c in keep if c in frame]].copy()


def match_epilepsiae(rows: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.copy()
    inv["seizure_id"] = inv.seizure_id.astype(str)
    inv["subject"] = "epilepsiae_" + inv.subject.astype(str)
    out = rows.merge(inv[["subject", "seizure_id", "eeg_onset_epoch", "clin_onset_epoch",
                          "recording_id", "block_id", "classification", "pattern"]],
                     on=["subject", "seizure_id"], how="left", indicator=True)
    out["canonical_seizure_id"] = np.where(out._merge == "both", out.seizure_id, None)
    out["match_route"] = np.where(out._merge == "both", "seizure_id", "unmatched")
    return out.drop(columns=["_merge"])


def match_yuquan(rows: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Explicit record-code mapping; never a string or positional guess."""
    inv = inventory.copy()
    inv["subject"] = "yuquan_" + inv.subject.astype(str)
    inv["record"] = inv.record.astype(str)
    # within each record, order the seizures by onset so the H2B index means something
    inv = inv.sort_values(["record", "eeg_onset_epoch"]).reset_index(drop=True)
    inv["index_within_record"] = inv.groupby("record").cumcount()

    parsed = rows.seizure_id.astype(str).str.extract(YUQUAN_ID)
    rows = rows.copy()
    rows["record"] = parsed["record"]
    rows["index_within_record"] = pd.to_numeric(parsed["index"], errors="coerce")
    rows["id_parse_ok"] = parsed["record"].notna()

    out = rows.merge(
        inv[["subject", "record", "index_within_record", "seizure_id",
             "eeg_onset_epoch", "recording_id", "patient_code"]]
        .rename(columns={"seizure_id": "canonical_seizure_id"}),
        on=["subject", "record", "index_within_record"], how="left", indicator=True)
    out["clin_onset_epoch"] = np.nan
    out["match_route"] = np.where(out._merge == "both", "record_code+index", "unmatched")
    out.loc[~out.id_parse_ok, "match_route"] = "id_did_not_parse"
    return out.drop(columns=["_merge"])


def audit_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Every accepted match must agree on the onset it claims to describe."""
    reference = np.where(frame.onset_kind.eq("clinical") & frame.clin_onset_epoch.notna(),
                         frame.clin_onset_epoch, frame.eeg_onset_epoch)
    frame = frame.copy()
    frame["canonical_onset_epoch"] = reference
    frame["onset_difference_seconds"] = frame.onset_epoch - reference
    frame["timestamp_verified"] = frame.onset_difference_seconds.abs() <= TOLERANCE_SECONDS
    frame.loc[frame.match_route.eq("unmatched"), "timestamp_verified"] = False
    return frame


def attach_subtypes(frame: pd.DataFrame, band: str) -> pd.DataFrame:
    import json
    labels = {}
    for path in sorted(SUBTYPE_ROOT.glob("*__zer_binned.json")):
        record = json.loads(path.read_text())
        subject = record["subject"].replace("/", "_")
        block = (record.get("per_band") or {}).get(band) or {}
        if block.get("status") != "ok":
            continue
        kept = [str(s) for s in block.get("seizure_ids_kept", [])]
        for sid, label, outlier in zip(kept, block.get("subtype_label", []),
                                       block.get("outlier_flag", [])):
            labels[(subject, sid)] = {
                f"{band}__subtype_label": int(label),
                f"{band}__outlier": bool(outlier),
                f"{band}__n_subtypes": block.get("n_subtypes"),
                f"{band}__subtype_sizes": json.dumps(block.get("subtype_sizes")),
                f"{band}__chosen_k": block.get("chosen_k"),
            }
    extra = pd.DataFrame([
        {**{"subject": s, "canonical_seizure_id": i}, **v} for (s, i), v in labels.items()])
    if extra.empty:
        return frame
    return frame.merge(extra, on=["subject", "canonical_seizure_id"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="linear_graph_recurrent")
    parser.add_argument("--lead", default="lead30m")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = h2b_seizures(args.layer, args.lead)
    epi = match_epilepsiae(rows[rows.dataset == "epilepsiae"],
                           pd.read_csv(EPI_INVENTORY))
    yuq = match_yuquan(rows[rows.dataset == "yuquan"], pd.read_csv(YUQ_INVENTORY))
    joined = audit_timestamps(pd.concat([epi, yuq], ignore_index=True, sort=False))

    # one-to-one is a requirement, not an assumption
    counts = joined.groupby(["subject", "canonical_seizure_id"]).size()
    ambiguous = counts[counts > 1]
    joined["ambiguous"] = joined.set_index(["subject", "canonical_seizure_id"]).index.isin(
        ambiguous.index)

    for band in ("broad_ER", "gamma_ER"):
        joined = attach_subtypes(joined, band)

    atomic_write_csv(OUT / f"crosswalk__{args.layer}__{args.lead}.csv", joined)
    matched = joined[joined.timestamp_verified & ~joined.ambiguous]
    summary = {
        "contract": "topic5_epi_prssm_seizure_crosswalk",
        "layer": args.layer, "lead": args.lead,
        "tolerance_seconds": TOLERANCE_SECONDS,
        "n_h2b_seizures": int(len(joined)),
        "n_matched_and_verified": int(len(matched)),
        "n_unmatched": int((joined.match_route == "unmatched").sum()),
        "n_id_did_not_parse": int((joined.match_route == "id_did_not_parse").sum()),
        "n_ambiguous": int(joined.ambiguous.sum()),
        "n_matched_but_timestamp_off": int(
            ((joined.match_route != "unmatched") & ~joined.timestamp_verified).sum()),
        "by_route": joined.match_route.value_counts().to_dict(),
        "by_dataset": {
            d: {"n": int((joined.dataset == d).sum()),
                "verified": int(((joined.dataset == d) & joined.timestamp_verified).sum())}
            for d in sorted(joined.dataset.unique())},
        "onset_difference_seconds": {
            "median": float(np.nanmedian(joined.onset_difference_seconds)),
            "p95_abs": float(np.nanpercentile(joined.onset_difference_seconds.abs(), 95)),
            "max_abs": float(np.nanmax(np.abs(joined.onset_difference_seconds))),
        },
        "subtype_coverage": {
            band: int(matched[f"{band}__subtype_label"].notna().sum())
            for band in ("broad_ER", "gamma_ER")
            if f"{band}__subtype_label" in matched},
        "why_not_a_string_join":
            "Yuquan H2B ids are <record code>_<index> while the inventory keys on "
            "<subject>_sz_<counter> and carries the record code in its own column, so a "
            "string join silently keeps every Epilepsiae seizure and drops every Yuquan "
            "one. Positional joins are worse: the tables differ in order and length.",
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    atomic_write_json(OUT / f"CROSSWALK_SUMMARY__{args.layer}__{args.lead}.json", summary)
    import json as _json
    print(_json.dumps({k: v for k, v in summary.items()
                       if k not in ("why_not_a_string_join", "code_revision",
                                    "package_hash")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
