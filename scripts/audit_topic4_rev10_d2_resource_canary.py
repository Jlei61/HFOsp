"""Compare local q-resource arms against matched global and exact off."""
from __future__ import annotations

import argparse, hashlib, json, os, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d2_inhibitory_resource_canary.json"


def _sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adjudicate(summary):
    rows = summary["candidate_rows"]
    off = next(row for row in rows if row["candidate_id"] == "edge_noop")
    local = {float(row["resource_k_q_per_ms"]): row for row in rows if row["resource_mode"] == "local"}
    global_ = {float(row["resource_k_q_per_ms"]): row for row in rows if row["resource_mode"] == "global"}
    if set(local) != set(global_) or len(local) != 3: raise RuntimeError("resource grid is incomplete")
    comparisons=[]; passed=[]
    for k_q in sorted(local):
        l=local[k_q]; g=global_[k_q]
        ok=bool(l["n_runaway_networks"] == 0 and l["networks_with_both_clean_modes"] >= 2
                and l["networks_with_clean_B"] >= 2
                and l["networks_with_both_clean_modes"] > g["networks_with_both_clean_modes"]
                and l["networks_with_both_clean_modes"] > off["networks_with_both_clean_modes"])
        row={"k_q_per_ms": k_q, "local_candidate_id": l["candidate_id"], "global_candidate_id": g["candidate_id"],
             "local_networks_with_A": l["networks_with_clean_A"], "global_networks_with_A": g["networks_with_clean_A"],
             "off_networks_with_A": off["networks_with_clean_A"], "local_networks_with_B": l["networks_with_clean_B"],
             "local_networks_with_both": l["networks_with_both_clean_modes"], "global_networks_with_both": g["networks_with_both_clean_modes"],
             "off_networks_with_both": off["networks_with_both_clean_modes"], "local_score": l["selection_score_equal_network"],
             "global_score": g["selection_score_equal_network"], "off_score": off["selection_score_equal_network"],
             "local_specific_route_access": ok}
        comparisons.append(row)
        if ok: passed.append(row)
    passed.sort(key=lambda row: row["local_score"])
    return {"status": "REV10D2_LOCAL_INHIBITORY_RESOURCE_ACCESS_OBSERVED" if passed else "REV10D2_LOCAL_INHIBITORY_RESOURCE_ACCESS_NOT_OBSERVED",
            "selected_local_candidate_id": passed[0]["local_candidate_id"] if passed else None,
            "matched_global_candidate_id": passed[0]["global_candidate_id"] if passed else None,
            "comparisons": comparisons, "off_baseline": off,
            "claim_boundary": "three-network development canary; returned-only; not patient-blind or an ictal lifecycle test"}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config", default=str(DEFAULT_CONFIG)); args=parser.parse_args()
    config_path=Path(args.config).resolve(); config=json.loads(config_path.read_text()); root=ROOT/config["output_root"]
    summary_path=root/"canary_summary_returned_only.json"; summary=json.loads(summary_path.read_text())
    if summary.get("status") != "REV10D2_RETURNED_ONLY_CANARY_COMPLETE": raise RuntimeError("D2 summary incomplete")
    payload=adjudicate(summary); payload["inputs"]={"config":{"path":str(config_path.relative_to(ROOT)),"sha256":_sha256(config_path)},"summary":{"path":str(summary_path),"sha256":_sha256(summary_path)}}
    out=root/"canary_verdict.json"; out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=out.parent,suffix=".tmp"); os.close(fd)
    try: Path(tmp).write_text(json.dumps(payload,indent=2,sort_keys=True)); os.replace(tmp,out)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(json.dumps({"status":payload["status"],"selected_local_candidate_id":payload["selected_local_candidate_id"],"output":str(out)},indent=2))


if __name__ == "__main__": main()
