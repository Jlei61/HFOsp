#!/usr/bin/env python3
"""Persist per-Yuquan-subject clinical network (onset + interictal + ictal-spread
contacts) parsed from the intracranial-EEG report text, for use as the anatomical
reference in broad-lagPat QC + SOZ inside/outside analysis.

Reads converted report text from /tmp/yqdoc/*颅内eeg.txt (re-converts via
libreoffice if absent). Writes results/lagpat_broad/yuquan_clinical_networks.json:
  { "<pinyin>": {"network": ["E8","E9",...], "raw_lines": [...]} }

Network contacts come from the ictal/interictal/summary free-text lines
(脑电图发作 / 发作期 / 间歇期 / 间期 / 总结 / 起源 / 起始). Ranges like 'E8-15'
expand to {E8..E15}; prime electrodes (C') normalized. Fuzzy by nature (free
text) — flagged as approximate.
"""
from __future__ import annotations
import os, re, glob, json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BING = Path("/mnt/yuquan_data/yuquan_24h_bingli")
TMP = Path("/tmp/yqdoc")
OUT = REPO / "results" / "lagpat_broad" / "yuquan_clinical_networks.json"

PRIME = {"’": "'", "‘": "'", "ʼ": "'"}
def norm(s):
    for k, v in PRIME.items():
        s = s.replace(k, v)
    return s

def expand_tokens(text):
    out = set()
    for el, a, b, el2, n2 in re.findall(r"([A-Z]+'?)(\d+)-(\d+)|([A-Z]+'?)(\d+)", norm(text)):
        if el:
            for n in range(int(a), int(b) + 1):
                out.add(f"{el.upper()}{n}")
        else:
            out.add(f"{el2.upper()}{n2}")
    return out

def ensure_txt(pin):
    cands = [f for f in glob.glob(str(BING / pin / "*颅内eeg.doc"))
             if not os.path.basename(f).startswith(".~")]
    if not cands:
        return None
    base = os.path.basename(cands[0])[:-4]
    txt = TMP / f"{base}.txt"
    if txt.exists():
        return txt
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run(["libreoffice", "--headless", "--convert-to", "txt:Text",
                    "--outdir", str(TMP), cands[0]],
                   capture_output=True, timeout=120)
    return txt if txt.exists() else None

def parse(pin):
    txt = ensure_txt(pin)
    if not txt:
        return None
    t = open(txt, encoding="utf-8", errors="ignore").read()
    lines = re.findall(r"(?:脑电图发作|发作期|间歇期|间期|总结|起源|起始)[：:].{0,160}", t)
    net = set()
    for ln in lines:
        net |= expand_tokens(ln)
    return {"network": sorted(net), "n_contacts": len(net),
            "raw_lines": [re.sub(r"\s+", " ", l)[:200] for l in lines]}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pins = sorted(d.name for d in BING.iterdir() if d.is_dir())
    res = {}
    for pin in pins:
        p = parse(pin)
        if p:
            res[pin] = p
            print(f"{pin:<16} network n={p['n_contacts']:<4} {p['network'][:12]}")
        else:
            print(f"{pin:<16} (no doc)")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {OUT} ({len(res)} subjects)")

if __name__ == "__main__":
    main()
