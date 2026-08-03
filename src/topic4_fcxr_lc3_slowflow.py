"""Deterministic landmark selection for LC3 local D-X slow-flow probes."""
from __future__ import annotations


D_ORDER = ("D_healthy", "D10", "D30", "D50", "D70", "Dmax")
X_ORDER = (1.0, 0.9, 0.8, 0.65, 0.5, 0.3, 0.1)
HIGH = {"FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT"}


def _coord(row):
    return row["d_label"], float(row["a_x"]), row["state_kind"]


def _adjacent(a, b):
    if a["state_kind"] != b["state_kind"]:
        return False
    di, dj = D_ORDER.index(a["d_label"]), D_ORDER.index(b["d_label"])
    xi, xj = X_ORDER.index(float(a["a_x"])), X_ORDER.index(float(b["a_x"]))
    return abs(di - dj) + abs(xi - xj) == 1


def select_slowflow_landmarks(geometry_rows: list[dict], *, min_n=12, max_n=20) -> list[dict]:
    """Select both sides of observed H1 label boundaries, then a fixed fallback grid."""

    h1 = [row for row in geometry_rows if not row.get("sentinel", False)]
    if len(h1) != 84:
        raise ValueError(f"complete 84-row H1 map required, got {len(h1)}")
    chosen = {}
    ordered = sorted(h1, key=lambda r: (
        r["state_kind"], D_ORDER.index(r["d_label"]),
        X_ORDER.index(float(r["a_x"])), r["row_id"]))
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if not _adjacent(a, b):
                continue
            ah = a["resolved_label"] in HIGH
            bh = b["resolved_label"] in HIGH
            if ah != bh:
                chosen[_coord(a)] = a
                chosen[_coord(b)] = b

    # Fixed no-bracket/supplement grid from spec §6.1.  The state rule is locked:
    # healthy and D50 use low-start to probe entry drift; Dmax uses high-start to
    # probe return drift.  It does not inspect outcome labels.
    for d_label, state in (("D_healthy", "low"), ("D50", "low"), ("Dmax", "high")):
        for a_x in (1.0, 0.65, 0.3, 0.1):
            if len(chosen) >= max(min_n, 12):
                break
            matches = [r for r in h1 if r["d_label"] == d_label
                       and float(r["a_x"]) == a_x and r["state_kind"] == state]
            if len(matches) != 1:
                raise ValueError(f"missing fixed landmark {d_label}/{a_x}/{state}")
            chosen.setdefault(_coord(matches[0]), matches[0])

    rows = sorted(chosen.values(), key=lambda r: (
        r["state_kind"], D_ORDER.index(r["d_label"]),
        X_ORDER.index(float(r["a_x"])), r["row_id"]))[:int(max_n)]
    if not (int(min_n) <= len(rows) <= int(max_n)):
        raise ValueError(f"landmark count outside [{min_n},{max_n}]: {len(rows)}")
    return rows

