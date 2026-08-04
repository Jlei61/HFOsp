"""Did a short re-simulation reproduce the trajectory already on disk?

A 20 s entry ledger is only readable if it is the *same* trajectory as the 45 s
reconnaissance run it re-derives, so the runs are compared event by event before
anything is published.

The comparison has one subtlety that a naive cut gets wrong, and it produced a
false alarm on the first seed it met.  An event still in progress when the short
run stops is **truncated** there and whole in the long run, so the same event
appears with two different end times: 19985-19999 against 19985-20014.  Filtering
each side by "ends before the cut" then keeps the truncated copy and drops the
whole one, and the short run reports one event too many while being bit-identical
everywhere it could be compared.

So the comparable span ends a margin short of the cut, and the margin is the
longest event either run produced.  Anything that could have been clipped is
excluded from both sides rather than counted as a mismatch on one.
"""
from __future__ import annotations

REPRODUCTION_SCHEMA = "fcxr-lc3-reproduction-1.0"
FIELD_TOL = 1e-6
MIN_MARGIN_MS = 1.0


def comparable_margin_ms(fresh, recorded) -> float:
    """How far short of the cut the comparison must stop.

    The longest event in either run: nothing longer can be in progress at the
    cut, so nothing beyond this margin can have been truncated.
    """
    durations = [float(e["dur_ms"]) for e in fresh] + [float(e["dur_ms"]) for e in recorded]
    return max(MIN_MARGIN_MS, max(durations, default=MIN_MARGIN_MS))


def events_reproduce(fresh, recorded, *, cut_ms) -> dict:
    """Compare the two event lists over the span both runs could see whole.

    ``fresh`` carries the runner's ``t_on``/``t_off`` keys, ``recorded`` the
    stored ``t_on_ms``/``t_off_ms``; both are read through the same accessor so a
    key rename on either side fails loudly instead of silently comparing nothing.
    """
    def _get(e, name):
        for key in (name, f"{name}_ms"):
            if key in e:
                return float(e[key])
        raise KeyError(f"event is missing {name}: {sorted(e)}")

    margin = comparable_margin_ms(fresh, recorded)
    edge = float(cut_ms) - margin
    got = [e for e in fresh if _get(e, "t_off") < edge]
    ref = [e for e in recorded if _get(e, "t_off") < edge]
    common = dict(schema=REPRODUCTION_SCHEMA, cut_ms=float(cut_ms),
                  margin_ms=margin, comparable_until_ms=edge,
                  n_compared=len(got), n_recorded_in_span=len(ref))
    if len(got) != len(ref):
        return dict(common, reproduces=False,
                    detail=(f"{len(got)} events against {len(ref)} recorded within "
                            f"{edge:.0f} ms"))
    for i, (a, b) in enumerate(zip(got, ref)):
        for name in ("t_on", "t_off", "dur_ms", "peak_ext"):
            x = _get(a, name) if name in ("t_on", "t_off") else float(a[name])
            y = _get(b, name) if name in ("t_on", "t_off") else float(b[name])
            if abs(x - y) > FIELD_TOL:
                return dict(common, reproduces=False,
                            detail=f"event {i} {name}: {x!r} against recorded {y!r}")
    return dict(common, reproduces=True,
                detail=f"{len(got)} events reproduced exactly within {edge:.0f} ms")
