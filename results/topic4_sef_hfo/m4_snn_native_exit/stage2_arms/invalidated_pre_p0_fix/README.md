# ⚠️ INVALIDATED — pre-P0-fix runs (do not cite)

These `arms_s2dasym*` runs were labelled "asymmetric" but the `d_sweep` code path dropped `tau_p_down`
(P0 bug, review 2026-07-21), so they actually ran SYMMETRIC `tau_p=3000`. Their "runaway / periodic
runaway-burst" behaviour is NOT the asymmetric hold. Superseded by `../arms_asymfix_seed1.*` (real
asymmetric, `tau_p_down=12000` confirmed in `cfg_effective`). Kept only as evidence of the bug.
