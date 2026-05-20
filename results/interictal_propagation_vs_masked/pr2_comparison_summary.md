# Step 5a — PR-2 phantom vs masked comparison summary

- subjects: total **40**, status=ok **40**

- **chosen_k flips**: 4/40
  - `yuquan_huangwanling`: chosen_k 4 → 3
  - `yuquan_zhaojinrui`: chosen_k 5 → 6
  - `yuquan_zhangjinhan`: chosen_k 6 → 5
  - `epilepsiae_916`: chosen_k 2 → 4
- **stable_k flips**: 4/40
  - `yuquan_huangwanling`: stable_k 4 → 3
  - `yuquan_zhaojinrui`: stable_k 5 → 6
  - `yuquan_zhangjinhan`: stable_k 6 → 5
  - `epilepsiae_916`: stable_k 2 → 4

### Label-level shift (subjects where chosen_k unchanged, n=35)

| metric | median | range |
|---|---|---|
| Jaccard (macro, best-perm) | 0.703 | [0.265, 0.897] |
| exact agreement (best-perm) | 0.846 | [0.422, 0.950] |
| AMI(orig labels, masked labels) | 0.372 | [0.001, 0.700] |

Sanity vs lagpatrank_audit: Spearman ρ(audit Δ, PR-2 AMI) = 0.961 (p=6.65e-20, n=35). Same-direction = audit and PR-2-level rerun agree on which subjects are most affected.