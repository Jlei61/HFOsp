# Stage 0C transfer-support audit v1.1

- 结论：`EXTRA_FINE_VALID_NO_SUPPORTED_OBJECT_WITH_UNRESOLVED_TRANSIENTS`
- all-102 final counts：`{'becomes_over_100': 17, 'numerical_unresolved': 84, 'candidate_survives': 1}`
- primary-23 final counts：`{'becomes_over_100': 12, 'numerical_unresolved': 10, 'candidate_survives': 1}`
- supported points：0
- screen / confirm survivor forks：1 / 1
- extra-fine overlap / trajectory direct：True / True
- wall / peak RSS：422.47 s / 0.181 GiB
- 解释：extra-fine transfer 数值验证通过，但没有两初态支持的有限对象，且仍有长瞬态或分类未决。

v1 按 implementation/spec provenance mismatch 保留 unresolved；coarse 不参与 v1.1 authoritative 判定。
