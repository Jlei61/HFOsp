import pytest

from scripts.audit_topic4_rev12_causal_fit_v1_invalidation import accounting


def test_invalidated_fit_queue_accounting_is_exact():
    result = accounting(
        expected=108, completed=14, parity_failed=2,
        resource_stopped=8, never_launched=84,
    )
    assert result["accounting_exact"] is True
    with pytest.raises(RuntimeError):
        accounting(
            expected=108, completed=14, parity_failed=2,
            resource_stopped=8, never_launched=83,
        )

