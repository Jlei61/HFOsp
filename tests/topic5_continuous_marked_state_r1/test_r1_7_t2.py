from src.topic5_continuous_marked_state_r1.r1_7_t2 import (
    R1_7_T2_REVISION, is_expected_support_limit,
)


def test_r1_7_t2_revision_is_frozen() -> None:
    assert R1_7_T2_REVISION == "r1_7a_d_mechanism_t2_r2_n100_v1"


def test_only_declared_support_failures_are_non_estimable() -> None:
    assert is_expected_support_limit(
        ValueError("state-matched placebo has too few TRAIN donors")
    )
    assert is_expected_support_limit(
        ValueError("T2-R2.0 H10 has no within-segment pairs")
    )
    assert not is_expected_support_limit(ValueError("placebo arrays disagree"))
    assert not is_expected_support_limit(ValueError("checkpoint hash mismatch"))


def test_every_query_states_call_site_passes_state_permutation() -> None:
    """`state_permutation` is keyword-only with no default.

    Omitting it at any one of the many call sites is a TypeError that only
    surfaces when that specific analysis path is executed for real, so guard
    every call site statically instead of waiting for the run.
    """
    import ast
    from pathlib import Path

    root = Path("src/topic5_continuous_marked_state_r1")
    missing = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            if name != "_query_states":
                continue
            if not any(kw.arg == "state_permutation" for kw in node.keywords):
                missing.append(f"{path.name}:{node.lineno}")
    assert not missing, f"_query_states call sites missing state_permutation: {missing}"
