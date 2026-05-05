import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v2_validate_layer_c import extract_layer_c


def test_extract_layer_c_strict_and(tmp_path):
    payload = {
        "time_split_reproducibility": {
            "grade": "strong",
            "splits": {
                "first_half_second_half": {"forward_reverse_reproduced": True},
                "odd_even_block": {"forward_reverse_reproduced": True},
            },
        },
        "adaptive_cluster": {"stable_k": 2},
    }
    p = tmp_path / "fake.json"
    p.write_text(json.dumps(payload))
    res = extract_layer_c(p)
    assert res["forward_reverse_reproduced_strict"] is True
    assert res["forward_reverse_reproduced_lenient"] is True
    assert res["passes_layer_c"] is True


def test_extract_layer_c_lenient_only_fails_strict(tmp_path):
    payload = {
        "time_split_reproducibility": {
            "grade": "moderate",
            "splits": {
                "first_half_second_half": {"forward_reverse_reproduced": True},
                "odd_even_block": {"forward_reverse_reproduced": False},
            },
        },
        "adaptive_cluster": {"stable_k": 2},
    }
    p = tmp_path / "fake.json"
    p.write_text(json.dumps(payload))
    res = extract_layer_c(p)
    assert res["forward_reverse_reproduced_strict"] is False
    assert res["forward_reverse_reproduced_lenient"] is True
    assert res["passes_layer_c"] is False  # strict failed
