"""Regression checks for descriptive final-export validation."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_final_comparability import paired_interval, validate_cell
from test_nominal_comparability import _record_sets


def test_interval_direction_and_pair_order():
    e, a = _record_sets(both=180, expert_only=8, learner_only=6, neither=6)
    result, lo, hi = paired_interval(e, a)
    assert result['learner_minus_expert'] == pytest.approx(-.01)
    assert lo <= -1 <= hi
    assert paired_interval(e[::-1], a[::-1]) == (result, lo, hi)


def test_final_validator_rejects_duplicate_ids_before_other_fields():
    e, _ = _record_sets(both=200, expert_only=0, learner_only=0, neither=0)
    e[-1]['initial_condition_id'] = 0
    with pytest.raises(ValueError, match='invalid IDs'):
        validate_cell(e, {}, 'peg', 'expert', 1, 2, None)


def test_final_validator_rejects_partial_cell():
    with pytest.raises(ValueError, match='200 episodes'):
        validate_cell([], {}, 'peg', 'expert', 1, 2, None)
