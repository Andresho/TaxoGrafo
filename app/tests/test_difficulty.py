import pytest

import pytest

from scripts.pipeline_tasks import _calculate_final_difficulty_from_raw

def test_calculate_final_difficulty_single_eval():
    # Single UC with one evaluation
    generated = [{"uc_id": "uc1", "other": "x"}]
    raw = [{"uc_id": "uc1", "difficulty_score": 80, "justification": "well done"}]
    updated, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(generated, raw)
    # One UC evaluated, meets minimum evaluations (1)
    assert evaluated_count == 1
    assert min_evals_met_count == 1
    assert len(updated) == 1
    uc = updated[0]
    assert uc["uc_id"] == "uc1"
    assert uc["difficulty_score"] == 80
    assert uc["difficulty_justification"] == "well done"
    assert uc["evaluation_count"] == 1

def test_calculate_final_difficulty_multiple_evals():
    # Single UC with multiple evaluations; average and concatenated justifications
    generated = [{"uc_id": "uc2"}]
    raw = [
        {"uc_id": "uc2", "difficulty_score": 10, "justification": "ok"},
        {"uc_id": "uc2", "difficulty_score": 20, "justification": "good"},
    ]
    updated, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(generated, raw)
    assert evaluated_count == 1
    assert min_evals_met_count == 1
    uc = updated[0]
    # Average of 10 and 20
    assert uc["difficulty_score"] == 15
    # Justifications joined by ' | '
    assert uc["difficulty_justification"] == "ok | good"
    # Evaluation count is number of justifications
    assert uc["evaluation_count"] == 2

def test_calculate_final_difficulty_no_evals():
    # Single UC with no evaluations
    generated = [{"uc_id": "uc3"}]
    raw = []
    updated, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(generated, raw)
    assert evaluated_count == 0
    assert min_evals_met_count == 0
    uc = updated[0]
    assert uc["uc_id"] == "uc3"
    # No score or justification => default to None / 'Não avaliado'
    assert uc["difficulty_score"] is None
    assert uc["difficulty_justification"] == "Não avaliado"
    assert uc["evaluation_count"] == 0

def test_calculate_final_difficulty_mixed_multiple_ucs():
    # Multiple UCs, only one evaluated
    generated = [{"uc_id": "a"}, {"uc_id": "b"}]
    raw = [{"uc_id": "a", "difficulty_score": 100, "justification": "perfect"}]
    updated, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(generated, raw)
    assert evaluated_count == 1
    assert min_evals_met_count == 1
    # Preserve order of generated list
    assert [u["uc_id"] for u in updated] == ["a", "b"]
    ua, ub = updated
    # 'a' evaluated
    assert ua["difficulty_score"] == 100
    assert ua["difficulty_justification"] == "perfect"
    assert ua["evaluation_count"] == 1
    # 'b' not evaluated
    assert ub["difficulty_score"] is None
    assert ub["difficulty_justification"] == "Não avaliado"
    assert ub["evaluation_count"] == 0

def test_calculate_final_difficulty_multiple_ucs_multiple_evals():
    # Two UCs, each with two evaluations: verify averages and justifications
    generated = [{"uc_id": "x"}, {"uc_id": "y"}]
    raw = [
        {"uc_id": "x", "difficulty_score": 30, "justification": "a"},
        {"uc_id": "x", "difficulty_score": 50, "justification": "b"},
        {"uc_id": "y", "difficulty_score": 100, "justification": "c"},
        {"uc_id": "y", "difficulty_score": 0, "justification": "d"},
    ]
    updated, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(generated, raw)
    # Both UCs should be evaluated and meet minimum evaluations
    assert evaluated_count == 2
    assert min_evals_met_count == 2
    # Check UC 'x'
    ux = next(u for u in updated if u.get("uc_id") == "x")
    assert ux["difficulty_score"] == 40  # (30+50)/2
    assert ux["difficulty_justification"] == "a | b"
    assert ux["evaluation_count"] == 2
    # Check UC 'y'
    uy = next(u for u in updated if u.get("uc_id") == "y")
    assert uy["difficulty_score"] == 50  # (100+0)/2
    assert uy["difficulty_justification"] == "c | d"
    assert uy["evaluation_count"] == 2

def test_calculate_final_difficulty_invalid_score_type():
    # A UC with a float score should have no valid score counted
    generated = [{"uc_id": "ucX"}]
    raw = [{"uc_id": "ucX", "difficulty_score": 50.0, "justification": "just"}]
    updated, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(generated, raw)
    # No integer scores => evaluated_count remains 0
    assert evaluated_count == 0
    # Justification counts as an evaluation but no valid score => min_evals_met_count is 1
    assert min_evals_met_count == 1
    uc = updated[0]
    # Since no valid scores, difficulty fields are not set
    assert uc.get("difficulty_score") is None
    assert uc.get("difficulty_justification") is None
    assert uc.get("evaluation_count") is None