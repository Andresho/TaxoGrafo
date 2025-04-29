import pytest

from scripts.pipeline_tasks import (
    _get_sort_key,
    _add_relationships_avoiding_duplicates,
    _format_difficulty_prompt,
)

@pytest.mark.parametrize("origin,expected", [
    # Community report: priority 1, score = -(10000 - level)
    ({"origin_type": "community_report", "level": 5},
     (1, -(10000 - 5))),
    # Entity person: priority 3, score = -(degree*10 + frequency)
    ({"origin_type": "entity", "degree": 2, "frequency": 3, "entity_type": "person"},
     (3, -(2*10 + 3))),
    # Entity organization: priority 2, score = -(degree*10 + frequency)
    ({"origin_type": "entity", "degree": 1, "frequency": 4, "entity_type": "organization"},
     (2, -(1*10 + 4))),
    # Entity default/unknown: priority 2, zero score
    ({"origin_type": "entity", "degree": 0, "frequency": 0, "entity_type": "unknown"},
     (2, 0)),
    # Unrecognized type: priority 2, zero score
    ({"origin_type": "other"}, (2, 0)),
])
def test_get_sort_key(origin, expected):
    assert _get_sort_key(origin) == expected

def test_add_relationships_avoiding_duplicates_empty_new():
    existing = [{"source": "a", "target": "b", "type": "X"}]
    # When new_rels is empty, should return existing list unchanged
    result = _add_relationships_avoiding_duplicates(existing, [])
    assert result == existing

def test_add_relationships_avoiding_duplicates_with_duplicates_and_new():
    existing = [{"source": "a", "target": "b", "type": "X"}]
    new = [
        {"source": "a", "target": "b", "type": "X"},  # duplicate
        {"source": "b", "target": "c", "type": "Y"},  # new
    ]
    result = _add_relationships_avoiding_duplicates(existing, new)
    # Should include only one new relation, preserving existing
    assert {"source": "b", "target": "c", "type": "Y"} in result
    # Original relation remains
    assert {"source": "a", "target": "b", "type": "X"} in result
    # Total length should be 2
    assert len(result) == 2

def test_format_difficulty_prompt_simple():
    batch = [
        {"uc_id": "1", "uc_text": "text1"},
        {"uc_id": "2", "uc_text": "text2"},
    ]
    template = "Header\n{{BATCH_OF_UCS}}\nFooter"
    prompt = _format_difficulty_prompt(batch, template)
    # Check that header and footer are preserved
    assert prompt.startswith("Header")
    assert prompt.endswith("Footer")
    # Check that each UC appears in the batch block
    expected_block = "- ID: 1\n  Texto: text1\n- ID: 2\n  Texto: text2"
    assert expected_block in prompt