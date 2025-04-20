import sys
import pathlib

# Ensure 'airflow-pipeline' root is in sys.path for importing 'scripts'
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest

from scripts.pipeline_tasks import (
    _get_sort_key,
    _add_relationships_avoiding_duplicates,
    _format_difficulty_prompt,
)

def test_get_sort_key_community_report():
    origin = {"origin_type": "community_report", "level": 5}
    # score = 10000 - level = 9995, key = (1, -9995)
    expected = (1, -(10000 - 5))
    assert _get_sort_key(origin) == expected

@pytest.mark.parametrize(
    ("degree", "frequency", "entity_type", "expected_priority", "expected_score"),
    [
        (2, 3, "person", 3, 2*10 + 3),
        (1, 4, "organization", 2, 1*10 + 4),
        (0, 0, "unknown", 2, 0),
    ],
)
def test_get_sort_key_entity(degree, frequency, entity_type, expected_priority, expected_score):
    origin = {"origin_type": "entity", "degree": degree, "frequency": frequency, "entity_type": entity_type}
    assert _get_sort_key(origin) == (expected_priority, -expected_score)

def test_get_sort_key_default():
    # Unrecognized origin_type should yield default priority 2 and score 0
    origin = {"origin_type": "other"}
    assert _get_sort_key(origin) == (2, 0)

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