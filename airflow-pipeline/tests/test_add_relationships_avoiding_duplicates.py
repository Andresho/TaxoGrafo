
import pytest

from scripts.pipeline_tasks import _add_relationships_avoiding_duplicates

def test_no_new_rels_returns_existing():
    existing = [{'source': 'A', 'target': 'B', 'type': 'T'}]
    result = _add_relationships_avoiding_duplicates(existing, [])
    # Should return the original list unchanged
    assert result == existing
    assert result is existing

def test_adds_only_unique_relations():
    existing = [
        {'source': '1', 'target': '2', 'type': 'X'},
    ]
    new_rels = [
        {'source': '1', 'target': '2', 'type': 'X'},  # duplicate
        {'source': '2', 'target': '3', 'type': 'Y'},  # new
        {'source': '1', 'target': '2', 'type': 'Z'},  # new because type different
    ]
    result = _add_relationships_avoiding_duplicates(existing, new_rels)
    # Expect original plus the two unique new relations
    assert len(result) == 3
    # Check members
    assert {'source': '2', 'target': '3', 'type': 'Y'} in result
    assert {'source': '1', 'target': '2', 'type': 'Z'} in result

def test_empty_existing_adds_all_new():
    existing = []
    new_rels = [
        {'source': 'A', 'target': 'B', 'type': 'T'},
        {'source': 'C', 'target': 'D', 'type': 'T'},
    ]
    result = _add_relationships_avoiding_duplicates(existing, new_rels)
    assert result == new_rels
    assert len(result) == 2

def test_existing_and_new_empty_both():
    existing = []
    new_rels = []
    result = _add_relationships_avoiding_duplicates(existing, new_rels)
    # Returns existing, which is empty list
    assert result == existing