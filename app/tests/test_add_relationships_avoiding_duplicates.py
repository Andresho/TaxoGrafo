
import pytest

import pytest
from scripts.pipeline_tasks import _add_relationships_avoiding_duplicates

@pytest.mark.parametrize("existing,new,expected", [
    # No new relations: returns existing only
    ([{'source': 'A', 'target': 'B', 'type': 'T'}], [],
     [{'source': 'A', 'target': 'B', 'type': 'T'}]),
    # Duplicate + new: duplicate skipped, new added
    ([{'source': '1', 'target': '2', 'type': 'X'}],
     [
         {'source': '1', 'target': '2', 'type': 'X'},
         {'source': '2', 'target': '3', 'type': 'Y'},
         {'source': '1', 'target': '2', 'type': 'Z'},
     ],
     [
         {'source': '1', 'target': '2', 'type': 'X'},
         {'source': '2', 'target': '3', 'type': 'Y'},
         {'source': '1', 'target': '2', 'type': 'Z'},
     ]),
    # Empty existing: all new added
    ([],
     [
         {'source': 'A', 'target': 'B', 'type': 'T'},
         {'source': 'C', 'target': 'D', 'type': 'T'},
     ],
     [
         {'source': 'A', 'target': 'B', 'type': 'T'},
         {'source': 'C', 'target': 'D', 'type': 'T'},
     ]),
    # Both empty: remains empty
    ([], [], []),
])
def test_add_relationships_avoiding_duplicates(existing, new, expected):
    # Ensure we operate on a copy
    result = _add_relationships_avoiding_duplicates(list(existing), new)
    assert result == expected