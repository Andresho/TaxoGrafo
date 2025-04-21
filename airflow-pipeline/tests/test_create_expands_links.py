import sys
import pathlib

# Ensure project root is in sys.path for importing 'scripts'
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from scripts.pipeline_tasks import _create_expands_links

def test_create_expands_links_missing_columns():
    df = pd.DataFrame({"other": [1, 2]})
    # Missing 'source'/'target' returns empty list
    rels = _create_expands_links(df, {}, {})
    assert rels == []

def test_create_expands_links_missing_mapping():
    # Even with proper columns, missing name->id mapping skips all
    df = pd.DataFrame({"source": ["A"], "target": ["B"]})
    rels = _create_expands_links(df, {}, {"e1": {"Lembrar": ["uc1"]}})
    assert rels == []

def test_create_expands_links_same_id():
    # Mapping both names to same id -> skip self-relations
    df = pd.DataFrame({"source": ["X"], "target": ["Y"]})
    mapping = {"X": "e", "Y": "e"}
    ucs = {"e": {"Lembrar": ["u1"], "Entender": ["u2"]}}
    rels = _create_expands_links(df, mapping, ucs)
    assert rels == []

def test_create_expands_links_no_ucs_for_levels():
    # Both entities mapped but no UCs at required levels -> no relations
    df = pd.DataFrame({"source": ["S"], "target": ["T"], "weight": [5.0], "description": ["d"]})
    mapping = {"S": "e1", "T": "e2"}
    # ucs only at level not in LEVELS_TO_CONNECT
    ucs = {"e1": {"Aplicar": ["u1"]}, "e2": {"Aplicar": ["u2"]}}
    rels = _create_expands_links(df, mapping, ucs)
    assert rels == []

def test_create_expands_links_basic_weight_and_desc():
    # Proper entities, UCs at both levels, weight and desc propagated
    df = pd.DataFrame({
        "source": ["SourceName"],
        "target": ["TargetName"],
        "weight": [2.5],
        "description": ["relation desc"]
    })
    mapping = {"SourceName": "e1", "TargetName": "e2"}
    # Define UCs at both LEVELS_TO_CONNECT: 'Lembrar' and 'Entender'
    ucs = {
        "e1": {"Lembrar": ["ucA1"], "Entender": ["ucA2"]},
        "e2": {"Lembrar": ["ucB1"], "Entender": ["ucB2"]},
    }
    rels = _create_expands_links(df, mapping, ucs)
    # Expect one relation per level => total 2
    assert len(rels) == 2
    # Build set of tuples for easier assertion (nodes tuple, weight, desc)
    rel_set = {(tuple(r["nodes"]), r["weight"], r["graphrag_rel_desc"]) for r in rels}
    # Expected relations: node pairs per level
    expected = {
        (("ucA1", "ucB1"), 2.5, "relation desc"),
        (("ucA2", "ucB2"), 2.5, "relation desc"),
    }
    assert rel_set == expected
    # Check that 'nodes' array lists both UC IDs
    for r in rels:
        assert "nodes" in r
        assert isinstance(r["nodes"], list) and len(r["nodes"]) == 2

def test_create_expands_links_default_weight_and_no_desc():
    # No weight/description columns -> default weight 1.0 and desc None
    df = pd.DataFrame({"source": ["A"], "target": ["B"]})
    mapping = {"A": "e1", "B": "e2"}
    ucs = {"e1": {"Lembrar": ["u1"]}, "e2": {"Lembrar": ["u2"]}}
    rels = _create_expands_links(df, mapping, ucs)
    # Only one level 'Lembrar' => one undirected relation
    assert len(rels) == 1
    for r in rels:
        assert r["weight"] == 1.0
        assert r["graphrag_rel_desc"] is None
        assert r["type"] == "EXPANDS"
        # Check nodes array for undirected edge
        assert "nodes" in r
        assert isinstance(r["nodes"], list) and len(r["nodes"]) == 2