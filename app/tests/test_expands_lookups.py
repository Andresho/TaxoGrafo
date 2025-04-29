import pandas as pd
import pytest

import pandas as pd
import pytest

from scripts.pipeline_tasks import _prepare_expands_lookups, BLOOM_ORDER_MAP

def test_prepare_expands_lookups_standard():
    # Entities DataFrame with id-title mapping
    entities_df = pd.DataFrame([
        {"id": "e1", "title": "Title1"},
        {"id": "e2", "title": "Title2"},
    ])
    # Generated UCs for multiple origins and levels
    generated_ucs = [
        {"origin_id": "e1", "bloom_level": "Lembrar", "uc_id": "uc1"},
        {"origin_id": "e1", "bloom_level": "Entender", "uc_id": "uc2"},
        {"origin_id": "e2", "bloom_level": "Aplicar", "uc_id": "uc3"},
        # Bloom level not in BLOOM_ORDER_MAP should be skipped
        {"origin_id": "e2", "bloom_level": "UnknownLevel", "uc_id": "uc4"},
    ]
    name_map, ucs_by_origin_level = _prepare_expands_lookups(entities_df, generated_ucs)
    # Check name map
    assert name_map == {"Title1": "e1", "Title2": "e2"}
    # Check ucs_by_origin_level keys and values
    assert set(ucs_by_origin_level.keys()) == {"e1", "e2"}
    # For e1:
    assert ucs_by_origin_level["e1"]["Lembrar"] == ["uc1"]
    assert ucs_by_origin_level["e1"]["Entender"] == ["uc2"]
    # For e2:
    assert ucs_by_origin_level["e2"]["Aplicar"] == ["uc3"]
    # uc4 should be skipped due to invalid bloom_level
    assert "UnknownLevel" not in ucs_by_origin_level["e2"]

def test_prepare_expands_lookups_no_entities():
    # No entities_df provided
    generated_ucs = [
        {"origin_id": "o1", "bloom_level": lvl, "uc_id": f"uc{idx}"}
        for idx, lvl in enumerate(BLOOM_ORDER_MAP.keys(), start=1)
    ]
    name_map, ucs_by_origin_level = _prepare_expands_lookups(None, generated_ucs)
    # Name map should be empty
    assert name_map == {}
    # ucs_by_origin_level should include one entry per unique origin_id
    origin_ids = {uc["origin_id"] for uc in generated_ucs}
    assert set(ucs_by_origin_level.keys()) == origin_ids
    # Each bloom_level has exactly one UC as per generated_ucs
    for uc in generated_ucs:
        origin = uc["origin_id"]
        level = uc["bloom_level"]
        uc_id = uc["uc_id"]
        assert ucs_by_origin_level[origin][level] == [uc_id]

@pytest.mark.parametrize("invalid_uc", [
    {"origin_id": None, "bloom_level": "Lembrar", "uc_id": "uc"},
    {"origin_id": "e", "bloom_level": None, "uc_id": "uc"},
    {"origin_id": "e", "bloom_level": "Lembrar", "uc_id": None},
])
def test_prepare_expands_lookups_invalid_generated_entries(invalid_uc):
    # Entities present but generated_uc entries missing fields should be skipped
    entities_df = pd.DataFrame([{"id": "e", "title": "T"}])
    name_map, ucs_by_origin_level = _prepare_expands_lookups(entities_df, [invalid_uc])
    # Map remains correct
    assert name_map == {"T": "e"}
    # No entries should be added to ucs_by_origin_level
    assert ucs_by_origin_level == {}