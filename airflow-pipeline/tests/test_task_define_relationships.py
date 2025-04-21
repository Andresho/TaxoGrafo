import sys
import pathlib
import pandas as pd
import pytest

# Ajusta caminho para importar módulo pipeline_tasks
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import scripts.pipeline_tasks as pt

def test_task_define_relationships_no_ucs(monkeypatch):
    # Sem UCs gerados: save_dataframe com colunas mínimas
    monkeypatch.setattr(pt, 'load_dataframe', lambda stage, name: None)
    called = {}
    def fake_save(df, stage_dir, filename):
        called['df'] = df.copy()
        called['stage_dir'] = stage_dir
        called['filename'] = filename
    monkeypatch.setattr(pt, 'save_dataframe', fake_save)
    # Executa
    pt.task_define_relationships()
    # Verifica
    assert called['filename'] == pt.REL_INTERMEDIATE
    df = called['df']
    assert list(df.columns) == ['source', 'target', 'type']
    assert df.empty

def test_task_define_relationships_requires(monkeypatch):
    # Duas UCs consecutivas para relação REQUIRES
    df_input = pd.DataFrame([
        {'uc_id': 'u1', 'origin_id': 'o1', 'bloom_level': 'Lembrar'},
        {'uc_id': 'u2', 'origin_id': 'o1', 'bloom_level': 'Entender'},
    ])
    def fake_load(stage, name):
        if name == pt.GENERATED_UCS_RAW:
            return df_input
        # skip EXPANDS
        return None
    monkeypatch.setattr(pt, 'load_dataframe', fake_load)
    called = {}
    def fake_save(df, stage_dir, filename):
        called['df'] = df.copy()
        called['stage_dir'] = stage_dir
        called['filename'] = filename
    monkeypatch.setattr(pt, 'save_dataframe', fake_save)
    # Executa
    pt.task_define_relationships()
    # Verifica resultado
    assert called['filename'] == pt.REL_INTERMEDIATE
    df_out = called['df']
    # Deve haver uma relação REQUIRES
    assert len(df_out) == 1
    row = df_out.iloc[0]
    assert row['source'] == 'u1' and row['target'] == 'u2' and row['type'] == 'REQUIRES'
    # stage_dir correto
    assert called['stage_dir'] == pt.stage3_dir