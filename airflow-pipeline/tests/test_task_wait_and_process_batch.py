import sys
import pathlib
import pandas as pd
import pytest

# Ajusta caminho para importar módulo pipeline_tasks
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import scripts.pipeline_tasks as pt

class FakeTI:
    def __init__(self, value):
        self.value = value
    def xcom_pull(self, task_ids=None, key=None):
        return self.value

def test_task_wait_and_process_batch_no_batch(monkeypatch, tmp_path):
    # Sem batch_id -> save_dataframe é chamado com DataFrame vazio
    fake_ti = FakeTI(None)
    called = {}
    def fake_save(df, stage_dir, filename):
        called['df'] = df.copy()
        called['stage_dir'] = stage_dir
        called['filename'] = filename
    monkeypatch.setattr(pt, 'save_dataframe', fake_save)
    # Chama com batch_id_key 'uc_generation'
    pt.task_wait_and_process_batch_generic('uc_generation', tmp_path, pt.GENERATED_UCS_RAW, ti=fake_ti)
    # Verifica chamada
    assert 'df' in called
    df = called['df']
    # Deve ter colunas conforme DEFAULT_OUTPUT_COLUMNS
    expected_cols = pt.DEFAULT_OUTPUT_COLUMNS.get(pt.GENERATED_UCS_RAW)
    assert list(df.columns) == expected_cols
    assert called['stage_dir'] == tmp_path
    assert called['filename'] == pt.GENERATED_UCS_RAW

def test_task_wait_and_process_batch_success(monkeypatch, tmp_path):
    # Com batch_id e processamento bem-sucedido
    fake_ti = FakeTI('batch123')
    # Monkeypatch xcom_pull via fake_ti
    # Monkeypatch check_batch_status e process_batch_results
    monkeypatch.setattr(pt, 'check_batch_status', lambda batch_id: ('completed', 'out1', None))
    called = {}
    def fake_process(batch_id, output_file_id, error_file_id, stage_dir, filename):
        called['args'] = (batch_id, output_file_id, error_file_id, stage_dir, filename)
        return True
    monkeypatch.setattr(pt, 'process_batch_results', fake_process)
    # Chama a tarefa; não deve lançar
    result = pt.task_wait_and_process_batch_generic('uc_generation', tmp_path, pt.GENERATED_UCS_RAW, ti=fake_ti)
    # Deve ter chamado process_batch_results com valores corretos
    assert called['args'][0] == 'batch123'
    assert called['args'][1] == 'out1'
    assert called['args'][2] is None
    assert called['args'][3] == tmp_path
    assert called['args'][4] == pt.GENERATED_UCS_RAW
    # Sucesso não retorna valor
    assert result is None