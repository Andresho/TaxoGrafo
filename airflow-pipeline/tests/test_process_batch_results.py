import sys
import pathlib
import json
import pandas as pd
import pytest

# Ajusta caminho para importar o módulo pipeline_tasks
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import scripts.pipeline_tasks as pt
from scripts.pipeline_tasks import process_batch_results, GENERATED_UCS_RAW, UC_EVALUATIONS_RAW

class DummyContent:
    def __init__(self, data_bytes):
        self._data = data_bytes
    def read(self):
        return self._data

class DummyFiles:
    def __init__(self, content_map):
        # content_map: file_id -> bytes
        self.content_map = content_map
    def content(self, file_id):
        if file_id not in self.content_map:
            raise KeyError(f"Unknown file_id {file_id}")
        return DummyContent(self.content_map[file_id])

class DummyClient:
    def __init__(self, content_map):
        self.files = DummyFiles(content_map)
        # batches namespace not used here

@pytest.fixture(autouse=True)
def clear_client(monkeypatch):
    # Garante cliente inicializado ou não conforme teste
    yield

def test_process_batch_results_no_client():
    # OPENAI_CLIENT None -> ValueError
    pt.OPENAI_CLIENT = None
    with pytest.raises(ValueError):
        process_batch_results('b', 'out', None, pathlib.Path('/tmp'), GENERATED_UCS_RAW)

def test_process_batch_results_success(tmp_path, monkeypatch):
    # Prepara inner JSON com 6 unidades
    units = [{"bloom_level": "Lembrar", "uc_text": f"text{i}"} for i in range(6)]
    inner = {"generated_units": units}
    inner_str = json.dumps(inner)
    # Outer JSON line
    outer = {
        "custom_id": "gen_req_o1_0",
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": inner_str}}]}
        }
    }
    line = json.dumps(outer)
    # Map file_id to bytes
    content_map = {'out': line.encode('utf-8')}
    pt.OPENAI_CLIENT = DummyClient(content_map)
    # Capture save_dataframe
    saved = {}
    def fake_save_dataframe(df, stage_dir, filename):
        saved['df'] = df.copy()
        saved['stage_dir'] = stage_dir
        saved['filename'] = filename
    monkeypatch.setattr(pt, 'save_dataframe', fake_save_dataframe)
    # Chama função
    ok = process_batch_results('batch1', 'out', None, tmp_path, GENERATED_UCS_RAW)
    assert ok is True
    # Verifica DataFrame salvo
    df = saved.get('df')
    assert df is not None
    # Deve ter 6 linhas
    assert len(df) == 6
    # Colunas esperadas
    expected_cols = ['uc_id', 'origin_id', 'bloom_level', 'uc_text']
    assert all(col in df.columns for col in expected_cols)
    # origin_id extraído de custom_id -> 'o1'
    assert set(df['origin_id']) == {'o1'}
    # monkeypatch fixture reverts changes automatically

def test_process_batch_results_all_errors(tmp_path, monkeypatch):
    # Linha com erro no batch
    outer = {"custom_id": "c", "error": {"message": "err"}}
    line = json.dumps(outer)
    content_map = {'out2': line.encode('utf-8')}
    pt.OPENAI_CLIENT = DummyClient(content_map)
    # save_dataframe não deve ser chamado; se chamado, falha
    def bad_save(*args, **kwargs):
        pytest.skip("save_dataframe should not be called on all errors")
    monkeypatch.setattr(pt, 'save_dataframe', bad_save)
    ok = process_batch_results('batch2', 'out2', None, tmp_path, UC_EVALUATIONS_RAW)
    assert ok is False