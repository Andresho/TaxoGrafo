import sys
import pathlib
import json
import pandas as pd
import pytest

# Ajusta caminho para importar módulo pipeline_tasks
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import scripts.pipeline_tasks as pt

class DummyFiles:
    def __init__(self):
        self.last_file = None
        self.last_purpose = None
    def create(self, file, purpose):
        self.last_file = file
        self.last_purpose = purpose
        class FileObj:
            id = 'file_diff'
        return FileObj()

class DummyBatches:
    def __init__(self):
        self.last_kwargs = None
    def create(self, *args, **kwargs):
        self.last_kwargs = kwargs
        class BatchObj:
            id = 'batch_diff'
        return BatchObj()

class DummyClient:
    def __init__(self):
        self.files = DummyFiles()
        self.batches = DummyBatches()

@pytest.fixture(autouse=True)
def fixed_datetime(monkeypatch):
    # Controla timestamp para nome de arquivo
    import datetime
    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2020,1,1,0,0,0)
    monkeypatch.setattr(pt, 'datetime', datetime)
    monkeypatch.setattr(pt.datetime, 'datetime', FakeDateTime)
    yield

def test_submit_difficulty_batch_no_client(monkeypatch):
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', None)
    with pytest.raises(ValueError) as exc:
        pt.task_submit_difficulty_batch()
    assert 'Cliente OpenAI não inicializado' in str(exc.value)

def test_submit_difficulty_batch_empty(monkeypatch):
    client = DummyClient()
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', client)
    # Sem UCs retorna None
    monkeypatch.setattr(pt, 'load_dataframe', lambda stage, name: None)
    rv = pt.task_submit_difficulty_batch()
    assert rv is None
    # DataFrame vazio
    monkeypatch.setattr(pt, 'load_dataframe', lambda stage, name: pd.DataFrame())
    rv2 = pt.task_submit_difficulty_batch()
    assert rv2 is None

def test_submit_difficulty_batch_success(monkeypatch, tmp_path):
    client = DummyClient()
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', client)
    # DataFrame com 1 UC
    df = pd.DataFrame([
        {'uc_id': 'uA', 'bloom_level': 'Lembrar', 'uc_text': 'textA'}
    ])
    monkeypatch.setattr(pt, 'load_dataframe', lambda stage, name: df)
    # Prompt template com placeholder
    prompt_file = tmp_path / 'prompt_diff.txt'
    prompt_file.write_text('BEGIN\n{{BATCH_OF_UCS}}\nEND')
    monkeypatch.setattr(pt, 'PROMPT_UC_DIFFICULTY_FILE', prompt_file)
    # Usa tmp_path para batch dir
    monkeypatch.setattr(pt, 'BATCH_FILES_DIR', tmp_path)
    # Executa
    rv = pt.task_submit_difficulty_batch()
    assert rv == 'batch_diff'
    # Verifica arquivo gerado
    files = list(tmp_path.glob('uc_difficulty_batch_*.jsonl'))
    assert len(files) == 1
    lines = files[0].read_text(encoding='utf-8').splitlines()
    # Deve haver uma linha JSON
    assert len(lines) == 1
    obj = json.loads(lines[0])
    # custom_id tem prefixo diff_eval
    assert obj['custom_id'].startswith('diff_eval_Lembrar_0')
    assert obj['method'] == 'POST'
    # Verifica prompt interpolado
    body = obj['body']
    messages = body.get('messages', [])
    assert any('BEGIN' in m.get('content','') for m in messages)
    assert any('END' in m.get('content','') for m in messages)
    assert 'ID: uA' in body['messages'][1]['content']
    # Verifica chamadas do cliente
    assert client.files.last_purpose == 'batch'
    assert client.batches.last_kwargs.get('input_file_id') == 'file_diff'
    assert client.batches.last_kwargs.get('endpoint') == '/v1/chat/completions'