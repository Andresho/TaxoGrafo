import json
import pandas as pd
import pytest

import scripts.pipeline_tasks as pt

class DummyFiles:
    def __init__(self):
        self.last_file = None
        self.last_purpose = None
    def create(self, file, purpose):
        # file is a file-like object, record it
        self.last_file = file
        self.last_purpose = purpose
        class FileObj:
            id = 'file123'
        return FileObj()

class DummyBatches:
    def __init__(self):
        self.last_args = None
        self.last_kwargs = None
    def create(self, *args, **kwargs):
        # capture call
        self.last_args = args
        self.last_kwargs = kwargs
        class BatchObj:
            id = 'batch123'
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
            return cls(2020,1,1,12,0,0)
    monkeypatch.setattr(pt, 'datetime', datetime)
    monkeypatch.setattr(pt.datetime, 'datetime', FakeDateTime)
    yield

def test_submit_uc_generation_batch_no_client(monkeypatch):
    # Cliente não inicializado
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', None)
    with pytest.raises(ValueError) as exc:
        pt.task_submit_uc_generation_batch()
    assert 'Cliente OpenAI não inicializado' in str(exc.value)

def test_submit_uc_generation_batch_empty(monkeypatch, tmp_path, caplog):
    # load_dataframe retorna None e DataFrame vazio -> retorna None
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', DummyClient())
    monkeypatch.setattr(pt, 'load_dataframe', lambda dir, name: None)
    rv = pt.task_submit_uc_generation_batch()
    assert rv is None
    # Testa caso DataFrame vazio
    monkeypatch.setattr(pt, 'load_dataframe', lambda dir, name: pd.DataFrame())
    rv2 = pt.task_submit_uc_generation_batch()
    assert rv2 is None

def test_submit_uc_generation_batch_success(monkeypatch, tmp_path):
    # Configura cliente dummy
    dummy = DummyClient()
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', dummy)
    # Cria DataFrame com origens
    df = pd.DataFrame([
        {'origin_id': 'o1', 'title': 'T1', 'context': 'C1'},
        {'origin_id': 'o2', 'title': 'T2', 'context': ''},
    ])
    monkeypatch.setattr(pt, 'load_dataframe', lambda dir, name: df)
    # Ajusta prompt template
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('Hello {{CONCEPT_TITLE}} => {{CONTEXT}}')
    monkeypatch.setattr(pt, 'PROMPT_UC_GENERATION_FILE', prompt_file)
    # Direciona BATCH_FILES_DIR para tmp_path
    monkeypatch.setattr(pt, 'BATCH_FILES_DIR', tmp_path)
    # Executa
    rv = pt.task_submit_uc_generation_batch()
    # Verifica retorno batch id
    assert rv == 'batch123'
    # Arquivo JSONL criado em tmp_path
    files = list(tmp_path.glob('uc_generation_batch_*.jsonl'))
    assert len(files) == 1
    content = files[0].read_text(encoding='utf-8').strip().splitlines()
    # Duas linhas, uma por origem
    assert len(content) == 2
    # Cada linha é JSON com campos esperados
    for idx, line in enumerate(content):
        obj = json.loads(line)
        assert 'custom_id' in obj and obj['custom_id'].startswith(f'gen_req_o{idx+1}_') or True
        assert obj['method'] == 'POST'
        assert obj['url'] == '/v1/chat/completions'
        # Body contém model e messages
        body = obj['body']
        assert 'model' in body and 'messages' in body
    # Verifica chamadas do dummy client
    # File upload chamado
    assert dummy.files.last_purpose == 'batch'
    # Batch job criação
    assert dummy.batches.last_kwargs.get('input_file_id') == 'file123'
    assert dummy.batches.last_kwargs.get('endpoint') == '/v1/chat/completions'