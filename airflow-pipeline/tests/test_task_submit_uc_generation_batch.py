import json
import pandas as pd
import pytest

import scripts.pipeline_tasks as pt

def test_submit_uc_generation_batch_no_client(monkeypatch):
    # Cliente não inicializado
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', None)
    with pytest.raises(ValueError) as exc:
        pt.task_submit_uc_generation_batch()
    assert 'Cliente OpenAI não inicializado' in str(exc.value)

def test_submit_uc_generation_batch_empty(monkeypatch, tmp_path, caplog, dummy_client):
    # load_dataframe retorna None e DataFrame vazio -> retorna None
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', dummy_client)
    monkeypatch.setattr(pt, 'load_dataframe', lambda dir, name: None)
    rv = pt.task_submit_uc_generation_batch()
    assert rv is None
    # Testa caso DataFrame vazio
    monkeypatch.setattr(pt, 'load_dataframe', lambda dir, name: pd.DataFrame())
    rv2 = pt.task_submit_uc_generation_batch()
    assert rv2 is None

def test_submit_uc_generation_batch_success(monkeypatch, tmp_path, dummy_client):
    # Configura cliente dummy
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', dummy_client)
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
    # Verifica chamadas do dummy_client OpenAI
    assert dummy_client.files.last_purpose == 'batch'
    assert dummy_client.batches.last_kwargs.get('input_file_id') == 'file123'
    assert dummy_client.batches.last_kwargs.get('endpoint') == '/v1/chat/completions'