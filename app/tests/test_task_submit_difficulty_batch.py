import json
import pandas as pd
import pytest

import scripts.pipeline_tasks as pt
import scripts.llm_client as llm_client

def test_submit_difficulty_batch_no_client(monkeypatch):
    with pytest.raises(ValueError) as exc:
        pt.task_submit_difficulty_batch()
    # Deve utilizar get_llm_strategy e reportar erro de cliente não inicializado
    assert 'OpenAI client não inicializado' in str(exc.value)

def test_submit_difficulty_batch_empty(monkeypatch, dummy_client):
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
    # Sem UCs retorna None
    monkeypatch.setattr(pt, 'load_dataframe', lambda stage, name: None)
    rv = pt.task_submit_difficulty_batch()
    assert rv is None
    # DataFrame vazio
    monkeypatch.setattr(pt, 'load_dataframe', lambda stage, name: pd.DataFrame())
    rv2 = pt.task_submit_difficulty_batch()
    assert rv2 is None

def test_submit_difficulty_batch_success(monkeypatch, tmp_path, dummy_client):
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
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
    # DummyBatch returns id 'batch123'
    assert rv == 'batch123'
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
    # Verifica chamadas do cliente (dummy_client)
    assert dummy_client.files.last_purpose == 'batch'
    assert dummy_client.batches.last_kwargs.get('input_file_id') == 'file123'
    assert dummy_client.batches.last_kwargs.get('endpoint') == '/v1/chat/completions'