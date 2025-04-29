import json
import pandas as pd
import pathlib
import pytest

import scripts.pipeline_tasks as pt
from scripts.pipeline_tasks import process_batch_results, GENERATED_UCS_RAW, UC_EVALUATIONS_RAW
import scripts.llm_client as llm_client

def test_process_batch_results_no_client(monkeypatch):
    # Sem cliente injetado -> ValueError
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', None)
    with pytest.raises(ValueError):
        process_batch_results('b', 'out', None, pathlib.Path('/tmp'), GENERATED_UCS_RAW)

def test_process_batch_results_success(tmp_path, monkeypatch, dummy_client):
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
    # Usa dummy_client fornecido por fixture
    dummy_client.files.content_map = content_map
    # Injeção do cliente dummy para leitura de arquivo
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
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

def test_process_batch_results_all_errors(tmp_path, monkeypatch, dummy_client):
    # Linha com erro no batch
    outer = {"custom_id": "c", "error": {"message": "err"}}
    line = json.dumps(outer)
    content_map = {'out2': line.encode('utf-8')}
    dummy_client.files.content_map = content_map
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
    # save_dataframe não deve ser chamado; se chamado, falha
    def bad_save(*args, **kwargs):
        pytest.skip("save_dataframe should not be called on all errors")
    monkeypatch.setattr(pt, 'save_dataframe', bad_save)
    ok = process_batch_results('batch2', 'out2', None, tmp_path, UC_EVALUATIONS_RAW)
    assert ok is False
    # monkeypatch fixture reverts changes automatically

def test_process_batch_results_with_code_fence(tmp_path, monkeypatch, dummy_client):
    # Verifica remoção de blocos de código ```json e parsing correto
    # Prepara inner JSON com 6 unidades
    units = [{"bloom_level": "Lembrar", "uc_text": f"txt{i}"} for i in range(6)]
    inner = {"generated_units": units}
    inner_str = json.dumps(inner)
    # Encapsula em code fence
    fenced = f"```json\n{inner_str}\n```"
    # Outer JSON line com message content fenced
    outer = {
        "custom_id": "gen_req_o2_0",
        "response": {"status_code": 200, "body": {"choices": [{"message": {"content": fenced}}]}}
    }
    line = json.dumps(outer)
    content_map = {'outf': line.encode('utf-8')}
    dummy_client.files.content_map = content_map
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
    # Intercepta save_dataframe
    saved = {}
    def fake_save(df, stage_dir, filename):
        saved['df'] = df.copy()
    monkeypatch.setattr(pt, 'save_dataframe', fake_save)
    ok = process_batch_results('batchX', 'outf', None, tmp_path, GENERATED_UCS_RAW)
    assert ok is True
    df = saved.get('df')
    assert df is not None and len(df) == 6
    # Conteúdo do DataFrame confere bloom_level e uc_text
    assert all(df['bloom_level'] == 'Lembrar')
    expected_texts = {f"txt{i}" for i in range(6)}
    assert set(df['uc_text']) == expected_texts
    # monkeypatch fixture reverts changes automatically

def test_process_batch_results_plain_code_fence(tmp_path, monkeypatch, dummy_client):
    # Testa triple backticks sem 'json' prefix
    units = [{"bloom_level": "Entender", "uc_text": f"x{i}"} for i in range(6)]
    inner = {"generated_units": units}
    inner_str = json.dumps(inner)
    fenced = f"```\n{inner_str}\n```"
    outer = {"custom_id": "id", "response": {"status_code": 200,
             "body": {"choices": [{"message": {"content": fenced}}]}}}
    line = json.dumps(outer)
    dummy_client.files.content_map = {'f': line.encode()}
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
    saved = {}
    monkeypatch.setattr(pt, 'save_dataframe', lambda df, sd, fn: saved.update(df=df))
    ok = process_batch_results('b', 'f', None, tmp_path, GENERATED_UCS_RAW)
    assert ok
    df = saved.get('df')
    assert df is not None and len(df) == 6
    assert all(df['bloom_level'] == 'Entender')

def test_process_batch_results_mixed_valid_and_invalid(tmp_path, monkeypatch, dummy_client):
    # Uma linha válida, outra com JSON interno inválido
    valid_units = [{"bloom_level": "Aplicar", "uc_text": "ok"} for _ in range(6)]
    valid_inner = {"generated_units": valid_units}
    valid_str = json.dumps(valid_inner)
    outer_valid = {"custom_id": "gen_req_o3_0", "response": {"status_code": 200,
                    "body": {"choices": [{"message": {"content": valid_str}}]}}}
    # invalid inner JSON
    outer_invalid = {"custom_id": "gen_req_o3_1", "response": {"status_code": 200,
                      "body": {"choices": [{"message": {"content": "{not a json}"}}]}}}
    content = '\n'.join([json.dumps(outer_valid), json.dumps(outer_invalid)])
    dummy_client.files.content_map = {'mix': content.encode()}
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
    # Capture save_dataframe
    saved = {}
    monkeypatch.setattr(pt, 'save_dataframe', lambda df, sd, fn: saved.update(df=df))
    ok = process_batch_results('batch', 'mix', None, tmp_path, GENERATED_UCS_RAW)
    # Mesmo com erro interno, deve processar a linha válida e retornar True
    assert ok
    df = saved.get('df')
    assert df is not None and len(df) == 6

def test_process_batch_results_difficulty_success(tmp_path, monkeypatch, dummy_client):
    # Testa processamento de avaliações de dificuldade (UC_EVALUATIONS_RAW)
    assessments = [
        {"uc_id": "u1", "difficulty_score": 10, "justification": "a"},
        {"uc_id": "u2", "difficulty_score": 20, "justification": "b"},
    ]
    inner = {"difficulty_assessments": assessments}
    inner_str = json.dumps(inner)
    # sem code fence
    outer = {"custom_id": "diff_0", "response": {"status_code": 200,
             "body": {"choices": [{"message": {"content": inner_str}}]}}}
    content = json.dumps(outer)
    dummy_client.files.content_map = {'dout': content.encode()}
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
    saved = {}
    def fake_save(df, sd, fn): saved.update(df=df)
    monkeypatch.setattr(pt, 'save_dataframe', fake_save)
    ok = process_batch_results('b', 'dout', None, tmp_path, UC_EVALUATIONS_RAW)
    assert ok
    df = saved.get('df')
    # Deve ter duas linhas correspondentes a assessments
    assert df is not None and len(df) == 2
    assert set(df['uc_id']) == {"u1", "u2"}
    assert set(df['difficulty_score']) == {10, 20}
    assert set(df['justification']) == {"a", "b"}

def test_process_batch_results_difficulty_missing_list(tmp_path, monkeypatch, dummy_client):
    # JSON interno sem 'difficulty_assessments'
    inner = {}
    inner_str = json.dumps(inner)
    outer = {"custom_id": "diff_1", "response": {"status_code": 200,
             "body": {"choices": [{"message": {"content": inner_str}}]}}}
    content = json.dumps(outer)
    dummy_client.files.content_map = {'d2': content.encode()}
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', dummy_client)
    # save_dataframe não deve ser chamado, retorno False
    monkeypatch.setattr(pt, 'save_dataframe', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Não deve salvar")))
    # Quando não há assessments mas sem erros, retorna True e não salva
    ok = process_batch_results('b', 'd2', None, tmp_path, UC_EVALUATIONS_RAW)
    assert ok is True