import pathlib
import pandas as pd
import json
import uuid
import pytest

import scripts.pipeline_tasks as pt
import scripts.llm_client as llm_client

@pytest.fixture
def setup_dirs(tmp_path, monkeypatch):
    # Cria estruturas de input e work em tmp_path
    input_dir = tmp_path / 'input'
    work_dir = tmp_path / 'work'
    for d in (input_dir, work_dir):
        d.mkdir()
    # Monkeypatch dos diretórios no módulo
    monkeypatch.setattr(pt, 'BASE_INPUT_DIR', input_dir)
    monkeypatch.setattr(pt, 'PIPELINE_WORK_DIR', work_dir)
    monkeypatch.setattr(pt, 'BATCH_FILES_DIR', work_dir / 'batch_files')
    monkeypatch.setattr(pt, 'stage1_dir', work_dir / '1_origins')
    monkeypatch.setattr(pt, 'stage2_output_ucs_dir', work_dir / '2_generated_ucs')
    monkeypatch.setattr(pt, 'stage3_dir', work_dir / '3_relationships')
    monkeypatch.setattr(pt, 'stage4_input_batch_dir', work_dir / 'batch_files')
    monkeypatch.setattr(pt, 'stage4_output_eval_dir', work_dir / '4_difficulty_evals')
    monkeypatch.setattr(pt, 'stage5_dir', work_dir / '5_final_outputs')
    # Cria prompt templates
    gen_prompt = tmp_path / 'prompt_gen.txt'
    diff_prompt = tmp_path / 'prompt_diff.txt'
    gen_prompt.write_text('Generate UC for {{CONCEPT_TITLE}}: {{CONTEXT}}')
    diff_prompt.write_text('Evaluate difficulty:\n{{BATCH_OF_UCS}}')
    monkeypatch.setattr(pt, 'PROMPT_UC_GENERATION_FILE', gen_prompt)
    monkeypatch.setattr(pt, 'PROMPT_UC_DIFFICULTY_FILE', diff_prompt)
    return input_dir, work_dir

def test_full_pipeline_integration(setup_dirs, monkeypatch, dummy_client):
    input_dir, work_dir = setup_dirs
    # 1) Cria arquivos de input
    # Entidades: uma entidade simples
    df_entities = pd.DataFrame([
        {'id': 'ent1', 'title': 'Entity1', 'description': 'Desc', 'frequency': 1, 'degree': 1, 'type': 'unknown'}
    ])
    df_reports = pd.DataFrame([], columns=['id', 'community', 'title', 'summary', 'level'])
    df_rels = pd.DataFrame([], columns=['source', 'target', 'weight', 'description'])
    df_entities.to_parquet(input_dir / 'entities.parquet', index=False)
    df_reports.to_parquet(input_dir / 'community_reports.parquet', index=False)
    df_rels.to_parquet(input_dir / 'relationships.parquet', index=False)

    # 2) Run prepare_origins
    pt.task_prepare_origins()
    # Verifica arquivo intermediário
    origins_path = work_dir / '1_origins' / 'uc_origins.parquet'
    assert origins_path.exists()
    # 3) Run generation submission e processing
    # Injeção de cliente dummy para geração
    client_gen = dummy_client
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', client_gen)
    gen_batch = pt.task_submit_uc_generation_batch()
    # DummyBatches.create retorna id 'batch123'
    assert gen_batch == 'batch123'
    # Prepare dummy generation result for processing
    # Build JSONL with 6 units for the single origin
    origins = pd.read_parquet(origins_path).to_dict('records')
    units = []
    for o in origins:
        for i in range(6):
            units.append({'bloom_level': 'Lembrar', 'uc_text': f"UC_{o['origin_id']}_{i}"})
    inner = {'generated_units': units}
    data_str = json.dumps(inner)
    line = json.dumps({'custom_id': f"gen_req_{origins[0]['origin_id']}_0", 'response': {'status_code': 200, 'body': {'choices': [{'message': {'content': data_str}}]}}})
    # Setup conteúdo de retorno para arquivo gerado (file123)
    file_id = client_gen.files.last_file_obj.id
    client_gen.files.content_map = {file_id: line.encode('utf-8')}
    # Processa geração
    ok_gen = pt.process_batch_results(gen_batch, file_id, None, work_dir / '2_generated_ucs', pt.GENERATED_UCS_RAW)
    assert ok_gen
    gen_path = work_dir / '2_generated_ucs' / 'generated_ucs_raw.parquet'
    assert gen_path.exists()

    # 4) Define relationships
    # Use empty relationships -> only REQUIRES within same origin
    pt.task_define_relationships()
    # Verifica arquivo intermediário de relações
    rels_file = work_dir / '3_relationships' / (pt.REL_INTERMEDIATE + '.parquet')
    assert rels_file.exists()

    # 5) Submit and process difficulty
    # Monkeypatch LLM client para usar cliente de dificuldade customizado
    # Prepara lista de assessments
    df_gen = pd.read_parquet(work_dir / '2_generated_ucs' / (pt.GENERATED_UCS_RAW + '.parquet'))
    assess = [{"uc_id": uc, "difficulty_score": 50, "justification": "auto"} for uc in df_gen['uc_id']]
    inner_diff = {"difficulty_assessments": assess}
    diff_line = json.dumps({"custom_id": "diff_eval_Lembrar_0",
                            "response": {"status_code": 200,
                                         "body": {"choices": [{"message": {"content": json.dumps(inner_diff)}}]}}})
    # Novo cliente para dificuldade
    ClientClass = type(client_gen)
    client_diff = ClientClass({file_id: diff_line.encode('utf-8')})
    # Monkeypatcha o cliente bruto para as chamadas de dificuldade
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', client_diff)
    diff_batch = pt.task_submit_difficulty_batch()
    assert diff_batch == 'batch123'
    ok_diff = pt.process_batch_results(diff_batch, file_id, None, work_dir / '4_difficulty_evals', pt.UC_EVALUATIONS_RAW)
    assert ok_diff
    diff_path = work_dir / '4_difficulty_evals' / (pt.UC_EVALUATIONS_RAW + '.parquet')
    assert diff_path.exists()

    # 6) Finalize outputs
    pt.task_finalize_outputs()
    # Verifica arquivos finais
    final_ucs = work_dir / '5_final_outputs' / (pt.FINAL_UC_FILE + '.parquet')
    final_rels = work_dir / '5_final_outputs' / (pt.FINAL_REL_FILE + '.parquet')
    assert final_ucs.exists()
    assert final_rels.exists()
    # Carrega e valida schema mínimo
    df_final_ucs = pd.read_parquet(final_ucs)
    # Deve conter colunas de UC gerada + dificuldade
    assert 'uc_id' in df_final_ucs.columns
    assert 'difficulty_score' in df_final_ucs.columns
    df_final_rels = pd.read_parquet(final_rels)
    # Como só had REQUIRES, pode estar vazio ou ter colunas mínimas
    assert set(df_final_rels.columns) >= {'source', 'target', 'type'}