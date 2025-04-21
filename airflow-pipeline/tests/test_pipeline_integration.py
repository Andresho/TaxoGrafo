import pathlib
import pandas as pd
import json
import uuid
import pytest


import scripts.pipeline_tasks as pt

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

class DummyGenFiles:
    def __init__(self):
        self.id = 'gen_file'
    def create(self, file, purpose):
        return type('F', (), {'id': self.id})
    def content(self, file_id):
        # Não usado em geração
        raise RuntimeError('Unexpected content call')

class DummyGenBatches:
    def __init__(self): pass
    def create(self, *args, **kwargs):
        return type('B', (), {'id': 'gen_batch'})

class DummyGenClient:
    def __init__(self):
        self.files = DummyGenFiles()
        self.batches = DummyGenBatches()

class DummyDiffFiles:
    def __init__(self, work_dir, raw_generated_path):
        self.work_dir = work_dir
        # Prepare content map for difficulty based on generated units
        # Load generated UCs to get uc_ids
        df = pd.read_parquet(raw_generated_path)
        # Create difficulty assessments for each UC
        assess = []
        for uc_id in df['uc_id']:
            assess.append({'uc_id': uc_id, 'difficulty_score': 50, 'justification': 'auto'})
        inner = {'difficulty_assessments': assess}
        inner_str = json.dumps(inner)
        line = json.dumps({'custom_id': 'diff_eval_Lembrar_0', 'response': {'status_code': 200, 'body': {'choices': [{'message': {'content': inner_str}}]}}})
        self.data = line.encode('utf-8')
    def create(self, file, purpose):
        # For upload create
        return type('F', (), {'id': 'diff_file'})
    def content(self, file_id):
        # Retorna objeto com método read() que devolve os bytes preparados
        class C:
            def __init__(inner): inner._data = self.data
            def read(inner): return inner._data
        return C()

class DummyDiffBatches:
    def create(self, *args, **kwargs):
        return type('B', (), {'id': 'diff_batch'})

class DummyDiffClient:
    def __init__(self, work_dir):
        # Path to generated UCs raw parquet
        raw_generated = work_dir / '2_generated_ucs' / (pt.GENERATED_UCS_RAW + '.parquet')
        self.files = DummyDiffFiles(work_dir, raw_generated)
        self.batches = DummyDiffBatches()

def test_full_pipeline_integration(setup_dirs, monkeypatch):
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
    # 3) Run generation submission and processing
    # Monkeypatch OPENAI_CLIENT to generation dummy
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', DummyGenClient())
    gen_batch = pt.task_submit_uc_generation_batch()
    assert gen_batch == 'gen_batch'
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
    # Monkeypatch files.content to return this generation line
    class DummyGenContent:
        def __init__(self, data): self._data = data
        def read(self): return self._data
    # Override content method on files
    monkeypatch.setattr(pt.OPENAI_CLIENT.files, 'content', lambda fid: DummyGenContent(line.encode('utf-8')))
    # Process generation
    ok_gen = pt.process_batch_results(gen_batch, 'gen_file', None, work_dir / '2_generated_ucs', pt.GENERATED_UCS_RAW)
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
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', DummyDiffClient(work_dir))
    diff_batch = pt.task_submit_difficulty_batch()
    assert diff_batch == 'diff_batch'
    ok_diff = pt.process_batch_results(diff_batch, 'diff_file', None, work_dir / '4_difficulty_evals', pt.UC_EVALUATIONS_RAW)
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