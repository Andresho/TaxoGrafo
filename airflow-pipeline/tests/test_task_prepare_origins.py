import pandas as pd
import pytest

import scripts.pipeline_tasks as pt

def test_task_prepare_origins_success(monkeypatch):
    # Mocka load_dataframe: entidades retornam DataFrame, relatórios None
    df_entities = pd.DataFrame([
        {"id": "e1", "title": "Ent1", "description": "Desc1", "frequency": 5, "degree": 3, "type": "person"}
    ])
    def fake_load_dataframe(stage_dir, name):
        if name == 'entities':
            return df_entities
        if name == 'community_reports':
            return None
        return None
    monkeypatch.setattr(pt, 'load_dataframe', fake_load_dataframe)
    # Captura chamada de save_dataframe
    saved = {}
    def fake_save_dataframe(df, stage_dir, filename):
        # Copia DataFrame e parâmetros
        saved['df'] = df.copy()
        saved['stage_dir'] = stage_dir
        saved['filename'] = filename
    monkeypatch.setattr(pt, 'save_dataframe', fake_save_dataframe)
    # Executa a tarefa
    pt.task_prepare_origins()
    # Verifica que save_dataframe foi chamado corretamente
    assert 'df' in saved
    df_out = saved['df']
    # Deve ter uma linha com os valores preparados
    assert df_out.shape[0] == 1
    row = df_out.iloc[0]
    assert row['origin_id'] == 'e1'
    assert row['origin_type'] == 'entity'
    assert row['title'] == 'Ent1'
    assert row['context'] == 'Desc1'
    assert row['frequency'] == 5
    assert row['degree'] == 3
    assert row['entity_type'] == 'person'
    # Verifica parâmetros de save
    assert saved['filename'] == 'uc_origins'
    assert saved['stage_dir'] == pt.stage1_dir

def test_task_prepare_origins_failure_raises(monkeypatch):
    # Ambos load_dataframe retornam None -> erro
    monkeypatch.setattr(pt, 'load_dataframe', lambda dir, name: None)
    # Se save_dataframe for chamado, falha o teste
    monkeypatch.setattr(pt, 'save_dataframe', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('save_dataframe should not be called')))
    with pytest.raises(ValueError) as excinfo:
        pt.task_prepare_origins()
    # Mensagem de entrada não carregada
    assert 'Inputs entities/reports não carregados' in str(excinfo.value)