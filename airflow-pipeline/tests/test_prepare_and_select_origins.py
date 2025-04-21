import pandas as pd
import pytest

from scripts.pipeline_tasks import prepare_uc_origins, _select_origins_for_testing, _get_sort_key

def make_df(cols, rows):
    # Conveniência para criar DataFrame com colunas faltantes
    return pd.DataFrame(rows, columns=cols)

def test_prepare_uc_origins_basic_entity_and_report():
    # DataFrame de entidades com valores completos
    entities = pd.DataFrame([
        {"id": "e1", "title": "Ent1", "description": "Desc1", "frequency": 3, "degree": 2, "type": "person"},
        {"id": "e2", "title": "Ent2", "description": None, "frequency": None, "degree": None, "type": None},
    ])
    # DataFrame de relatórios de comunidade
    reports = pd.DataFrame([
        {"id": "r1", "community": "c1", "title": "Rep1", "summary": "Sum1", "level": 1},
        {"id": "r2", "community": "c2", "title": "Rep2", "summary": None, "level": None},
    ])
    origins = prepare_uc_origins(entities, reports)
    # Deve combinar duas entidades e dois relatórios
    assert len(origins) == 4
    # Verifica campos padrão e coerência de tipos
    ent1 = [o for o in origins if o['origin_id'] == 'e1'][0]
    assert ent1['origin_type'] == 'entity'
    assert ent1['frequency'] == 3 and ent1['degree'] == 2 and ent1['entity_type'] == 'person'
    ent2 = [o for o in origins if o['origin_id'] == 'e2'][0]
    # Valores nulos devem virar 0 ou empty string
    assert ent2['frequency'] == 0 and ent2['degree'] == 0 and ent2['entity_type'] == 'unknown'
    rep1 = [o for o in origins if o['origin_id'] == 'r1'][0]
    assert rep1['origin_type'] == 'community_report' and rep1['community_human_id'] == 'c1'
    rep2 = [o for o in origins if o['origin_id'] == 'r2'][0]
    # summary None vira string vazia, level None vira 99
    assert rep2['context'] == '' and rep2['level'] == 99

def test_prepare_uc_origins_missing_columns():
    # Colunas faltantes em entidades e relatórios devem gerar apenas warnings e não levantar
    entities = make_df(['id', 'title'], [{'id': 'e1', 'title': 'T'}])
    reports = make_df(['id'], [{'id': 'r1'}])
    origins = prepare_uc_origins(entities, reports)
    # Nenhuma origem válida extraída
    assert origins == []

def test_select_origins_for_testing_no_max(monkeypatch, tmp_path):
    # Quando total <= max, retorna tudo
    all_origins = [{'origin_id': f'o{i}', 'origin_type': 'entity', 'title': f'T{i}', 'frequency': 0, 'degree': 0, 'entity_type': 'unknown'} for i in range(3)]
    selected = _select_origins_for_testing(all_origins, tmp_path, max_origins=5)
    assert selected == all_origins

def test_select_origins_for_testing_no_entity(monkeypatch, tmp_path):
    # Sem entity_origins, deve ordenar por sort_key e pegar primeiros
    all_origins = [ {'origin_id': f'o{i}', 'origin_type': 'community_report', 'level': i} for i in range(5) ]
    selected = _select_origins_for_testing(all_origins, tmp_path, max_origins=2)
    # Espera os dois com menor level (maior score) após sort
    # sort_key community: (1, -(10000-level)) => maior level => menor -score
    expected = sorted(all_origins, key=_get_sort_key)[:2]
    assert selected == expected

def test_select_origins_for_testing_with_neighbors(monkeypatch, tmp_path):
    # Cria origens e simula relationships e entities para vizinhança
    hub = {'origin_id': 'hub', 'origin_type': 'entity', 'title': 'H', 'frequency': 0, 'degree': 0, 'entity_type': 'unknown'}
    nei1 = {'origin_id': 'n1', 'origin_type': 'entity', 'title': 'N1', 'frequency': 0, 'degree': 0, 'entity_type': 'unknown'}
    nei2 = {'origin_id': 'n2', 'origin_type': 'entity', 'title': 'N2', 'frequency': 0, 'degree': 0, 'entity_type': 'unknown'}
    other = {'origin_id': 'x', 'origin_type': 'entity', 'title': 'X', 'frequency': 0, 'degree': 0, 'entity_type': 'unknown'}
    all_origins = [hub, nei1, nei2, other]
    # Cria DataFrame de entities para mapping title->id
    entities_df = pd.DataFrame([
        {'id': 'hub', 'title': 'H'}, {'id': 'n1', 'title': 'N1'}, {'id': 'n2', 'title': 'N2'}, {'id': 'x', 'title': 'X'},
    ])
    # relationships apontando hub->nei1 e nei2->hub
    rels = pd.DataFrame([
        {'source': 'H', 'target': 'N1'},
        {'source': 'N2', 'target': 'H'},
        {'source': 'X', 'target': 'H'},
    ])
    # Monkeypatch load_dataframe para retornar nossos DataFrames
    import scripts.pipeline_tasks as pt
    def fake_load_dataframe(stage_dir, filename):
        if filename == 'entities':
            return entities_df
        if filename == 'relationships':
            return rels
        return None
    monkeypatch.setattr(pt, 'load_dataframe', fake_load_dataframe)
    # Seleciona max 3 origens
    selected = _select_origins_for_testing(all_origins, tmp_path, max_origins=3)
    # Deve incluir hub e dois vizinhos (seleciona 2 de {n1, n2, x})
    ids = [o['origin_id'] for o in selected]
    # Sempre inclui o hub
    assert 'hub' in ids
    # Seleciona exatamente 3 IDs (hub + 2 vizinhos)
    assert len(ids) == 3
    # Vizinhos devem ser do conjunto original
    neighbors = set(ids) - {'hub'}
    assert len(neighbors) == 2
    assert neighbors.issubset({'n1', 'n2', 'x'})