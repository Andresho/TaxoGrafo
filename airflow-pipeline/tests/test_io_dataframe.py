import pandas as pd
import pytest

from scripts.pipeline_tasks import save_dataframe, load_dataframe

def test_save_and_load_dataframe_roundtrip(tmp_path):
    # Cria DataFrame de teste
    df = pd.DataFrame([
        {"col1": 1, "col2": "a"},
        {"col1": 2, "col2": "b"},
    ])
    stage_dir = tmp_path / "stage"
    # Salva e carrega
    save_dataframe(df, stage_dir, "mydata")
    loaded = load_dataframe(stage_dir, "mydata")
    # Deve retornar DataFrame equivalente
    assert loaded is not None
    # Ordenação e índice devem corresponder
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded.reset_index(drop=True))

def test_load_dataframe_missing_file(tmp_path, caplog):
    # Diretório existe mas arquivo não
    stage_dir = tmp_path / "stage_missing"
    stage_dir.mkdir()
    # Tentativa de carregar retorna None e log de erro
    result = load_dataframe(stage_dir, "nonexistent")
    assert result is None
    assert "Arquivo de input não encontrado" in caplog.text

def test_save_dataframe_creates_directory(tmp_path):
    # Se o diretório não existir, save_dataframe deve criá-lo
    df = pd.DataFrame([{"x": 10}])
    stage_dir = tmp_path / "new" / "nested"
    # Diretório não existe antes
    assert not stage_dir.exists()
    save_dataframe(df, stage_dir, "datafile")
    # Agora o diretório e arquivo devem existir
    assert stage_dir.exists() and stage_dir.is_dir()
    file_path = stage_dir / "datafile.parquet"
    assert file_path.is_file()
    # Carrega sem erro
    loaded = load_dataframe(stage_dir, "datafile")
    assert loaded is not None and loaded.shape == df.shape