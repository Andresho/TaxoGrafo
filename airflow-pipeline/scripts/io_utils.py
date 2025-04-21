import pandas as pd
from pathlib import Path
import logging
from typing import Optional

def save_dataframe(df: pd.DataFrame, stage_dir: Path, filename: str):
    """Salva um DataFrame em formato Parquet no diretório do estágio."""
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        output_path = stage_dir / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        logging.info(f"Salvo {len(df)} linhas em {output_path}")
    except Exception:
        logging.exception(f"Falha ao salvar DataFrame em {stage_dir}/{filename}.parquet")
        raise

def load_dataframe(stage_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    """Carrega um DataFrame Parquet de um diretório de estágio."""
    file_path = stage_dir / f"{filename}.parquet"
    if not file_path.is_file():
        logging.error(f"Arquivo de input não encontrado: {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Carregado {len(df)} linhas de {file_path}")
        return df
    except Exception:
        logging.exception(f"Falha ao carregar DataFrame de {file_path}")
        return None