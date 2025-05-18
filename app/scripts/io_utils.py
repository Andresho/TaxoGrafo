import pandas as pd
from pathlib import Path
from typing import Optional
from app.scripts.data_lake import DataLake

def save_dataframe(df: pd.DataFrame, stage_dir: Path, filename: str):
    """Salva um DataFrame em formato Parquet no diretório do estágio via DataLake."""
    DataLake.save_parquet(df, stage_dir, filename)

def load_dataframe(stage_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    """Carrega um DataFrame Parquet de um diretório de estágio via DataLake."""
    return DataLake.load_parquet(stage_dir, filename)