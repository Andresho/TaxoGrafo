import pandas as pd
from pathlib import Path
import json
import logging
from typing import Optional, List, Dict, Any

class DataLake:
    """Fachada para operações de I/O: Parquet, JSON, JSONL."""
    @staticmethod
    def save_parquet(df: pd.DataFrame, stage_dir: Path, filename: str) -> None:
        try:
            stage_dir.mkdir(parents=True, exist_ok=True)
            output_path = stage_dir / f"{filename}.parquet"
            df.to_parquet(output_path, index=False)
            logging.info(f"Salvo {len(df)} linhas em {output_path}")
        except Exception:
            logging.exception(f"Falha ao salvar Parquet em {stage_dir}/{filename}.parquet")
            raise

    @staticmethod
    def load_parquet(stage_dir: Path, filename: str) -> Optional[pd.DataFrame]:
        file_path = stage_dir / f"{filename}.parquet"
        if not file_path.is_file():
            logging.error(f"Arquivo de input não encontrado: {file_path}")
            return None
        try:
            df = pd.read_parquet(file_path)
            logging.info(f"Carregado {len(df)} linhas de {file_path}")
            return df
        except Exception:
            logging.exception(f"Falha ao carregar Parquet de {file_path}")
            return None

    @staticmethod
    def write_json(data: Any, file_path: Path, indent: int = 2) -> None:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            logging.info(f"Salvo JSON em {file_path}")
        except Exception:
            logging.exception(f"Falha ao salvar JSON em {file_path}")
            raise

    @staticmethod
    def load_json(file_path: Path) -> Optional[Any]:
        if not file_path.is_file():
            logging.error(f"JSON input não encontrado: {file_path}")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Carregado JSON de {file_path}")
            return data
        except Exception:
            logging.exception(f"Falha ao carregar JSON de {file_path}")
            return None

    @staticmethod
    def write_jsonl(records: List[Dict[str, Any]], file_path: Path) -> None:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            logging.info(f"Salvo JSONL ({len(records)} linhas) em {file_path}")
        except Exception:
            logging.exception(f"Falha ao salvar JSONL em {file_path}")
            raise

    @staticmethod
    def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not file_path.is_file():
            logging.error(f"JSONL input não encontrado: {file_path}")
            return records
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logging.warning(f"Falha parsing JSONL linha: {line[:100]}...")
            logging.info(f"Carregado JSONL ({len(records)} linhas) de {file_path}")
        except Exception:
            logging.exception(f"Falha ao ler JSONL de {file_path}")
        return records