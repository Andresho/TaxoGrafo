import sys
import pathlib
# Adiciona raiz do projeto (airflow-pipeline) ao PYTHONPATH para imports de scripts/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# --- Dummy Classes for Tests ---
import pandas as pd
import pytest
import scripts.pipeline_tasks as pt

class DummyContent:
    """Wraps bytes and provides a read() method."""
    def __init__(self, data_bytes):
        self._data = data_bytes
    def read(self):
        return self._data

class DummyFile:
    """Represents a fake file resource with an ID."""
    def __init__(self, file_id: str):
        self.id = file_id

class DummyFiles:
    """Simulates OpenAI file upload and download."""
    def __init__(self, content_map=None):
        self.last_file = None
        self.last_purpose = None
        # Mapping file_id -> bytes data for content()
        self.content_map = content_map or {}
        self.last_file_obj = None
    def create(self, file, purpose):
        self.last_file = file
        self.last_purpose = purpose
        # Create and store a DummyFile instance
        file_obj = DummyFile('file123')
        self.last_file_obj = file_obj
        return file_obj
    def content(self, file_id):
        if file_id not in self.content_map:
            raise KeyError(f"Unknown file_id {file_id}")
        return DummyContent(self.content_map[file_id])

class DummyBatch:
    """Represents a fake batch job resource with an ID."""
    def __init__(self, batch_id: str):
        self.id = batch_id

class DummyBatches:
    """Simulates OpenAI batch creation."""
    def __init__(self):
        self.last_args = None
        self.last_kwargs = None
    def create(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return DummyBatch('batch123')

class DummyClient:
    """Combines DummyFiles and DummyBatches for a fake OpenAI client."""
    def __init__(self, content_map=None):
        self.files = DummyFiles(content_map)
        self.batches = DummyBatches()

import datetime
@pytest.fixture(autouse=True)
def fixed_datetime(monkeypatch):
    """Fixa a data/hora para testes que dependem de timestamp em nomes de arquivos."""
    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2020, 1, 1, 0, 0, 0)
    # Monkeypatch datetime.datetime in pipeline_tasks
    monkeypatch.setattr(pt, 'datetime', datetime)
    monkeypatch.setattr(pt.datetime, 'datetime', FakeDateTime)
    yield

@pytest.fixture
def dummy_client():
    """Retorna um DummyClient com content_map vazio por padrão."""
    return DummyClient()
    
@pytest.fixture
def sample_entities_df():
    """DataFrame mínimo de entidades para tests."""
    return pd.DataFrame([
        {"id": "e1", "title": "Ent1", "description": "Desc1", "frequency": 3, "degree": 2, "type": "person"},
        {"id": "e2", "title": "Ent2", "description": None, "frequency": None, "degree": None, "type": None},
    ])

@pytest.fixture
def sample_reports_df():
    """DataFrame mínimo de community_reports para tests."""
    return pd.DataFrame([
        {"id": "r1", "community": "c1", "title": "Rep1", "summary": "Sum1", "level": 1},
        {"id": "r2", "community": "c2", "title": "Rep2", "summary": None, "level": None},
    ])