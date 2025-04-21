import sys
import pathlib
# Adiciona raiz do projeto (airflow-pipeline) ao PYTHONPATH para imports de scripts/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# --- Dummy Classes for Tests ---
import pytest
import scripts.pipeline_tasks as pt

class DummyContent:
    """Wraps bytes and provides a read() method."""
    def __init__(self, data_bytes):
        self._data = data_bytes
    def read(self):
        return self._data

class DummyFiles:
    """Simulates OpenAI file upload and download."""
    def __init__(self, content_map=None):
        self.last_file = None
        self.last_purpose = None
        # Mapping file_id -> bytes data for content()
        self.content_map = content_map or {}
    def create(self, file, purpose):
        self.last_file = file
        self.last_purpose = purpose
        # Return object with id attribute
        return type('F', (), {'id': 'file123'})
    def content(self, file_id):
        if file_id not in self.content_map:
            raise KeyError(f"Unknown file_id {file_id}")
        return DummyContent(self.content_map[file_id])

class DummyBatches:
    """Simulates OpenAI batch creation."""
    def __init__(self):
        self.last_args = None
        self.last_kwargs = None
    def create(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return type('B', (), {'id': 'batch123'})

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