import pytest

import scripts.pipeline_tasks as pt
from scripts.pipeline_tasks import check_batch_status

class DummyBatchSuccess:
    def __init__(self):
        self.status = 'completed'
        self.output_file_id = 'out123'
        self.error_file_id = 'err123'

class DummyClientSuccess:
    def __init__(self):
        # Simula namespace batches
        self.batches = self
    def retrieve(self, batch_id):
        return DummyBatchSuccess()

class DummyClientFailure:
    def __init__(self):
        self.batches = self
    def retrieve(self, batch_id):
        raise RuntimeError('API failure')

def test_check_batch_status_success(monkeypatch):
    # Cliente válido, retrieve bem-sucedido
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', DummyClientSuccess())
    status, out_id, err_id = check_batch_status('batch1')
    assert status == 'completed'
    assert out_id == 'out123'
    assert err_id == 'err123'

def test_check_batch_status_api_error(monkeypatch, caplog):
    # Cliente lança exceção -> retorna API_ERROR
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', DummyClientFailure())
    status, out_id, err_id = check_batch_status('batch2')
    assert status == 'API_ERROR'
    assert out_id is None and err_id is None
    assert 'Erro ao verificar status do batch' in caplog.text

def test_check_batch_status_no_client(monkeypatch):
    # OPENAI_CLIENT None -> ValueError
    monkeypatch.setattr(pt, 'OPENAI_CLIENT', None)
    with pytest.raises(ValueError) as excinfo:
        check_batch_status('batch3')
    assert 'OpenAI client não inicializado' in str(excinfo.value)