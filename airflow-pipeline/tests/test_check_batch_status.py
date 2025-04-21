import pytest

import scripts.pipeline_tasks as pt
from scripts.pipeline_tasks import check_batch_status
import scripts.llm_client as llm_client

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

@pytest.mark.parametrize("client,expected_status,expected_out,expected_err,log_snippet", [
    (DummyClientSuccess(), 'completed', 'out123', 'err123', None),
    (DummyClientFailure(), 'API_ERROR', None, None, 'Erro ao verificar status do batch'),
])
def test_check_batch_status(monkeypatch, caplog, client, expected_status, expected_out, expected_err, log_snippet):
    # Configura o cliente e executa
    # Injeção do cliente dummy para llm_client
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', client)
    status, out_id, err_id = check_batch_status('batchX')
    assert status == expected_status
    assert out_id == expected_out
    assert err_id == expected_err
    # Verifica log quando aplicável
    if log_snippet:
        assert log_snippet in caplog.text

def test_check_batch_status_no_client(monkeypatch):
    # Sem cliente inicializado -> ValueError
    monkeypatch.setattr(llm_client, 'OPENAI_CLIENT', None)
    with pytest.raises(ValueError) as excinfo:
        check_batch_status('batchZ')
    assert 'OpenAI client não inicializado' in str(excinfo.value)