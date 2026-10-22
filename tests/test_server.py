import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import mlflow_server as srv


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Each test gets a fresh audit log and an empty token store."""
    monkeypatch.setattr(srv, "AUDIT_LOG", tmp_path / "audit.jsonl")
    srv.PENDING.clear()


def test_read_only_blocks_writes(monkeypatch):
    monkeypatch.setattr(srv, "READ_ONLY", True)
    assert srv.trigger_retrain()["status"] == "BLOCKED"


def test_preview_issues_token_and_trains_nothing(monkeypatch):
    monkeypatch.setattr(srv, "READ_ONLY", False)
    result = srv.trigger_retrain(n_estimators=50)
    assert result["status"] == "CONFIRMATION_REQUIRED"
    assert result["confirm_token"] in srv.PENDING


def test_unknown_token_rejected(monkeypatch):
    monkeypatch.setattr(srv, "READ_ONLY", False)
    result = srv.trigger_retrain(n_estimators=50, confirm_token="nope1234")
    assert result["status"] == "REJECTED"


def test_token_cannot_be_used_with_different_params(monkeypatch):
    monkeypatch.setattr(srv, "READ_ONLY", False)
    token = srv.trigger_retrain(n_estimators=50)["confirm_token"]
    result = srv.trigger_retrain(n_estimators=60, confirm_token=token)
    assert result["status"] == "REJECTED"


def test_expired_token_rejected(monkeypatch):
    monkeypatch.setattr(srv, "READ_ONLY", False)
    token = srv.trigger_retrain(n_estimators=50)["confirm_token"]
    srv.PENDING[token]["expires"] = 0
    result = srv.trigger_retrain(n_estimators=50, confirm_token=token)
    assert result["status"] == "REJECTED"


def test_out_of_range_inputs_raise():
    with pytest.raises(ValueError):
        srv.trigger_retrain(n_estimators=100000)
    with pytest.raises(ValueError):
        srv.trigger_retrain(max_depth=0)


def test_audit_log_records_events(monkeypatch):
    monkeypatch.setattr(srv, "READ_ONLY", True)
    srv.trigger_retrain()
    assert "blocked_read_only" in srv.AUDIT_LOG.read_text()


def test_get_logs_redacts_tokens():
    srv.audit("preview_issued", params={}, token="secret123")
    result = srv.get_logs("audit", 5)
    assert "secret123" not in str(result)