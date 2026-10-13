"""API validation and restricted audit metadata endpoints."""
import pytest
from fastapi.testclient import TestClient
from agents.api import app

client = TestClient(app)


def test_root_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Operations Console" in response.text


def test_audit_rejects_nonfinite_primary_metric():
    payload = {"task_id": "T1", "target_identifier": "SAMPLE-1", "primary_metric": "nan"}
    assert client.post("/api/audit", json=payload).status_code == 422


def test_audit_log_requires_explicit_token_configuration(monkeypatch):
    monkeypatch.delenv("AUDIT_API_TOKEN", raising=False)
    assert client.get("/api/audit/logs").status_code == 503


def test_audit_log_requires_correct_bearer_token(monkeypatch):
    monkeypatch.setenv("AUDIT_API_TOKEN", "local-test-token")
    assert client.get("/api/audit/logs").status_code == 403
    assert client.get("/api/audit/logs", headers={"Authorization": "Bearer wrong"}).status_code == 403
    response = client.get("/api/audit/logs", headers={"Authorization": "Bearer local-test-token"})
    assert response.status_code == 200
    assert response.json()["verified"] is True


def test_mock_does_not_claim_undelivered_verification():
    from agents.llm_factory import LLMFactory
    assert "no external model" in LLMFactory.create("mock").invoke("hello")
    with pytest.raises(ValueError):
        LLMFactory.create("openai")
