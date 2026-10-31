"""Token persistence, secrecy and authenticated inference requests."""

from dataclasses import replace
import json

import httpx
from fastapi.testclient import TestClient

from llmhelper.client import LocalLLM
from llmhelper.config import load_config
from llmhelper import webui


def test_authenticated_requests(sandbox):
    cfg = replace(sandbox, server=replace(sandbox.server, api_token="test-token"))
    paths = []

    def respond(request):
        assert request.headers["Authorization"] == "Bearer test-token"
        paths.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test/model"}]})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = LocalLLM(cfg, client)
        assert llm.healthy()
        assert llm.models() == ["test/model"]
        assert llm.ask("hi", cfg.tier("fast")).text == "ok"
    assert paths == ["/v1/models", "/v1/models", "/v1/chat/completions"]
    assert "test-token" not in repr(cfg.server)


def test_token_settings(sandbox, tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(webui, "cfg", lambda: load_config(path))
    monkeypatch.setattr(webui, "config_invalidate", lambda: None)
    with TestClient(webui.app) as client:
        assert client.get("/api/settings").json()["api_token_set"] is False
        # Quotes and backslashes must survive TOML serialization.
        token = 'test-"token\\value'
        webui._probe_cache["value"] = {"stale": True}
        result = client.post("/api/settings", json={"api_token": token})
        assert result.status_code == 200
        assert webui._probe_cache["value"] is None
        assert load_config(path).server.api_token == token
        settings = client.get("/api/settings")
        assert settings.json()["api_token_set"] is True
        assert token not in json.dumps(settings.json())
        assert "api_token" not in settings.json()["editable"]
        assert token not in result.text
        assert client.post("/api/settings", json={"max_attempts": 4}).status_code == 200
        assert load_config(path).server.api_token == token
        assert client.post("/api/settings", json={"api_token": "replacement"}).status_code == 200
        assert load_config(path).server.api_token == "replacement"
        assert client.post("/api/settings", json={"api_token": "bad\ntoken"}).status_code == 400
        assert load_config(path).server.api_token == "replacement"
        assert client.post("/api/settings", json={"api_token": ""}).status_code == 200
        assert load_config(path).server.api_token == ""
        assert client.get("/api/settings").json()["api_token_set"] is False


def test_no_token_keeps_unauthenticated_servers_working(sandbox):
    def respond(request):
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"data": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert LocalLLM(sandbox, client).healthy()
