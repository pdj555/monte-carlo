from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "vercel_api_engine", REPO_ROOT / "api" / "engine.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _post(engine, body: bytes) -> tuple[int, dict[str, str], dict]:
    handler = engine.handler.__new__(engine.handler)
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    captured: dict[str, object] = {}
    handler.send_response = lambda status: captured.setdefault("status", status)
    handler.send_header = lambda key, value: captured.setdefault(
        "headers", {}
    ).__setitem__(key, value)
    handler.end_headers = lambda: None
    handler.do_POST()
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    return int(captured["status"]), dict(captured["headers"]), payload


def test_failure_payload_shape() -> None:
    engine = _load_engine()

    payload = engine._failure_payload("boom")

    assert payload["error"] == "boom"
    assert payload["summary"] == "boom"
    assert payload["detailsText"] == "boom"
    assert payload["title"] == "Run failed."
    assert payload["sourceNote"] == "No run completed."
    assert payload["request"]["job"] == "simulate"
    assert payload["notes"] == []
    assert payload["metrics"] == []


def test_post_valid_body_returns_bridge_payload(monkeypatch) -> None:
    engine = _load_engine()
    seen: dict[str, object] = {}

    def fake_create_payload(body):
        seen["body"] = body
        return {"title": "Done.", "error": None}

    monkeypatch.setattr(engine, "create_payload", fake_create_payload)

    status, headers, payload = _post(engine, b'{"job": "simulate", "tickers": "MSFT"}')

    assert seen["body"] == {"job": "simulate", "tickers": "MSFT"}
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert payload == {"error": None, "title": "Done."}


def test_post_non_object_json_returns_failure_payload() -> None:
    engine = _load_engine()

    status, _headers, payload = _post(engine, b'["not", "an", "object"]')

    assert status == 500
    assert payload["error"] == "Request payload must be a JSON object."
    assert payload["title"] == "Run failed."


def test_post_invalid_json_returns_failure_payload() -> None:
    engine = _load_engine()

    status, _headers, payload = _post(engine, b"{not json")

    assert status == 500
    assert payload["title"] == "Run failed."
    assert payload["error"]


def test_post_bridge_error_returns_failure_payload(monkeypatch) -> None:
    engine = _load_engine()

    def failing_create_payload(body):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(engine, "create_payload", failing_create_payload)

    status, _headers, payload = _post(engine, b'{"job": "simulate"}')

    assert status == 500
    assert payload["error"] == "engine exploded"
