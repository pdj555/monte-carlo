"""Vercel Python entrypoint for the decision engine."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_bridge import create_payload  # noqa: E402


def _failure_payload(message: str) -> dict[str, object]:
    return {
        "request": {
            "job": "simulate",
            "tickers": "AAPL",
            "source": "demo",
            "data_path": None,
        },
        "sourceNote": "No run completed.",
        "eyebrow": "Needs attention",
        "title": "Run failed.",
        "summary": message,
        "notes": [],
        "metrics": [],
        "chartSvg": "",
        "chartAlt": "",
        "detailsText": message,
        "error": message,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(raw or "{}")
            if not isinstance(body, dict):
                raise ValueError("Request payload must be a JSON object.")
            payload = create_payload(body)
            status = 200
        except Exception as exc:
            payload = _failure_payload(str(exc))
            status = 500

        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)
