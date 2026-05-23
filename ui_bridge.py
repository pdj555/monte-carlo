"""JSON bridge between the Next.js UI and the Python decision engine."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any

from ui_state import Metric, PageState, create_page_state, request_from_payload


def _metric_payload(metric: Metric) -> dict[str, str]:
    return {"label": metric.label, "value": metric.value}


def serialize_page_state(state: PageState) -> dict[str, Any]:
    return {
        "request": asdict(state.request),
        "sourceNote": state.source_note,
        "eyebrow": state.eyebrow,
        "title": state.title,
        "summary": state.summary,
        "notes": list(state.notes),
        "metrics": [_metric_payload(metric) for metric in state.metrics],
        "chartSvg": state.chart_svg or "",
        "chartAlt": state.chart_alt,
        "detailsText": state.details_text,
        "error": state.error,
    }


def create_payload(payload: dict[str, object]) -> dict[str, Any]:
    return serialize_page_state(create_page_state(request_from_payload(payload)))


def main() -> int:
    raw = sys.stdin.read().strip() or "{}"
    try:
        request_payload = json.loads(raw)
        if not isinstance(request_payload, dict):
            raise ValueError("Request payload must be a JSON object.")
        payload = create_payload(request_payload)
    except Exception as exc:
        payload = {
            "request": {
                "job": "simulate",
                "tickers": "AAPL",
                "source": "demo",
                "data_path": None,
            },
            "sourceNote": "No run completed.",
            "eyebrow": "Needs attention",
            "title": "Run failed.",
            "summary": str(exc),
            "notes": [],
            "metrics": [],
            "chartSvg": "",
            "chartAlt": "",
            "detailsText": str(exc),
            "error": str(exc),
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
