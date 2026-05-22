"""OpenAI-compatible narrative summaries for simulation output.

Optional: used when the CLI is invoked with ``--ai-summary`` and
``OLLAMA_API_KEY`` or ``OPENAI_API_KEY`` is available.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_OLLAMA_BASE_URL = "https://ollama.com/v1"
DEFAULT_OLLAMA_MODEL = "gpt-oss:120b"


class OpenAIConfigurationError(RuntimeError):
    """Raised when OpenAI settings are missing or invalid."""


class OpenAIRequestError(RuntimeError):
    """Raised when an OpenAI API request fails."""


def _resolve_openai_credentials() -> tuple[str, str, str]:
    ollama_key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    api_key = ollama_key or openai_key
    if not api_key:
        raise OpenAIConfigurationError(
            "OLLAMA_API_KEY or OPENAI_API_KEY is required when --ai-summary is enabled."
        )

    if ollama_key:
        base_url = (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        )
        model = (
            os.getenv("OLLAMA_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_OLLAMA_MODEL
        )
    else:
        base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        model = os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    return api_key, base_url.rstrip("/"), model


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise OpenAIRequestError(
            f"OpenAI request failed ({exc.code}): {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OpenAIRequestError(f"OpenAI request failed: {exc.reason}") from exc

    try:
        parsed: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenAIRequestError(f"OpenAI returned invalid JSON: {exc}") from exc

    return parsed


def _extract_responses_text(response: dict[str, Any]) -> str:
    """Extract assistant text from a Responses API payload."""

    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    fragments: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())

    text = "\n".join(fragments).strip()
    if text:
        return text

    raise OpenAIRequestError(f"Unexpected OpenAI response format: {response}")


def _build_summary_payload(
    *,
    ticker: str,
    summary: pd.Series,
    simulation_model: str,
    days: int,
    scenarios: int,
) -> dict[str, object]:
    numeric_summary = {
        str(key): float(value)
        for key, value in summary.to_dict().items()
        if value is not None
    }

    return {
        "ticker": ticker,
        "simulation_model": simulation_model,
        "horizon_days": days,
        "scenarios": scenarios,
        "metrics": numeric_summary,
        "requested_output": (
            "Return Markdown with 4-8 short bullets. Include expected return, "
            "probability above/below current, downside risk (VaR/CVaR), and one "
            "clear assumptions/limitations bullet. Do not provide personalized "
            "financial advice."
        ),
    }


def generate_ai_summary(
    *,
    ticker: str,
    summary: pd.Series,
    simulation_model: str,
    days: int,
    scenarios: int,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
) -> str:
    """Generate a concise narrative summary for a single ticker simulation."""

    api_key, resolved_base_url, resolved_model = _resolve_openai_credentials()
    base_url = (base_url or resolved_base_url).rstrip("/")
    model = model or resolved_model

    instructions = (
        "You are a quantitative finance assistant. Summarize Monte Carlo "
        "simulation output in plain English for a technical user. Focus on risk, "
        "uncertainty, and what the metrics imply. Never present the output as "
        "personalized financial advice."
    )

    payload = {
        "model": model,
        "instructions": instructions,
        "input": json.dumps(
            _build_summary_payload(
                ticker=ticker,
                summary=summary,
                simulation_model=simulation_model,
                days=days,
                scenarios=scenarios,
            ),
            indent=2,
            sort_keys=True,
        ),
        "store": False,
    }

    response = _post_json(
        url=f"{base_url}/responses",
        payload=payload,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    return _extract_responses_text(response)
